"""Telegram bot — a thin client over the HTTP backend (spec section 7).

aiogram 3 long polling. The bot holds no mode state: it detects URL-only messages
and document attachments locally (stdlib only) and lets the backend decide the
rest — ordinary text goes to ``/ingest/message`` and falls through to ``/chat``
on 409 (no active session). Per the layering rules this module imports only
httpx, config and the Telegram SDK.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from kb.config import get_settings

logger = logging.getLogger("kb.channels.telegram")

TELEGRAM_LIMIT = 4000
CHAT_TIMEOUT = 120.0
INGEST_TIMEOUT = 300.0
TYPING_INTERVAL = 4.5
ACCESS_DENIED = "нет доступа, обратитесь к администратору"
GENERIC_ERROR = "Произошла ошибка, попробуйте позже."

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".pptx", ".html", ".htm"}
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_BLANK_LINE_RE = re.compile(r"\n\s*\n")

dp = Dispatcher()


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split a long answer into <=limit-char parts, preferring paragraph breaks."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    current = ""
    for block in _BLANK_LINE_RE.split(text):
        block = block.strip()
        if not block:
            continue
        if len(block) > limit:
            if current:
                parts.append(current)
                current = ""
            for start in range(0, len(block), limit):
                parts.append(block[start : start + limit])
            continue
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > limit:
            parts.append(current)
            current = block
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def message_urls(text: str) -> list[str] | None:
    """Return the URLs if the whole message is one or more http(s) URLs, else None."""
    tokens = text.split()
    if not tokens or not all(_URL_RE.match(token) for token in tokens):
        return None
    return tokens


def _supported_attachment(filename: str | None) -> bool:
    if not filename:
        return False
    suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
    return suffix in SUPPORTED_EXTENSIONS


class Backend:
    """httpx client for the KB Agent backend."""

    def __init__(self, base_url: str) -> None:
        secret = get_settings().backend_shared_secret
        headers = {"X-KB-Secret": secret} if secret else None
        self._client = httpx.AsyncClient(base_url=base_url, timeout=CHAT_TIMEOUT, headers=headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, telegram_id: int, text: str) -> httpx.Response:
        return await self._client.post(
            "/chat", json={"telegram_id": telegram_id, "text": text}, timeout=CHAT_TIMEOUT
        )

    async def reset(self, telegram_id: int) -> httpx.Response:
        return await self._client.post("/reset", json={"telegram_id": telegram_id})

    async def toggle_session(self, telegram_id: int) -> httpx.Response:
        return await self._client.post("/ingest/session", json={"telegram_id": telegram_id})

    async def ingest_message(self, telegram_id: int, text: str) -> httpx.Response:
        return await self._client.post(
            "/ingest/message", json={"telegram_id": telegram_id, "text": text}
        )

    async def ingest_url(self, telegram_id: int, url: str) -> httpx.Response:
        return await self._client.post(
            "/ingest/url", json={"telegram_id": telegram_id, "url": url}, timeout=INGEST_TIMEOUT
        )

    async def ingest_file(self, telegram_id: int, filename: str, data: bytes) -> httpx.Response:
        return await self._client.post(
            "/ingest/file",
            data={"telegram_id": str(telegram_id), "origin": "telegram", "filename": filename},
            files={"file": (filename, data)},
            timeout=INGEST_TIMEOUT,
        )

    async def facts(self, telegram_id: int) -> httpx.Response:
        return await self._client.get("/facts", params={"telegram_id": telegram_id})

    async def delete_fact(self, telegram_id: int, topic: str) -> httpx.Response:
        return await self._client.delete(f"/facts/{topic}", params={"telegram_id": telegram_id})


_backend: Backend | None = None


def _get_backend() -> Backend:
    assert _backend is not None, "backend client not initialised"
    return _backend


def _document_card(card: dict[str, Any]) -> str:
    line = (
        f"Тип: {card['doc_type']} · v{card['version']} · "
        f"чанков: {card['chunks']} · {card['action']}"
    )
    if card.get("extraction_status") == "failed":
        line += "\nСохранён как generic, извлечение позже (reextract)."
    return line


async def _typing_loop(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:
            return  # keep typing best-effort; stop on any transport error
        try:
            await asyncio.wait_for(stop.wait(), timeout=TYPING_INTERVAL)
        except TimeoutError:
            continue


@dp.message(Command("start"))
async def on_start(message: Message) -> None:
    await message.answer(
        "Привет! Я корпоративная база знаний. Задайте вопрос текстом.\n"
        "Команды: /reset — сброс контекста, /ingest — режим набора, "
        "/facts — темы фактов, /del <тема> — удалить факт."
    )


@dp.message(Command("reset"))
async def on_reset(message: Message) -> None:
    if message.from_user is None:
        return
    response = await _get_backend().reset(message.from_user.id)
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    await message.answer("Контекст диалога сброшен.")


@dp.message(Command("ingest"))
async def on_ingest(message: Message) -> None:
    if message.from_user is None:
        return
    response = await _get_backend().toggle_session(message.from_user.id)
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    data = response.json()
    if data.get("state") == "opened":
        await message.answer("Режим набора включён. Отправляйте текст, файлы и ссылки.")
        return
    action = data.get("action")
    documents = data.get("documents", 0)
    if action == "saved":
        await message.answer(f"Сохранено: {data.get('topic')}; документов: {documents}")
    elif action == "documents":
        await message.answer(f"Сессия закрыта; документов: {documents}")
    else:
        await message.answer("Сессия пуста, ничего не сохранено.")


@dp.message(Command("facts"))
async def on_facts(message: Message) -> None:
    if message.from_user is None:
        return
    response = await _get_backend().facts(message.from_user.id)
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    facts = response.json().get("facts", [])
    if not facts:
        await message.answer("Фактов пока нет.")
        return
    await message.answer("Темы фактов:\n" + "\n".join(f"• {fact['topic']}" for fact in facts))


@dp.message(Command("del"))
async def on_del(message: Message, command: CommandObject) -> None:
    if message.from_user is None:
        return
    topic = (command.args or "").strip()
    if not topic:
        await message.answer("Укажите тему: /del <тема>")
        return
    response = await _get_backend().delete_fact(message.from_user.id, topic)
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    if response.status_code == 404:
        await message.answer(f"Тема «{topic}» не найдена.")
        return
    await message.answer(f"Факт «{topic}» удалён.")


@dp.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    if message.from_user is None or message.document is None:
        return
    filename = message.document.file_name or "attachment"
    if not _supported_attachment(filename):
        await message.answer("Неподдерживаемый тип файла.")
        return
    buffer = io.BytesIO()
    await bot.download(message.document, destination=buffer)
    response = await _get_backend().ingest_file(message.from_user.id, filename, buffer.getvalue())
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    if response.status_code >= 400:
        await message.answer(GENERIC_ERROR)
        return
    await message.answer(_document_card(response.json()))


async def _handle_urls(message: Message, telegram_id: int, urls: list[str]) -> None:
    backend = _get_backend()
    for url in urls:
        response = await backend.ingest_url(telegram_id, url)
        if response.status_code == 403:
            await message.answer(ACCESS_DENIED)
            return
        if response.status_code >= 400:
            await message.answer(f"Не удалось загрузить {url}.")
            continue
        await message.answer(_document_card(response.json()))


async def _handle_chat(message: Message, bot: Bot, telegram_id: int, text: str) -> None:
    stop = asyncio.Event()
    typing = asyncio.create_task(_typing_loop(bot, message.chat.id, stop))
    try:
        response = await _get_backend().chat(telegram_id, text)
    finally:
        stop.set()
        await typing
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    if response.status_code >= 400:
        await message.answer(GENERIC_ERROR)
        return
    for part in split_message(response.json().get("answer", "")):
        await message.answer(part)


@dp.message(F.text)
async def on_text(message: Message, bot: Bot) -> None:
    if message.from_user is None or message.text is None:
        return
    telegram_id = message.from_user.id
    text = message.text

    urls = message_urls(text)
    if urls is not None:
        await _handle_urls(message, telegram_id, urls)
        return

    response = await _get_backend().ingest_message(telegram_id, text)
    if response.status_code == 403:
        await message.answer(ACCESS_DENIED)
        return
    if response.status_code == 409:
        await _handle_chat(message, bot, telegram_id, text)
        return
    if response.status_code >= 400:
        await message.answer(GENERIC_ERROR)
        return
    await message.answer(f"принято: {response.json().get('seq')}")


async def main() -> None:
    """Run the bot with long polling."""
    global _backend
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    _backend = Backend(settings.backend_url)
    bot = Bot(token=settings.telegram_bot_token)
    try:
        await dp.start_polling(bot)
    finally:
        await _backend.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
