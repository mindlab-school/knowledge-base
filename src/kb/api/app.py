"""FastAPI backend (spec sections 6, 15).

The single HTTP surface over the core. Every endpoint resolves the caller by
``telegram_id`` (whitelist) and returns 403 for unknown ids. Channels (Telegram
bot, MCP server) are thin clients over these endpoints.
"""

from __future__ import annotations

import hmac
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kb.agent import loop as agent_loop
from kb.config import get_settings
from kb.db.pool import close_pool, get_pool
from kb.db.repo import conversations as conversations_repo
from kb.db.repo import users as users_repo
from kb.ingestion.pipeline import IngestResult, ingest_source
from kb.ingestion.session import (
    append_text,
    get_active_session,
    maybe_autoclose,
    note_document,
    toggle_session,
)
from kb.ingestion.sources.files import FileSource, is_supported, telegram_file_source
from kb.ingestion.sources.urls import UrlSource
from kb.search import entities as entities_search
from kb.search import semantic, structured
from kb.services import facts as facts_service

MAX_FILE_SIZE = 20 * 1024 * 1024
FORBIDDEN_MESSAGE = "нет доступа, обратитесь к администратору"
UNAUTHENTICATED_MESSAGE = "неверный или отсутствующий ключ доступа"
SECRET_HEADER = "X-KB-Secret"

logger = logging.getLogger("kb.api.app")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not get_settings().backend_shared_secret:
        logger.warning(
            "BACKEND_SHARED_SECRET is unset: the backend HTTP surface is unauthenticated."
        )
    await get_pool()
    try:
        yield
    finally:
        await close_pool()


app = FastAPI(title="KB Agent API", lifespan=lifespan)


@app.middleware("http")
async def require_shared_secret(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject callers lacking the shared secret when one is configured.

    ``GET /health`` stays public so container health checks work without the
    secret. When ``backend_shared_secret`` is empty, auth is disabled entirely.
    """
    secret = get_settings().backend_shared_secret
    exempt = request.method == "GET" and request.url.path == "/health"
    if secret and not exempt:
        provided = request.headers.get(SECRET_HEADER, "")
        if not hmac.compare_digest(provided.encode("utf-8"), secret.encode("utf-8")):
            return JSONResponse(status_code=403, content={"detail": UNAUTHENTICATED_MESSAGE})
    return await call_next(request)


# --- request models -----------------------------------------------------


class ChatRequest(BaseModel):
    telegram_id: int
    text: str


class ResetRequest(BaseModel):
    telegram_id: int


class UrlRequest(BaseModel):
    telegram_id: int
    url: str


class SessionRequest(BaseModel):
    telegram_id: int


class MessageRequest(BaseModel):
    telegram_id: int
    text: str


class SearchRequest(BaseModel):
    telegram_id: int
    query: str
    filters: dict[str, Any] | None = None


class DocumentFilterRequest(BaseModel):
    telegram_id: int
    filters: dict[str, Any] | None = None


class EntityRequest(BaseModel):
    telegram_id: int
    name: str
    depth: int = 1


# --- helpers ------------------------------------------------------------


async def _require_user(telegram_id: int) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        user = await users_repo.get_by_telegram_id(conn, telegram_id)
    if user is None:
        raise HTTPException(status_code=403, detail=FORBIDDEN_MESSAGE)
    return user


def _card(result: IngestResult) -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "title": result.title,
        "doc_type": result.doc_type,
        "version": result.version,
        "chunks": result.chunks,
        "extraction_status": result.extraction_status,
        "action": result.action,
    }


# --- endpoints ----------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    user = await _require_user(request.telegram_id)
    pool = await get_pool()
    await maybe_autoclose(int(user["id"]), pool=pool)

    async with pool.acquire() as conn:
        conversation_id = await conversations_repo.get_or_create_active(conn, int(user["id"]))
        history = await conversations_repo.last_messages(conn, conversation_id, limit=8)

    result = await agent_loop.answer(
        request.text, history=history, conversation_id=conversation_id, pool=pool
    )

    async with pool.acquire() as conn, conn.transaction():
        await conversations_repo.add_message(conn, conversation_id, "user", request.text)
        await conversations_repo.add_message(
            conn, conversation_id, "assistant", result.answer, usage=result.usage
        )
    return {"answer": result.answer}


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, str]:
    user = await _require_user(request.telegram_id)
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conversations_repo.reset(conn, int(user["id"]))
    return {"status": "reset"}


@app.post("/ingest/file")
async def ingest_file(
    telegram_id: int = Form(...),
    file: UploadFile = File(...),
    origin: str = Form("file"),
    filename: str | None = Form(None),
) -> dict[str, Any]:
    user = await _require_user(telegram_id)
    name = filename or file.filename or "upload"
    if not is_supported(name):
        raise HTTPException(status_code=400, detail=f"неподдерживаемый тип файла: {name}")

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="файл превышает 20 MB")

    pool = await get_pool()
    await maybe_autoclose(int(user["id"]), pool=pool)
    active = await get_active_session(int(user["id"]), pool=pool)
    session_id = int(active["id"]) if active else None

    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        source = (
            telegram_file_source(tmp_path, name, int(user["id"]))
            if origin == "telegram"
            else FileSource(
                tmp_path, source_path=f"file:{user['id']}:{name}", title=Path(name).stem
            )
        )
        result = await ingest_source(source, session_id=session_id, pool=pool)
    finally:
        os.unlink(tmp_path)

    if session_id is not None:
        await note_document(session_id, pool=pool)
    return _card(result)


@app.post("/ingest/url")
async def ingest_url(request: UrlRequest) -> dict[str, Any]:
    user = await _require_user(request.telegram_id)
    pool = await get_pool()
    await maybe_autoclose(int(user["id"]), pool=pool)
    active = await get_active_session(int(user["id"]), pool=pool)
    session_id = int(active["id"]) if active else None
    result = await ingest_source(UrlSource(request.url), session_id=session_id, pool=pool)
    if session_id is not None:
        await note_document(session_id, pool=pool)
    return _card(result)


@app.post("/ingest/session")
async def ingest_session(request: SessionRequest) -> dict[str, Any]:
    user = await _require_user(request.telegram_id)
    pool = await get_pool()
    toggle = await toggle_session(int(user["id"]), pool=pool)
    if toggle.state == "opened":
        return {"state": "opened"}
    close = toggle.close
    assert close is not None
    return {
        "state": "closed",
        "topic": close.topic,
        "action": close.action,
        "documents": close.documents,
    }


@app.post("/ingest/message")
async def ingest_message(request: MessageRequest) -> dict[str, int]:
    user = await _require_user(request.telegram_id)
    pool = await get_pool()
    await maybe_autoclose(int(user["id"]), pool=pool)
    active = await get_active_session(int(user["id"]), pool=pool)
    if active is None:
        raise HTTPException(status_code=409, detail="нет активной сессии набора")
    seq = await append_text(int(active["id"]), request.text, pool=pool)
    return {"seq": seq}


@app.get("/facts")
async def get_facts(telegram_id: int) -> dict[str, Any]:
    await _require_user(telegram_id)
    pool = await get_pool()
    facts = await facts_service.list_facts(pool=pool)
    return {"facts": [{"topic": fact["topic"], "updated_at": fact["valid_from"]} for fact in facts]}


@app.delete("/facts/{topic}")
async def delete_fact(topic: str, telegram_id: int) -> dict[str, str]:
    await _require_user(telegram_id)
    pool = await get_pool()
    deleted = await facts_service.delete_fact(topic, pool=pool)
    if not deleted:
        raise HTTPException(status_code=404, detail="тема не найдена")
    return {"status": "deleted", "topic": topic}


@app.get("/documents/{document_id}")
async def get_document(document_id: int, telegram_id: int) -> dict[str, Any]:
    await _require_user(telegram_id)
    pool = await get_pool()
    card = await structured.get_document_card(document_id, pool=pool)
    if card is None:
        raise HTTPException(status_code=404, detail="документ не найден")
    return card


# --- read endpoints for the MCP plugin (section 15) ---------------------


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    await _require_user(request.telegram_id)
    pool = await get_pool()
    results = await semantic.search_chunks(request.query, request.filters, pool=pool)
    return {"results": results}


@app.post("/search/documents")
async def search_documents(request: DocumentFilterRequest) -> dict[str, Any]:
    await _require_user(request.telegram_id)
    pool = await get_pool()
    documents = await structured.query_documents(request.filters or {}, pool=pool)
    return {"documents": documents}


@app.post("/search/entity")
async def search_entity(request: EntityRequest) -> dict[str, Any]:
    await _require_user(request.telegram_id)
    pool = await get_pool()
    card = await entities_search.explore_entity(request.name, depth=request.depth, pool=pool)
    return card if card is not None else {"found": False, "name": request.name}
