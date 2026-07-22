"""Agent tool schemas and dispatcher (spec section 5).

Four tools map onto the search layer. The dispatcher parses tool arguments,
calls the matching search function, and returns a JSON string for the tool
result message. Result sizes are bounded (top-4 chunks, 15 documents, 8k-char
document body) to keep the context small.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from kb.config import Settings, get_settings
from kb.ingestion.embeddings import Embedder, make_embedder
from kb.search import entities as entities_search
from kb.search import semantic, structured

logger = logging.getLogger("kb.agent")

DOCUMENT_BODY_LIMIT = 8_000
_DOCUMENT_CAP = structured.DEFAULT_DOCUMENT_LIMIT

_FILTERS_SCHEMA = {
    "type": "object",
    "description": (
        'Фильтры (AND). Примеры: {"doc_type": "regulation"}, '
        '{"review_by_before": "2026-07-18"}, {"created_after": "2026-01-01"}, '
        '{"title_contains": "отпуск"}, {"attributes": {"status": "active"}}, '
        '{"entity": {"name": "Иван Петров", "role": "owner"}}.'
    ),
    "properties": {
        "doc_type": {
            "type": "string",
            "description": "Тип: regulation|instruction|faq|meeting_notes|generic",
        },
        "title_contains": {"type": "string"},
        "created_before": {"type": "string", "description": "ISO-дата"},
        "created_after": {"type": "string", "description": "ISO-дата"},
        "review_by_before": {"type": "string", "description": "ISO-дата (атрибут review_by)"},
        "review_by_after": {"type": "string"},
        "attributes": {
            "type": "object",
            "description": "Точное совпадение по атрибутам (JSONB @>)",
        },
        "entity": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "role": {"type": "string"}},
        },
    },
    "additionalProperties": True,
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Семантический+полнотекстовый поиск фрагментов по содержанию. "
                "Возвращает до 4 фрагментов с источниками."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "filters": _FILTERS_SCHEMA,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_documents",
            "description": (
                "Структурированный поиск документов по метаданным (тип, атрибуты, даты, "
                "сущности). Для вопросов про наборы, статусы, просроченные регламенты."
            ),
            "parameters": {
                "type": "object",
                "properties": {"filters": _FILTERS_SCHEMA},
                "required": ["filters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_entity",
            "description": (
                "Карточка сущности (человек, отдел, продукт, процесс, проект): документы с "
                "ролями и граф связей. depth 0–3 (дефолт 1); depth=2 для вопросов о связях."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Полный текст документа по id (усечение до 8000 символов).",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "integer"}},
                "required": ["document_id"],
            },
        },
    },
]

TOOL_NAMES = frozenset(tool["function"]["name"] for tool in TOOLS)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _coerce_filters(value: Any) -> dict[str, Any] | None:
    """Normalise a ``filters`` tool argument to a dict.

    The model occasionally sends ``filters`` as a JSON-encoded string instead of an
    object; parse it when possible, and drop anything that is not a dict so the
    search layer never receives a malformed value.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_int(value: Any, default: int) -> int:
    """Coerce a numeric tool argument to an ``int``, falling back to ``default``.

    The model sometimes sends numbers as JSON strings, ``null``, or non-numeric
    text; mirror :func:`_coerce_filters` by never letting a bad value raise.
    ``bool`` is rejected so a stray ``true``/``false`` does not become ``1``/``0``.
    """
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


async def _search_knowledge_base(
    args: dict[str, Any], *, pool: asyncpg.Pool, embedder: Embedder, settings: Settings
) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return _json({"error": "пустой запрос"})
    results = await semantic.search_chunks(
        query, _coerce_filters(args.get("filters")), pool=pool, embedder=embedder, settings=settings
    )
    return _json(results)


async def _query_documents(args: dict[str, Any], *, pool: asyncpg.Pool) -> str:
    filters = _coerce_filters(args.get("filters")) or {}
    documents = await structured.query_documents(filters, limit=_DOCUMENT_CAP, pool=pool)
    # A COUNT is only needed when the page is full: only then can the true total
    # exceed what was returned. A short page is already the whole result set.
    if len(documents) == _DOCUMENT_CAP:
        total = await structured.count_documents(filters, pool=pool)
    else:
        total = len(documents)
    payload: dict[str, Any] = {"documents": documents, "applied_filters": filters}
    if total > _DOCUMENT_CAP:
        payload["total"] = total
        payload["note"] = "Слишком много результатов — уточните фильтры."
    return _json(payload)


async def _explore_entity(args: dict[str, Any], *, pool: asyncpg.Pool) -> str:
    name = str(args.get("name", "")).strip()
    if not name:
        return _json({"error": "не указано имя сущности"})
    depth = _coerce_int(args.get("depth"), 1)
    card = await entities_search.explore_entity(name, depth=depth, pool=pool)
    if card is None:
        return _json({"found": False, "name": name})
    return _json(card)


async def _get_document(args: dict[str, Any], *, pool: asyncpg.Pool) -> str:
    document_id = _coerce_int(args.get("document_id"), 0)
    if document_id <= 0:
        return _json({"found": False, "document_id": document_id})
    document = await structured.get_document(document_id, pool=pool)
    if document is None:
        return _json({"found": False, "document_id": document_id})
    raw = document["raw_content"]
    truncated = len(raw) > DOCUMENT_BODY_LIMIT
    body = raw[:DOCUMENT_BODY_LIMIT] + "\n…[усечено]" if truncated else raw
    return _json(
        {
            "id": document["id"],
            "title": document["title"],
            "doc_type": document["doc_type"],
            "version": document["version"],
            "content": body,
            "truncated": truncated,
        }
    )


async def dispatch(
    name: str,
    arguments: str | dict[str, Any],
    *,
    pool: asyncpg.Pool,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> str:
    """Execute a tool call and return its JSON string result."""
    settings = settings or get_settings()
    embedder = embedder or make_embedder(settings)

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return _json({"error": "не удалось разобрать аргументы инструмента"})
    else:
        args = arguments
    if not isinstance(args, dict):
        args = {}

    try:
        if name == "search_knowledge_base":
            return await _search_knowledge_base(
                args, pool=pool, embedder=embedder, settings=settings
            )
        if name == "query_documents":
            return await _query_documents(args, pool=pool)
        if name == "explore_entity":
            return await _explore_entity(args, pool=pool)
        if name == "get_document":
            return await _get_document(args, pool=pool)
    except Exception as exc:
        # Resilience boundary: a failing tool must not crash the answer loop —
        # return an error result the model can recover from, like the parse case.
        logger.exception("tool %s failed", name)
        return _json({"error": f"ошибка инструмента: {exc}"})
    return _json({"error": f"неизвестный инструмент: {name}"})
