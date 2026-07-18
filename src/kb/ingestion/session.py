"""Ingest-session (``/ingest``) orchestration (spec sections 0.2, 4).

While a session is open, text messages are buffered and files/URLs are ingested
as documents attached to the session. Closing the session glues the buffered
text into a single fact (topic assigned without an LLM) and reports what was
saved. An idle session (60 minutes) is closed on the user's next contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from kb.config import Settings, get_settings
from kb.db.pool import get_pool
from kb.db.repo import sessions as sessions_repo
from kb.ingestion.embeddings import Embedder, make_embedder
from kb.search import facts as facts_service

AUTOCLOSE_IDLE_MINUTES = 60

_BUFFER_JOIN = "\n\n"


@dataclass(slots=True)
class SessionCloseResult:
    """Outcome of closing an ingest session."""

    topic: str | None
    action: str  # saved | documents | empty
    documents: int


@dataclass(slots=True)
class SessionToggleResult:
    """Outcome of toggling the ingest session (``/ingest``)."""

    state: str  # opened | closed
    session_id: int
    close: SessionCloseResult | None = None


async def open_session(user_id: int, *, pool: asyncpg.Pool | None = None) -> int:
    """Open a new active session and return its id."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        return await sessions_repo.open_session(conn, user_id)


async def get_active_session(
    user_id: int, *, pool: asyncpg.Pool | None = None
) -> dict[str, Any] | None:
    """Return the user's active session, or ``None``."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        return await sessions_repo.get_active(conn, user_id)


async def append_text(session_id: int, text: str, *, pool: asyncpg.Pool | None = None) -> int:
    """Buffer a text message and return its ``seq``."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        return await sessions_repo.append_text(conn, session_id, text)


async def note_document(session_id: int, *, pool: asyncpg.Pool | None = None) -> None:
    """Mark session activity when a document is attached to it."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await sessions_repo.touch(conn, session_id)


async def _close(
    session: dict[str, Any], *, embedder: Embedder, pool: asyncpg.Pool
) -> SessionCloseResult:
    session_id = int(session["id"])
    user_id = int(session["user_id"])

    async with pool.acquire() as conn:
        buffer = await sessions_repo.buffer_texts(conn, session_id)
        documents = await sessions_repo.count_documents(conn, session_id)

    topic: str | None = None
    if buffer:
        content = _BUFFER_JOIN.join(buffer)
        topic = await facts_service.assign_topic(content, embedder=embedder, pool=pool)
        await facts_service.upsert_fact(topic, content, user_id, session_id, pool=pool)
        action = "saved"
    elif documents > 0:
        action = "documents"
    else:
        action = "empty"

    async with pool.acquire() as conn, conn.transaction():
        await sessions_repo.close(conn, session_id)

    return SessionCloseResult(topic=topic, action=action, documents=documents)


async def toggle_session(
    user_id: int,
    *,
    pool: asyncpg.Pool | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> SessionToggleResult:
    """Open the session if none is active, otherwise close the active one."""
    settings = settings or get_settings()
    pool = pool or await get_pool()
    embedder = embedder or make_embedder(settings)

    async with pool.acquire() as conn:
        active = await sessions_repo.get_active(conn, user_id)

    if active is None:
        session_id = await open_session(user_id, pool=pool)
        return SessionToggleResult(state="opened", session_id=session_id)

    result = await _close(active, embedder=embedder, pool=pool)
    return SessionToggleResult(state="closed", session_id=int(active["id"]), close=result)


async def maybe_autoclose(
    user_id: int,
    *,
    pool: asyncpg.Pool | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    idle_minutes: int = AUTOCLOSE_IDLE_MINUTES,
) -> SessionCloseResult | None:
    """Close the user's session if it has been idle beyond ``idle_minutes``."""
    settings = settings or get_settings()
    pool = pool or await get_pool()
    embedder = embedder or make_embedder(settings)

    async with pool.acquire() as conn:
        stale = await sessions_repo.stale_active(conn, user_id, idle_minutes)
    if stale is None:
        return None
    return await _close(stale, embedder=embedder, pool=pool)
