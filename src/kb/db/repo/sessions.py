"""Ingest-session repository: the ``/ingest`` buffer.

A session collects text messages (buffered by ``seq``) and counts documents
attached during it. The buffer is never deleted after close — it is the
provenance (trace) of the fact born from the session.
"""

from __future__ import annotations

from typing import Any

import asyncpg

Row = dict[str, Any]


async def open_session(conn: asyncpg.Connection, user_id: int) -> int:
    """Open a new active session for the user and return its id."""
    record = await conn.fetchrow(
        "INSERT INTO ingest_sessions (user_id) VALUES ($1) RETURNING id", user_id
    )
    return int(record["id"])


async def get_active(conn: asyncpg.Connection, user_id: int) -> Row | None:
    """Return the user's active session, or ``None``."""
    record = await conn.fetchrow(
        """
        SELECT id, user_id, active, started_at, last_active_at, closed_at
        FROM ingest_sessions
        WHERE user_id = $1 AND active
        """,
        user_id,
    )
    return dict(record) if record is not None else None


async def get(conn: asyncpg.Connection, session_id: int) -> Row | None:
    """Return a session by id."""
    record = await conn.fetchrow(
        """
        SELECT id, user_id, active, started_at, last_active_at, closed_at
        FROM ingest_sessions WHERE id = $1
        """,
        session_id,
    )
    return dict(record) if record is not None else None


async def append_text(conn: asyncpg.Connection, session_id: int, text: str) -> int:
    """Buffer a text message; return the assigned ``seq`` (max+1)."""
    record = await conn.fetchrow(
        """
        INSERT INTO ingest_buffer (session_id, seq, text)
        VALUES (
            $1,
            (SELECT COALESCE(MAX(seq), 0) + 1 FROM ingest_buffer WHERE session_id = $1),
            $2
        )
        RETURNING seq
        """,
        session_id,
        text,
    )
    await touch(conn, session_id)
    return int(record["seq"])


async def touch(conn: asyncpg.Connection, session_id: int) -> None:
    """Mark session activity (used for the 60-minute autoclose)."""
    await conn.execute(
        "UPDATE ingest_sessions SET last_active_at = now() WHERE id = $1", session_id
    )


async def buffer_texts(conn: asyncpg.Connection, session_id: int) -> list[str]:
    """Return buffered texts in send order."""
    records = await conn.fetch(
        "SELECT text FROM ingest_buffer WHERE session_id = $1 ORDER BY seq",
        session_id,
    )
    return [r["text"] for r in records]


async def buffer_messages(conn: asyncpg.Connection, session_id: int) -> list[Row]:
    """Return buffered messages with ``seq`` (for fact tracing)."""
    records = await conn.fetch(
        "SELECT seq, text FROM ingest_buffer WHERE session_id = $1 ORDER BY seq",
        session_id,
    )
    return [dict(r) for r in records]


async def count_documents(conn: asyncpg.Connection, session_id: int) -> int:
    """Count documents attached to the session."""
    value = await conn.fetchval(
        "SELECT count(*) FROM documents WHERE ingest_session_id = $1", session_id
    )
    return int(value)


async def close(conn: asyncpg.Connection, session_id: int) -> None:
    """Close the session (keeps the buffer as provenance)."""
    await conn.execute(
        "UPDATE ingest_sessions SET active = FALSE, closed_at = now() WHERE id = $1",
        session_id,
    )


async def stale_active(conn: asyncpg.Connection, user_id: int, idle_minutes: int) -> Row | None:
    """Return the user's active session if it has been idle beyond ``idle_minutes``."""
    record = await conn.fetchrow(
        """
        SELECT id, user_id, active, started_at, last_active_at, closed_at
        FROM ingest_sessions
        WHERE user_id = $1 AND active
          AND last_active_at < now() - make_interval(mins => $2)
        """,
        user_id,
        idle_minutes,
    )
    return dict(record) if record is not None else None
