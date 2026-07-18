"""Fact repository: bitemporal business-context records.

One active fact per topic (``invalid_at IS NULL``). An update invalidates the
previous version and inserts a new one in the same transaction — history is
preserved and traceable to the ingest session that produced it.
"""

from __future__ import annotations

from typing import Any

import asyncpg

Row = dict[str, Any]


async def list_active(conn: asyncpg.Connection) -> list[Row]:
    """Return active facts (topic, content, valid_from) ordered by topic."""
    records = await conn.fetch(
        """
        SELECT topic, content, valid_from
        FROM facts WHERE invalid_at IS NULL
        ORDER BY topic
        """
    )
    return [dict(r) for r in records]


async def get_active(conn: asyncpg.Connection, topic: str) -> Row | None:
    """Return the active fact for a topic, or ``None``."""
    record = await conn.fetchrow(
        "SELECT id, topic, content, valid_from FROM facts WHERE topic = $1 AND invalid_at IS NULL",
        topic,
    )
    return dict(record) if record is not None else None


async def upsert(
    conn: asyncpg.Connection,
    topic: str,
    content: str,
    user_id: int | None,
    session_id: int | None,
) -> None:
    """Invalidate the current active version (if any) and insert a new one.

    Must run inside a transaction so the invalidate+insert are atomic and the
    ``facts_topic_active`` partial-unique index is never violated.
    """
    await conn.execute(
        "UPDATE facts SET invalid_at = now() WHERE topic = $1 AND invalid_at IS NULL",
        topic,
    )
    await conn.execute(
        """
        INSERT INTO facts (topic, content, created_by, session_id)
        VALUES ($1, $2, $3, $4)
        """,
        topic,
        content,
        user_id,
        session_id,
    )


async def soft_delete(conn: asyncpg.Connection, topic: str) -> bool:
    """Invalidate the active fact for a topic. Return whether one existed."""
    result = await conn.execute(
        "UPDATE facts SET invalid_at = now() WHERE topic = $1 AND invalid_at IS NULL",
        topic,
    )
    return int(str(result).split()[-1]) > 0


async def hard_delete(conn: asyncpg.Connection, topic: str) -> int:
    """Physically delete every version of a topic (admin ``del-fact``). Return rows."""
    result = await conn.execute("DELETE FROM facts WHERE topic = $1", topic)
    return int(str(result).split()[-1])


async def history(conn: asyncpg.Connection, topic: str) -> list[Row]:
    """Return every version of a topic ordered oldest-first."""
    records = await conn.fetch(
        """
        SELECT id, topic, content, valid_from, invalid_at, created_by, session_id
        FROM facts WHERE topic = $1
        ORDER BY valid_from ASC, id ASC
        """,
        topic,
    )
    return [dict(r) for r in records]


async def topics_with_updated(conn: asyncpg.Connection) -> list[Row]:
    """Return active topics with their update time and content length."""
    records = await conn.fetch(
        """
        SELECT topic, valid_from AS updated_at, length(content) AS length
        FROM facts WHERE invalid_at IS NULL
        ORDER BY topic
        """
    )
    return [dict(r) for r in records]
