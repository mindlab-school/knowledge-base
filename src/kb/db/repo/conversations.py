"""Conversation and message repository.

A user has at most one active conversation; ``/reset`` deactivates it. The agent
reads the last N messages for context and appends both turns after answering.
"""

from __future__ import annotations

from typing import Any

import asyncpg

Row = dict[str, Any]


async def get_or_create_active(conn: asyncpg.Connection, user_id: int) -> int:
    """Return the id of the user's active conversation, creating one if needed."""
    record = await conn.fetchrow(
        "SELECT id FROM conversations WHERE user_id = $1 AND active ORDER BY id DESC LIMIT 1",
        user_id,
    )
    if record is not None:
        return int(record["id"])
    created = await conn.fetchrow(
        "INSERT INTO conversations (user_id) VALUES ($1) RETURNING id", user_id
    )
    return int(created["id"])


async def reset(conn: asyncpg.Connection, user_id: int) -> None:
    """Deactivate every active conversation for the user."""
    await conn.execute(
        "UPDATE conversations SET active = FALSE WHERE user_id = $1 AND active",
        user_id,
    )


async def add_message(
    conn: asyncpg.Connection,
    conversation_id: int,
    role: str,
    content: str,
    usage: dict[str, Any] | None = None,
) -> int:
    """Append a message and return its id. ``usage`` stores the OpenRouter cost payload."""
    record = await conn.fetchrow(
        """
        INSERT INTO messages (conversation_id, role, content, usage)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        conversation_id,
        role,
        content,
        usage,
    )
    return int(record["id"])


async def last_messages(
    conn: asyncpg.Connection, conversation_id: int, limit: int = 8
) -> list[Row]:
    """Return the last ``limit`` messages in chronological order."""
    records = await conn.fetch(
        """
        SELECT role, content FROM (
            SELECT id, role, content FROM messages
            WHERE conversation_id = $1
            ORDER BY id DESC
            LIMIT $2
        ) recent
        ORDER BY id ASC
        """,
        conversation_id,
        limit,
    )
    return [dict(r) for r in records]
