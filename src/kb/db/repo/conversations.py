"""Conversation and message repository.

A user has at most one active conversation; ``/reset`` deactivates it. The agent
reads the last N messages for context and appends both turns after answering.
"""

from __future__ import annotations

from typing import Any

import asyncpg

Row = dict[str, Any]


async def get_or_create_active(conn: asyncpg.Connection, user_id: int) -> int:
    """Return the id of the user's active conversation, creating one if needed.

    Race-safe against concurrent first-messages: the insert defers to the
    ``conversations_one_active`` partial unique index, so a connection that loses
    the race reads the winner's row instead of creating a duplicate. The
    select-then-insert is retried a few times so that a ``/reset`` deactivating
    the winner in the same instant cannot leave the caller without a row.
    """
    for _ in range(3):
        record = await conn.fetchrow(
            "SELECT id FROM conversations WHERE user_id = $1 AND active ORDER BY id DESC LIMIT 1",
            user_id,
        )
        if record is not None:
            return int(record["id"])
        created = await conn.fetchrow(
            """
            INSERT INTO conversations (user_id) VALUES ($1)
            ON CONFLICT (user_id) WHERE active DO NOTHING
            RETURNING id
            """,
            user_id,
        )
        if created is not None:
            return int(created["id"])
        # Lost the race and the winner was deactivated before we could read it;
        # loop to re-select or create afresh.
    raise RuntimeError(f"could not obtain an active conversation for user {user_id}")


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


async def month_to_date_cost(conn: asyncpg.Connection) -> float:
    """Sum the OpenRouter ``cost`` field over this calendar month's messages (USD)."""
    value = await conn.fetchval(
        """
        SELECT COALESCE(SUM((usage->>'cost')::float), 0)
        FROM messages
        WHERE usage ? 'cost'
          AND created_at >= date_trunc('month', now())
        """
    )
    return float(value)
