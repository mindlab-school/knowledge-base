"""User (whitelist) repository — writing SQL for the ``users`` table.

Access is whitelist-only: every endpoint resolves a user by ``telegram_id`` and
rejects unknown ids. There is no per-user authorization beyond membership.
"""

from __future__ import annotations

from typing import Any

import asyncpg

Row = dict[str, Any]


async def get_by_telegram_id(conn: asyncpg.Connection, telegram_id: int) -> Row | None:
    """Return the user row for ``telegram_id`` or ``None`` if not whitelisted."""
    record = await conn.fetchrow(
        "SELECT id, telegram_id, name, created_at FROM users WHERE telegram_id = $1",
        telegram_id,
    )
    return dict(record) if record is not None else None


async def create(conn: asyncpg.Connection, telegram_id: int, name: str) -> Row:
    """Add (or rename) a whitelist user and return the row."""
    record = await conn.fetchrow(
        """
        INSERT INTO users (telegram_id, name) VALUES ($1, $2)
        ON CONFLICT (telegram_id) DO UPDATE SET name = EXCLUDED.name
        RETURNING id, telegram_id, name, created_at
        """,
        telegram_id,
        name,
    )
    return dict(record)


async def list_users(conn: asyncpg.Connection) -> list[Row]:
    """Return all whitelist users ordered by id."""
    records = await conn.fetch("SELECT id, telegram_id, name, created_at FROM users ORDER BY id")
    return [dict(r) for r in records]
