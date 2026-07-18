"""asyncpg connection pool with pgvector and JSONB codecs registered.

The pool is the only gateway to PostgreSQL. Every pooled connection registers
the pgvector codecs (so ``halfvec`` round-trips through
``pgvector.HalfVector``) and a JSONB codec (so ``attributes``/``usage`` columns
decode to Python ``dict`` instead of raw strings).
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from kb.config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup: pgvector types + JSONB <-> dict codec."""
    await register_vector(conn)
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def create_pool(dsn: str | None = None, **kwargs: Any) -> asyncpg.Pool:
    """Create a new asyncpg pool with codecs registered.

    :param dsn: Database URL; defaults to ``settings.database_url``.
    :param kwargs: Extra arguments forwarded to :func:`asyncpg.create_pool`.
    """
    settings = get_settings()
    return await asyncpg.create_pool(
        dsn or settings.database_url,
        init=_init_connection,
        **kwargs,
    )


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = await create_pool()
    return _pool


async def close_pool() -> None:
    """Close the process-wide pool if it was opened."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
