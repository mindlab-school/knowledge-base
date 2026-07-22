"""asyncpg connection pool with pgvector and JSONB codecs registered.

The pool is the only gateway to PostgreSQL. Every pooled connection registers
the pgvector codecs (so ``halfvec`` round-trips through
``pgvector.HalfVector``) and a JSONB codec (so ``attributes``/``usage`` columns
decode to Python ``dict`` instead of raw strings).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from kb.config import get_settings

logger = logging.getLogger("kb.db.pool")

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


async def _check_embed_dim(pool: asyncpg.Pool, expected_dim: int) -> None:
    """Fail fast if the deployed ``chunks.embedding`` dimension has drifted.

    pgvector stores the declared dimension of a ``halfvec(N)`` column directly in
    ``pg_attribute.atttypmod`` (no ``+4`` offset, unlike ``varchar``), so a
    mismatch with ``settings.embed_dim`` means the schema and configuration
    disagree and every write/search would fail at runtime — a reindex is needed.

    A fresh database whose ``chunks`` table does not exist yet is not an error:
    migrations simply have not run, so log a warning and skip.

    :param pool: Pool to query the catalog through.
    :param expected_dim: Dimension the application is configured for.
    :raises RuntimeError: If the column exists with a different dimension.
    """
    async with pool.acquire() as conn:
        typmod = await conn.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
        )
    if typmod is None:
        logger.warning(
            "chunks.embedding not found; skipping embedding-dimension check "
            "(run migrations before serving)"
        )
        return
    if typmod != expected_dim:
        raise RuntimeError(
            f"embedding dimension mismatch: chunks.embedding is halfvec({typmod}) "
            f"but settings.embed_dim is {expected_dim}; a reindex is required"
        )


async def create_pool(dsn: str | None = None, **kwargs: Any) -> asyncpg.Pool:
    """Create a new asyncpg pool with codecs registered.

    :param dsn: Database URL; defaults to ``settings.database_url``.
    :param kwargs: Extra arguments forwarded to :func:`asyncpg.create_pool`.
    """
    settings = get_settings()
    pool = await asyncpg.create_pool(
        dsn or settings.database_url,
        init=_init_connection,
        **kwargs,
    )
    await _check_embed_dim(pool, settings.embed_dim)
    return pool


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
