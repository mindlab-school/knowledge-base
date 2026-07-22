"""Integration tests for DB integrity guards and startup checks (unit U4).

Covers the one-active-conversation invariant (partial unique index + race-safe
``get_or_create_active``), the cost-query partial index, and the embedding
dimension boot guard in :mod:`kb.db.pool`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kb.db.pool import _check_embed_dim
from kb.db.repo import conversations as conversations_repo

pytestmark = pytest.mark.integration


async def test_concurrent_get_or_create_active_yields_one_row(
    pool: Any, whitelisted_user: dict[str, Any]
) -> None:
    """Two first-messages racing on separate connections converge on one row."""
    user_id = whitelisted_user["id"]

    async def call() -> int:
        async with pool.acquire() as conn:
            return await conversations_repo.get_or_create_active(conn, user_id)

    first, second = await asyncio.gather(call(), call())

    assert first == second
    async with pool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT count(*) FROM conversations WHERE user_id = $1 AND active", user_id
        )
        total = await conn.fetchval(
            "SELECT count(*) FROM conversations WHERE user_id = $1", user_id
        )
    assert active == 1
    assert total == 1


async def test_get_or_create_active_is_idempotent(
    pool: Any, whitelisted_user: dict[str, Any]
) -> None:
    """Sequential calls return the same active conversation."""
    user_id = whitelisted_user["id"]
    async with pool.acquire() as conn:
        first = await conversations_repo.get_or_create_active(conn, user_id)
        second = await conversations_repo.get_or_create_active(conn, user_id)
    assert first == second


async def test_integrity_indexes_exist(pool: Any) -> None:
    """The new partial indexes are present after migration."""
    async with pool.acquire() as conn:
        names = {
            row["indexname"]
            for row in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE indexname = ANY($1::text[])",
                ["conversations_one_active", "messages_cost_idx"],
            )
        }
    assert names == {"conversations_one_active", "messages_cost_idx"}


async def test_embedding_column_typmod_is_the_dimension(pool: Any) -> None:
    """pgvector stores the halfvec dimension directly in ``atttypmod`` (no +4)."""
    async with pool.acquire() as conn:
        typmod = await conn.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
        )
    assert typmod == 1024


async def test_embed_dim_guard_passes_on_matching_schema(pool: Any) -> None:
    """The guard accepts the deployed 1024-dim schema."""
    await _check_embed_dim(pool, 1024)


async def test_embed_dim_guard_raises_on_mismatch(pool: Any) -> None:
    """A configured dimension that disagrees with the schema fails fast."""
    with pytest.raises(RuntimeError, match="embedding dimension"):
        await _check_embed_dim(pool, 999)
