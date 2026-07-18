"""Integration tests for bitemporal facts (spec section 10)."""

from __future__ import annotations

from typing import Any

import pytest

from kb.search import facts as facts_service

pytestmark = pytest.mark.integration


async def test_update_preserves_history(pool: Any, whitelisted_user: dict[str, Any]) -> None:
    user_id = whitelisted_user["id"]
    await facts_service.upsert_fact("цены-и-пакеты", "старые цены", user_id, None, pool=pool)
    await facts_service.upsert_fact("цены-и-пакеты", "новые цены", user_id, None, pool=pool)

    active = await facts_service.list_facts(pool=pool)
    assert len(active) == 1
    assert active[0]["content"] == "новые цены"

    history = await facts_service.fact_history("цены-и-пакеты", pool=pool)
    assert len(history) == 2
    assert history[0]["content"] == "старые цены"
    assert history[0]["invalid_at"] is not None
    assert history[1]["invalid_at"] is None


async def test_delete_is_soft(pool: Any, whitelisted_user: dict[str, Any]) -> None:
    await facts_service.upsert_fact("расписание", "пн-пт", whitelisted_user["id"], None, pool=pool)
    assert await facts_service.delete_fact("расписание", pool=pool) is True
    assert await facts_service.list_facts(pool=pool) == []
    history = await facts_service.fact_history("расписание", pool=pool)
    assert len(history) == 1
    assert history[0]["invalid_at"] is not None


async def test_delete_missing_topic(pool: Any) -> None:
    assert await facts_service.delete_fact("нет-такой", pool=pool) is False
