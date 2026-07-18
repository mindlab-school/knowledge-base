"""Integration tests for ingest sessions and traceability (spec section 10)."""

from __future__ import annotations

from typing import Any

import pytest

from kb.db.repo import facts as facts_repo
from kb.db.repo import sessions as sessions_repo
from kb.ingestion import session as session_service

pytestmark = pytest.mark.integration


async def test_session_glues_buffer_into_fact(
    pool: Any, fake_embedder: Any, whitelisted_user: dict[str, Any]
) -> None:
    user_id = whitelisted_user["id"]
    opened = await session_service.toggle_session(user_id, pool=pool, embedder=fake_embedder)
    assert opened.state == "opened"
    session_id = opened.session_id

    for text in ["первое сообщение", "второе сообщение", "третье сообщение"]:
        await session_service.append_text(session_id, text, pool=pool)

    closed = await session_service.toggle_session(user_id, pool=pool, embedder=fake_embedder)
    assert closed.state == "closed"
    assert closed.close is not None
    assert closed.close.action == "saved"
    topic = closed.close.topic
    assert topic is not None

    async with pool.acquire() as conn:
        fact = await facts_repo.get_active(conn, topic)
        buffer = await sessions_repo.buffer_messages(conn, session_id)
    assert fact is not None
    assert "первое сообщение" in fact["content"]
    assert "третье сообщение" in fact["content"]
    # Buffer preserved as provenance; fact carries the session id.
    assert [row["text"] for row in buffer] == [
        "первое сообщение",
        "второе сообщение",
        "третье сообщение",
    ]
    async with pool.acquire() as conn:
        session_of_fact = await conn.fetchval(
            "SELECT session_id FROM facts WHERE topic = $1 AND invalid_at IS NULL", topic
        )
    assert session_of_fact == session_id


async def test_empty_session_saves_nothing(
    pool: Any, fake_embedder: Any, whitelisted_user: dict[str, Any]
) -> None:
    user_id = whitelisted_user["id"]
    await session_service.toggle_session(user_id, pool=pool, embedder=fake_embedder)
    closed = await session_service.toggle_session(user_id, pool=pool, embedder=fake_embedder)
    assert closed.state == "closed"
    assert closed.close is not None
    assert closed.close.action == "empty"
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM facts") == 0
    assert await session_service.get_active_session(user_id, pool=pool) is None


async def test_second_ingest_closes_not_opens(
    pool: Any, fake_embedder: Any, whitelisted_user: dict[str, Any]
) -> None:
    user_id = whitelisted_user["id"]
    first = await session_service.toggle_session(user_id, pool=pool, embedder=fake_embedder)
    second = await session_service.toggle_session(user_id, pool=pool, embedder=fake_embedder)
    assert first.state == "opened"
    assert second.state == "closed"
