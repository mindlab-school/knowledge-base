"""Integration test for the /chat endpoint's persistence and history (section 6)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kb.agent.loop import AnswerResult
from kb.api import app as app_module
from kb.services import chat as chat_service

pytestmark = pytest.mark.integration


async def test_chat_persists_and_builds_history(
    pool: Any, whitelisted_user: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_history: list[int] = []

    async def fake_answer(
        question: str, *, history: list[Any], conversation_id: int, **kwargs: Any
    ) -> AnswerResult:
        seen_history.append(len(history))
        return AnswerResult(
            answer=f"ответ на: {question}",
            model="fake/model",
            usage={"cost": 0.001, "model": "fake/model"},
            rounds=1,
        )

    # The chat service reaches the agent loop through ``agent_loop.answer``; patch
    # that shared module attribute so the HTTP path returns the fake result.
    monkeypatch.setattr(chat_service.agent_loop, "answer", fake_answer)

    transport = httpx.ASGITransport(app=app_module.app)
    telegram_id = whitelisted_user["telegram_id"]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/chat", json={"telegram_id": telegram_id, "text": "привет"})
        second = await client.post("/chat", json={"telegram_id": telegram_id, "text": "ещё"})

    assert first.status_code == 200
    assert first.json() == {"answer": "ответ на: привет"}
    assert second.json() == {"answer": "ответ на: ещё"}

    # First call saw empty history; the second saw the first exchange (2 messages).
    assert seen_history == [0, 2]

    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM messages")
        cost = await conn.fetchval(
            "SELECT (usage->>'cost')::float FROM messages WHERE role = 'assistant' LIMIT 1"
        )
    assert count == 4  # two user + two assistant
    assert cost == 0.001
