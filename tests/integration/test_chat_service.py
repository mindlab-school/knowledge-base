"""Integration test for the chat application-service (spec section 6).

Drives ``chat_service.answer_question`` directly — no HTTP, no OpenRouter — with
a ``FakeLLM`` so the full agent loop runs offline. Mirrors the ``/chat`` endpoint
test one layer down, proving the extracted service persists both turns and feeds
prior history back into the loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from kb.config import Settings
from kb.services import chat as chat_service
from tests.fakes import FakeLLM, content_response

pytestmark = pytest.mark.integration


async def test_answer_question_persists_and_builds_history(
    pool: Any, fake_embedder: Any, whitelisted_user: dict[str, Any]
) -> None:
    llm = FakeLLM([content_response("ответ", usage={"cost": 0.002})])
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    user_id = int(whitelisted_user["id"])

    first = await chat_service.answer_question(
        user_id, "привет", pool=pool, llm=llm, embedder=fake_embedder, settings=settings
    )
    second = await chat_service.answer_question(
        user_id, "ещё", pool=pool, llm=llm, embedder=fake_embedder, settings=settings
    )

    assert first.answer == "ответ"
    assert second.answer == "ответ"
    # Both turns land in the same active conversation.
    assert first.conversation_id == second.conversation_id
    assert second.usage["cost"] == 0.002

    # The second loop call carried the first exchange back as history: the prior
    # user turn precedes the new one. (Filtering by role is stable against the
    # assistant message the loop appends to the same list after the call.)
    second_user_turns = [m["content"] for m in llm.calls[1]["messages"] if m.get("role") == "user"]
    assert second_user_turns == ["привет", "ещё"]

    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM messages")
        cost = await conn.fetchval(
            "SELECT (usage->>'cost')::float FROM messages WHERE role = 'assistant' LIMIT 1"
        )
    assert count == 4  # two user + two assistant
    assert cost == 0.002
