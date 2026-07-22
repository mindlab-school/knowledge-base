"""Chat use-case service (spec section 6).

The application-service entry point behind ``POST /chat``: it autocloses a stale
ingest session, resolves the user's active conversation, feeds recent history to
the agent loop, and persists both turns. Handlers (HTTP) and other channels
(CLI/MCP) call :func:`answer_question` instead of re-implementing this sequence.
The agent loop is reached through the module attribute ``agent_loop.answer`` so
the write path stays a single monkeypatchable seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from kb.agent import loop as agent_loop
from kb.config import Settings
from kb.db.pool import get_pool
from kb.db.repo import conversations as conversations_repo
from kb.embeddings import Embedder
from kb.ingestion.session import maybe_autoclose
from kb.llm.client import LLMClient

HISTORY_LIMIT = 8


@dataclass(slots=True)
class ChatResult:
    """Outcome of answering one user question, after both turns are persisted."""

    answer: str
    conversation_id: int
    model: str | None
    usage: dict[str, Any]
    rounds: int
    limited: bool


async def answer_question(
    user_id: int,
    text: str,
    *,
    pool: asyncpg.Pool | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> ChatResult:
    """Answer ``text`` for ``user_id`` and persist the user + assistant turns.

    ``settings``/``llm``/``embedder`` are forwarded to the agent loop; leaving
    them unset uses the configured defaults (and touches OpenRouter). Injecting
    fakes drives the full loop without any network call.
    """
    pool = pool or await get_pool()
    await maybe_autoclose(user_id, pool=pool)

    async with pool.acquire() as conn:
        conversation_id = await conversations_repo.get_or_create_active(conn, user_id)
        history = await conversations_repo.last_messages(conn, conversation_id, limit=HISTORY_LIMIT)

    result = await agent_loop.answer(
        text,
        history=history,
        conversation_id=conversation_id,
        pool=pool,
        settings=settings,
        llm=llm,
        embedder=embedder,
    )

    async with pool.acquire() as conn, conn.transaction():
        await conversations_repo.add_message(conn, conversation_id, "user", text)
        await conversations_repo.add_message(
            conn, conversation_id, "assistant", result.answer, usage=result.usage
        )

    return ChatResult(
        answer=result.answer,
        conversation_id=conversation_id,
        model=result.model,
        usage=result.usage,
        rounds=result.rounds,
        limited=result.limited,
    )
