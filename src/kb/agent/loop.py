"""Agent tool-use loop (spec sections 5, 14).

Runs the Chat Completions tool-use cycle until the model stops requesting tools
(or ``MAX_TOOL_ROUNDS`` is reached, after which one tool-less call forces an
answer). Applies the conditional prompt cache, sticky ``session_id`` routing,
accumulates usage (incl. cache fields) for cost logging, and enforces the soft
monthly spend limit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg
import openai

from kb.agent import tools
from kb.agent.prompts import (
    build_facts_block,
    build_system_content,
    estimate_tokens,
)
from kb.agent.tools import TOOLS
from kb.config import Settings, get_settings
from kb.db.pool import get_pool
from kb.db.repo import conversations as conversations_repo
from kb.embeddings import Embedder, make_embedder
from kb.llm.client import LLMClient, response_usage
from kb.services import facts as facts_service

logger = logging.getLogger("kb.agent")

MAX_TOOL_ROUNDS = 5
MAX_TOKENS = 1200
LIMIT_MESSAGE = "Лимит расходов исчерпан, обратитесь к администратору."
NO_ANSWER = "в базе знаний этого нет"
LLM_ERROR_MESSAGE = "Сервис временно недоступен, попробуйте позже."


@dataclass(slots=True)
class AnswerResult:
    """Result of answering one question."""

    answer: str
    model: str | None
    usage: dict[str, Any]
    rounds: int
    limited: bool = False


def _empty_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
    }


def _accumulate_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    total["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    total["total_tokens"] += int(usage.get("total_tokens") or 0)
    total["cost"] += float(usage.get("cost") or 0.0)
    details = usage.get("prompt_tokens_details") or {}
    total["cached_tokens"] += int(details.get("cached_tokens") or 0)
    total["cache_write_tokens"] += int(
        usage.get("cache_write_tokens") or usage.get("cache_creation_input_tokens") or 0
    )


def _log_usage(model: str | None, usage: dict[str, Any]) -> None:
    details = usage.get("prompt_tokens_details") or {}
    logger.info(
        "agent call model=%s cached_tokens=%s cache_write=%s cost=%s",
        model,
        details.get("cached_tokens"),
        usage.get("cache_write_tokens") or usage.get("cache_creation_input_tokens"),
        usage.get("cost"),
    )


def _llm_error_result(usage: dict[str, Any], model: str | None, rounds: int) -> AnswerResult:
    """Degrade a failed LLM call to a friendly answer, keeping usage-so-far."""
    usage["model"] = model
    return AnswerResult(answer=LLM_ERROR_MESSAGE, model=model, usage=usage, rounds=rounds)


def _system_message(text: str, use_cache: bool) -> dict[str, Any]:
    """Build the system message, caching its (static) content block when enabled."""
    if use_cache:
        return {
            "role": "system",
            "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}],
        }
    return {"role": "system", "content": text}


def _assistant_message(message: Any) -> dict[str, Any]:
    """Reconstruct an assistant message (with tool calls) for the next round."""
    data: dict[str, Any] = {"role": "assistant", "content": message.content}
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for call in (message.tool_calls or [])
        if call.type == "function"
    ]
    if tool_calls:
        data["tool_calls"] = tool_calls
    return data


def _should_cache(system_text: str, settings: Settings) -> bool:
    if settings.prompt_cache != "auto":
        return False
    prefix = system_text + json.dumps(TOOLS, ensure_ascii=False)
    return estimate_tokens(prefix) >= settings.cache_min_tokens


async def answer(
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
    conversation_id: int,
    pool: asyncpg.Pool | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> AnswerResult:
    """Answer a question via the tool-use loop, grounded in the knowledge base."""
    settings = settings or get_settings()
    pool = pool or await get_pool()
    llm = llm or LLMClient(settings)
    embedder = embedder or make_embedder(settings)
    history = history or []

    async with pool.acquire() as conn:
        spent = await conversations_repo.month_to_date_cost(conn)
    if spent >= settings.or_monthly_limit:
        logger.warning("monthly spend %.4f >= limit %.2f", spent, settings.or_monthly_limit)
        return AnswerResult(
            answer=LIMIT_MESSAGE, model=None, usage=_empty_usage(), rounds=0, limited=True
        )

    facts = await facts_service.list_facts(pool=pool)
    facts_block, truncated = build_facts_block(facts, settings.facts_block_limit)
    if truncated:
        logger.warning("facts block truncated to %d chars", settings.facts_block_limit)
    system_text = build_system_content(facts_block)

    messages: list[dict[str, Any]] = [
        _system_message(system_text, _should_cache(system_text, settings)),
        *history,
        {"role": "user", "content": question},
    ]

    reasoning = {"effort": settings.agent_effort} if settings.agent_effort else None
    session_id = str(conversation_id)
    total_usage = _empty_usage()
    model_used: str | None = None
    final_answer = ""
    rounds = 0

    for _ in range(MAX_TOOL_ROUNDS):
        rounds += 1
        try:
            response = await llm.chat(
                model=settings.agent_model_slug,
                messages=messages,
                tools=TOOLS,
                max_tokens=MAX_TOKENS,
                session_id=session_id,
                reasoning=reasoning,
            )
        except openai.OpenAIError:
            logger.exception("agent LLM call failed")
            return _llm_error_result(total_usage, model_used, rounds)
        model_used = response.model
        usage = response_usage(response)
        _accumulate_usage(total_usage, usage)
        _log_usage(model_used, usage)

        message = response.choices[0].message
        messages.append(_assistant_message(message))

        if message.tool_calls:
            for call in message.tool_calls:
                if call.type != "function":
                    continue
                result = await tools.dispatch(
                    call.function.name,
                    call.function.arguments,
                    pool=pool,
                    embedder=embedder,
                    settings=settings,
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            continue

        final_answer = message.content or ""
        break
    else:
        # Rounds exhausted with tool calls still pending — force a text answer.
        rounds += 1
        try:
            response = await llm.chat(
                model=settings.agent_model_slug,
                messages=messages,
                max_tokens=MAX_TOKENS,
                session_id=session_id,
                reasoning=reasoning,
            )
        except openai.OpenAIError:
            logger.exception("agent LLM call failed")
            return _llm_error_result(total_usage, model_used, rounds)
        model_used = response.model
        usage = response_usage(response)
        _accumulate_usage(total_usage, usage)
        _log_usage(model_used, usage)
        final_answer = response.choices[0].message.content or ""

    if not final_answer.strip():
        final_answer = NO_ANSWER
    total_usage["model"] = model_used
    return AnswerResult(answer=final_answer, model=model_used, usage=total_usage, rounds=rounds)
