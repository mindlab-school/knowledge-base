"""Unit tests for agent read/answer-path resilience (loop + tool dispatch).

These cover the containment boundaries added so a transient LLM error, a tool
that raises mid-execution, or malformed numeric tool arguments degrade to a
recoverable result instead of crashing ``/chat``.
"""

from __future__ import annotations

import json
from typing import Any

import openai
import pytest

from kb.agent import loop, tools
from kb.config import Settings
from tests.fakes import FakeLLM, HashingEmbedder


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


class _FakeAcquire:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakePool:
    """Minimal pool whose ``acquire`` yields no real connection."""

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()


class _ExplodingPool:
    """Pool that fails if the tool ever reaches the database."""

    def acquire(self) -> Any:
        raise AssertionError("tool must not touch the database")


# --- (c) LLM error containment in loop.answer ---------------------------------


async def test_answer_contains_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _zero_cost(conn: object) -> float:
        return 0.0

    async def _no_facts(*, pool: object) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(loop.conversations_repo, "month_to_date_cost", _zero_cost)
    monkeypatch.setattr(loop.facts_service, "list_facts", _no_facts)

    llm = FakeLLM([openai.OpenAIError("transient upstream error")])
    result = await loop.answer(
        "вопрос",
        conversation_id=1,
        pool=_FakePool(),  # type: ignore[arg-type]
        settings=_settings(),
        llm=llm,  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
    )

    assert result.answer == loop.LLM_ERROR_MESSAGE
    assert result.model is None
    assert result.rounds == 1
    assert result.usage["cost"] == 0.0


# --- (b) numeric coercion helper ---------------------------------------------


def test_coerce_int_accepts_numeric_forms() -> None:
    assert tools._coerce_int(5, 0) == 5
    assert tools._coerce_int(2.9, 0) == 2
    assert tools._coerce_int("7", 0) == 7
    assert tools._coerce_int(" 3 ", 0) == 3


def test_coerce_int_falls_back_on_bad_values() -> None:
    assert tools._coerce_int("abc", 1) == 1
    assert tools._coerce_int("5.0", 1) == 1
    assert tools._coerce_int(None, 1) == 1
    assert tools._coerce_int([], 1) == 1
    assert tools._coerce_int(True, 1) == 1
    assert tools._coerce_int(float("inf"), 1) == 1
    assert tools._coerce_int(float("nan"), 1) == 1


async def test_dispatch_get_document_bad_id_is_not_found() -> None:
    result = await tools.dispatch(
        "get_document",
        '{"document_id": "abc"}',
        pool=_ExplodingPool(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        settings=_settings(),
    )
    assert json.loads(result) == {"found": False, "document_id": 0}


async def test_dispatch_explore_entity_bad_depth_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def _spy(name: str, *, depth: int, pool: object) -> None:
        seen["depth"] = depth
        return None

    monkeypatch.setattr(tools.entities_search, "explore_entity", _spy)

    result = await tools.dispatch(
        "explore_entity",
        '{"name": "Иван", "depth": "x"}',
        pool=object(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        settings=_settings(),
    )

    assert seen["depth"] == 1
    assert json.loads(result) == {"found": False, "name": "Иван"}


# --- (a) tool-execution error containment ------------------------------------


async def test_dispatch_contains_tool_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise RuntimeError("db exploded")

    monkeypatch.setattr(tools.structured, "query_documents", _boom)

    result = await tools.dispatch(
        "query_documents",
        '{"filters": {"doc_type": "faq"}}',
        pool=object(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        settings=_settings(),
    )
    payload = json.loads(result)
    assert "error" in payload
    assert "db exploded" in payload["error"]


# --- (d) redundant COUNT skip -------------------------------------------------


async def test_query_documents_skips_count_below_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count_calls: list[Any] = []

    async def _docs(filters: dict[str, Any], *, limit: int, pool: object) -> list[dict[str, Any]]:
        return [{"id": 1}, {"id": 2}]

    async def _count(filters: dict[str, Any], *, pool: object) -> int:
        count_calls.append(filters)
        return 999

    monkeypatch.setattr(tools.structured, "query_documents", _docs)
    monkeypatch.setattr(tools.structured, "count_documents", _count)

    result = await tools.dispatch(
        "query_documents",
        '{"filters": {}}',
        pool=object(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        settings=_settings(),
    )
    payload = json.loads(result)
    assert count_calls == []
    assert "total" not in payload
    assert "note" not in payload
    assert len(payload["documents"]) == 2


async def test_query_documents_counts_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    documents = [{"id": i} for i in range(tools._DOCUMENT_CAP)]
    count_calls: list[Any] = []

    async def _docs(filters: dict[str, Any], *, limit: int, pool: object) -> list[dict[str, Any]]:
        return documents

    async def _count(filters: dict[str, Any], *, pool: object) -> int:
        count_calls.append(filters)
        return tools._DOCUMENT_CAP + 5

    monkeypatch.setattr(tools.structured, "query_documents", _docs)
    monkeypatch.setattr(tools.structured, "count_documents", _count)

    result = await tools.dispatch(
        "query_documents",
        '{"filters": {}}',
        pool=object(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        settings=_settings(),
    )
    payload = json.loads(result)
    assert len(count_calls) == 1
    assert payload["total"] == tools._DOCUMENT_CAP + 5
    assert "note" in payload
