"""Unit tests for LLM extraction handling (spec section 3)."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from kb.config import Settings
from kb.ingestion.extraction import extract_structure, reset_structured_support
from tests.fakes import FakeLLM, content_response, tool_response


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_structured_support()


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _bad_request() -> openai.BadRequestError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError("structured outputs unsupported", response=response, body=None)


REGULATION = json.dumps(
    {
        "doc_type": "regulation",
        "confidence": 0.95,
        "attributes": {"owner": "Иван Петров", "status": "active"},
        "entities": [{"entity_type": "person", "name": "Иван Петров", "role": "owner"}],
        "relations": [],
        "links": [],
    }
)


async def test_successful_extraction() -> None:
    result = await extract_structure(
        "текст регламента", llm=FakeLLM([content_response(REGULATION)]), settings=_settings()
    )
    assert result.doc_type == "regulation"
    assert result.status == "done"
    assert result.attributes["owner"] == "Иван Петров"
    assert result.entities[0].name == "Иван Петров"


async def test_low_confidence_downgrades_to_generic() -> None:
    payload = json.dumps(
        {
            "doc_type": "regulation",
            "confidence": 0.3,
            "attributes": {"owner": "X"},
            "entities": [{"entity_type": "person", "name": "X"}],
            "relations": [],
            "links": [],
        }
    )
    result = await extract_structure(
        "текст", llm=FakeLLM([content_response(payload)]), settings=_settings()
    )
    assert result.doc_type == "generic"
    assert result.attributes == {}
    assert result.status == "done"
    assert len(result.entities) == 1  # entities are preserved


async def test_retry_then_success() -> None:
    llm = FakeLLM([content_response("не json"), content_response(REGULATION)])
    result = await extract_structure("текст", llm=llm, settings=_settings())
    assert result.status == "done"
    assert result.doc_type == "regulation"
    assert len(llm.calls) == 2


async def test_two_failures_yield_generic_failed() -> None:
    llm = FakeLLM([content_response("мусор"), content_response("тоже мусор")])
    result = await extract_structure("текст", llm=llm, settings=_settings())
    assert result.doc_type == "generic"
    assert result.status == "failed"
    assert result.attributes == {}


async def test_forced_tool_fallback() -> None:
    llm = FakeLLM([_bad_request(), tool_response("emit_extraction", REGULATION)])
    result = await extract_structure("текст", llm=llm, settings=_settings())
    assert result.doc_type == "regulation"
    assert result.status == "done"
