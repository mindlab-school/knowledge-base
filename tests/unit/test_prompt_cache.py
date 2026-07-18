"""Unit tests for the facts block and the conditional prompt cache (spec section 5)."""

from __future__ import annotations

from kb.agent.loop import _should_cache
from kb.agent.prompts import (
    FACTS_HEADER,
    build_facts_block,
    build_system_content,
    estimate_tokens,
)
from kb.config import Settings


def test_estimate_tokens() -> None:
    assert estimate_tokens("a" * 400) == 100


def test_facts_block_format() -> None:
    facts = [{"topic": "цены", "content": "48000/38400"}]
    block, truncated = build_facts_block(facts, 4000)
    assert block.startswith(FACTS_HEADER)
    assert "[цены]: 48000/38400" in block
    assert not truncated


def test_facts_block_truncation() -> None:
    facts = [{"topic": "t", "content": "x" * 5000}]
    block, truncated = build_facts_block(facts, 100)
    assert len(block) == 100
    assert truncated


def test_no_facts_gives_empty_block() -> None:
    block, truncated = build_facts_block([], 4000)
    assert block == ""
    assert not truncated
    assert build_system_content("") != ""


def test_should_not_cache_small_prefix() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert not _should_cache(build_system_content(""), settings)


def test_should_cache_large_prefix() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    huge = "текст " * 5000
    assert _should_cache(huge, settings)


def test_prompt_cache_off_disables() -> None:
    settings = Settings(_env_file=None, prompt_cache="off")  # type: ignore[call-arg]
    assert not _should_cache("текст " * 5000, settings)
