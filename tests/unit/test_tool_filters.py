"""Unit tests for the agent tool ``filters`` coercion (robustness to bad LLM args)."""

from __future__ import annotations

from kb.agent.tools import _coerce_filters


def test_dict_passthrough() -> None:
    value = {"doc_type": "regulation"}
    assert _coerce_filters(value) is value


def test_json_string_parsed() -> None:
    assert _coerce_filters('{"doc_type": "faq"}') == {"doc_type": "faq"}


def test_non_dict_json_string_dropped() -> None:
    assert _coerce_filters('"regulation"') is None
    assert _coerce_filters("[1, 2]") is None


def test_garbage_string_dropped() -> None:
    assert _coerce_filters("regulation") is None
    assert _coerce_filters("") is None


def test_non_string_non_dict_dropped() -> None:
    assert _coerce_filters(None) is None
    assert _coerce_filters(42) is None
