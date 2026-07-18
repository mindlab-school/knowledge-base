"""Unit tests for Telegram answer splitting (spec section 7)."""

from __future__ import annotations

from kb.channels.telegram import split_message


def test_short_text_single_part() -> None:
    assert split_message("короткий ответ") == ["короткий ответ"]


def test_empty_text() -> None:
    assert split_message("   ") == []


def test_splits_on_paragraphs_within_limit() -> None:
    text = "\n\n".join(["абзац " * 30] * 4)
    parts = split_message(text, limit=200)
    assert len(parts) > 1
    assert all(len(part) <= 200 for part in parts)


def test_oversized_paragraph_hard_split() -> None:
    text = "x" * 950
    parts = split_message(text, limit=400)
    assert len(parts) == 3
    assert all(len(part) <= 400 for part in parts)
    assert "".join(parts) == text
