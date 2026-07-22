"""Unit tests for file source path construction (spec section 11)."""

from __future__ import annotations

from kb.ingestion.sources.files import telegram_file_source


def test_telegram_source_path_namespaces_by_user() -> None:
    source = telegram_file_source("/tmp/x.pdf", "Policy.PDF", 1001)
    assert source.source_path == "tg:1001:policy.pdf"


def test_telegram_source_path_isolates_users() -> None:
    a = telegram_file_source("/tmp/x.pdf", "policy.pdf", 1001)
    b = telegram_file_source("/tmp/y.pdf", "policy.pdf", 2002)
    assert a.source_path != b.source_path
