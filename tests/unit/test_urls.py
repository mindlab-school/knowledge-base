"""Unit tests for URL normalisation and detection (spec section 11)."""

from __future__ import annotations

from kb.ingestion.sources.urls import looks_like_url, message_urls, normalize_url


def test_normalize_strips_utm_and_trailing_slash() -> None:
    url = "HTTPS://Example.com/Page/?utm_source=x&id=7&utm_medium=y#frag"
    assert normalize_url(url) == "https://example.com/Page?id=7"


def test_normalize_root_path() -> None:
    assert normalize_url("https://example.com/") == "https://example.com"


def test_normalize_is_idempotent() -> None:
    once = normalize_url("https://example.com/a/?utm_x=1")
    assert normalize_url(once) == once


def test_looks_like_url() -> None:
    assert looks_like_url("https://example.com")
    assert looks_like_url("http://x.io/y")
    assert not looks_like_url("example.com")
    assert not looks_like_url("текст")


def test_message_urls_pure_links() -> None:
    assert message_urls("https://a.com https://b.com") == ["https://a.com", "https://b.com"]
    assert message_urls("https://a.com\nhttps://b.com") == ["https://a.com", "https://b.com"]


def test_message_urls_rejects_mixed_text() -> None:
    assert message_urls("смотри https://a.com здесь") is None
    assert message_urls("обычный текст") is None
    assert message_urls("") is None
