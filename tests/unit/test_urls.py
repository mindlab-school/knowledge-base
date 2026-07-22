"""Unit tests for URL normalisation, detection and fetch hardening (spec section 11)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from kb.ingestion.sources import urls
from kb.ingestion.sources.urls import (
    UrlSource,
    _assert_fetchable,
    looks_like_url,
    message_urls,
    normalize_url,
)

_ARTICLE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>Политика отпусков</title></head>
<body>
<header><nav>Меню компании</nav></header>
<article>
<h1>Политика отпусков</h1>
<p>Каждый сотрудник имеет право на 28 календарных дней ежегодного оплачиваемого
отпуска. Заявление подаётся не менее чем за две недели до предполагаемой даты.</p>
<p>Неиспользованные дни отпуска переносятся на следующий год, но не более чем на
14 дней. Компенсация выплачивается только при увольнении сотрудника.</p>
<p>График отпусков утверждается руководителем отдела в декабре на следующий год и
является обязательным для всех сотрудников подразделения.</p>
</article>
<footer>© Компания 2026</footer>
</body>
</html>"""


class _FakeStreamResponse:
    """Duck-typed stand-in for a streamed ``httpx.Response``."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        encoding: str = "utf-8",
        chunk_size: int = 64 * 1024,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = encoding
        self._body = body
        self._chunk_size = chunk_size

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected error status {self.status_code}")

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for start in range(0, max(len(self._body), 1), self._chunk_size):
            yield self._body[start : start + self._chunk_size]


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeStreamClient:
    """Injectable ``httpx.AsyncClient`` double keyed on request URL."""

    def __init__(
        self,
        response: _FakeStreamResponse | None = None,
        *,
        by_url: dict[str, _FakeStreamResponse] | None = None,
    ) -> None:
        self._default = response
        self._by_url = by_url or {}
        self.requests: list[str] = []

    def stream(self, method: str, url: str) -> _FakeStreamContext:
        self.requests.append(url)
        response = self._by_url.get(url, self._default)
        assert response is not None, f"no fake response registered for {url}"
        return _FakeStreamContext(response)

    async def aclose(self) -> None:
        return None


# --- normalisation / detection ------------------------------------------------


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


# --- _assert_fetchable (SSRF guard, no DNS) -----------------------------------


def test_metadata_endpoint_rejected() -> None:
    with pytest.raises(ValueError, match=r"169\.254\.169\.254"):
        _assert_fetchable("http://169.254.169.254/latest/meta-data")


def test_loopback_literal_rejected() -> None:
    with pytest.raises(ValueError):
        _assert_fetchable("http://127.0.0.1/")


def test_private_literal_rejected() -> None:
    with pytest.raises(ValueError):
        _assert_fetchable("http://10.0.0.1/")


def test_ipv6_loopback_rejected() -> None:
    with pytest.raises(ValueError):
        _assert_fetchable("http://[::1]/")


def test_non_http_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="схем"):
        _assert_fetchable("ftp://example.com/x")
    with pytest.raises(ValueError, match="схем"):
        _assert_fetchable("file:///etc/passwd")


def test_public_literal_ip_allowed() -> None:
    # Public address validates without any DNS lookup.
    assert _assert_fetchable("http://93.184.216.34/") is None


# --- UrlSource.load fetch hardening -------------------------------------------


async def test_load_rejects_private_target() -> None:
    source = UrlSource("http://127.0.0.1/", client=FakeStreamClient())
    with pytest.raises(ValueError):
        await source.load()


async def test_load_rejects_oversize_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urls, "MAX_URL_BYTES", 16)
    client = FakeStreamClient(_FakeStreamResponse(body=b"x" * 64))
    source = UrlSource("http://93.184.216.34/", client=client)
    with pytest.raises(ValueError, match="лимит"):
        await source.load()


async def test_load_redirect_to_internal_rejected() -> None:
    client = FakeStreamClient(
        by_url={
            "http://93.184.216.34/": _FakeStreamResponse(
                status_code=302, headers={"location": "http://169.254.169.254/"}
            ),
        }
    )
    source = UrlSource("http://93.184.216.34/", client=client)
    with pytest.raises(ValueError, match=r"169\.254\.169\.254"):
        await source.load()


async def test_load_happy_path_literal_ip() -> None:
    client = FakeStreamClient(_FakeStreamResponse(body=_ARTICLE_HTML.encode("utf-8")))
    source = UrlSource("http://93.184.216.34/", client=client)
    doc = await source.load()
    assert doc.source_kind == "url"
    assert doc.title == "Политика отпусков"
    assert "28 календарных дней" in doc.text
    assert client.requests == ["http://93.184.216.34/"]
