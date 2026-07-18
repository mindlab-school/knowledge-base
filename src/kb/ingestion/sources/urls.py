"""URL source adapter and pure URL helpers (spec section 11).

``normalize_url`` (no I/O) yields the canonical ``source_path`` for a page:
lower-cased host, no UTM params, no trailing slash, no fragment. ``UrlSource``
fetches the page and extracts readable text with trafilatura. SPAs are not
supported (no readable text -> error).
"""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from kb.ingestion.sources.base import LoadedDoc

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_URL_FETCH_TIMEOUT = 30.0


def normalize_url(url: str) -> str:
    """Return a canonical URL: no UTM params, no trailing slash, no fragment."""
    parsed = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            urlencode(query),
            "",
        )
    )


def looks_like_url(token: str) -> bool:
    """Return whether a whitespace token is an http(s) URL."""
    return bool(_URL_RE.match(token.strip()))


def message_urls(text: str) -> list[str] | None:
    """Return the URLs if the whole message is one or more URLs, else ``None``.

    A URL embedded inside ordinary prose is not treated as a link message.
    """
    tokens = text.split()
    if not tokens or not all(looks_like_url(token) for token in tokens):
        return None
    return tokens


def _extract_title(html: str) -> str | None:
    match = _TITLE_RE.search(html)
    if match is None:
        return None
    title = html_lib.unescape(match.group(1)).strip()
    return title or None


class UrlSource:
    """Load a web page and extract its readable text."""

    def __init__(
        self,
        url: str,
        *,
        client: object | None = None,
        timeout: float = _URL_FETCH_TIMEOUT,
    ) -> None:
        self._url = url.strip()
        self.source_path = normalize_url(url)
        self._client = client
        self._timeout = timeout

    async def load(self) -> LoadedDoc:
        import httpx
        import trafilatura

        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        try:
            response = await client.get(self._url)  # type: ignore[attr-defined]
            response.raise_for_status()
            html = response.text
        finally:
            if owns_client:
                await client.aclose()  # type: ignore[attr-defined]

        text = trafilatura.extract(html) or ""
        if not text.strip():
            raise ValueError(f"не удалось извлечь текст из {self._url} (SPA не поддерживаются)")
        title = _extract_title(html) or self.source_path
        return LoadedDoc(
            source_path=self.source_path,
            title=title,
            text=text,
            source_kind="url",
        )
