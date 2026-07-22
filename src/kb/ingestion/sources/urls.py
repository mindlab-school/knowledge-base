"""URL source adapter and pure URL helpers (spec section 11).

``normalize_url`` (no I/O) yields the canonical ``source_path`` for a page:
lower-cased host, no UTM params, no trailing slash, no fragment. ``UrlSource``
fetches the page and extracts readable text with trafilatura. SPAs are not
supported (no readable text -> error).
"""

from __future__ import annotations

import html as html_lib
import ipaddress
import re
import socket
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from kb.ingestion.sources.base import LoadedDoc

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_URL_FETCH_TIMEOUT = 30.0

#: Reject a response whose body grows past this size (defence against zip/HTML bombs).
MAX_URL_BYTES = 10 * 1024 * 1024
_MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
#: Always blocked even though it is also link-local: the cloud metadata endpoint.
_BLOCKED_IPS = frozenset({"169.254.169.254"})


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


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether ``ip`` points at a non-routable or internal address."""
    return (
        str(ip) in _BLOCKED_IPS
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _assert_fetchable(url: str) -> None:
    """Validate ``url`` against the SSRF policy before it is fetched.

    Only ``http``/``https`` are allowed, and every address the host resolves to
    must be publicly routable. A literal-IP host is checked directly (no DNS); a
    hostname is resolved with :func:`socket.getaddrinfo` and *all* returned
    addresses are validated. Raises :class:`ValueError` on any violation.
    """
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"недопустимая схема URL {scheme!r} (разрешены только http/https): {url}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL без имени хоста: {url}")

    try:
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [
            ipaddress.ip_address(host)
        ]
    except ValueError:
        try:
            # Port is irrelevant to which addresses a host resolves to.
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise ValueError(f"не удалось разрешить хост {host!r}: {url}") from exc
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]

    for ip in addresses:
        if _is_forbidden_ip(ip):
            raise ValueError(
                f"доступ к внутреннему/зарезервированному адресу запрещён ({ip}): {url}"
            )


async def _read_capped(response: object) -> str:
    """Stream ``response`` into text, aborting if it exceeds :data:`MAX_URL_BYTES`."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():  # type: ignore[attr-defined]
        total += len(chunk)
        if total > MAX_URL_BYTES:
            raise ValueError(f"ответ превышает лимит {MAX_URL_BYTES} байт и был прерван")
        chunks.append(chunk)
    encoding = getattr(response, "encoding", None) or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


async def _fetch(client: object, url: str) -> str:
    """Fetch ``url`` with SSRF and size guards, following redirects manually.

    Auto-redirects are disabled; each hop is re-validated with
    :func:`_assert_fetchable` so a redirect cannot smuggle the request to an
    internal target. Raises :class:`ValueError` on policy violations or after
    :data:`_MAX_REDIRECTS` hops.
    """
    for _ in range(_MAX_REDIRECTS + 1):
        _assert_fetchable(url)
        async with client.stream("GET", url) as response:  # type: ignore[attr-defined]
            if response.status_code in _REDIRECT_CODES and (
                location := response.headers.get("location")
            ):
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            return await _read_capped(response)
    raise ValueError(f"слишком много перенаправлений при загрузке {url}")


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
            client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=False)
        try:
            html = await _fetch(client, self._url)
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
