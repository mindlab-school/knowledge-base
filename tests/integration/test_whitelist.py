"""Integration tests for whitelist enforcement (spec section 10)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kb.api.app import app

pytestmark = pytest.mark.integration

_UNKNOWN = 999999


async def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health_is_public(pool: Any) -> None:
    async with await _client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_unknown_user_forbidden_everywhere(pool: Any) -> None:
    async with await _client() as client:
        checks = [
            await client.post("/chat", json={"telegram_id": _UNKNOWN, "text": "hi"}),
            await client.post("/reset", json={"telegram_id": _UNKNOWN}),
            await client.post("/ingest/session", json={"telegram_id": _UNKNOWN}),
            await client.post("/ingest/message", json={"telegram_id": _UNKNOWN, "text": "x"}),
            await client.post("/ingest/url", json={"telegram_id": _UNKNOWN, "url": "http://x"}),
            await client.get("/facts", params={"telegram_id": _UNKNOWN}),
            await client.get("/documents/1", params={"telegram_id": _UNKNOWN}),
            await client.post("/search", json={"telegram_id": _UNKNOWN, "query": "q"}),
            await client.post("/search/documents", json={"telegram_id": _UNKNOWN}),
            await client.post("/search/entity", json={"telegram_id": _UNKNOWN, "name": "n"}),
        ]
    assert all(response.status_code == 403 for response in checks)


async def test_whitelisted_user_allowed(pool: Any, whitelisted_user: dict[str, Any]) -> None:
    async with await _client() as client:
        response = await client.get(
            "/facts", params={"telegram_id": whitelisted_user["telegram_id"]}
        )
    assert response.status_code == 200
    assert response.json() == {"facts": []}
