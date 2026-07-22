"""Integration tests for the shared-secret backend gate (spec section 6).

The middleware reads ``settings.backend_shared_secret`` per request, so each test
overrides ``kb.api.app.get_settings`` — the name the middleware resolves — rather
than touching the lru-cached singleton other tests rely on.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kb.api import app as app_module
from kb.config import Settings

pytestmark = pytest.mark.integration

_SECRET = "s3cr3t-token"


async def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _with_secret(secret: str) -> Any:
    def _fake_get_settings() -> Settings:
        return Settings(_env_file=None, backend_shared_secret=secret)  # type: ignore[call-arg]

    return _fake_get_settings


async def test_missing_header_forbidden(
    pool: Any, whitelisted_user: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "get_settings", _with_secret(_SECRET))
    async with await _client() as client:
        response = await client.get(
            "/facts", params={"telegram_id": whitelisted_user["telegram_id"]}
        )
    assert response.status_code == 403


async def test_wrong_header_forbidden(
    pool: Any, whitelisted_user: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "get_settings", _with_secret(_SECRET))
    async with await _client() as client:
        response = await client.get(
            "/facts",
            params={"telegram_id": whitelisted_user["telegram_id"]},
            headers={"X-KB-Secret": "wrong"},
        )
    assert response.status_code == 403


async def test_correct_header_allowed(
    pool: Any, whitelisted_user: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "get_settings", _with_secret(_SECRET))
    async with await _client() as client:
        response = await client.get(
            "/facts",
            params={"telegram_id": whitelisted_user["telegram_id"]},
            headers={"X-KB-Secret": _SECRET},
        )
    assert response.status_code == 200
    assert response.json() == {"facts": []}


async def test_health_public_even_with_secret(pool: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", _with_secret(_SECRET))
    async with await _client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_no_secret_no_header_allowed(
    pool: Any, whitelisted_user: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "get_settings", _with_secret(""))
    async with await _client() as client:
        response = await client.get(
            "/facts", params={"telegram_id": whitelisted_user["telegram_id"]}
        )
    assert response.status_code == 200
