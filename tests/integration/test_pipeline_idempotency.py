"""Integration tests for ingestion idempotency and versioning (spec section 10)."""

from __future__ import annotations

from typing import Any

import pytest

from kb.config import Settings
from kb.ingestion.pipeline import ingest_loaded
from kb.ingestion.sources.base import LoadedDoc
from tests.fakes import extraction_llm, extraction_payload

pytestmark = pytest.mark.integration

_SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
_GENERIC = extraction_payload("generic", attributes={"summary": "s", "language": "ru"})


async def _ingest(pool: Any, embedder: Any, source_path: str, text: str) -> Any:
    loaded = LoadedDoc(source_path=source_path, title="Документ", text=text)
    return await ingest_loaded(
        loaded, pool=pool, embedder=embedder, llm=extraction_llm(_GENERIC), settings=_SETTINGS
    )


async def test_reingest_same_content_is_noop(pool: Any, fake_embedder: Any) -> None:
    text = "Первый абзац документа.\n\nВторой абзац документа."
    first = await _ingest(pool, fake_embedder, "docs/a.md", text)
    assert first.action == "created"
    assert first.version == 1
    assert first.chunks >= 1

    second = await _ingest(pool, fake_embedder, "docs/a.md", text)
    assert second.action == "unchanged"
    assert second.version == 1
    assert second.document_id == first.document_id


async def test_changed_content_bumps_version(pool: Any, fake_embedder: Any) -> None:
    first = await _ingest(pool, fake_embedder, "docs/b.md", "Старый текст.\n\nЕщё абзац.")
    updated = await _ingest(
        pool, fake_embedder, "docs/b.md", "Новый текст.\n\nДругой абзац.\n\nТретий."
    )
    assert updated.action == "updated"
    assert updated.version == 2
    assert updated.document_id == first.document_id

    async with pool.acquire() as conn:
        chunk_count = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE document_id = $1", first.document_id
        )
        content = await conn.fetchval(
            "SELECT string_agg(content, ' ') FROM chunks WHERE document_id = $1", first.document_id
        )
    assert chunk_count == updated.chunks
    assert "Новый текст" in content
    assert "Старый текст" not in content


async def test_telegram_filename_versioning(pool: Any, fake_embedder: Any) -> None:
    first = await _ingest(pool, fake_embedder, "tg:report.pdf", "Версия один текста.")
    second = await _ingest(pool, fake_embedder, "tg:report.pdf", "Версия два текста.")
    assert first.document_id == second.document_id
    assert second.version == 2


async def test_same_filename_different_users_are_distinct(pool: Any, fake_embedder: Any) -> None:
    alice = await _ingest(pool, fake_embedder, "tg:1001:policy.pdf", "Политика Алисы.")
    bob = await _ingest(pool, fake_embedder, "tg:2002:policy.pdf", "Политика Боба.")
    assert alice.action == "created"
    assert bob.action == "created"
    assert alice.document_id != bob.document_id

    # Each user's re-upload of the same name maps back to their own document.
    alice_again = await _ingest(pool, fake_embedder, "tg:1001:policy.pdf", "Политика Алисы.")
    assert alice_again.action == "unchanged"
    assert alice_again.document_id == alice.document_id
