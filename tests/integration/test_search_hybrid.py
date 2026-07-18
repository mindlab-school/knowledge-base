"""Integration tests for hybrid semantic + full-text search (spec section 10)."""

from __future__ import annotations

from typing import Any

import pytest

from kb.config import Settings
from kb.ingestion.pipeline import ingest_loaded
from kb.ingestion.sources.base import LoadedDoc
from kb.search import semantic
from tests.fakes import extraction_llm, extraction_payload

pytestmark = pytest.mark.integration

_SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
_GENERIC = extraction_payload("generic", attributes={"summary": "s", "language": "ru"})


async def _ingest(pool: Any, embedder: Any, *, title: str, source_path: str, text: str) -> Any:
    loaded = LoadedDoc(source_path=source_path, title=title, text=text)
    return await ingest_loaded(
        loaded, pool=pool, embedder=embedder, llm=extraction_llm(_GENERIC), settings=_SETTINGS
    )


async def test_rare_abbreviation_found_by_fulltext(pool: Any, fake_embedder: Any) -> None:
    rare = await _ingest(
        pool,
        fake_embedder,
        title="Коды",
        source_path="codes.md",
        text="Внутренний код ОКВЭД12345 применяется для годовой отчётности компании.",
    )
    await _ingest(
        pool,
        fake_embedder,
        title="Отпуска",
        source_path="vac.md",
        text="Порядок оформления ежегодного оплачиваемого отпуска сотрудника компании.",
    )

    results = await semantic.search_chunks("ОКВЭД12345", pool=pool, embedder=fake_embedder)
    found = {item["document_id"] for item in results}
    assert rare.document_id in found


async def test_semantic_query_returns_relevant_document(pool: Any, fake_embedder: Any) -> None:
    await _ingest(
        pool,
        fake_embedder,
        title="Коды",
        source_path="codes.md",
        text="Внутренний код ОКВЭД12345 применяется для годовой отчётности компании.",
    )
    vacations = await _ingest(
        pool,
        fake_embedder,
        title="Отпуска",
        source_path="vac.md",
        text="Порядок оформления ежегодного оплачиваемого отпуска сотрудника компании.",
    )

    results = await semantic.search_chunks(
        "оформление отпуска сотрудника", pool=pool, embedder=fake_embedder
    )
    assert results
    assert vacations.document_id in {item["document_id"] for item in results}
    assert all("score" in item and "fragment" in item for item in results)


async def test_filtered_search_uses_iterative_scan(pool: Any, fake_embedder: Any) -> None:
    loaded = LoadedDoc(
        source_path="reg.md",
        title="Регламент",
        text="# Регламент\nПорядок оформления отпуска сотрудника компании.",
    )
    regulation = await ingest_loaded(
        loaded,
        pool=pool,
        embedder=fake_embedder,
        llm=extraction_llm(
            extraction_payload("regulation", attributes={"owner": "X", "status": "active"})
        ),
        settings=_SETTINGS,
    )
    await _ingest(
        pool,
        fake_embedder,
        title="Заметка",
        source_path="note.md",
        text="Порядок оформления отпуска сотрудника в свободной форме.",
    )

    results = await semantic.search_chunks(
        "оформление отпуска", filters={"doc_type": "regulation"}, pool=pool, embedder=fake_embedder
    )
    found = {item["document_id"] for item in results}
    assert regulation.document_id in found
    assert found == {regulation.document_id}
