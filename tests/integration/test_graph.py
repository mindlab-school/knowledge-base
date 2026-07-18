"""Integration tests for graph traversal and document links (spec section 10)."""

from __future__ import annotations

from typing import Any

import pytest

from kb.config import Settings
from kb.ingestion.pipeline import ingest_loaded, reextract_document
from kb.ingestion.sources.base import LoadedDoc
from kb.search import graph
from tests.fakes import extraction_llm, extraction_payload

pytestmark = pytest.mark.integration

_SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]


async def _ingest(
    pool: Any, embedder: Any, *, title: str, source_path: str, payload: dict[str, Any]
) -> Any:
    loaded = LoadedDoc(source_path=source_path, title=title, text=f"Текст {title}.")
    return await ingest_loaded(
        loaded, pool=pool, embedder=embedder, llm=extraction_llm(payload), settings=_SETTINGS
    )


async def _entity_id(pool: Any, name: str) -> int:
    async with pool.acquire() as conn:
        value = await conn.fetchval("SELECT id FROM entities WHERE canonical = $1", name.lower())
    return int(value)


async def test_ring_traversal_terminates(pool: Any, fake_embedder: Any) -> None:
    payload = extraction_payload(
        "generic",
        attributes={"summary": "s", "language": "ru"},
        entities=[
            {"entity_type": "person", "name": "Алиса"},
            {"entity_type": "person", "name": "Борис"},
            {"entity_type": "person", "name": "Виктор"},
        ],
        relations=[
            {"source": "Алиса", "target": "Борис", "relation": "related_to"},
            {"source": "Борис", "target": "Виктор", "relation": "related_to"},
            {"source": "Виктор", "target": "Алиса", "relation": "related_to"},
        ],
    )
    await _ingest(pool, fake_embedder, title="Кольцо", source_path="ring.md", payload=payload)

    alice = await _entity_id(pool, "Алиса")
    result = await graph.neighbors(alice, depth=3, pool=pool)
    names = {node["name"] for node in result["nodes"]}
    assert names == {"Алиса", "Борис", "Виктор"}
    assert len(result["edges"]) == 3


async def test_reindex_replaces_only_own_edges(pool: Any, fake_embedder: Any) -> None:
    with_edge = extraction_payload(
        "generic",
        attributes={"summary": "s", "language": "ru"},
        entities=[
            {"entity_type": "person", "name": "Пётр"},
            {"entity_type": "department", "name": "Продажи"},
        ],
        relations=[{"source": "Пётр", "target": "Продажи", "relation": "part_of"}],
    )
    doc = await _ingest(
        pool, fake_embedder, title="Связь", source_path="link.md", payload=with_edge
    )
    peter = await _entity_id(pool, "пётр")
    assert len((await graph.neighbors(peter, depth=1, pool=pool))["edges"]) == 1

    # Re-extraction with no relations invalidates this document's edge only.
    without_edge = extraction_payload(
        "generic",
        attributes={"summary": "s", "language": "ru"},
        entities=[{"entity_type": "person", "name": "Пётр"}],
        relations=[],
    )
    await reextract_document(
        doc.document_id, pool=pool, llm=extraction_llm(without_edge), settings=_SETTINGS
    )
    async with pool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT count(*) FROM entity_relations WHERE invalid_at IS NULL"
        )
        total = await conn.fetchval("SELECT count(*) FROM entity_relations")
    assert active == 0
    assert total == 1  # the edge is invalidated, not deleted


async def test_pending_link_resolves_on_target_ingest(pool: Any, fake_embedder: Any) -> None:
    beta = await _ingest(
        pool,
        fake_embedder,
        title="Бета",
        source_path="beta.md",
        payload=extraction_payload(
            "generic", attributes={"summary": "s", "language": "ru"}, links=["Альфа"]
        ),
    )
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM document_links") == 0
        assert await conn.fetchval("SELECT count(*) FROM pending_document_links") == 1

    alpha = await _ingest(
        pool,
        fake_embedder,
        title="Альфа",
        source_path="alpha.md",
        payload=extraction_payload("generic", attributes={"summary": "s", "language": "ru"}),
    )
    async with pool.acquire() as conn:
        link = await conn.fetchrow("SELECT source_id, target_id FROM document_links")
        pending = await conn.fetchval("SELECT count(*) FROM pending_document_links")
    assert link is not None
    assert link["source_id"] == beta.document_id
    assert link["target_id"] == alpha.document_id
    assert pending == 0


async def test_related_documents_marks_superseded_version(pool: Any, fake_embedder: Any) -> None:
    payload = extraction_payload("generic", attributes={"summary": "s", "language": "ru"})
    new = await _ingest(pool, fake_embedder, title="Новая", source_path="new.md", payload=payload)
    old = await _ingest(pool, fake_embedder, title="Старая", source_path="old.md", payload=payload)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO document_links (source_id, target_id, kind) VALUES ($1, $2, 'supersedes')",
            new.document_id,
            old.document_id,
        )

    related_to_new = await graph.related_documents(new.document_id, pool=pool)
    assert [row["id"] for row in related_to_new] == [old.document_id]
    assert related_to_new[0]["current"] is False  # old is superseded

    related_to_old = await graph.related_documents(old.document_id, pool=pool)
    assert related_to_old[0]["id"] == new.document_id
    assert related_to_old[0]["current"] is True


async def test_duplicate_null_role_mention_deduped(pool: Any, fake_embedder: Any) -> None:
    payload = extraction_payload(
        "generic",
        attributes={"summary": "s", "language": "ru"},
        entities=[
            {"entity_type": "person", "name": "Дубль"},
            {"entity_type": "person", "name": "Дубль"},
        ],
    )
    await _ingest(pool, fake_embedder, title="Док", source_path="dup.md", payload=payload)
    async with pool.acquire() as conn:
        mentions = await conn.fetchval("SELECT count(*) FROM entity_mentions")
        entities = await conn.fetchval("SELECT count(*) FROM entities")
    assert entities == 1
    assert mentions == 1
