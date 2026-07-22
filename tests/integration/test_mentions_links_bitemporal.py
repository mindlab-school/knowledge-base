"""Integration tests for bitemporal entity mentions and document links (R10).

Re-ingesting a document must invalidate the mentions/links it no longer produces
(keeping them as history) and activate the current set, mirroring the proven
``entity_relations`` model. Active reads must see only current rows.
"""

from __future__ import annotations

from typing import Any

import pytest

from kb.config import Settings
from kb.ingestion.pipeline import ingest_loaded, reextract_document
from kb.ingestion.sources.base import LoadedDoc
from kb.search import entities as entities_search
from kb.search import graph, structured
from tests.fakes import extraction_llm, extraction_payload

pytestmark = pytest.mark.integration

_SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]


async def _ingest(
    pool: Any, embedder: Any, *, title: str, source_path: str, text: str, payload: dict[str, Any]
) -> Any:
    loaded = LoadedDoc(source_path=source_path, title=title, text=text)
    return await ingest_loaded(
        loaded, pool=pool, embedder=embedder, llm=extraction_llm(payload), settings=_SETTINGS
    )


def _mentions_payload(entities: list[dict[str, Any]]) -> dict[str, Any]:
    return extraction_payload(
        "generic", attributes={"summary": "s", "language": "ru"}, entities=entities
    )


async def _mention_counts(pool: Any, document_id: int) -> tuple[int, int]:
    async with pool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT count(*) FROM entity_mentions WHERE document_id = $1 AND invalid_at IS NULL",
            document_id,
        )
        total = await conn.fetchval(
            "SELECT count(*) FROM entity_mentions WHERE document_id = $1", document_id
        )
    return int(active), int(total)


async def test_reingest_invalidates_old_mentions_keeps_history(
    pool: Any, fake_embedder: Any
) -> None:
    first = await _ingest(
        pool,
        fake_embedder,
        title="Док",
        source_path="mentions.md",
        text="Первая версия текста.",
        payload=_mentions_payload(
            [
                {"entity_type": "person", "name": "Алиса", "role": "owner"},
                {"entity_type": "person", "name": "Борис", "role": "participant"},
            ]
        ),
    )
    assert first.action == "created"

    updated = await _ingest(
        pool,
        fake_embedder,
        title="Док",
        source_path="mentions.md",
        text="Вторая версия текста, другая.",
        payload=_mentions_payload(
            [
                {"entity_type": "person", "name": "Алиса", "role": "owner"},
                {"entity_type": "person", "name": "Виктор", "role": "participant"},
            ]
        ),
    )
    assert updated.action == "updated"
    assert updated.document_id == first.document_id

    # Two active rows (Алиса, Виктор); Борис + the old Алиса row survive invalidated.
    active, total = await _mention_counts(pool, first.document_id)
    assert active == 2
    assert total == 4

    card = await structured.get_document_card(first.document_id, pool=pool)
    assert card is not None
    names = {row["name"] for row in card["entities"]}
    assert names == {"Алиса", "Виктор"}
    assert "Борис" not in names

    # Борис's only mention is invalidated: gone from active reads, alive in history.
    async with pool.acquire() as conn:
        boris_active = await conn.fetchval(
            """
            SELECT count(*) FROM entity_mentions m
            JOIN entities e ON e.id = m.entity_id
            WHERE e.canonical = 'борис' AND m.invalid_at IS NULL
            """
        )
        boris_history = await conn.fetchval(
            """
            SELECT count(*) FROM entity_mentions m
            JOIN entities e ON e.id = m.entity_id
            WHERE e.canonical = 'борис' AND m.invalid_at IS NOT NULL
            """
        )
    assert boris_active == 0
    assert boris_history == 1


async def test_active_reads_filter_invalidated_mentions(pool: Any, fake_embedder: Any) -> None:
    doc = await _ingest(
        pool,
        fake_embedder,
        title="Профиль",
        source_path="profile.md",
        text="Исходный текст.",
        payload=_mentions_payload(
            [
                {"entity_type": "person", "name": "Нина", "role": "owner"},
                {"entity_type": "person", "name": "Олег", "role": "participant"},
            ]
        ),
    )
    await _ingest(
        pool,
        fake_embedder,
        title="Профиль",
        source_path="profile.md",
        text="Переписанный текст полностью.",
        payload=_mentions_payload([{"entity_type": "person", "name": "Нина", "role": "owner"}]),
    )

    # explore_entity: Нина keeps its active doc; Олег is retained but doc-less.
    nina = await entities_search.explore_entity("Нина", pool=pool)
    assert nina is not None
    assert [row["id"] for row in nina["documents"]] == [doc.document_id]

    oleg = await entities_search.explore_entity("Олег", pool=pool)
    assert oleg is not None  # entity survives (not an orphan: it has history)
    assert oleg["documents"] == []

    # entity filter (build_document_where EXISTS predicate) sees only active mentions.
    by_nina = await structured.query_documents({"entity": {"name": "Нина"}}, pool=pool)
    assert [row["id"] for row in by_nina] == [doc.document_id]
    by_oleg = await structured.query_documents({"entity": {"name": "Олег"}}, pool=pool)
    assert by_oleg == []
    by_role = await structured.query_documents(
        {"entity": {"name": "Нина", "role": "owner"}}, pool=pool
    )
    assert [row["id"] for row in by_role] == [doc.document_id]


async def test_reextract_removed_mention_survives_in_history(pool: Any, fake_embedder: Any) -> None:
    doc = await _ingest(
        pool,
        fake_embedder,
        title="Отчёт",
        source_path="report.md",
        text="Текст отчёта.",
        payload=_mentions_payload(
            [
                {"entity_type": "person", "name": "Пётр", "role": "owner"},
                {"entity_type": "person", "name": "Мария", "role": "participant"},
            ]
        ),
    )

    await reextract_document(
        doc.document_id,
        pool=pool,
        llm=extraction_llm(
            _mentions_payload([{"entity_type": "person", "name": "Пётр", "role": "owner"}])
        ),
        settings=_SETTINGS,
    )

    active, total = await _mention_counts(pool, doc.document_id)
    assert active == 1  # only Пётр
    assert total == 3  # Пётр(old, invalid) + Мария(invalid) + Пётр(new, active)

    card = await structured.get_document_card(doc.document_id, pool=pool)
    assert card is not None
    assert [row["name"] for row in card["entities"]] == ["Пётр"]

    maria = await entities_search.explore_entity("Мария", pool=pool)
    assert maria is not None
    assert maria["documents"] == []


async def test_null_role_mention_reingest_keeps_single_active(
    pool: Any, fake_embedder: Any
) -> None:
    first = await _ingest(
        pool,
        fake_embedder,
        title="Безроль",
        source_path="norole.md",
        text="Первый текст.",
        payload=_mentions_payload([{"entity_type": "person", "name": "Тень"}]),
    )
    await _ingest(
        pool,
        fake_embedder,
        title="Безроль",
        source_path="norole.md",
        text="Второй текст, иной.",
        payload=_mentions_payload([{"entity_type": "person", "name": "Тень"}]),
    )

    active, total = await _mention_counts(pool, first.document_id)
    assert active == 1  # a single active NULL-role mention
    assert total == 2  # the prior NULL-role mention is kept as history

    card = await structured.get_document_card(first.document_id, pool=pool)
    assert card is not None
    assert [row["role"] for row in card["entities"]] == [None]


async def _link_counts(pool: Any, source_id: int) -> tuple[int, int]:
    async with pool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT count(*) FROM document_links WHERE source_id = $1 AND invalid_at IS NULL",
            source_id,
        )
        total = await conn.fetchval(
            "SELECT count(*) FROM document_links WHERE source_id = $1", source_id
        )
    return int(active), int(total)


async def test_reingest_invalidates_and_reactivates_links(pool: Any, fake_embedder: Any) -> None:
    target = await _ingest(
        pool,
        fake_embedder,
        title="Цель",
        source_path="target.md",
        text="Целевой документ.",
        payload=extraction_payload("generic", attributes={"summary": "s", "language": "ru"}),
    )
    source = await _ingest(
        pool,
        fake_embedder,
        title="Источник",
        source_path="source.md",
        text="Первый текст со ссылкой.",
        payload=extraction_payload(
            "generic", attributes={"summary": "s", "language": "ru"}, links=["Цель"]
        ),
    )
    active, total = await _link_counts(pool, source.document_id)
    assert (active, total) == (1, 1)
    related = await graph.related_documents(source.document_id, pool=pool)
    assert [row["id"] for row in related] == [target.document_id]

    # Re-ingest with no links: the outgoing link is invalidated, not deleted.
    await _ingest(
        pool,
        fake_embedder,
        title="Источник",
        source_path="source.md",
        text="Второй текст без ссылок.",
        payload=extraction_payload("generic", attributes={"summary": "s", "language": "ru"}),
    )
    active, total = await _link_counts(pool, source.document_id)
    assert (active, total) == (0, 1)  # invalidated, survives as history
    assert await graph.related_documents(source.document_id, pool=pool) == []
    assert await graph.related_documents(target.document_id, pool=pool) == []

    # Re-ingest with the link again: a fresh active row, history preserved.
    await _ingest(
        pool,
        fake_embedder,
        title="Источник",
        source_path="source.md",
        text="Третий текст со ссылкой снова.",
        payload=extraction_payload(
            "generic", attributes={"summary": "s", "language": "ru"}, links=["Цель"]
        ),
    )
    active, total = await _link_counts(pool, source.document_id)
    assert (active, total) == (1, 2)  # one active, one invalidated in history
    related = await graph.related_documents(source.document_id, pool=pool)
    assert [row["id"] for row in related] == [target.document_id]
