"""Integration tests: date filters in structured document queries must execute
against a real database.

Regression for the ``created_before`` / ``created_after`` bug where an ISO date
string was bound directly against the ``TIMESTAMPTZ`` ``created_at`` column,
raising ``asyncpg.DataError`` (HTTP 500) instead of filtering. The pure unit test
only inspected the generated SQL string, so the type mismatch slipped through.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from kb.search import structured

pytestmark = pytest.mark.integration


async def _insert(
    conn: Any, *, title: str, created_at: datetime, attributes: dict[str, Any]
) -> int:
    return await conn.fetchval(
        """
        INSERT INTO documents (
            title, source_path, source_kind, content_hash, raw_content,
            doc_type, attributes, extraction_status, created_at
        )
        VALUES ($1, $2, 'file', $3, 'body', 'regulation', $4, 'done', $5)
        RETURNING id
        """,
        title,
        f"{title}.md",
        title,
        attributes,
        created_at,
    )


@pytest.fixture
async def _two_dated_docs(pool: Any) -> dict[str, int]:
    async with pool.acquire() as conn:
        old = await _insert(
            conn,
            title="old",
            created_at=datetime(2018, 6, 1, tzinfo=UTC),
            attributes={"review_by": "2019-01-01"},
        )
        new = await _insert(
            conn,
            title="new",
            created_at=datetime(2025, 6, 1, tzinfo=UTC),
            attributes={"review_by": "2030-01-01"},
        )
    return {"old": old, "new": new}


async def test_created_before_executes_and_filters(
    pool: Any, _two_dated_docs: dict[str, int]
) -> None:
    rows = await structured.query_documents({"created_before": "2020-01-01"}, pool=pool)
    ids = {row["id"] for row in rows}
    assert ids == {_two_dated_docs["old"]}


async def test_created_after_executes_and_filters(
    pool: Any, _two_dated_docs: dict[str, int]
) -> None:
    rows = await structured.query_documents({"created_after": "2020-01-01"}, pool=pool)
    ids = {row["id"] for row in rows}
    assert ids == {_two_dated_docs["new"]}


async def test_count_documents_with_created_range(
    pool: Any, _two_dated_docs: dict[str, int]
) -> None:
    assert await structured.count_documents({"created_after": "2020-01-01"}, pool=pool) == 1


async def test_attribute_date_filter_executes(pool: Any, _two_dated_docs: dict[str, int]) -> None:
    rows = await structured.query_documents({"review_by_before": "2020-01-01"}, pool=pool)
    ids = {row["id"] for row in rows}
    assert ids == {_two_dated_docs["old"]}


async def test_non_date_attribute_cast_does_not_error(pool: Any) -> None:
    # ``status`` is a non-date attribute; a ``status_before`` filter would cast
    # ``'active'::date`` and raise ``asyncpg.DataError`` (HTTP 500) without the
    # guard. It must instead execute and exclude the non-date row.
    async with pool.acquire() as conn:
        await _insert(
            conn,
            title="statusy",
            created_at=datetime(2022, 1, 1, tzinfo=UTC),
            attributes={"status": "active"},
        )
    rows = await structured.query_documents({"status_before": "2026-01-01"}, pool=pool)
    assert rows == []
    assert await structured.count_documents({"status_before": "2026-01-01"}, pool=pool) == 0
