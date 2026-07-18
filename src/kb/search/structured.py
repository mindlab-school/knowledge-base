"""Structured metadata queries over documents (spec section 4).

``query_documents`` returns document metadata (no ``raw_content``) filtered by an
AND-combination of predicates. ``build_document_where`` is a pure function that
turns a filter dict into a parameterised SQL ``WHERE`` fragment, so it is easy to
unit-test in isolation.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from kb.db.pool import get_pool
from kb.db.repo import documents as documents_repo

DEFAULT_DOCUMENT_LIMIT = 15


class _Where:
    """Accumulates SQL conditions and positional parameters."""

    def __init__(self) -> None:
        self.conditions: list[str] = []
        self.params: list[Any] = []

    def placeholder(self, value: Any) -> str:
        self.params.append(value)
        return f"${len(self.params)}"

    def add(self, condition: str) -> None:
        self.conditions.append(condition)

    @property
    def sql(self) -> str:
        return " AND ".join(self.conditions) if self.conditions else "TRUE"


def build_document_where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a ``WHERE`` fragment and parameters from a filter dict.

    Supported keys (all AND-combined):

    * ``doc_type`` — exact type;
    * ``attributes`` — JSONB containment (``@>``);
    * ``title_contains`` — case-insensitive substring;
    * ``created_before`` / ``created_after`` — on ``created_at``;
    * ``<attr>_before`` / ``<attr>_after`` — date compare on an attribute
      (e.g. ``review_by_before``);
    * ``entity`` — ``{"name": ..., "role"?: ...}`` mention existence.

    Unknown keys are ignored.
    """
    where = _Where()
    for key, value in filters.items():
        if value is None:
            continue
        if key == "doc_type":
            where.add(f"d.doc_type = {where.placeholder(value)}")
        elif key == "attributes" and isinstance(value, dict):
            where.add(f"d.attributes @> {where.placeholder(value)}::jsonb")
        elif key == "title_contains":
            where.add(f"d.title ILIKE '%' || {where.placeholder(value)} || '%'")
        elif key == "created_before":
            where.add(f"d.created_at < {where.placeholder(value)}")
        elif key == "created_after":
            where.add(f"d.created_at > {where.placeholder(value)}")
        elif key == "entity" and isinstance(value, dict) and value.get("name"):
            canonical = str(value["name"]).strip().lower()
            role = value.get("role")
            base = (
                "EXISTS (SELECT 1 FROM entity_mentions em "
                "JOIN entities e ON e.id = em.entity_id "
                f"WHERE em.document_id = d.id AND e.canonical = {where.placeholder(canonical)}"
            )
            if role:
                base += f" AND em.role = {where.placeholder(str(role))}"
            where.add(base + ")")
        elif key.endswith("_before"):
            field = key[: -len("_before")]
            column = where.placeholder(field)
            where.add(f"(d.attributes ->> {column})::date < {where.placeholder(value)}::date")
        elif key.endswith("_after"):
            field = key[: -len("_after")]
            column = where.placeholder(field)
            where.add(f"(d.attributes ->> {column})::date > {where.placeholder(value)}::date")
    return where.sql, where.params


async def query_documents(
    filters: dict[str, Any] | None = None,
    *,
    limit: int = DEFAULT_DOCUMENT_LIMIT,
    pool: asyncpg.Pool | None = None,
) -> list[dict[str, Any]]:
    """Return document metadata matching ``filters`` (no ``raw_content``)."""
    filters = filters or {}
    where_sql, params = build_document_where(filters)
    pool = pool or await get_pool()
    sql = f"""
        SELECT id, title, doc_type, attributes, version, created_at
        FROM documents d
        WHERE {where_sql}
        ORDER BY created_at DESC
        LIMIT ${len(params) + 1}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params, limit)
    return [dict(row) for row in rows]


async def count_documents(
    filters: dict[str, Any] | None = None, *, pool: asyncpg.Pool | None = None
) -> int:
    """Return the number of documents matching ``filters`` (for cap reporting)."""
    filters = filters or {}
    where_sql, params = build_document_where(filters)
    pool = pool or await get_pool()
    sql = f"SELECT count(*) FROM documents d WHERE {where_sql}"
    async with pool.acquire() as conn:
        value = await conn.fetchval(sql, *params)
    return int(value)


async def get_document(
    document_id: int, *, pool: asyncpg.Pool | None = None
) -> dict[str, Any] | None:
    """Return a full document card (including ``raw_content``)."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        return await documents_repo.get(conn, document_id)


_MENTIONS_SQL = """
SELECT e.entity_type, e.name, em.role
FROM entity_mentions em
JOIN entities e ON e.id = em.entity_id
WHERE em.document_id = $1
ORDER BY e.id
"""


async def get_document_card(
    document_id: int, *, pool: asyncpg.Pool | None = None
) -> dict[str, Any] | None:
    """Return a document with its chunk count and entity mentions (with roles)."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        document = await documents_repo.get(conn, document_id)
        if document is None:
            return None
        chunks = await documents_repo.chunk_count(conn, document_id)
        mentions = await conn.fetch(_MENTIONS_SQL, document_id)
    card = dict(document)
    card["chunks"] = chunks
    card["entities"] = [dict(row) for row in mentions]
    return card
