"""Structured metadata queries over documents (spec section 4).

``query_documents`` returns document metadata (no ``raw_content``) filtered by an
AND-combination of predicates. ``build_document_where`` is a pure function that
turns a filter dict into a parameterised SQL ``WHERE`` fragment, so it is easy to
unit-test in isolation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import asyncpg

from kb.db.pool import get_pool
from kb.db.repo import documents as documents_repo
from kb.db.repo import entities as entities_repo

DEFAULT_DOCUMENT_LIMIT = 15


def _parse_iso_date(value: Any) -> date | None:
    """Parse an ISO date/datetime string into a ``date``.

    ``asyncpg`` binds a date/timestamp parameter from a Python ``date``/``datetime``
    object, not from a string — a bare ISO string raises ``DataError`` even with a
    ``::date`` SQL cast. Returns ``None`` for unparseable input so the caller can
    skip the predicate instead of failing the whole query.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None


def _attr_date_expr(column: str) -> str:
    """SQL that casts a JSONB attribute to ``date`` only when it looks like one.

    ``column`` is a bound placeholder holding the attribute name. A filter such as
    ``status_before`` targets an arbitrary attribute, so the value may not be a
    date (e.g. ``'active'``). Casting it directly raises Postgres ``DataError``
    (HTTP 500); the ``CASE`` guard yields ``NULL`` for a non-date value, which
    excludes the row from the comparison instead.
    """
    ref = f"(d.attributes ->> {column})"
    return f"CASE WHEN {ref} ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN {ref}::date END"


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

    Unknown keys are ignored. A non-dict ``filters`` (e.g. a stray string from a
    malformed tool call) yields an empty ``WHERE`` instead of raising.
    """
    where = _Where()
    if not isinstance(filters, dict):
        return where.sql, where.params
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
            parsed = _parse_iso_date(value)
            if parsed is not None:
                where.add(f"d.created_at::date < {where.placeholder(parsed)}")
        elif key == "created_after":
            parsed = _parse_iso_date(value)
            if parsed is not None:
                where.add(f"d.created_at::date > {where.placeholder(parsed)}")
        elif key == "entity" and isinstance(value, dict) and value.get("name"):
            canonical = entities_repo.canonicalize(str(value["name"]))
            role = value.get("role")
            base = (
                "EXISTS (SELECT 1 FROM entity_mentions em "
                "JOIN entities e ON e.id = em.entity_id "
                "WHERE em.document_id = d.id AND em.invalid_at IS NULL "
                f"AND e.canonical = {where.placeholder(canonical)}"
            )
            if role:
                base += f" AND em.role = {where.placeholder(str(role))}"
            where.add(base + ")")
        elif key.endswith("_before"):
            parsed = _parse_iso_date(value)
            if parsed is None:
                continue
            column = where.placeholder(key[: -len("_before")])
            where.add(f"{_attr_date_expr(column)} < {where.placeholder(parsed)}")
        elif key.endswith("_after"):
            parsed = _parse_iso_date(value)
            if parsed is None:
                continue
            column = where.placeholder(key[: -len("_after")])
            where.add(f"{_attr_date_expr(column)} > {where.placeholder(parsed)}")
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
WHERE em.document_id = $1 AND em.invalid_at IS NULL
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
