"""Entity repository: entities, mentions, and bitemporal graph edges.

Entities are upserted by ``(entity_type, canonical)``. Mentions and edges
belonging to a document are *replaced* on re-ingestion by invalidating the ones
no longer produced (``invalid_at = now()``) and inserting the current set — rows
are never physically deleted, and active reads default to ``invalid_at IS NULL``.
The current set is guarded by a partial unique index on active rows, so a
superseded mention/edge falls out of the index and survives as history.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg

Row = dict[str, Any]


def canonicalize(name: str) -> str:
    """Return the canonical key for an entity name: ``lower(trim(name))``."""
    return name.strip().lower()


async def upsert_entity(
    conn: asyncpg.Connection,
    entity_type: str,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> int:
    """Upsert an entity by ``(entity_type, canonical)`` and return its id."""
    record = await conn.fetchrow(
        """
        INSERT INTO entities (entity_type, name, canonical, attributes)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (entity_type, canonical)
        DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        entity_type,
        name,
        canonicalize(name),
        attributes or {},
    )
    return int(record["id"])


async def add_mention(
    conn: asyncpg.Connection, entity_id: int, document_id: int, role: str | None
) -> None:
    """Activate an entity's mention of a document with a role (idempotent).

    Mirrors ``replace_relations``: an active duplicate is refreshed rather than
    duplicated. A ``NULL`` role needs an explicit existence check because Postgres
    treats NULLs as distinct in the partial unique index, so ``ON CONFLICT`` would
    not dedupe two active role-less mentions of the same entity in one document;
    the check is scoped to active rows so a re-ingest still inserts a fresh row.
    """
    if role is None:
        await conn.execute(
            """
            INSERT INTO entity_mentions (entity_id, document_id, role)
            SELECT $1, $2, NULL
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_mentions
                WHERE entity_id = $1 AND document_id = $2
                  AND role IS NULL AND invalid_at IS NULL
            )
            """,
            entity_id,
            document_id,
        )
        return
    await conn.execute(
        """
        INSERT INTO entity_mentions (entity_id, document_id, role)
        VALUES ($1, $2, $3)
        ON CONFLICT (entity_id, document_id, role) WHERE invalid_at IS NULL
        DO UPDATE SET invalid_at = NULL, valid_from = now()
        """,
        entity_id,
        document_id,
        role,
    )


async def delete_mentions_for_document(conn: asyncpg.Connection, document_id: int) -> None:
    """Invalidate all active mentions of a document (before recreating them).

    Rows are kept with ``invalid_at`` set — the mention history/provenance of
    prior versions survives re-ingestion, mirroring ``replace_relations``.
    """
    await conn.execute(
        "UPDATE entity_mentions SET invalid_at = now() "
        "WHERE document_id = $1 AND invalid_at IS NULL",
        document_id,
    )


async def replace_relations(
    conn: asyncpg.Connection,
    document_id: int,
    edges: Sequence[tuple[int, int, str]],
) -> None:
    """Replace this document's graph edges.

    Active edges absent from ``edges`` are invalidated; edges in ``edges`` are
    (re)activated. Self-edges are ignored (they violate the table CHECK).
    """
    wanted = {(s, t, r) for s, t, r in edges if s != t}

    current = await conn.fetch(
        """
        SELECT source_id, target_id, relation FROM entity_relations
        WHERE document_id = $1 AND invalid_at IS NULL
        """,
        document_id,
    )
    for row in current:
        key = (int(row["source_id"]), int(row["target_id"]), row["relation"])
        if key not in wanted:
            await conn.execute(
                """
                UPDATE entity_relations SET invalid_at = now()
                WHERE document_id = $1 AND source_id = $2 AND target_id = $3
                  AND relation = $4 AND invalid_at IS NULL
                """,
                document_id,
                key[0],
                key[1],
                key[2],
            )

    for source_id, target_id, relation in wanted:
        await conn.execute(
            """
            INSERT INTO entity_relations (source_id, target_id, relation, document_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (source_id, target_id, relation, document_id)
            DO UPDATE SET invalid_at = NULL, valid_from = now()
            """,
            source_id,
            target_id,
            relation,
            document_id,
        )


async def cleanup_orphans(conn: asyncpg.Connection) -> int:
    """Delete entities with no mentions and no edges. Return how many were removed.

    Rows are counted regardless of ``invalid_at``: an entity whose mentions/edges
    are all invalidated still carries history and is deliberately *not* an orphan.
    Deleting it would cascade (``ON DELETE CASCADE``) onto those invalidated rows
    and destroy the provenance this bitemporal model exists to keep.
    """
    value = await conn.fetchval(
        """
        WITH deleted AS (
            DELETE FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = e.id)
              AND NOT EXISTS (
                  SELECT 1 FROM entity_relations r
                  WHERE r.source_id = e.id OR r.target_id = e.id
              )
            RETURNING e.id
        )
        SELECT count(*) FROM deleted
        """
    )
    return int(value)
