"""Graph traversal over entity and document edges (spec section 4).

Traversal is a single recursive CTE — no embeddings, no extra model rounds. Depth
is capped at 3 and the node count at 40: beyond that the walk stops being useful
and only bloats the agent context. Cycles are prevented by tracking the visited
path.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from kb.db.pool import get_pool

MAX_DEPTH = 3
MAX_NODES = 40

_NEIGHBORS_SQL = """
WITH RECURSIVE walk(id, path, d) AS (
    SELECT $1::int, ARRAY[$1::int], 0
    UNION ALL
    SELECT nb.nid, w.path || nb.nid, w.d + 1
    FROM walk w
    JOIN LATERAL (
        SELECT CASE WHEN r.source_id = w.id THEN r.target_id ELSE r.source_id END AS nid
        FROM entity_relations r
        WHERE (r.source_id = w.id OR r.target_id = w.id) AND r.invalid_at IS NULL
    ) nb ON TRUE
    WHERE w.d < $2 AND NOT (nb.nid = ANY(w.path))
)
SELECT e.id, e.entity_type AS type, e.name, MIN(walk.d) AS depth
FROM walk
JOIN entities e ON e.id = walk.id
GROUP BY e.id, e.entity_type, e.name
ORDER BY depth, e.id
LIMIT $3
"""

_EDGES_SQL = """
SELECT source_id AS source, target_id AS target, relation
FROM entity_relations
WHERE invalid_at IS NULL
  AND source_id = ANY($1::int[])
  AND target_id = ANY($1::int[])
"""

_RELATED_DOCS_SQL = """
WITH RECURSIVE walk(id, path, d) AS (
    SELECT $1::int, ARRAY[$1::int], 0
    UNION ALL
    SELECT nb.nid, w.path || nb.nid, w.d + 1
    FROM walk w
    JOIN LATERAL (
        SELECT CASE WHEN l.source_id = w.id THEN l.target_id ELSE l.source_id END AS nid
        FROM document_links l
        WHERE l.source_id = w.id OR l.target_id = w.id
    ) nb ON TRUE
    WHERE w.d < $2 AND NOT (nb.nid = ANY(w.path))
)
SELECT d.id, d.title, d.doc_type, d.version, MIN(walk.d) AS depth,
       NOT EXISTS (
           SELECT 1 FROM document_links s
           WHERE s.kind = 'supersedes' AND s.target_id = d.id
       ) AS current
FROM walk
JOIN documents d ON d.id = walk.id
WHERE walk.id <> $1
GROUP BY d.id, d.title, d.doc_type, d.version
ORDER BY depth, d.id
"""


def _clamp_depth(depth: int) -> int:
    return max(0, min(depth, MAX_DEPTH))


async def neighbors(
    entity_id: int,
    *,
    depth: int = 2,
    limit: int = MAX_NODES,
    pool: asyncpg.Pool | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return the ``{nodes, edges}`` reachable from an entity within ``depth``."""
    depth = _clamp_depth(depth)
    limit = min(limit, MAX_NODES)
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        node_rows = await conn.fetch(_NEIGHBORS_SQL, entity_id, depth, limit)
        ids = [int(row["id"]) for row in node_rows]
        edge_rows = await conn.fetch(_EDGES_SQL, ids) if ids else []
    return {
        "nodes": [dict(row) for row in node_rows],
        "edges": [dict(row) for row in edge_rows],
    }


async def related_documents(
    document_id: int, *, depth: int = 1, pool: asyncpg.Pool | None = None
) -> list[dict[str, Any]]:
    """Return documents linked to ``document_id`` within ``depth`` (both directions).

    Each result carries ``current`` (False when a ``supersedes`` link marks it as
    an outdated version).
    """
    depth = _clamp_depth(depth)
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(_RELATED_DOCS_SQL, document_id, depth)
    return [dict(row) for row in rows]
