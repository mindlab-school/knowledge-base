"""Entity exploration (spec section 4).

``explore_entity`` resolves an entity by canonical name (falling back to a fuzzy
name match) and returns a card: the entity, the documents that mention it with
roles, and its neighbourhood in the graph at the requested depth.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from kb.db.pool import get_pool
from kb.db.repo import entities as entities_repo
from kb.search import graph

_FIND_EXACT_SQL = """
SELECT id, entity_type, name, canonical, attributes
FROM entities
WHERE canonical = $1
ORDER BY id
LIMIT 1
"""

_FIND_FUZZY_SQL = r"""
SELECT id, entity_type, name, canonical, attributes
FROM entities
WHERE name ILIKE '%' || $1 || '%' ESCAPE '\'
ORDER BY id
LIMIT 1
"""


def _escape_like(value: str) -> str:
    """Escape LIKE/ILIKE wildcards so a name is matched literally.

    ``%`` and ``_`` are treated as wildcards by ``ILIKE``; escape them (and the
    escape character ``\\`` itself) so a name such as ``50%`` or ``a_b`` cannot
    silently widen the fuzzy match. Pair with ``ESCAPE '\\'`` in the query.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_DOCUMENTS_SQL = """
SELECT d.id, d.title, d.doc_type, em.role
FROM entity_mentions em
JOIN documents d ON d.id = em.document_id
WHERE em.entity_id = $1
ORDER BY d.id
"""


async def explore_entity(
    name: str, *, depth: int = 1, pool: asyncpg.Pool | None = None
) -> dict[str, Any] | None:
    """Return a card for the named entity, or ``None`` if not found.

    :param depth: Graph traversal depth (0–3); ``0`` returns no neighbours.
    """
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        entity = await conn.fetchrow(_FIND_EXACT_SQL, entities_repo.canonicalize(name))
        if entity is None:
            entity = await conn.fetchrow(_FIND_FUZZY_SQL, _escape_like(name))
        if entity is None:
            return None
        documents = await conn.fetch(_DOCUMENTS_SQL, entity["id"])

    entity_graph = await graph.neighbors(int(entity["id"]), depth=depth, pool=pool)
    return {
        "entity": {
            "id": entity["id"],
            "type": entity["entity_type"],
            "name": entity["name"],
            "attributes": entity["attributes"],
        },
        "documents": [dict(row) for row in documents],
        "graph": entity_graph,
    }
