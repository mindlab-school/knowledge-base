"""Hybrid semantic + full-text chunk search with RRF (spec section 4).

Two ranked candidate lists — vector (HNSW cosine) and full-text
(``websearch_to_tsquery`` + ``ts_rank_cd``) — are merged by Reciprocal Rank
Fusion (``k = 60``). When filters narrow the set, ``hnsw.iterative_scan`` is set
to ``relaxed_order`` so the ANN branch keeps collecting candidates past the
filter (pgvector 0.8+); otherwise HNSW would select candidates before the filter
and under-fill.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import asyncpg
from pgvector import HalfVector

from kb.config import Settings, get_settings
from kb.db.pool import get_pool
from kb.embeddings import Embedder, make_embedder
from kb.search.structured import build_document_where

RRF_K = 60
DEFAULT_LIMIT = 4
_CANDIDATES = 20


def reciprocal_rank_fusion[K: Hashable](
    ranked_lists: Sequence[Sequence[K]], *, k: int = RRF_K
) -> list[tuple[K, float]]:
    """Fuse ranked id lists into one ranking.

    ``score(id) = sum(1 / (k + rank_i))`` over the lists in which the id appears
    (rank is 1-based). Returns ``(id, score)`` pairs sorted by descending score.
    """
    scores: dict[K, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


_VECTOR_SQL = """
SELECT c.id AS chunk_id, c.content, c.document_id, d.title
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE {where}
ORDER BY c.embedding <=> ${qpos}::halfvec
LIMIT ${limit_pos}
"""

_FTS_SQL = """
SELECT c.id AS chunk_id, c.content, c.document_id, d.title
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE {where} AND c.tsv @@ websearch_to_tsquery('russian', ${qpos})
ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('russian', ${qpos})) DESC
LIMIT ${limit_pos}
"""


async def search_chunks(
    query: str,
    filters: dict[str, Any] | None = None,
    *,
    limit: int = DEFAULT_LIMIT,
    pool: asyncpg.Pool | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Return the top ``limit`` chunks for a query as
    ``{document_id, document, score, fragment}`` dicts."""
    filters = filters or {}
    settings = settings or get_settings()
    pool = pool or await get_pool()
    embedder = embedder or make_embedder(settings)

    query_vector = await embedder.embed_query(query)
    where_sql, base_params = build_document_where(filters)
    next_pos = len(base_params) + 1

    vector_params = [*base_params, HalfVector(list(query_vector)), _CANDIDATES]
    vector_sql = _VECTOR_SQL.format(where=where_sql, qpos=next_pos, limit_pos=next_pos + 1)
    fts_params = [*base_params, query, _CANDIDATES]
    fts_sql = _FTS_SQL.format(where=where_sql, qpos=next_pos, limit_pos=next_pos + 1)

    async with pool.acquire() as conn:
        if filters:
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                vector_rows = await conn.fetch(vector_sql, *vector_params)
        else:
            vector_rows = await conn.fetch(vector_sql, *vector_params)
        fts_rows = await conn.fetch(fts_sql, *fts_params)

    by_chunk: dict[int, dict[str, Any]] = {}
    for row in (*vector_rows, *fts_rows):
        by_chunk.setdefault(int(row["chunk_id"]), dict(row))

    vector_ids = [int(row["chunk_id"]) for row in vector_rows]
    fts_ids = [int(row["chunk_id"]) for row in fts_rows]
    fused = reciprocal_rank_fusion([vector_ids, fts_ids])

    results: list[dict[str, Any]] = []
    for chunk_id, score in fused[:limit]:
        row = by_chunk[chunk_id]
        results.append(
            {
                "document_id": row["document_id"],
                "document": row["title"],
                "score": round(score, 6),
                "fragment": row["content"],
            }
        )
    return results
