"""Fact orchestration over the bitemporal facts repository (spec section 4).

Facts carry no embeddings or chunks — they go into the agent context whole. Topic
assignment is done *without* an LLM: the new text's embedding is compared to the
embeddings of existing active facts; a close match (cosine > threshold) reuses
that topic, otherwise a slug is built from the first significant words.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import asyncpg
import numpy as np

from kb.db.pool import get_pool
from kb.db.repo import facts as facts_repo

if TYPE_CHECKING:
    from kb.ingestion.embeddings import Embedder

TOPIC_SIMILARITY_THRESHOLD = 0.75

_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "о",
    "об",
    "из",
    "к",
    "у",
    "за",
    "от",
    "до",
    "для",
    "что",
    "как",
    "это",
    "наш",
    "наша",
    "наши",
    "мы",
    "нас",
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "for",
}


def slugify_topic(text: str) -> str:
    """Build a topic slug from the first significant words of the text."""
    tokens = _TOKEN_RE.findall(text.lower())
    significant = [token for token in tokens if len(token) >= 3 and token not in _STOPWORDS]
    words = significant or tokens
    slug = "-".join(words[:4])
    return slug or "fakt"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def assign_topic(
    content: str,
    *,
    embedder: Embedder,
    pool: asyncpg.Pool | None = None,
    threshold: float = TOPIC_SIMILARITY_THRESHOLD,
) -> str:
    """Return an existing topic if semantically close, otherwise a fresh slug."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        existing = await facts_repo.list_active(conn)
    if not existing:
        return slugify_topic(content)

    vectors = await embedder.embed_documents([content, *[f["content"] for f in existing]])
    query = np.asarray(vectors[0], dtype=np.float32)
    best_topic: str | None = None
    best_score = threshold
    for fact, vector in zip(existing, vectors[1:], strict=True):
        score = _cosine(query, np.asarray(vector, dtype=np.float32))
        if score > best_score:
            best_score = score
            best_topic = fact["topic"]
    return best_topic if best_topic is not None else slugify_topic(content)


async def list_facts(pool: asyncpg.Pool | None = None) -> list[dict[str, Any]]:
    """Return active facts (topic, content, valid_from)."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn:
        return await facts_repo.list_active(conn)


async def upsert_fact(
    topic: str,
    content: str,
    user_id: int | None,
    session_id: int | None,
    *,
    pool: asyncpg.Pool | None = None,
) -> None:
    """Invalidate the current version of a topic and insert a new one atomically."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await facts_repo.upsert(conn, topic, content, user_id, session_id)


async def delete_fact(topic: str, *, pool: asyncpg.Pool | None = None) -> bool:
    """Soft-delete (invalidate) the active fact for a topic."""
    pool = pool or await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        return await facts_repo.soft_delete(conn, topic)
