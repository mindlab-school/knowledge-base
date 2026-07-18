"""Chunk repository: derived fragments with half-precision embeddings.

Chunks are recreated whenever a document's content changes. Embedding values are
wrapped in :class:`pgvector.HalfVector` so asyncpg stores them into the
``halfvec(1024)`` column.
"""

from __future__ import annotations

from collections.abc import Sequence

import asyncpg
from pgvector import HalfVector


async def delete_for_document(conn: asyncpg.Connection, document_id: int) -> None:
    """Delete all chunks of a document (before recreating a new version)."""
    await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)


async def insert_many(
    conn: asyncpg.Connection,
    document_id: int,
    chunks: Sequence[tuple[int, str, Sequence[float]]],
) -> int:
    """Insert ``(chunk_index, content, embedding)`` rows; return the count.

    Each embedding is a length-``EMBED_DIM`` sequence of floats, stored as a
    ``halfvec``.
    """
    if not chunks:
        return 0
    rows = [
        (document_id, index, content, HalfVector(list(embedding)))
        for index, content, embedding in chunks
    ]
    await conn.executemany(
        """
        INSERT INTO chunks (document_id, chunk_index, content, embedding)
        VALUES ($1, $2, $3, $4)
        """,
        rows,
    )
    return len(rows)
