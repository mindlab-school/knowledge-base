"""Document repository: versioning, cards, and document->document links.

The original ``raw_content`` is the source of truth; everything else is derived
and recreated on a new version. Writes here always run inside the caller's
ingestion transaction.
"""

from __future__ import annotations

from typing import Any

import asyncpg

Row = dict[str, Any]

_TERMINAL_STATUSES = ("done", "failed", "skipped")


async def get_by_source_path(conn: asyncpg.Connection, source_path: str) -> Row | None:
    """Return ``{id, content_hash, version}`` for a source path, or ``None``."""
    record = await conn.fetchrow(
        "SELECT id, content_hash, version FROM documents WHERE source_path = $1",
        source_path,
    )
    return dict(record) if record is not None else None


async def get(conn: asyncpg.Connection, document_id: int) -> Row | None:
    """Return the full document card including ``raw_content``."""
    record = await conn.fetchrow(
        """
        SELECT id, title, source_path, source_kind, ingest_session_id, content_hash,
               version, raw_content, doc_type, attributes, extraction_status,
               extraction_model, extracted_at, created_at
        FROM documents WHERE id = $1
        """,
        document_id,
    )
    return dict(record) if record is not None else None


async def insert_document(
    conn: asyncpg.Connection,
    *,
    title: str,
    source_path: str,
    source_kind: str,
    content_hash: str,
    raw_content: str,
    doc_type: str,
    attributes: dict[str, Any],
    extraction_status: str,
    extraction_model: str | None,
    ingest_session_id: int | None,
) -> Row:
    """Insert a brand-new document (version 1) and return ``{id, version}``."""
    record = await conn.fetchrow(
        """
        INSERT INTO documents (
            title, source_path, source_kind, content_hash, raw_content,
            doc_type, attributes, extraction_status, extraction_model,
            extracted_at, ingest_session_id
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                CASE WHEN $8 = ANY($10::text[]) THEN now() ELSE NULL END, $11)
        RETURNING id, version
        """,
        title,
        source_path,
        source_kind,
        content_hash,
        raw_content,
        doc_type,
        attributes,
        extraction_status,
        extraction_model,
        list(_TERMINAL_STATUSES),
        ingest_session_id,
    )
    return dict(record)


async def bump_version(
    conn: asyncpg.Connection,
    document_id: int,
    *,
    title: str,
    content_hash: str,
    raw_content: str,
    doc_type: str,
    attributes: dict[str, Any],
    extraction_status: str,
    extraction_model: str | None,
    ingest_session_id: int | None,
) -> int:
    """Replace document content with a new version (+1). Derived rows are the
    caller's responsibility (delete chunks/mentions before, recreate after)."""
    record = await conn.fetchrow(
        """
        UPDATE documents SET
            title = $2,
            content_hash = $3,
            raw_content = $4,
            doc_type = $5,
            attributes = $6,
            extraction_status = $7,
            extraction_model = $8,
            extracted_at = CASE WHEN $7 = ANY($9::text[]) THEN now() ELSE NULL END,
            ingest_session_id = COALESCE($10, ingest_session_id),
            version = version + 1
        WHERE id = $1
        RETURNING version
        """,
        document_id,
        title,
        content_hash,
        raw_content,
        doc_type,
        attributes,
        extraction_status,
        extraction_model,
        list(_TERMINAL_STATUSES),
        ingest_session_id,
    )
    return int(record["version"])


async def update_extraction(
    conn: asyncpg.Connection,
    document_id: int,
    *,
    doc_type: str,
    attributes: dict[str, Any],
    extraction_status: str,
    extraction_model: str | None,
) -> None:
    """Overwrite type/attributes/status after a (re)extraction. Chunks untouched."""
    await conn.execute(
        """
        UPDATE documents SET
            doc_type = $2,
            attributes = $3,
            extraction_status = $4,
            extraction_model = $5,
            extracted_at = CASE WHEN $4 = ANY($6::text[]) THEN now() ELSE NULL END
        WHERE id = $1
        """,
        document_id,
        doc_type,
        attributes,
        extraction_status,
        extraction_model,
        list(_TERMINAL_STATUSES),
    )


async def set_type(conn: asyncpg.Connection, document_id: int, doc_type: str) -> None:
    """Set a document's type (admin ``set-type``), resetting extraction to pending."""
    await conn.execute(
        """
        UPDATE documents
        SET doc_type = $2, extraction_status = 'pending', attributes = '{}'::jsonb
        WHERE id = $1
        """,
        document_id,
        doc_type,
    )


async def chunk_count(conn: asyncpg.Connection, document_id: int) -> int:
    """Return the number of chunks stored for a document."""
    value = await conn.fetchval("SELECT count(*) FROM chunks WHERE document_id = $1", document_id)
    return int(value)


# --- document -> document links -----------------------------------------


async def resolve_document_id(conn: asyncpg.Connection, name: str) -> int | None:
    """Resolve a link target (title or source_path) to a document id."""
    value = await conn.fetchval(
        """
        SELECT id FROM documents
        WHERE source_path = $1 OR lower(title) = lower($1)
        ORDER BY id LIMIT 1
        """,
        name,
    )
    return int(value) if value is not None else None


async def add_link(
    conn: asyncpg.Connection, source_id: int, target_id: int, kind: str = "reference"
) -> None:
    """Create a document link (no-op on self-links or duplicates)."""
    if source_id == target_id:
        return
    await conn.execute(
        """
        INSERT INTO document_links (source_id, target_id, kind)
        VALUES ($1, $2, $3)
        ON CONFLICT (source_id, target_id, kind) DO NOTHING
        """,
        source_id,
        target_id,
        kind,
    )


async def add_pending_link(
    conn: asyncpg.Connection, source_id: int, target_name: str, kind: str = "reference"
) -> None:
    """Remember a link whose target document does not exist yet."""
    await conn.execute(
        """
        INSERT INTO pending_document_links (source_id, target_name, kind)
        VALUES ($1, $2, $3)
        ON CONFLICT (source_id, target_name, kind) DO NOTHING
        """,
        source_id,
        target_name,
        kind,
    )


async def resolve_pending_links_for(
    conn: asyncpg.Connection, document_id: int, title: str, source_path: str
) -> int:
    """Materialise pending links that point at this newly-available document.

    Returns the number of links created. Matches pending targets against the
    document's ``title`` (case-insensitive) or ``source_path``.
    """
    pending = await conn.fetch(
        """
        SELECT id, source_id, kind FROM pending_document_links
        WHERE lower(target_name) = lower($1) OR target_name = $2
        """,
        title,
        source_path,
    )
    created = 0
    for row in pending:
        if int(row["source_id"]) != document_id:
            await add_link(conn, int(row["source_id"]), document_id, row["kind"])
            created += 1
        await conn.execute("DELETE FROM pending_document_links WHERE id = $1", int(row["id"]))
    return created


async def clear_outgoing_links(conn: asyncpg.Connection, document_id: int) -> None:
    """Remove this document's outgoing links (materialised and pending).

    Called before re-writing links on re-ingestion so stale references do not
    accumulate across versions.
    """
    await conn.execute("DELETE FROM document_links WHERE source_id = $1", document_id)
    await conn.execute("DELETE FROM pending_document_links WHERE source_id = $1", document_id)
