"""Ingestion pipeline: source -> extract -> chunk -> embed -> write (spec section 3).

Extraction and embedding (slow, network-bound) run before the short write
transaction so a DB transaction is never held open across an LLM call. Ingestion
is idempotent: identical content is a no-op; changed content at the same
``source_path`` bumps the version and recreates all derived rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import asyncpg

from kb.config import Settings, get_settings
from kb.db.pool import get_pool
from kb.db.repo import chunks as chunks_repo
from kb.db.repo import documents as documents_repo
from kb.db.repo import entities as entities_repo
from kb.ingestion.chunking import chunk_document
from kb.ingestion.embeddings import Embedder, make_embedder
from kb.ingestion.extraction import ExtractionResult, extract_structure
from kb.ingestion.sources.base import LoadedDoc, Source
from kb.llm.client import LLMClient


@dataclass(slots=True)
class IngestResult:
    """Result card for a single ingestion (spec section 6)."""

    document_id: int
    title: str
    doc_type: str
    version: int
    chunks: int
    extraction_status: str
    action: str  # created | updated | unchanged


def content_hash(text: str) -> str:
    """Return the sha256 hex digest of the document text (idempotency key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _write_graph(
    conn: asyncpg.Connection, document_id: int, extraction: ExtractionResult
) -> None:
    """Upsert entities + mentions and replace this document's relation edges."""
    canonical_to_id: dict[str, int] = {}
    for entity in extraction.entities:
        entity_id = await entities_repo.upsert_entity(conn, entity.entity_type, entity.name)
        canonical_to_id[entities_repo.canonicalize(entity.name)] = entity_id
        await entities_repo.add_mention(conn, entity_id, document_id, entity.role)

    edges: list[tuple[int, int, str]] = []
    for relation in extraction.relations:
        source_id = canonical_to_id.get(entities_repo.canonicalize(relation.source))
        target_id = canonical_to_id.get(entities_repo.canonicalize(relation.target))
        if source_id is not None and target_id is not None:
            edges.append((source_id, target_id, relation.relation))
    await entities_repo.replace_relations(conn, document_id, edges)


async def _write_links(
    conn: asyncpg.Connection, document_id: int, loaded: LoadedDoc, extraction: ExtractionResult
) -> None:
    """Resolve extracted links to document ids; defer unresolved ones."""
    await documents_repo.clear_outgoing_links(conn, document_id)
    for name in extraction.links:
        target_id = await documents_repo.resolve_document_id(conn, name)
        if target_id is not None and target_id != document_id:
            await documents_repo.add_link(conn, document_id, target_id)
        elif target_id is None:
            await documents_repo.add_pending_link(conn, document_id, name)
    await documents_repo.resolve_pending_links_for(
        conn, document_id, loaded.title, loaded.source_path
    )


async def _write(
    conn: asyncpg.Connection,
    loaded: LoadedDoc,
    digest: str,
    extraction: ExtractionResult,
    chunk_texts: list[str],
    embeddings: list[list[float]],
    session_id: int | None,
    existing: dict[str, Any] | None,
) -> IngestResult:
    if existing is None:
        row = await documents_repo.insert_document(
            conn,
            title=loaded.title,
            source_path=loaded.source_path,
            source_kind=loaded.source_kind,
            content_hash=digest,
            raw_content=loaded.text,
            doc_type=extraction.doc_type,
            attributes=extraction.attributes,
            extraction_status=extraction.status,
            extraction_model=extraction.model,
            ingest_session_id=session_id,
        )
        document_id = int(row["id"])
        version = int(row["version"])
        action = "created"
    else:
        document_id = int(existing["id"])
        await chunks_repo.delete_for_document(conn, document_id)
        await entities_repo.delete_mentions_for_document(conn, document_id)
        version = await documents_repo.bump_version(
            conn,
            document_id,
            title=loaded.title,
            content_hash=digest,
            raw_content=loaded.text,
            doc_type=extraction.doc_type,
            attributes=extraction.attributes,
            extraction_status=extraction.status,
            extraction_model=extraction.model,
            ingest_session_id=session_id,
        )
        action = "updated"

    n_chunks = await chunks_repo.insert_many(
        conn,
        document_id,
        [
            (index, text, embedding)
            for index, (text, embedding) in enumerate(zip(chunk_texts, embeddings, strict=True))
        ],
    )
    await _write_graph(conn, document_id, extraction)
    await _write_links(conn, document_id, loaded, extraction)

    return IngestResult(
        document_id=document_id,
        title=loaded.title,
        doc_type=extraction.doc_type,
        version=version,
        chunks=n_chunks,
        extraction_status=extraction.status,
        action=action,
    )


async def ingest_loaded(
    loaded: LoadedDoc,
    *,
    session_id: int | None = None,
    pool: asyncpg.Pool | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """Ingest an already-loaded document through the full pipeline."""
    settings = settings or get_settings()
    pool = pool or await get_pool()
    embedder = embedder or make_embedder(settings)
    llm = llm or LLMClient(settings)

    digest = content_hash(loaded.text)
    async with pool.acquire() as conn:
        existing = await documents_repo.get_by_source_path(conn, loaded.source_path)

    if existing is not None and existing["content_hash"] == digest:
        async with pool.acquire() as conn:
            document = await documents_repo.get(conn, int(existing["id"]))
            n_chunks = await documents_repo.chunk_count(conn, int(existing["id"]))
        assert document is not None
        return IngestResult(
            document_id=int(document["id"]),
            title=document["title"],
            doc_type=document["doc_type"],
            version=int(document["version"]),
            chunks=n_chunks,
            extraction_status=document["extraction_status"],
            action="unchanged",
        )

    extraction = await extract_structure(loaded.text, llm=llm, settings=settings)
    chunk_texts = chunk_document(loaded.text, extraction.doc_type, extraction.attributes)
    embeddings = await embedder.embed_documents(chunk_texts) if chunk_texts else []

    async with pool.acquire() as conn, conn.transaction():
        return await _write(
            conn, loaded, digest, extraction, chunk_texts, embeddings, session_id, existing
        )


async def ingest_source(
    source: Source,
    *,
    session_id: int | None = None,
    pool: asyncpg.Pool | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """Load a source and ingest it."""
    loaded = await source.load()
    return await ingest_loaded(
        loaded,
        session_id=session_id,
        pool=pool,
        embedder=embedder,
        llm=llm,
        settings=settings,
    )


async def reextract_document(
    document_id: int,
    *,
    pool: asyncpg.Pool | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Re-run extraction for a stored document, rewriting type/attributes/graph.

    Chunks and embeddings are left untouched (spec section 3).
    """
    settings = settings or get_settings()
    pool = pool or await get_pool()
    llm = llm or LLMClient(settings)

    async with pool.acquire() as conn:
        document = await documents_repo.get(conn, document_id)
    if document is None:
        raise ValueError(f"документ {document_id} не найден")

    extraction = await extract_structure(document["raw_content"], llm=llm, settings=settings)

    async with pool.acquire() as conn, conn.transaction():
        await documents_repo.update_extraction(
            conn,
            document_id,
            doc_type=extraction.doc_type,
            attributes=extraction.attributes,
            extraction_status=extraction.status,
            extraction_model=extraction.model,
        )
        await entities_repo.delete_mentions_for_document(conn, document_id)
        await _write_graph(conn, document_id, extraction)
    return extraction
