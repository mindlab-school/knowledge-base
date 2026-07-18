"""Bulk ingestion CLI (spec section 11).

Usage::

    python -m kb.cli.ingest <file> [<file> ...]   # ingest files
    python -m kb.cli.ingest url <url> [<url> ...]  # ingest URLs
    python -m kb.cli.ingest refresh-urls           # re-fetch all url documents
"""

from __future__ import annotations

import asyncio
import sys

from kb.db.pool import close_pool, get_pool
from kb.db.repo import documents as documents_repo
from kb.ingestion.pipeline import IngestResult, ingest_source
from kb.ingestion.sources.files import FileSource
from kb.ingestion.sources.urls import UrlSource

_USAGE = (
    "usage:\n"
    "  python -m kb.cli.ingest <file> [<file> ...]\n"
    "  python -m kb.cli.ingest url <url> [<url> ...]\n"
    "  python -m kb.cli.ingest refresh-urls"
)


def _print(result: IngestResult) -> None:
    print(
        f"[{result.action}] #{result.document_id} {result.title} "
        f"({result.doc_type} v{result.version}, chunks={result.chunks}, "
        f"extraction={result.extraction_status})"
    )


async def _ingest_files(paths: list[str]) -> None:
    for path in paths:
        try:
            result = await ingest_source(FileSource(path))
        except (ValueError, RuntimeError) as exc:
            print(f"[error] {path}: {exc}", file=sys.stderr)
            continue
        _print(result)


async def _ingest_urls(urls: list[str]) -> None:
    for url in urls:
        try:
            result = await ingest_source(UrlSource(url))
        except (ValueError, RuntimeError) as exc:
            print(f"[error] {url}: {exc}", file=sys.stderr)
            continue
        _print(result)


async def _refresh_urls() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        source_paths = await documents_repo.list_url_source_paths(conn)
    if not source_paths:
        print("no url-sourced documents")
        return
    await _ingest_urls(source_paths)


async def _run(argv: list[str]) -> int:
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 1
    try:
        command = argv[0]
        if command == "url":
            if len(argv) < 2:
                print(_USAGE, file=sys.stderr)
                return 1
            await _ingest_urls(argv[1:])
        elif command == "refresh-urls":
            await _refresh_urls()
        else:
            await _ingest_files(argv)
    finally:
        await close_pool()
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
