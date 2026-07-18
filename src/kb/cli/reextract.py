"""Re-extraction CLI (spec section 3).

Re-runs extraction (type + attributes + graph) for stored documents selected by
filter; chunks and embeddings are left untouched.

Usage::

    python -m kb.cli.reextract [--status failed|pending] [--type generic] \\
        [--model-older-than <model>] [--all]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from kb.db.pool import close_pool, get_pool
from kb.db.repo import documents as documents_repo
from kb.ingestion.pipeline import reextract_document


async def _run(args: argparse.Namespace) -> int:
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            ids = await documents_repo.list_ids_for_reextract(
                conn,
                status=args.status,
                doc_type=args.type,
                not_model=args.model_older_than,
            )
        if not ids:
            print("no documents matched")
            return 0
        print(f"re-extracting {len(ids)} document(s)…")
        for document_id in ids:
            result = await reextract_document(document_id, pool=pool)
            print(
                f"#{document_id}: {result.doc_type} "
                f"(status={result.status}, entities={len(result.entities)})"
            )
    finally:
        await close_pool()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-extract stored documents.")
    parser.add_argument("--status", choices=["failed", "pending", "done", "skipped"])
    parser.add_argument("--type", dest="type")
    parser.add_argument("--model-older-than", dest="model_older_than")
    parser.add_argument("--all", action="store_true", help="re-extract every document")
    args = parser.parse_args(argv)

    if not (args.status or args.type or args.model_older_than or args.all):
        parser.error("choose a filter (--status/--type/--model-older-than) or --all")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
