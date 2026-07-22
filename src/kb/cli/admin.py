"""Admin CLI (spec section 8).

Operational commands over the knowledge base: statistics, entity/graph
maintenance, document inspection, fact tracing, cost reporting, whitelist
management, and the quality evaluation harness (section 13).

Usage: ``python -m kb.cli.admin <command> [args]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import asyncpg

from kb.agent import loop as agent_loop
from kb.agent.loop import NO_ANSWER
from kb.db.pool import close_pool, get_pool
from kb.db.repo import conversations as conversations_repo
from kb.db.repo import documents as documents_repo
from kb.db.repo import entities as entities_repo
from kb.db.repo import facts as facts_repo
from kb.db.repo import sessions as sessions_repo
from kb.db.repo import users as users_repo
from kb.doc_types.registry import DOC_TYPE_REGISTRY, is_known_type
from kb.ingestion.pipeline import reextract_document
from kb.search import semantic, structured

DEFAULT_EVAL_DATASET = Path("tests/fixtures/eval.json")

_THRESHOLDS = {
    "type_accuracy": 0.90,
    "recall_at_4": 0.85,
    "attribute_completeness": 0.80,
    "graph_accuracy": 0.75,
    "refusals": 1.00,
}


# --- reporting commands -------------------------------------------------


async def cmd_stats(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    if args.cost:
        await _stats_cost(conn)
        return
    print("documents by type:")
    for row in await conn.fetch(
        "SELECT doc_type, count(*) n FROM documents GROUP BY doc_type ORDER BY doc_type"
    ):
        print(f"  {row['doc_type']}: {row['n']}")
    print("documents by extraction_status:")
    for row in await conn.fetch(
        "SELECT extraction_status, count(*) n FROM documents GROUP BY extraction_status ORDER BY extraction_status"
    ):
        print(f"  {row['extraction_status']}: {row['n']}")
    print("entities by type:")
    for row in await conn.fetch(
        "SELECT entity_type, count(*) n FROM entities GROUP BY entity_type ORDER BY entity_type"
    ):
        print(f"  {row['entity_type']}: {row['n']}")
    chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    facts = await conn.fetchval("SELECT count(*) FROM facts WHERE invalid_at IS NULL")
    print(f"chunks: {chunks}")
    print(f"active facts: {facts}")


async def _stats_cost(conn: asyncpg.Connection) -> None:
    total = await conversations_repo.month_to_date_cost(conn)
    print(f"month-to-date cost: ${total:.4f}")
    print("by model:")
    for row in await conn.fetch(
        "SELECT usage->>'model' model, SUM((usage->>'cost')::float) cost, count(*) n "
        "FROM messages WHERE usage ? 'cost' GROUP BY usage->>'model' ORDER BY cost DESC"
    ):
        print(f"  {row['model']}: ${float(row['cost']):.4f} ({row['n']} calls)")
    print("by user:")
    for row in await conn.fetch(
        "SELECT u.name, SUM((m.usage->>'cost')::float) cost FROM messages m "
        "JOIN conversations c ON c.id = m.conversation_id JOIN users u ON u.id = c.user_id "
        "WHERE m.usage ? 'cost' GROUP BY u.name ORDER BY cost DESC"
    ):
        print(f"  {row['name']}: ${float(row['cost']):.4f}")


async def cmd_graph(_: argparse.Namespace, conn: asyncpg.Connection) -> None:
    print("nodes by type:")
    for row in await conn.fetch(
        "SELECT entity_type, count(*) n FROM entities GROUP BY entity_type ORDER BY n DESC"
    ):
        print(f"  {row['entity_type']}: {row['n']}")
    print("edges by relation (active):")
    for row in await conn.fetch(
        "SELECT relation, count(*) n FROM entity_relations WHERE invalid_at IS NULL GROUP BY relation ORDER BY n DESC"
    ):
        print(f"  {row['relation']}: {row['n']}")
    print("top-10 entities by degree:")
    for row in await conn.fetch(
        "SELECT e.name, e.entity_type, count(r.id) deg FROM entities e "
        "JOIN entity_relations r ON (r.source_id = e.id OR r.target_id = e.id) AND r.invalid_at IS NULL "
        "GROUP BY e.id, e.name, e.entity_type ORDER BY deg DESC LIMIT 10"
    ):
        print(f"  {row['name']} ({row['entity_type']}): {row['deg']}")
    isolated = await conn.fetchval(
        "SELECT count(*) FROM entities e WHERE NOT EXISTS ("
        "SELECT 1 FROM entity_relations r WHERE (r.source_id = e.id OR r.target_id = e.id) AND r.invalid_at IS NULL)"
    )
    print(f"isolated nodes: {isolated}")


async def cmd_sessions(_: argparse.Namespace, conn: asyncpg.Connection) -> None:
    rows = await conn.fetch(
        "SELECT s.id, u.name, s.started_at, s.last_active_at, "
        "(SELECT count(*) FROM ingest_buffer b WHERE b.session_id = s.id) buffered, "
        "(SELECT count(*) FROM documents d WHERE d.ingest_session_id = s.id) docs "
        "FROM ingest_sessions s JOIN users u ON u.id = s.user_id WHERE s.active ORDER BY s.id"
    )
    if not rows:
        print("no active sessions")
        return
    for row in rows:
        print(
            f"#{row['id']} {row['name']}: buffered={row['buffered']} docs={row['docs']} "
            f"last_active={row['last_active_at']}"
        )


async def cmd_facts(_: argparse.Namespace, conn: asyncpg.Connection) -> None:
    rows = await facts_repo.topics_with_updated(conn)
    if not rows:
        print("no facts")
        return
    for row in rows:
        print(f"{row['topic']}: {row['length']} chars, updated {row['updated_at']}")


async def cmd_show_doc(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    card = await structured.get_document_card(args.id, pool=await get_pool())
    if card is None:
        print("document not found", file=sys.stderr)
        return
    print(f"#{card['id']} {card['title']} ({card['doc_type']} v{card['version']})")
    print(f"source: {card['source_path']} · kind: {card['source_kind']} · chunks: {card['chunks']}")
    print(f"extraction: {card['extraction_status']} by {card['extraction_model']}")
    print(f"attributes: {json.dumps(card['attributes'], ensure_ascii=False)}")
    print("entities:")
    for entity in card["entities"]:
        print(f"  {entity['name']} ({entity['entity_type']}) — {entity['role']}")


async def cmd_set_type(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    if not is_known_type(args.type):
        print(f"unknown type: {args.type} (known: {', '.join(DOC_TYPE_REGISTRY)})", file=sys.stderr)
        return
    async with conn.transaction():
        await documents_repo.set_type(conn, args.id, args.type)
    result = await reextract_document(args.id, pool=await get_pool())
    if result.doc_type != args.type:
        async with conn.transaction():
            await documents_repo.update_extraction(
                conn,
                args.id,
                doc_type=args.type,
                attributes={},
                extraction_status="skipped",
                extraction_model=None,
            )
        print(
            f"forced type {args.type}; classifier suggested {result.doc_type}; attributes cleared"
        )
    else:
        print(f"type set to {args.type}; attributes extracted")


async def cmd_fact_history(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    versions = await facts_repo.history(conn, args.topic)
    if not versions:
        print("no such topic")
        return
    for version in versions:
        until = version["invalid_at"] or "now"
        print(f"[{version['valid_from']} → {until}] {version['content']}")


async def cmd_trace_fact(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    versions = await facts_repo.history(conn, args.topic)
    if not versions:
        print("no such topic")
        return
    for version in versions:
        print(f"version {version['id']} (from {version['valid_from']}):")
        print(f"  author user_id={version['created_by']} session_id={version['session_id']}")
        session_id = version["session_id"]
        if session_id is None:
            continue
        for message in await sessions_repo.buffer_messages(conn, session_id):
            print(f"    [{message['seq']}] {message['text']}")
        docs = await conn.fetch(
            "SELECT id, title FROM documents WHERE ingest_session_id = $1 ORDER BY id", session_id
        )
        for doc in docs:
            print(f"    doc #{doc['id']} {doc['title']}")


async def cmd_trace_doc(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    document = await documents_repo.get(conn, args.id)
    if document is None:
        print("document not found", file=sys.stderr)
        return
    session_id = document["ingest_session_id"]
    print(f"#{document['id']} {document['title']} — session_id={session_id}")
    if session_id is None:
        print("  (loaded outside a session)")
        return
    facts = await conn.fetch(
        "SELECT topic, valid_from FROM facts WHERE session_id = $1 ORDER BY id", session_id
    )
    for fact in facts:
        print(f"  fact «{fact['topic']}» created {fact['valid_from']}")


async def cmd_del_fact(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    async with conn.transaction():
        removed = await facts_repo.hard_delete(conn, args.topic)
    print(f"deleted {removed} version(s) of «{args.topic}»")


async def cmd_cleanup_entities(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    orphans = await conn.fetchval(
        "SELECT count(*) FROM entities e WHERE NOT EXISTS ("
        "SELECT 1 FROM entity_mentions m WHERE m.entity_id = e.id) AND NOT EXISTS ("
        "SELECT 1 FROM entity_relations r WHERE r.source_id = e.id OR r.target_id = e.id)"
    )
    if not orphans:
        print("no orphan entities")
        return
    if not args.yes and input(f"delete {orphans} orphan entities? [y/N] ").strip().lower() != "y":
        print("aborted")
        return
    async with conn.transaction():
        removed = await entities_repo.cleanup_orphans(conn)
    print(f"deleted {removed} entities")


async def cmd_add_user(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    async with conn.transaction():
        user = await users_repo.create(conn, args.telegram_id, args.name)
    print(f"whitelisted #{user['id']} {user['name']} (telegram_id={user['telegram_id']})")


# --- evaluation harness (section 13) ------------------------------------


async def cmd_eval(args: argparse.Namespace, conn: asyncpg.Connection) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"eval dataset not found: {dataset_path}", file=sys.stderr)
        return
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    pool = await get_pool()
    metrics: dict[str, float | None] = {}

    labels: dict[str, dict[str, Any]] = dataset.get("documents", {})
    if labels:
        metrics["type_accuracy"] = await _eval_type_accuracy(conn, labels)
        metrics["attribute_completeness"] = await _eval_attr_completeness(conn, labels)

    questions: list[dict[str, Any]] = dataset.get("questions", [])
    in_corpus = [q for q in questions if q.get("in_corpus", True) and q.get("expect_docs")]
    out_corpus = [q for q in questions if not q.get("in_corpus", True)]
    if in_corpus:
        metrics["recall_at_4"] = await _eval_recall(pool, in_corpus)
    if out_corpus:
        metrics["refusals"] = await _eval_refusals(pool, out_corpus)

    print(f"{'metric':<24}{'value':>8}{'threshold':>12}{'':>6}")
    for name, threshold in _THRESHOLDS.items():
        value = metrics.get(name)
        if value is None:
            print(f"{name:<24}{'n/a':>8}{threshold:>12.2f}{'—':>6}")
            continue
        mark = "ok" if value >= threshold else "FAIL"
        print(f"{name:<24}{value:>8.2f}{threshold:>12.2f}{mark:>6}")


async def _eval_type_accuracy(
    conn: asyncpg.Connection, labels: dict[str, dict[str, Any]]
) -> float | None:
    total = 0
    correct = 0
    for title, label in labels.items():
        row = await conn.fetchrow(
            "SELECT doc_type FROM documents WHERE lower(title) = lower($1) ORDER BY id LIMIT 1",
            title,
        )
        if row is None:
            continue
        total += 1
        if row["doc_type"] == label.get("doc_type"):
            correct += 1
    return correct / total if total else None


async def _eval_attr_completeness(
    conn: asyncpg.Connection, labels: dict[str, dict[str, Any]]
) -> float | None:
    required_total = 0
    present = 0
    for title, label in labels.items():
        required = label.get("required", [])
        if not required:
            continue
        row = await conn.fetchrow(
            "SELECT attributes FROM documents WHERE lower(title) = lower($1) ORDER BY id LIMIT 1",
            title,
        )
        if row is None:
            continue
        attributes = row["attributes"] or {}
        for field in required:
            required_total += 1
            if attributes.get(field) not in (None, "", []):
                present += 1
    return present / required_total if required_total else None


async def _eval_recall(pool: asyncpg.Pool, questions: list[dict[str, Any]]) -> float:
    hits = 0
    for question in questions:
        results = await semantic.search_chunks(question["q"], pool=pool)
        titles = {str(item["document"]) for item in results}
        ids = {int(item["document_id"]) for item in results}
        expected = question["expect_docs"]
        if any(str(exp) in titles or (str(exp).isdigit() and int(exp) in ids) for exp in expected):
            hits += 1
    return hits / len(questions)


async def _eval_refusals(pool: asyncpg.Pool, questions: list[dict[str, Any]]) -> float:
    refused = 0
    for question in questions:
        result = await agent_loop.answer(question["q"], conversation_id=0, pool=pool)
        if NO_ANSWER in result.answer.lower():
            refused += 1
    return refused / len(questions)


# --- dispatch -----------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KB Agent admin commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="counts (add --cost for spend)")
    stats.add_argument("--cost", action="store_true")
    stats.set_defaults(func=cmd_stats)

    sub.add_parser("graph", help="graph size report").set_defaults(func=cmd_graph)
    sub.add_parser("sessions", help="active ingest sessions").set_defaults(func=cmd_sessions)
    sub.add_parser("facts", help="fact topics").set_defaults(func=cmd_facts)

    show = sub.add_parser("show-doc", help="show a document")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show_doc)

    set_type = sub.add_parser("set-type", help="force a document type and re-extract")
    set_type.add_argument("id", type=int)
    set_type.add_argument("type")
    set_type.set_defaults(func=cmd_set_type)

    fact_history = sub.add_parser("fact-history", help="fact versions")
    fact_history.add_argument("topic")
    fact_history.set_defaults(func=cmd_fact_history)

    trace_fact = sub.add_parser("trace-fact", help="trace a fact to its sources")
    trace_fact.add_argument("topic")
    trace_fact.set_defaults(func=cmd_trace_fact)

    trace_doc = sub.add_parser("trace-doc", help="trace a document's session and facts")
    trace_doc.add_argument("id", type=int)
    trace_doc.set_defaults(func=cmd_trace_doc)

    del_fact = sub.add_parser("del-fact", help="hard-delete a fact topic")
    del_fact.add_argument("topic")
    del_fact.set_defaults(func=cmd_del_fact)

    cleanup = sub.add_parser("cleanup-entities", help="delete orphan entities")
    cleanup.add_argument("--yes", action="store_true")
    cleanup.set_defaults(func=cmd_cleanup_entities)

    add_user = sub.add_parser("add-user", help="whitelist a telegram user")
    add_user.add_argument("telegram_id", type=int)
    add_user.add_argument("name")
    add_user.set_defaults(func=cmd_add_user)

    evaluate = sub.add_parser("eval", help="run the quality evaluation set")
    evaluate.add_argument("dataset", nargs="?", default=str(DEFAULT_EVAL_DATASET))
    evaluate.set_defaults(func=cmd_eval)

    return parser


async def _run(args: argparse.Namespace) -> int:
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            await args.func(args, conn)
    finally:
        await close_pool()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
