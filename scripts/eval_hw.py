"""Reusable evaluation harness for the KB agent on the "Hello World" corpus.

One command re-loads the fixed corpus snapshot and re-asks the 100 benchmark
questions against the in-process agent loop (no HTTP server required), then
reports how the agent behaved.

Usage::

    uv run python scripts/eval_hw.py ingest          # reset DB + ingest corpus snapshot
    uv run python scripts/eval_hw.py run              # ask all 100 questions -> results JSON
    uv run python scripts/eval_hw.py run --only 31,33 # ask a subset
    uv run python scripts/eval_hw.py report           # metrics from the last results
    uv run python scripts/eval_hw.py all --judge      # ingest + run + LLM-graded report
    uv run python scripts/eval_hw.py all              # everything, heuristic report

Data lives under ``tests/fixtures/``: ``hw_corpus/`` (13 document snapshots +
``manifest.json``) and ``hw_questions.json`` (questions + expected answers +
``out_of_corpus`` flags). The harness never persists chat messages, so it does
not consume the agent's month-to-date spend budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.agent.loop import answer as agent_answer
from kb.config import Settings, get_settings
from kb.db.pool import close_pool, get_pool
from kb.embeddings import Embedder, make_embedder
from kb.ingestion.pipeline import ingest_loaded
from kb.ingestion.sources.base import LoadedDoc
from kb.llm.client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
CORPUS_DIR = FIXTURES / "hw_corpus"
QUESTIONS_PATH = FIXTURES / "hw_questions.json"
DEFAULT_RESULTS = ROOT / "experiments" / "hw_eval_results.json"

# Tables cleared on reset (children before parents), mirroring the test harness.
_RESET_TABLES = [
    "messages",
    "conversations",
    "facts",
    "pending_document_links",
    "document_links",
    "entity_relations",
    "entity_mentions",
    "entities",
    "chunks",
    "documents",
    "ingest_buffer",
    "ingest_sessions",
]

_REFUSAL = "в базе знаний этого нет"
_CLARIFY_MARKERS = ("уточн", "переформулир", "слишком общ", "больше информации", "не совсем понял")


@dataclass(slots=True)
class Question:
    """One benchmark question with its ground-truth expectation."""

    n: int
    section: str
    text: str
    expected: str
    out_of_corpus: bool


def load_questions() -> list[Question]:
    """Load the 100 benchmark questions from the fixture."""
    raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [
        Question(
            n=int(q["n"]),
            section=str(q["section"]),
            text=str(q["question"]),
            expected=str(q["expected"]),
            out_of_corpus=bool(q["out_of_corpus"]),
        )
        for q in raw
    ]


def load_corpus() -> list[LoadedDoc]:
    """Load the corpus snapshot as ``LoadedDoc`` objects (URL provenance preserved)."""
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    docs: list[LoadedDoc] = []
    for entry in manifest:
        text = (CORPUS_DIR / str(entry["file"])).read_text(encoding="utf-8")
        docs.append(
            LoadedDoc(
                source_path=str(entry["source_path"]),
                title=str(entry["title"]),
                text=text,
                source_kind=str(entry.get("source_kind", "url")),
            )
        )
    return docs


def _eval_settings() -> Settings:
    """Settings with the spend guard lifted (the harness never persists cost)."""
    return get_settings().model_copy(update={"or_monthly_limit": 1_000_000.0})


def categorize(answer: str) -> str:
    """Classify a raw answer as ``refused`` | ``clarification`` | ``answered``."""
    low = answer.lower()
    if _REFUSAL in low:
        return "refused"
    if any(m in low for m in _CLARIFY_MARKERS) and "источник" not in low:
        return "clarification"
    return "answered"


async def reset_db() -> None:
    """Truncate all content tables (fresh load)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(_RESET_TABLES)} RESTART IDENTITY CASCADE")


async def ingest_corpus() -> None:
    """Reset the database and ingest the corpus snapshot from scratch."""
    settings = _eval_settings()
    embedder = make_embedder(settings)
    llm = LLMClient(settings)
    await reset_db()
    docs = load_corpus()
    print(f"ingesting {len(docs)} documents…", flush=True)
    for doc in docs:
        result = await ingest_loaded(doc, embedder=embedder, llm=llm, settings=settings)
        print(
            f"  [{result.action}] #{result.document_id} {doc.title[:48]} "
            f"({result.doc_type}, chunks={result.chunks}, extraction={result.extraction_status})",
            flush=True,
        )
    print("ingest complete.", flush=True)


async def _ask(
    q: Question,
    *,
    settings: Settings,
    llm: LLMClient,
    embedder: Embedder,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        t0 = time.monotonic()
        try:
            res = await agent_answer(
                q.text,
                history=[],
                conversation_id=q.n,
                settings=settings,
                llm=llm,
                embedder=embedder,
            )
            answer, rounds, category = res.answer, res.rounds, categorize(res.answer)
        except Exception as exc:  # keep one bad question from aborting the whole run
            answer, rounds, category = f"<error: {type(exc).__name__}: {exc}>", 0, "error"
        dt = round(time.monotonic() - t0, 2)
    return {
        "n": q.n,
        "section": q.section,
        "question": q.text,
        "expected": q.expected,
        "out_of_corpus": q.out_of_corpus,
        "answer": answer,
        "rounds": rounds,
        "latency_s": dt,
        "category": category,
    }


async def run_questions(
    only: set[int] | None, concurrency: int, out_path: Path
) -> list[dict[str, Any]]:
    """Ask the (optionally filtered) questions concurrently and save results."""
    settings = _eval_settings()
    llm = LLMClient(settings)
    embedder = make_embedder(settings)
    questions = [q for q in load_questions() if only is None or q.n in only]
    sem = asyncio.Semaphore(concurrency)
    print(f"asking {len(questions)} questions (concurrency={concurrency})…", flush=True)
    tasks = [_ask(q, settings=settings, llm=llm, embedder=embedder, sem=sem) for q in questions]
    results: list[dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        print(
            f"  [{r['n']:>3}] ({r['latency_s']:>5}s) {r['category']:<13} {r['question'][:48]}",
            flush=True,
        )
    results.sort(key=lambda r: int(r["n"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(results)} results -> {out_path}", flush=True)
    return results


_JUDGE_SYSTEM = (
    "Ты — строгий оценщик. Дан вопрос к базе знаний, эталонный ответ (истина из документов) "
    "и ответ системы. Верни ОДНО слово-вердикт из списка: correct, partial, incorrect, "
    "hallucinated, false_refusal, unnecessary_clarification, correct_refusal. "
    "Для вопросов ВНЕ КОРПУСА корректный отказ = correct_refusal; выдумка = hallucinated. "
    "Отвечай только словом-вердиктом, без пояснений."
)


async def _judge_one(
    r: dict[str, Any], *, settings: Settings, llm: LLMClient, sem: asyncio.Semaphore
) -> str:
    prompt = (
        f"ВОПРОС: {r['question']}\n"
        f"ВНЕ КОРПУСА: {'да' if r['out_of_corpus'] else 'нет'}\n"
        f"ЭТАЛОН: {r['expected']}\n"
        f"ОТВЕТ СИСТЕМЫ: {r['answer']}\n\nВердикт:"
    )
    async with sem:
        resp = await llm.chat(
            model=settings.agent_model_slug,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
        )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    allowed = {
        "correct",
        "partial",
        "incorrect",
        "hallucinated",
        "false_refusal",
        "unnecessary_clarification",
        "correct_refusal",
    }
    for token in verdict.replace(",", " ").split():
        if token in allowed:
            return token
    return "unknown"


async def judge(results: list[dict[str, Any]], concurrency: int) -> list[dict[str, Any]]:
    """Attach an LLM verdict to each result."""
    settings = _eval_settings()
    llm = LLMClient(settings)
    sem = asyncio.Semaphore(concurrency)
    print(f"judging {len(results)} answers…", flush=True)
    verdicts = await asyncio.gather(
        *(_judge_one(r, settings=settings, llm=llm, sem=sem) for r in results)
    )
    for r, v in zip(results, verdicts, strict=True):
        r["verdict"] = v
    return results


def report(results: list[dict[str, Any]]) -> None:
    """Print behaviour + (if judged) correctness metrics."""
    in_corpus = [r for r in results if not r["out_of_corpus"]]
    out_corpus = [r for r in results if r["out_of_corpus"]]

    print("\n=== BEHAVIOUR (raw categories) ===")
    cats: dict[str, int] = {}
    for r in results:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    for k in ("answered", "clarification", "refused"):
        print(f"  {k:<14} {cats.get(k, 0)}")

    if in_corpus:
        deflected = sum(1 for r in in_corpus if r["category"] in ("refused", "clarification"))
        print(
            f"\n  in-corpus deflection (refused+clarify): {deflected}/{len(in_corpus)} "
            f"({deflected * 100 // len(in_corpus)}%)"
        )
    if out_corpus:
        refused_ok = sum(1 for r in out_corpus if r["category"] == "refused")
        print(f"  out-of-corpus refusals: {refused_ok}/{len(out_corpus)}")

    if results and "verdict" in results[0]:
        print("\n=== CORRECTNESS (LLM judge) ===")
        vc: dict[str, int] = {}
        for r in results:
            vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
        for k in (
            "correct",
            "partial",
            "incorrect",
            "hallucinated",
            "false_refusal",
            "unnecessary_clarification",
            "correct_refusal",
            "unknown",
        ):
            if vc.get(k):
                print(f"  {k:<26} {vc[k]}")
        ic = [r for r in in_corpus if "verdict" in r]
        attempted = sum(1 for r in ic if r["verdict"] in ("correct", "partial", "incorrect"))
        good = sum(1 for r in ic if r["verdict"] == "correct")
        if attempted:
            print(
                f"\n  in-corpus correct: {good}/{len(ic)} "
                f"(of {attempted} attempted: {good * 100 // attempted}%)"
            )


def _parse_only(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(x) for x in value.replace(" ", "").split(",") if x}


async def _main_async(args: argparse.Namespace) -> None:
    out_path = Path(args.out) if args.out else DEFAULT_RESULTS
    try:
        if args.command in ("ingest", "all"):
            await ingest_corpus()
        if args.command in ("run", "all"):
            results = await run_questions(_parse_only(args.only), args.concurrency, out_path)
        elif args.command == "report":
            results = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
        if args.command in ("run", "report", "all"):
            if args.judge and results:
                results = await judge(results, args.concurrency)
                out_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if results:
                report(results)
    finally:
        await close_pool()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KB agent evaluation harness (Hello World corpus)."
    )
    parser.add_argument("command", choices=["ingest", "run", "report", "all"])
    parser.add_argument("--only", help="Comma-separated question numbers to run (default: all).")
    parser.add_argument(
        "--concurrency", type=int, default=6, help="Concurrent requests (default 6)."
    )
    parser.add_argument("--judge", action="store_true", help="Grade answers with an LLM judge.")
    parser.add_argument("--out", help=f"Results JSON path (default {DEFAULT_RESULTS}).")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
