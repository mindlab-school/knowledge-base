# KB Agent

A corporate **agentic knowledge base**. Employees ask questions in Telegram — or
from Claude Code via MCP — and a Claude-based agent answers **strictly from company
documents, always citing its sources**. Documents are auto-typed and structured on
ingestion (entities, relations, attributes) by a single LLM call, so there is no
manual tagging. Volatile business context (prices, schedules, contacts) is kept as
compact **facts** injected into every answer.

## Highlights

- **Grounded answers only.** The agent replies from the knowledge base or says the
  information isn't there — never from the model's general knowledge or the open web.
- **Agentic retrieval.** A Claude model picks and combines four tools per question
  (semantic search, structured document query, entity/graph exploration, full
  document fetch) — no hard-coded pipeline.
- **Automatic structuring on ingest.** One LLM call classifies the document type and
  extracts typed attributes, entities, relations, and cross-document links.
- **Facts as first-class context.** Bitemporal, one-per-topic records are prepended
  to every prompt, answering common questions without a search round-trip.
- **Hybrid search.** Vector (HNSW cosine over half-precision embeddings) and Russian
  full-text candidates fused with Reciprocal Rank Fusion.
- **A knowledge graph without a graph database.** Entities and relations live in
  plain PostgreSQL; traversal is a single recursive CTE.
- **No heavyweight frameworks.** No ORM, no LangChain, no separate vector store.
  PostgreSQL + `asyncpg` + `pgvector` do it all.

## Architecture

```
                          QUERY PATH (online)
Telegram bot / MCP ──HTTP──▶ FastAPI backend ──▶ Agent loop (Claude, 4 tools)
 (thin clients)          (identity, history)              │
                                                          ▼
                          Search: semantic | structured | entity/graph | facts
                                                          ▼
                             PostgreSQL 18 + pgvector (halfvec, HNSW)
                                                          ▲
                          DATA PATH (offline)             │  one transaction
Files / URL ──▶ Ingestion: parse ▶ LLM extract ▶ chunk ▶ embed ▶ write
```

The core is **channel-agnostic**: the Telegram bot and the Claude Code MCP server
are thin HTTP clients over a single FastAPI backend. All writing SQL lives in
`src/kb/db/repo`; the original document text is the source of truth and everything
else (chunks, embeddings, entities, edges) is derived and recreated on change.

```
src/kb/
├── config.py       # single settings source (reads .env once)
├── db/             # asyncpg pool + repo/ (all writing SQL)
├── doc_types/      # Pydantic models + registry for document types
├── llm/            # OpenRouter chat client (OpenAI SDK)
├── ingestion/      # sources → extract → chunk → embed → write
├── search/         # semantic, structured, entities, graph, facts
├── agent/          # tool schemas + dispatcher + tool-use loop + prompts
├── api/            # FastAPI backend (only HTTP surface over the core)
├── channels/       # aiogram Telegram bot
└── cli/            # ingest, admin, reextract entry points
```

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.13 (managed by [`uv`](https://docs.astral.sh/uv/)) |
| Database | PostgreSQL 18 + `pgvector` 0.8.2 (`halfvec(1024)`, HNSW cosine) |
| DB driver | `asyncpg` (no ORM) |
| Backend | FastAPI + Uvicorn |
| Telegram | `aiogram` 3 (long polling) |
| LLM gateway | OpenRouter via the OpenAI SDK; default agent model `anthropic/claude-haiku-4.5` |
| Embeddings | `voyage-4-nano` — local ONNX (int8, CPU) by default, or OpenRouter |
| Parsing | `docling` (PDF/DOCX/PPTX), `trafilatura` (HTML/URL) |
| Tooling | `ruff`, `mypy --strict`, `pytest` + `testcontainers` |

Heavy dependencies are optional extras loaded lazily: `parse` (docling),
`embed-local` (onnxruntime + tokenizers), and `mcp` (plugin server only).

## Getting Started

**Prerequisites:** Python 3.13, [`uv`](https://docs.astral.sh/uv/), and Docker (for
PostgreSQL). An OpenRouter API key is required; a Telegram bot token is needed only
to run the bot.

```bash
uv sync                       # install core + dev dependencies
cp .env.example .env          # then fill in OPENROUTER_API_KEY (and TELEGRAM_BOT_TOKEN)
make up                       # start PostgreSQL + pgvector
make migrate                  # apply forward-only SQL migrations
```

## Usage

```bash
make ingest FILES="docs/policy.pdf docs/faq.md"   # ingest documents
make ingest-url URL="https://example.com/page"    # ingest a web page
make run-api                                       # FastAPI backend on :8000
make run-bot                                       # Telegram bot (long polling)
```

Only whitelisted Telegram users may query the backend; unknown callers get `403`.
Manage the whitelist with the admin CLI (`uv run python -m kb.cli.admin`).

The FastAPI backend exposes `/chat`, `/reset`, `/search`, `/facts`, the
`/ingest/*` endpoints, and `/health`. See `src/kb/api/app.py` for the full surface.

### Claude Code plugin

The `plugin/` directory ships an MCP stdio server and `/kb-add`, `/kb-search`, and
`/kb-facts` commands, giving Claude Code direct access to the knowledge base.

## Configuration

All settings are read once from `.env` (see `.env.example`). Key variables:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM gateway credential (required) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (bot only) |
| `DATABASE_URL` | PostgreSQL connection string |
| `AGENT_MODEL` | Agent model (default `anthropic/claude-haiku-4.5`) |
| `EMBED_BACKEND` | `local` (ONNX on CPU) or `openrouter` |
| `OR_MONTHLY_LIMIT` | Soft monthly spend limit, USD |

## Development

```bash
make check              # ruff lint + mypy --strict + unit tests
make test               # full suite (integration tests need Docker)
make test-unit          # unit tests only
make lint format typecheck
```

Integration tests spin up a real PostgreSQL + pgvector container via
`testcontainers`; deselect them with `-m "not integration"`. A behavioural eval
harness lives in `scripts/eval_hw.py` (`make eval`).

## Deployment

A single Docker image serves both processes; the entrypoint selects `api` or `bot`.
`docker-compose.yml` defines `db`, `api`, and `bot` services (the latter two behind
Compose profiles):

```bash
docker compose --profile api --profile bot up -d
```

## License

MIT
