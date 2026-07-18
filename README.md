# KB Agent

Corporate **agentic knowledge base**. Employees ask questions in Telegram; a
Claude-based agent (via OpenRouter) decides which tools to use and answers
strictly from company documents, always citing sources. It handles two classes
of questions:

- **content** — "what does the vacation policy say", "how to file a business
  trip" (semantic search over chunks);
- **documents and entities as objects** — "which regulations are overdue for
  review", "who owns the onboarding process", "what was decided on project X"
  (structured queries over metadata, entities and a graph).

On ingestion each document is auto-typed and an LLM extracts attributes and
entities (with relations) in a single call — no manual labelling. Alongside
documents, the system stores **facts**: compact, one-per-topic records of current
business context (prices, packages, schedule) that go into every answer.

## Architecture

```
                    QUERY PATH (online)
Telegram bot / MCP ──HTTP──▶ FastAPI backend ──▶ Agent loop (Claude, 4 tools)
 (thin clients)         (identity, history)        │ picks & combines
                                                   ▼
                              Search: semantic | structured | entity/graph
                                                   ▼
                                     PostgreSQL 18 + pgvector (halfvec, HNSW)
                                                   ▲
                    DATA PATH (offline)            │ one transaction
Files / URL / text ──▶ Ingestion: parse ▶ LLM extract ▶ chunk ▶ embed ▶ write
```

- **Core** is channel-agnostic; the Telegram bot and the Claude Code MCP server
  are thin clients over the HTTP API.
- **Writing SQL lives only in `src/kb/db/repo`**; reads are also allowed in
  `src/kb/search`. Dependency direction: `channels → api → agent → search →
  db/repo → pool`; `ingestion`/`doc_types → db/repo`.
- The original `raw_content` is the source of truth; chunks, embeddings,
  attributes, mentions and edges are all derived and recreated on change.

## Requirements

- Python 3.13 (managed by `uv`)
- [`uv`](https://docs.astral.sh/uv/) ≥ 0.5
- Docker + Docker Compose v2 (for PostgreSQL + pgvector, and integration tests)
- An OpenRouter API key and a Telegram bot token (for live use)

## Setup

```bash
uv sync                    # create the venv and install deps
cp .env.example .env       # then fill in OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
make up                    # start PostgreSQL (pgvector) via docker compose
make migrate               # apply forward-only SQL migrations
```

Configuration is read from `.env` in one place (`src/kb/config.py`); see
[`.env.example`](.env.example) for every setting.

## Running

```bash
make run-api               # FastAPI backend on :8000
make run-bot               # Telegram bot (long polling)
```

Or with Docker Compose (one image, entrypoint picks the process):

```bash
docker compose --profile api up --build     # db + api
docker compose --profile bot up --build     # db + bot
```

### Whitelist

Access is whitelist-only. Add a user before they can talk to the bot:

```bash
uv run python -m kb.cli.admin add-user <telegram_id> "<name>"
```

## Ingesting documents

Supported inputs: `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.html`, and URLs.
PDF/DOCX/PPTX parsing needs the `parse` extra (docling); local ONNX embeddings
need `embed-local` (or set `EMBED_BACKEND=openrouter`).

```bash
uv sync --extra parse --extra embed-local          # heavy optional deps
uv run python -m kb.cli.ingest docs/policy.pdf notes.md
uv run python -m kb.cli.ingest url https://example.com/page
uv run python -m kb.cli.ingest refresh-urls        # re-fetch all url documents
```

Inside Telegram, `/ingest` opens a capture session: text is buffered into a
single fact, while attachments and URL-only messages are saved as documents; a
second `/ingest` closes the session.

## Admin & maintenance

```bash
uv run python -m kb.cli.admin stats            # counts by type/status/entity
uv run python -m kb.cli.admin stats --cost     # OpenRouter spend by model/user
uv run python -m kb.cli.admin graph            # graph size and top entities
uv run python -m kb.cli.admin show-doc <id>
uv run python -m kb.cli.admin fact-history <topic>
uv run python -m kb.cli.admin trace-fact <topic>
uv run python -m kb.cli.admin eval             # quality metrics
uv run python -m kb.cli.reextract --status failed
```

## Testing & quality gates

```bash
make lint             # ruff check
make format           # ruff format
make typecheck        # mypy --strict on src/kb, scripts, plugin
make test-unit        # pure unit tests (no DB)
make test-integration # testcontainers-postgres (needs Docker)
make test             # everything
```

Integration tests spin up `pgvector/pgvector:0.8.2-pg18` via testcontainers; in
CI a PostgreSQL service is used instead when `KB_TEST_DSN` is set. CI runs
lint + mypy + unit on every PR and integration against a Postgres service.

## Claude Code plugin

`plugin/` packages an MCP server (thin HTTP client of the same backend), a skill
and slash commands (`/kb-search`, `/kb-add`, `/kb-facts`).

```
/plugin marketplace add mindlab-school/knowledge-base
/plugin install kb-agent@knowledge-base
```

Set `KB_BACKEND_URL` and `KB_USER_ID` (a whitelisted telegram_id) before use.

## Project layout

```
src/kb/            config, db (pool + repos), doc_types, llm, ingestion,
                   search, agent, api, channels (telegram), cli
migrations/        numbered forward-only SQL + scripts/migrate.py
plugin/            MCP server, skill, commands, manifests
tests/             unit/ integration/ fixtures/ + conftest, fakes
```

Stack: PostgreSQL 18 + pgvector (halfvec/HNSW), asyncpg, Pydantic, FastAPI,
aiogram, OpenRouter (OpenAI SDK), docling, trafilatura, voyage-4-nano ONNX
embeddings. No ORM, no LangChain, no separate vector store or graph database.
