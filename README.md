# KB Agent

Corporate **agentic knowledge base**. Employees ask questions in Telegram (or from
Claude Code via MCP); a Claude-based agent — reached through OpenRouter — decides
which tools to call and answers **strictly from company documents, always citing
sources**. On ingestion every document is auto-typed and an LLM extracts
attributes, entities and their relations in a single call, so there is no manual
labelling. Alongside documents the system keeps **facts**: compact, one-per-topic
records of current business context (prices, packages, schedule) that are injected
into every answer.

It answers two classes of question:

- **Content** — “what does the vacation policy say”, “how do I file a business
  trip” → semantic search over document chunks.
- **Documents & entities as objects** — “which regulations are overdue for
  review”, “who owns the onboarding process”, “what was decided on project X” →
  structured queries over metadata, entities and a graph.

---

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
  - [Query path (online)](#query-path-online)
  - [Data path (offline)](#data-path-offline)
- [Architecture](#architecture)
  - [Module layout & layering](#module-layout--layering)
  - [The agent loop](#the-agent-loop)
  - [The four agent tools](#the-four-agent-tools)
  - [Search internals](#search-internals)
  - [Ingestion internals](#ingestion-internals)
  - [Facts](#facts)
  - [Database schema](#database-schema)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running](#running)
- [Whitelist](#whitelist)
- [Ingesting Documents](#ingesting-documents)
- [Telegram Bot](#telegram-bot)
- [HTTP API Reference](#http-api-reference)
- [Admin & Maintenance CLI](#admin--maintenance-cli)
- [Claude Code Plugin](#claude-code-plugin)
- [Testing & Quality Gates](#testing--quality-gates)
  - [Behavioural eval harness](#behavioural-eval-harness-scriptseval_hwpy)
- [Continuous Integration](#continuous-integration)
- [Deployment](#deployment)
- [Cost & Token Usage](#cost--token-usage)
- [Troubleshooting](#troubleshooting)
- [Project Layout](#project-layout)
- [Design Decisions & Invariants](#design-decisions--invariants)
- [License](#license)

---

## Key Features

- **Grounded answers only.** The agent answers from the knowledge base or says
  “в базе знаний этого нет” (“this isn’t in the knowledge base”). It never uses the
  model’s general knowledge or the open internet.
- **Agentic tool use.** A Claude model picks and combines four tools
  (semantic search, structured document query, entity/graph exploration, full
  document fetch) per question — no hard-coded retrieval pipeline.
- **Automatic structuring on ingest.** One LLM call classifies the document type
  and extracts typed attributes, entities (with roles), entity relations and
  cross-document links. No manual tagging.
- **Facts as first-class context.** Volatile business context (prices, packages,
  schedule, contacts) is stored as bitemporal, one-per-topic facts and prepended
  to every prompt, so common questions are answered without a search round-trip.
- **Hybrid retrieval.** Vector (HNSW cosine over half-precision embeddings) and
  full-text (Russian `tsvector`) candidate lists fused with Reciprocal Rank Fusion.
- **A knowledge graph without a graph database.** Entities and their relations
  live in plain PostgreSQL tables; traversal is a single recursive CTE.
- **Bitemporal history.** Facts and entity relations keep their history — an
  update invalidates the old version instead of erasing it, so every fact is
  traceable back to the ingest session that produced it.
- **Channel-agnostic core.** The Telegram bot and the Claude Code MCP server are
  thin HTTP clients over one backend.
- **No heavyweight frameworks.** No ORM, no LangChain, no separate vector store or
  graph database. PostgreSQL + `asyncpg` + `pgvector` do all of it.

---

## Tech Stack

| Layer | Choice |
|---|---|
| **Language** | Python 3.13 (managed by [`uv`](https://docs.astral.sh/uv/)) |
| **Database** | PostgreSQL 18 + [`pgvector`](https://github.com/pgvector/pgvector) 0.8.2 (`halfvec(1024)`, HNSW cosine) |
| **DB driver** | `asyncpg` (no ORM) with `pgvector` and JSONB codecs |
| **Config/validation** | Pydantic v2 + `pydantic-settings` |
| **HTTP backend** | FastAPI + Uvicorn |
| **Telegram bot** | `aiogram` 3 (long polling) |
| **LLM gateway** | OpenRouter, called through the OpenAI SDK (`AsyncOpenAI`) |
| **Agent model** | `anthropic/claude-haiku-4.5` (default; configurable) |
| **Embeddings** | `voyage-4-nano` — local ONNX (int8, CPU) by default, or OpenRouter |
| **Document parsing** | `docling` (PDF/DOCX/PPTX → markdown), `trafilatura` (HTML/URL) |
| **MCP** | `mcp` (FastMCP stdio server) for the Claude Code plugin |
| **Tests** | `pytest`, `pytest-asyncio`, `testcontainers[postgres]` |
| **Tooling** | `ruff` (lint + format), `mypy --strict` |
| **Packaging/deploy** | `uv`, Docker (single image), Docker Compose |

Heavy dependencies are **optional extras**, loaded lazily: `parse` (docling) only in
the ingestion path; `embed-local` (onnxruntime + tokenizers) wherever embeddings are
computed under `EMBED_BACKEND=local` — both when ingesting documents **and** when
embedding queries at search time. The `mcp` extra is only needed by the plugin server.

---

## How It Works

```
                          QUERY PATH (online)
Telegram bot / MCP ──HTTP──▶ FastAPI backend ──▶ Agent loop (Claude, 4 tools)
 (thin clients)          (identity, history)         │ picks & combines
                                                      ▼
                                 Search: semantic | structured | entity/graph | facts
                                                      ▼
                                    PostgreSQL 18 + pgvector (halfvec, HNSW)
                                                      ▲
                          DATA PATH (offline)         │ one transaction
Files / URL ──▶ Ingestion: parse ▶ LLM extract ▶ chunk ▶ embed ▶ write
(free text in an /ingest session becomes a fact — no chunks/embeddings)
```

### Query path (online)

1. A channel (Telegram bot or MCP server) sends an HTTP request identifying the
   caller by their `telegram_id`.
2. The backend resolves the user against the **whitelist** (unknown → `403`), then
   auto-closes any idle ingest session.
3. It loads (or creates) the user’s active conversation and its **last 8 messages**
   of history.
4. The **agent loop** runs the tool-use cycle: the model may call any of the four
   tools, receive JSON results, and iterate up to `MAX_TOOL_ROUNDS = 5` before a
   final tool-less call forces a text answer. Active **facts** are injected into
   the system prompt, so fact-covered questions are answered without a tool call.
5. The user and assistant messages (with token/cost usage) are persisted; the
   answer is returned. The Telegram bot splits it into ≤ 4000-character parts.

### Data path (offline)

1. A **source adapter** normalises the input to plain text: `.txt`/`.md`/`.markdown`
   read directly, `.html`/`.htm`/URL via `trafilatura`, `.pdf`/`.docx`/`.pptx` via
   `docling`.
2. If the `content_hash` (SHA-256 of the text) already exists at that
   `source_path`, ingestion is a **no-op**. Otherwise one **LLM extraction** call
   determines the type, attributes, entities, relations and links.
3. The text is **chunked** by a type-specific strategy and each chunk is
   **embedded** (1024-dim unit vectors).
4. Everything derived is written in a **single short transaction**: document
   (new or version-bumped), chunks + embeddings, entities + mentions + relation
   edges, and resolved/deferred document links. Slow work (extraction, embedding)
   always happens *before* the transaction opens, so a DB transaction is never held
   across a network call.

---

## Architecture

### Module layout & layering

```
src/kb/
├── config.py            # single source of settings (reads .env / env once)
├── db/
│   ├── pool.py          # asyncpg pool; registers pgvector + JSONB codecs
│   └── repo/            # ALL writing SQL lives here (documents, chunks, entities,
│                        #   facts, sessions, users, conversations)
├── doc_types/           # Pydantic models + registry for document types
├── llm/client.py        # OpenRouter chat client (OpenAI SDK)
├── ingestion/
│   ├── sources/         # file / url / telegram source adapters → LoadedDoc
│   ├── extraction.py    # LLM structure extraction (type/attrs/entities/…)
│   ├── chunking.py      # pure per-type chunkers
│   ├── embeddings.py    # local ONNX / OpenRouter embedders
│   ├── pipeline.py      # source → extract → chunk → embed → write
│   └── session.py       # /ingest capture-session orchestration
├── search/              # read side: semantic, structured, entities, graph, facts
├── agent/               # tool schemas + dispatcher + the tool-use loop + prompts
├── api/app.py           # FastAPI backend (the only HTTP surface over the core)
├── channels/telegram.py # aiogram bot (thin HTTP client)
└── cli/                 # ingest, admin, reextract command-line entry points
```

**Dependency direction** (enforced by convention, checked in review):

```
channels → api → agent → search → db/repo → db/pool
ingestion / doc_types → db/repo
```

- The **core is channel-agnostic**; Telegram and MCP are thin clients.
- **Writing SQL lives only in `src/kb/db/repo`**; the `search` package may also
  read. `channels/telegram.py` imports only `httpx`, `config` and the Telegram SDK.
- The original `raw_content` is the **source of truth**; chunks, embeddings,
  attributes, mentions and edges are all derived and recreated on change.

### The agent loop

`src/kb/agent/loop.py` runs the Chat-Completions tool-use cycle:

- **Spend guard.** Before doing anything it compares month-to-date cost against
  `OR_MONTHLY_LIMIT` (default `$20`) and, if exceeded, returns
  “Лимит расходов исчерпан, обратитесь к администратору.” without calling the model.
- **System prompt.** A static prefix — the system instructions plus the
  **facts block** (all active facts, capped at `FACTS_BLOCK_LIMIT = 4000` chars) —
  is placed before history and the question so it can be prompt-cached.
- **Prompt caching.** When `PROMPT_CACHE=auto` and the prefix (tool schemas +
  system prompt + facts) is estimated at ≥ `CACHE_MIN_TOKENS = 4096` tokens
  (~4 chars/token), a `cache_control: ephemeral` block is attached to the system
  message.
- **Sticky routing.** `session_id = conversation_id` is passed through to
  OpenRouter so the same provider is reused across tool rounds.
- **Loop.** Up to `MAX_TOOL_ROUNDS = 5` rounds; each assistant tool call is
  dispatched and its JSON result fed back. If rounds are exhausted with tools still
  pending, a final **tool-less** call forces a text answer. `max_tokens = 1200`.
- **Usage accounting.** Prompt/completion/total tokens, cost, and cache
  read/write counters are accumulated and stored on the assistant message.
- **Refusal.** An empty final answer becomes “в базе знаний этого нет”.

### The four agent tools

Defined in `src/kb/agent/tools.py`. Results are size-bounded to keep the context
small.

| Tool | Arguments | Returns |
|---|---|---|
| `search_knowledge_base` | `query` (required), `filters?` | Up to **4** chunks (`{document_id, document, score, fragment}`) — hybrid semantic + full-text. |
| `query_documents` | `filters` (required) | Up to **15** document metadata rows; if more match, a `total` count + “refine your filters” note. |
| `explore_entity` | `name` (required), `depth` (0–3, default 1) | Entity card: the entity, the documents that mention it (with roles), and its graph neighbourhood. |
| `get_document` | `document_id` (required) | Full document text, truncated to **8000** characters. |

The **filter object** (shared by `search_knowledge_base` and `query_documents`)
supports, AND-combined:

- `doc_type` — one of `regulation | instruction | faq | meeting_notes | generic`
- `title_contains` — case-insensitive substring (`ILIKE`)
- `created_before` / `created_after` — ISO date on `created_at`
- `<attr>_before` / `<attr>_after` — date compare on a JSONB attribute
  (e.g. `review_by_before` for overdue regulations)
- `attributes` — exact JSONB containment (`@>`), e.g. `{"status": "active"}`
- `entity` — `{"name": ..., "role"?: owner|participant|mentioned|approver}`
  mention existence

### Search internals (`src/kb/search/`)

- **`semantic.py` — hybrid chunk search.** Two ranked lists are built: **vector**
  (`ORDER BY embedding <=> query::halfvec`, HNSW cosine) and **full-text**
  (`websearch_to_tsquery('russian', …)` ranked by `ts_rank_cd`). Each returns up to
  20 candidates; they are merged by **Reciprocal Rank Fusion** with `k = 60`
  (`score = Σ 1/(k + rank)`) and the top `limit` (default 4) are returned. When
  filters narrow the set, `SET LOCAL hnsw.iterative_scan = relaxed_order` lets the
  ANN branch keep collecting past the filter (pgvector 0.8+), avoiding under-fill.
- **`structured.py` — metadata queries.** `build_document_where` is a pure function
  turning a filter dict into a parameterised `WHERE` fragment (easy to unit-test);
  `query_documents` returns metadata (no `raw_content`), newest first, capped at 15.
  It is defensive against malformed tool arguments: ISO date filters
  (`created_before`/`created_after`, `<attr>_before`/`_after`) are parsed to
  `datetime.date` objects (a bare string would raise an `asyncpg` `DataError`), and a
  non-dict `filters` (e.g. a stray string) yields an empty `WHERE` instead of raising.
  The agent's tool layer additionally coerces a JSON-string `filters` into an object.
- **`entities.py` — entity cards.** Resolves a name by canonical match
  (`lower(trim(name))`), falling back to a fuzzy `ILIKE`, then returns the entity,
  its mentioning documents (with roles) and its graph neighbourhood.
- **`graph.py` — traversal.** A single **recursive CTE** walks `entity_relations`
  (undirected, following only currently-valid edges where `invalid_at IS NULL`), capped
  at `MAX_DEPTH = 3` and `MAX_NODES = 40`, cycle-safe via a visited-path array.
  `related_documents` walks `document_links` and flags versions superseded by a
  `supersedes` link (`current = false`).
- **`facts.py` — fact orchestration** (see [Facts](#facts)).

### Ingestion internals (`src/kb/ingestion/`)

- **Idempotency.** `content_hash` (SHA-256) is the key. Same hash at the same
  `source_path` → `unchanged`. Different content → the version is bumped and all
  derived rows are recreated. New `source_path` → a new document (version 1).
- **Extraction (`extraction.py`).** One model call requests structured output via
  `response_format=json_schema`; if the provider rejects it, the call is retried
  once through a forced `emit_extraction` tool, and the chosen path is cached for
  the process. Input is truncated to ~15k chars (12k head + 3k tail). Degrades
  gracefully: `confidence < 0.6` → `generic`; on any error, one retry, then
  `generic`/`failed` — the pipeline never dies.
- **Chunking (`chunking.py`, pure).** `CHUNK_SIZE = 1500`, `CHUNK_OVERLAP = 200`.
  Strategy by type: `faq` → one Q/A pair per chunk; `meeting_notes` → decisions and
  action-items blocks (date + participants repeated in each); `regulation` /
  `instruction` → by markdown headings; everything else → paragraph packing.
  Markdown tables are never split across chunks.
- **Embeddings (`embeddings.py`).** One `Embedder` interface, 1024-dim unit
  vectors. `local` = `voyage-4-nano` ONNX (int8) on CPU — native 2048 dims are
  Matryoshka-truncated to `EMBED_DIM` and renormalised; `onnxruntime`/`tokenizers`
  are imported only on first use. `openrouter` = the `/embeddings` endpoint with an
  explicit `dimensions`. Documents are embedded as-is; queries get a mandatory
  retrieval prefix.
- **Graph & links (`pipeline.py`).** Entities are upserted and canonicalised;
  mentions and this document’s relation edges are (re)written. Extracted links are
  resolved to document ids or, if the target doesn’t exist yet, stored in
  `pending_document_links` and materialised later when the target is ingested
  (reverse-by-name resolution).
- **Re-extraction.** `reextract_document` re-runs type/attribute/graph extraction;
  chunks, embeddings, and document links are left untouched.

### Facts

Facts are compact, one-per-topic records of current business context. They carry
**no embeddings or chunks** — they go into the agent context whole.

- **Topic assignment without an LLM.** On session close the buffered text is
  embedded and compared (cosine) to existing active facts; a match above
  `0.75` reuses that topic, otherwise a slug is built from the first significant
  words.
- **Bitemporal.** `upsert` invalidates the current active version and inserts a new
  one in the same transaction (a partial-unique index enforces one active fact per
  topic). `fact-history` / `trace-fact` reconstruct the full timeline and its
  source ingest session.

### Database schema

Migrations are **forward-only** numbered SQL files under `migrations/`, applied by
`scripts/migrate.py` and recorded in `schema_migrations`.

| Migration | Adds |
|---|---|
| `0001_init.sql` | Core schema — all tables below **except** `pending_document_links` (added by 0004) and `schema_migrations` (created by the migration runner) — + `pgvector` extension |
| `0002_source_kind.sql` | `documents.source_kind` (`file` \| `url`) |
| `0003_session_activity.sql` | `ingest_sessions.last_active_at` (idle autoclose) |
| `0004_pending_links.sql` | `pending_document_links` (deferred link resolution) |

**Tables**

| Table | Purpose |
|---|---|
| `users` | Whitelist: `telegram_id` → name. |
| `ingest_sessions` | `/ingest` capture sessions (one active per user). |
| `ingest_buffer` | Buffered text per session (kept — it is a fact’s source/trace). |
| `documents` | `raw_content` (source of truth), `doc_type`, `attributes` (JSONB), `content_hash`, `version`, `extraction_status`, `source_kind`. |
| `chunks` | `content` + `embedding halfvec(1024)` + generated `tsv` (Russian FTS). HNSW + GIN indexes. |
| `entities` | `entity_type`, `name`, `canonical` (unique per type). |
| `entity_mentions` | Entity ↔ document with a `role`. |
| `entity_relations` | Entity → entity edges (bitemporal: `valid_from`/`invalid_at`). |
| `document_links` | Document → document edges (`reference`/`supersedes`/`attachment`). |
| `pending_document_links` | Links whose target isn’t ingested yet. |
| `facts` | Bitemporal business-context records (one active per topic). |
| `conversations` / `messages` | Chat history + per-message OpenRouter `usage` (for cost). |
| `schema_migrations` | Applied migration versions. |

**Controlled vocabularies** (application-enforced):

- **Document types:** `generic`, `regulation`, `instruction`, `faq`, `meeting_notes`
- **Entity types:** `person`, `department`, `product`, `process`, `project`
- **Relations:** `owns`, `part_of`, `reports_to`, `responsible_for`, `related_to`
- **Mention roles:** `owner`, `participant`, `mentioned`, `approver`

---

## Prerequisites

- **Python 3.13** (pinned in `.python-version`; installed/managed by `uv`)
- **[`uv`](https://docs.astral.sh/uv/) ≥ 0.5** — the package manager and runner
- **Docker + Docker Compose v2** — for PostgreSQL + pgvector, and for integration
  tests via testcontainers
- **An OpenRouter API key** — for the agent and extraction models
- **A Telegram bot token** — only for running the live bot (get one from
  [@BotFather](https://t.me/BotFather))
- For local embeddings: the `voyage-4-nano` ONNX assets (`model.onnx` +
  `tokenizer.json`) under `EMBED_MODEL_PATH` — or set `EMBED_BACKEND=openrouter` to
  skip them.

---

## Getting Started

### 1. Clone and enter the repository

```bash
git clone https://github.com/mindlab-school/knowledge-base.git
cd knowledge-base
```

### 2. Install dependencies

```bash
uv sync                    # create the venv and install core + dev deps
```

For document parsing and local embeddings (heavy, optional):

```bash
uv sync --extra parse --extra embed-local
```

| Extra | Pulls in | Needed for |
|---|---|---|
| `parse` | `docling` | ingesting `.pdf` / `.docx` / `.pptx` |
| `embed-local` | `onnxruntime`, `tokenizers` | `EMBED_BACKEND=local` (the default) |
| `mcp` | `mcp` | running the Claude Code plugin’s MCP server |

### 3. Configure the environment

```bash
cp .env.example .env
```

Fill in at least `OPENROUTER_API_KEY` and (for the bot) `TELEGRAM_BOT_TOKEN`. See
[Configuration](#configuration) for every setting.

### 4. Start PostgreSQL

```bash
make up                    # docker compose up -d db  (pgvector/pgvector:0.8.2-pg18)
```

### 5. Apply migrations

```bash
make migrate               # uv run python scripts/migrate.py
```

Check status without applying:

```bash
uv run python scripts/migrate.py --status
```

### 6. Whitelist yourself and run

```bash
uv run python -m kb.cli.admin add-user <your_telegram_id> "Your Name"
make run-api               # in one terminal
make run-bot               # in another (needs TELEGRAM_BOT_TOKEN)
```

---

## Configuration

All configuration is read **once**, at process start, from `.env` / the
environment through `src/kb/config.py` (`get_settings()`). Nothing else in the
codebase reads `os.environ` directly.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required.** OpenRouter API key. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | LLM gateway base URL. |
| `TELEGRAM_BOT_TOKEN` | — | Required for the bot. |
| `DATABASE_URL` | `postgresql://kb:kb@localhost:5432/kb` | PostgreSQL DSN. |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL the bot calls. |
| `AGENT_MODEL` | `anthropic/claude-haiku-4.5` | Model that drives the agent loop. |
| `OR_MODEL_VARIANT` | *(empty)* | Appended to the agent slug: empty, `:exacto` (tool-calling accuracy), or `:nitro`. |
| `AGENT_EFFORT` | *(empty)* | Reasoning effort — only for adaptive-thinking models (e.g. Sonnet 5); leave empty for Haiku. |
| `EXTRACTION_MODEL` | `anthropic/claude-haiku-4.5` | Model used for ingestion extraction. |
| `EMBED_BACKEND` | `local` | `local` (voyage-4-nano ONNX on CPU) or `openrouter`. |
| `EMBED_MODEL` | `voyage-4-nano` | Embedding model name. |
| `EMBED_DIM` | `1024` | Embedding dimensions — **must match `halfvec(N)`** in the schema. |
| `EMBED_MODEL_PATH` | `models/voyage-4-nano` | Local backend: dir holding `model.onnx` + `tokenizer.json`. |
| `PROMPT_CACHE` | `auto` | `auto` (cache when the prefix is large enough) or `off`. |
| `CACHE_MIN_TOKENS` | `4096` | Minimum prefix size (tokens) before caching (Anthropic Haiku 4.5 minimum). |
| `FACTS_BLOCK_LIMIT` | `4000` | Max characters of facts injected into the prompt. |
| `OR_ALLOW_FALLBACKS` | `true` | Allow OpenRouter provider fallbacks. |
| `OR_SITE_URL` | *(empty)* | Optional `HTTP-Referer` header. |
| `OR_APP_NAME` | `kb-agent` | Optional `X-Title` header. |
| `OR_MONTHLY_LIMIT` | `20` | Soft month-to-date spend limit (USD); the agent refuses over it. |

> **Note on `EMBED_DIM`.** Changing it means changing `halfvec(N)` in the schema
> and re-embedding everything. The local backend truncates the native 2048-dim
> vectors to `EMBED_DIM`; the OpenRouter backend requests `dimensions=EMBED_DIM`
> directly (verify your embedding model supports that size).

---

## Running

### Locally with `uv`

```bash
make run-api    # uv run uvicorn kb.api.app:app --host 0.0.0.0 --port 8000
make run-bot    # uv run python -m kb.channels.telegram   (long polling)
```

The bot is a thin client: run the API first (or point `BACKEND_URL` at a running
backend).

### With Docker Compose

One image is built; the entrypoint selects the process via `KB_PROCESS`. Compose
profiles pick which service to bring up.

```bash
docker compose --profile api up --build     # db + api  (KB_PROCESS=api)
docker compose --profile bot up --build     # db + bot  (KB_PROCESS=bot)
```

Both services read `.env`. The `db` service uses a named volume `pgdata` and a
`pg_isready` healthcheck; `api`/`bot` wait for it to be healthy.

---

## Whitelist

Access is **whitelist-only**. Every endpoint resolves the caller by `telegram_id`
and returns `403` for unknown ids. Add a user before they can talk to the bot:

```bash
uv run python -m kb.cli.admin add-user <telegram_id> "<name>"
```

---

## Ingesting Documents

Supported inputs: `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.html`, and URLs.
PDF/DOCX/PPTX need the `parse` extra (docling); local embeddings need
`embed-local` (or set `EMBED_BACKEND=openrouter`).

### From the CLI

```bash
uv run python -m kb.cli.ingest docs/policy.pdf notes.md      # one or more files
uv run python -m kb.cli.ingest url https://example.com/page  # one or more URLs
uv run python -m kb.cli.ingest refresh-urls                  # re-fetch all url docs
```

Or via `make`:

```bash
make ingest FILES="a.pdf b.md"
make ingest-url URL="https://example.com/page"
make refresh-urls
```

Each result prints an action card: `[created|updated|unchanged] #id title
(doc_type vN, chunks=…, extraction=…)`.

### From Telegram (`/ingest` capture session)

Send `/ingest` to open a capture session, then:

- **Text messages** are buffered; on close they are glued into a **single fact**
  (topic auto-assigned).
- **File attachments** and **URL-only messages** are ingested as **documents**
  attached to the session.

A second `/ingest` closes the session and reports what was saved. An idle session
is auto-closed after **60 minutes** on the user’s next contact. URLs are canonical­
ised (host lower-cased, UTM params/fragment/trailing slash stripped) so re-sending
the same page updates one document.

### Re-extraction

Re-run extraction (type + attributes + graph) for stored documents; chunks and
embeddings are untouched:

```bash
uv run python -m kb.cli.reextract --status failed          # retry failed extractions
uv run python -m kb.cli.reextract --type generic           # re-classify generics
uv run python -m kb.cli.reextract --model-older-than anthropic/claude-haiku-4.5
uv run python -m kb.cli.reextract --all
```

At least one filter (`--status` / `--type` / `--model-older-than`) or `--all` is
required.

---

## Telegram Bot

`aiogram` 3 long polling. The bot holds **no mode state**: it detects URL-only
messages and document attachments locally, and lets the backend decide the rest —
ordinary text goes to `/ingest/message` and falls through to `/chat` on `409`
(no active session).

| Command | Effect |
|---|---|
| `/start` | Greeting and command list. |
| `/reset` | Reset the conversation context (deactivate the active conversation). |
| `/ingest` | Toggle the capture session (open / close-and-save). |
| `/facts` | List fact topics. |
| `/del <topic>` | Soft-delete (invalidate) the active fact for a topic. |
| *(any text)* | Buffered into the session if one is open, else answered by the agent. |
| *(a URL-only message)* | Ingest the page(s) as document(s). |
| *(a file attachment)* | Ingest the file (`.txt/.md/.markdown/.pdf/.docx/.pptx/.html/.htm`). |

While the agent thinks, the bot shows a “typing…” action. Long answers are split
into ≤ 4000-character parts on paragraph boundaries.

---

## HTTP API Reference

FastAPI backend (`kb.api.app:app`), served on `:8000`. Every endpoint (except
`/health`) resolves the caller by `telegram_id` and returns `403` for unknown ids.
The `/search*` endpoints exist for the MCP plugin.

| Method & Path | Body / Query | Description |
|---|---|---|
| `GET /health` | — | Liveness probe → `{"status": "ok"}`. |
| `POST /chat` | `{telegram_id, text}` | Run the agent loop; returns `{answer}`. Autocloses idle sessions, loads history, persists messages + usage. |
| `POST /reset` | `{telegram_id}` | Reset the caller’s active conversation. |
| `POST /ingest/file` | multipart: `telegram_id`, `file`, `origin` (`file`/`telegram`), `filename?` | Ingest an uploaded file. Returns an ingest card. Rejects unsupported types with `400` and files over **20 MB** with `413`. |
| `POST /ingest/url` | `{telegram_id, url}` | Ingest a web page. |
| `POST /ingest/session` | `{telegram_id}` | Toggle the capture session → `{state: opened}` or `{state: closed, topic, action, documents}`. |
| `POST /ingest/message` | `{telegram_id, text}` | Buffer text into the active session → `{seq}`, or `409` if none is active. |
| `GET /facts` | `?telegram_id` | List fact topics with `updated_at`. |
| `DELETE /facts/{topic}` | `?telegram_id` | Soft-delete a fact topic (`404` if none). |
| `GET /documents/{id}` | `?telegram_id` | Full document card (attributes, chunk count, entity mentions). |
| `POST /search` | `{telegram_id, query, filters?}` | Hybrid chunk search (MCP). |
| `POST /search/documents` | `{telegram_id, filters?}` | Structured document query (MCP). |
| `POST /search/entity` | `{telegram_id, name, depth?}` | Entity card + graph (MCP). |

---

## Admin & Maintenance CLI

```bash
uv run python -m kb.cli.admin <command> [args]
```

| Command | Description |
|---|---|
| `stats` | Counts by document type, extraction status, and entity type; total chunks and active facts. |
| `stats --cost` | Headline total is month-to-date OpenRouter spend; the per-model and per-user breakdowns below it are all-time totals. |
| `graph` | Graph size: nodes by type, active edges by relation, top-10 entities by degree, isolated-node count. |
| `sessions` | Active ingest sessions (buffered messages, attached docs, last activity). |
| `facts` | Fact topics with content length and update time. |
| `show-doc <id>` | Document details: source, attributes, extraction, entity mentions. |
| `set-type <id> <type>` | Force a document type and re-extract (falls back to `skipped` if the classifier disagrees). |
| `fact-history <topic>` | Every version of a fact with its validity interval. |
| `trace-fact <topic>` | Trace a fact to the ingest session, its buffered messages, and attached documents. |
| `trace-doc <id>` | Trace a document to its ingest session and any facts from that session. |
| `del-fact <topic>` | **Hard-delete** every version of a fact topic. |
| `cleanup-entities [--yes]` | Delete orphan entities (no mentions and no edges). |
| `add-user <telegram_id> <name>` | Whitelist a Telegram user. |
| `eval [dataset]` | Run the quality evaluation set (default `tests/fixtures/eval.json`). |

### Evaluation harness

`admin eval` scores retrieval and extraction quality against a labelled dataset and
prints each metric against its threshold:

| Metric | Threshold | What it checks |
|---|---|---|
| `type_accuracy` | 0.90 | Correct `doc_type` per labelled document. |
| `attribute_completeness` | 0.80 | Required attributes present after extraction. |
| `recall_at_4` | 0.85 | An expected document appears in the top-4 chunk search. |
| `refusals` | 1.00 | Out-of-corpus questions are correctly refused (“в базе знаний этого нет”). |
| `graph_accuracy` | 0.75 | Threshold is defined, but `admin eval` does not compute this metric yet — it is always reported as `n/a`. |

The dataset (`tests/fixtures/eval.json`) has a `documents` map (title → expected
type + required attributes) and a `questions` list (in-corpus questions with
`expect_docs`, and out-of-corpus questions expected to be refused).

---

## Claude Code Plugin

`plugin/` packages an **MCP server** (a thin stdio HTTP client of the same
backend), a **skill**, and three **slash commands** — bringing the knowledge base
into Claude Code.

**Install (via the marketplace):**

```
/plugin marketplace add mindlab-school/knowledge-base
/plugin install kb-agent@knowledge-base
```

Set `KB_BACKEND_URL` (the backend) and `KB_USER_ID` (a whitelisted `telegram_id`)
before use — the MCP server (`plugin/mcp/server.py`) reads both from the
environment (see `plugin/.mcp.json`). The `mcp` extra must be installed.

**MCP tools** (each is a single HTTP call to the backend): `kb_search`,
`kb_query_documents`, `kb_explore_entity`, `kb_get_document`, `kb_list_facts`,
`kb_add_url`, `kb_add_file`.

**Slash commands:** `/kb-search`, `/kb-add`, `/kb-facts`.

---

## Testing & Quality Gates

```bash
make lint             # ruff check .
make format           # ruff format .
make typecheck        # mypy --strict on src/kb, scripts, plugin/mcp
make test-unit        # unit tests only (no database)
make test-integration # testcontainers PostgreSQL+pgvector (needs Docker)
make test             # everything
make check            # lint + typecheck + unit
```

### Test layout

- **`tests/unit/`** — pure, no external services:
  `test_chunking`, `test_config_and_registry`, `test_embeddings_math`,
  `test_extraction`, `test_prompt_cache`, `test_rrf`, `test_split_message`,
  `test_structured_filters`, `test_tool_filters`, `test_urls`.
- **`tests/integration/`** — a real PostgreSQL + pgvector container:
  `test_chat_flow`, `test_facts_bitemporal`, `test_graph`,
  `test_pipeline_idempotency`, `test_search_hybrid`, `test_structured_dates`,
  `test_sessions`, `test_whitelist`.

Integration tests are marked `integration` (deselect with `-m "not integration"`).
The `pg_dsn` fixture uses `KB_TEST_DSN` if set (CI’s Postgres service), otherwise it
spins up `pgvector/pgvector:0.8.2-pg18` via testcontainers, applying migrations
once; if Docker/the image is unavailable, those tests **skip cleanly**. Each test
gets a truncated database (`pool` fixture). Integration tests use a deterministic
`HashingEmbedder` fake (`tests/fakes.py`, via the `fake_embedder` fixture) instead of a
real embedding backend.

### Behavioural eval harness (`scripts/eval_hw.py`)

A separate, **behavioural** benchmark that drives the real agent loop end-to-end
against a fixed corpus and grades how it answers. Unlike `admin eval` (retrieval/
extraction metrics against `eval.json`), this measures the *answering* behaviour —
correct answers, false refusals, unnecessary clarifications, hallucinations, and
correct refusals of out-of-corpus questions — and needs an `OPENROUTER_API_KEY`
(it makes real model calls).

It runs the agent **in-process** (no HTTP server) and never persists chat messages,
so it does not consume the month-to-date spend budget. Fixed data under
`tests/fixtures/` keeps runs comparable across system changes:

- `hw_corpus/` — a 13-document snapshot + `manifest.json` (URL/title provenance).
- `hw_questions.json` — 100 questions with expected answers and an `out_of_corpus`
  flag (94 in-corpus + 6 that must be refused).

```bash
make eval-ingest                 # reset DB + ingest the corpus snapshot (fresh load)
make eval-run ARGS="--judge"     # ask all 100 questions, LLM-graded report
make eval                        # ingest + run + graded report in one go
uv run python scripts/eval_hw.py run --only 31,33 --concurrency 6   # a subset
```

Results are written to `experiments/hw_eval_results.json` (git-ignored). See
`experiments/` for the two write-ups produced with this harness (a 100-question
evaluation and a round of prompt optimisation).

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every pull request and on pushes to `dev`/`main`:

1. **lint + mypy + unit** — `ruff check`, `ruff format --check`, `mypy --strict`,
   and `pytest -m "not integration"`.
2. **integration** — against a `pgvector/pgvector:0.8.2-pg18` service container,
   with `KB_TEST_DSN` pointed at it, running `pytest -m integration`.

Both jobs use `astral-sh/setup-uv` with Python 3.13 and dependency caching.

---

## Deployment

The project ships a single Docker image (`Dockerfile`) that serves **both**
processes; `docker/entrypoint.sh` selects one by `KB_PROCESS` (`api` | `bot`,
default `api`). The image installs the `parse`, `embed-local` and `mcp` extras and
prefetches docling models so ingestion needs no network at runtime.

### Docker Compose (single host)

```bash
docker compose --profile api up --build -d     # db + api
docker compose --profile bot up --build -d     # db + bot
```

Provide `.env` (at minimum `OPENROUTER_API_KEY`, and `TELEGRAM_BOT_TOKEN` for the
bot). Compose sets `DATABASE_URL`/`BACKEND_URL` for the containers automatically.

### Production notes

- **Migrations.** Run `uv run python scripts/migrate.py` (idempotent, forward-only)
  against the production database on each deploy before starting the new image.
- **Two processes.** Deploy one `api` container (exposes `:8000`, put it behind a
  reverse proxy / TLS) and one `bot` container (`KB_PROCESS=bot`, no ports, needs
  `BACKEND_URL` reachable). Long polling means the bot needs no inbound ports.
- **Embeddings.** For `EMBED_BACKEND=local`, mount or bake the `voyage-4-nano` ONNX
  assets at `EMBED_MODEL_PATH`. To avoid shipping weights, set
  `EMBED_BACKEND=openrouter` (the embedding dimension must match `EMBED_DIM`).
- **Spend control.** `OR_MONTHLY_LIMIT` is a *soft* guard checked against
  month-to-date message cost; monitor real spend with `admin stats --cost`.
- **Health.** `GET /health` is a cheap liveness probe for the API.
- **Backups.** All state is in PostgreSQL (the `pgdata` volume in Compose). Back it
  up; `raw_content` is the source of truth and everything else can be rebuilt by
  re-ingesting or re-extracting.

---

## Cost & Token Usage

Everything paid runs through OpenRouter. There are exactly three call sites:

| Call site | When | Model | Volume |
|---|---|---|---|
| **Agent loop** (`agent/loop.py`) | every question | `AGENT_MODEL` (haiku) | up to `MAX_TOOL_ROUNDS + 1` = **6 model calls per question** |
| **Extraction** (`ingestion/extraction.py`) | once per document on ingest | `EXTRACTION_MODEL` (haiku) | 1 call (+1 retry on failure) |
| **Embeddings** (`ingestion/embeddings.py`) | per chunk on ingest, per query on search | `EMBED_MODEL` | 1 vector each — **free** on `EMBED_BACKEND=local` |

**What actually costs money.** The recurring cost is the **agent loop**: each question
can trigger several `haiku` calls, and every round re-sends the static prefix (system
prompt + tool schemas + facts) plus the growing tool-result context. **Embeddings are
negligible** — the default `EMBED_BACKEND=local` (`voyage-4-nano` ONNX on CPU) is free;
even on `EMBED_BACKEND=openrouter`, embedding a query (~20 tokens) is a fraction of a
cent and a full corpus ingest is well under a cent. Extraction is a small one-time cost
per document.

**Levers, biggest first:**

- **Fewer tool rounds.** `MAX_TOOL_ROUNDS = 5` is the ceiling; most content questions
  resolve in 1–2 rounds. Lowering it caps the worst case directly.
- **Prompt caching.** With `PROMPT_CACHE=auto`, the static prefix is cached once it
  reaches `CACHE_MIN_TOKENS` (Anthropic Haiku minimum 4096). Below that, the prefix is
  re-billed every round — a reason to keep the loop short.
- **Free embeddings / offline.** `EMBED_BACKEND=local` (voyage-4-nano ONNX on CPU)
  removes all embedding API calls. This is about **autonomy/privacy, not money** — the
  embedding spend is already near zero. Requires the ONNX assets and re-embedding.
- **Spend guard.** `OR_MONTHLY_LIMIT` (default `$20`) is a soft month-to-date cap; the
  agent refuses above it. Track real spend with `admin stats --cost`.
- **Evaluation.** `scripts/eval_hw.py` makes many agent calls; prefer a cheap model for
  any LLM judge and run subsets (`--only …`) rather than all 100 on every iteration.

---

## Troubleshooting

**`could not connect to server: Connection refused`**
PostgreSQL isn’t up or `DATABASE_URL` is wrong. Check `docker ps` / `make up` and
that the DSN host/port match (Compose uses `db:5432` inside the network,
`localhost:5432` from the host).

**`Migrations are pending` / schema errors**
Run `uv run python scripts/migrate.py` (or `make migrate`); inspect with
`--status`.

**`Local embedding backend requires the 'embed-local' extra`**
Install it (`uv sync --extra embed-local`) or set `EMBED_BACKEND=openrouter`.

**`voyage-4-nano ONNX assets not found under models/voyage-4-nano`**
Provide `model.onnx` + `tokenizer.json` at `EMBED_MODEL_PATH`, or switch to the
OpenRouter backend.

**`Парсинг PDF/DOCX/PPTX требует extra 'parse'`**
Install docling: `uv sync --extra parse`.

**Extraction shows `failed` / documents land as `generic`**
The model call or JSON validation failed twice; the document is still stored and
searchable. Retry with `uv run python -m kb.cli.reextract --status failed`.

**“Лимит расходов исчерпан…” from the bot**
Month-to-date spend hit `OR_MONTHLY_LIMIT`. Review with `admin stats --cost` and
raise the limit if appropriate.

**“нет доступа, обратитесь к администратору” (403)**
The `telegram_id` isn’t whitelisted. Add it with `admin add-user`.

**URL ingestion fails with “SPA не поддерживаются”**
`trafilatura` found no readable text (client-rendered page). Ingest a static
export or a file instead.

---

## Project Layout

```
.
├── src/kb/               # application package (see Module layout above)
├── migrations/           # numbered forward-only SQL
├── scripts/              # migrate.py (migrations) + eval_hw.py (behavioural eval)
├── plugin/               # Claude Code plugin: MCP server, skill, commands, manifests
├── tests/                # unit/ integration/ fixtures/ (incl. hw_corpus) + conftest, fakes
├── experiments/          # eval write-ups (git-ignored results JSON)
├── docker/entrypoint.sh  # process selector (api | bot)
├── Dockerfile            # single image for both processes
├── docker-compose.yml    # db + api/bot profiles
├── Makefile              # dev/ops shortcuts
├── pyproject.toml        # deps, extras, ruff/mypy/pytest config
├── uv.lock               # locked dependencies
└── .env.example          # every setting, documented
```

---

## Design Decisions & Invariants

- **`raw_content` is the source of truth.** Chunks, embeddings, attributes,
  entity mentions and edges are all derived and recreated whenever a document
  changes. Nothing derived is edited in place.
- **No transaction across a network call.** Extraction and embedding (slow,
  network-bound) always complete before the short write transaction opens.
- **Extraction never dies.** Any failure degrades to `generic`/`failed`; the
  document is stored and remains searchable, and can be re-extracted later.
- **One place for settings.** Only `config.py` reads the environment.
- **Writing SQL only in `db/repo`.** The rest of the code composes reads and calls
  repositories; this keeps persistence auditable in one place.
- **Bitemporal, not destructive.** Facts and entity relations invalidate old
  versions instead of deleting them, preserving history and provenance.
- **Small agent context by design.** Tool results are capped (4 chunks, 15
  documents, 8k-char document body, 40 graph nodes) to keep prompts cheap and fast.

---

## License

MIT — see `pyproject.toml`.
