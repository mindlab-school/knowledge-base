-- 0001_init: core schema for KB Agent (spec section 1).
-- Forward-only. pgvector provides halfvec(1024) for half-precision embeddings.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id           SERIAL PRIMARY KEY,
    telegram_id  BIGINT UNIQUE NOT NULL,
    name         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ingest_sessions (
    id         SERIAL PRIMARY KEY,
    user_id    INT NOT NULL REFERENCES users(id),
    active     BOOLEAN NOT NULL DEFAULT TRUE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at  TIMESTAMPTZ
);
-- At most one active ingest session per user.
CREATE UNIQUE INDEX ingest_sessions_one_active ON ingest_sessions (user_id) WHERE active;

CREATE TABLE ingest_buffer (
    id         SERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES ingest_sessions(id) ON DELETE CASCADE,
    seq        INT NOT NULL,
    text       TEXT NOT NULL,
    UNIQUE (session_id, seq)
);
-- The session buffer is NOT deleted after close: it is the source text of facts (trace).

CREATE TABLE documents (
    id                SERIAL PRIMARY KEY,
    title             TEXT NOT NULL,
    source_path       TEXT UNIQUE NOT NULL,
    ingest_session_id INT REFERENCES ingest_sessions(id),  -- NULL for CLI loads
    content_hash      TEXT NOT NULL,                        -- sha256
    version           INT  NOT NULL DEFAULT 1,
    raw_content       TEXT NOT NULL,
    doc_type          TEXT NOT NULL DEFAULT 'generic',
    attributes        JSONB NOT NULL DEFAULT '{}',
    extraction_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (extraction_status IN ('pending','done','failed','skipped')),
    extraction_model  TEXT,
    extracted_at      TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX documents_type_idx       ON documents (doc_type);
CREATE INDEX documents_attributes_idx ON documents USING gin (attributes);

CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   halfvec(1024) NOT NULL,    -- half precision: index and RAM halved
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED
);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding halfvec_cosine_ops);

CREATE TABLE entities (
    id          SERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,            -- person | department | product | process | project
    name        TEXT NOT NULL,
    canonical   TEXT NOT NULL,            -- lower(trim(name))
    attributes  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_type, canonical)
);

CREATE TABLE entity_mentions (
    id          SERIAL PRIMARY KEY,
    entity_id   INT NOT NULL REFERENCES entities(id)  ON DELETE CASCADE,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    role        TEXT,                     -- owner | participant | mentioned | approver
    UNIQUE (entity_id, document_id, role)
);
CREATE INDEX entity_mentions_doc_idx ON entity_mentions (document_id);
CREATE INDEX entity_mentions_ent_idx ON entity_mentions (entity_id);

-- Graph layer: entity -> entity edges.
CREATE TABLE entity_relations (
    id          SERIAL PRIMARY KEY,
    source_id   INT  NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id   INT  NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,          -- owns | part_of | reports_to | responsible_for | related_to
    document_id INT  REFERENCES documents(id) ON DELETE CASCADE,  -- provenance
    valid_from  TIMESTAMPTZ NOT NULL DEFAULT now(),
    invalid_at  TIMESTAMPTZ,             -- NULL = edge is current
    CHECK (source_id <> target_id),
    UNIQUE (source_id, target_id, relation, document_id)
);
CREATE INDEX entity_relations_src_idx ON entity_relations (source_id);
CREATE INDEX entity_relations_tgt_idx ON entity_relations (target_id);

-- document -> document edges (links and explicit mentions).
CREATE TABLE document_links (
    id          SERIAL PRIMARY KEY,
    source_id   INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_id   INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'reference',   -- reference | supersedes | attachment
    CHECK (source_id <> target_id),
    UNIQUE (source_id, target_id, kind)
);
CREATE INDEX document_links_src_idx ON document_links (source_id);
CREATE INDEX document_links_tgt_idx ON document_links (target_id);

-- Facts are bitemporal: an update does not erase history (Graphiti model).
CREATE TABLE facts (
    id         SERIAL PRIMARY KEY,
    topic      TEXT NOT NULL,            -- topic slug: 'ceny-i-pakety', 'raspisanie'
    content    TEXT NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when it became true
    invalid_at TIMESTAMPTZ,                          -- when it stopped (NULL = current)
    created_by INT REFERENCES users(id),
    session_id INT REFERENCES ingest_sessions(id)    -- provenance
);
CREATE UNIQUE INDEX facts_topic_active ON facts (topic) WHERE invalid_at IS NULL;
CREATE INDEX facts_topic_hist_idx ON facts (topic, valid_from DESC);

CREATE TABLE conversations (
    id         SERIAL PRIMARY KEY,
    user_id    INT NOT NULL REFERENCES users(id),
    active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content         TEXT NOT NULL,
    usage           JSONB,                -- OpenRouter usage payload for cost aggregation
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX messages_conversation_idx ON messages (conversation_id);
