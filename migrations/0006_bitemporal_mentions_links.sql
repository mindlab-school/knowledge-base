-- 0006_bitemporal_mentions_links: make entity_mentions and document_links bitemporal.
-- Forward-only. Mirrors the entity_relations model from 0001 (invalidate, never
-- hard-delete): re-ingestion of a document invalidates the mentions/links it no
-- longer produces (invalid_at = now()) and inserts the current set, so history and
-- provenance survive across versions. The active set is a partial unique index on
-- invalid_at IS NULL rows; superseded rows fall out of the index and remain as
-- history. NULL role keeps its Postgres semantics (distinct in the unique index),
-- so the repo's add_mention still special-cases role-less mentions.

ALTER TABLE entity_mentions
    ADD COLUMN valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN invalid_at TIMESTAMPTZ;

ALTER TABLE entity_mentions
    DROP CONSTRAINT entity_mentions_entity_id_document_id_role_key;

CREATE UNIQUE INDEX entity_mentions_active
    ON entity_mentions (entity_id, document_id, role) WHERE invalid_at IS NULL;

-- get_document_card and the entity-filter EXISTS look up active mentions by
-- document_id; the active unique index above only leads with entity_id.
CREATE INDEX entity_mentions_active_doc_idx
    ON entity_mentions (document_id) WHERE invalid_at IS NULL;

ALTER TABLE document_links
    ADD COLUMN valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN invalid_at TIMESTAMPTZ;

ALTER TABLE document_links
    DROP CONSTRAINT document_links_source_id_target_id_kind_key;

CREATE UNIQUE INDEX document_links_active
    ON document_links (source_id, target_id, kind) WHERE invalid_at IS NULL;

-- The related-documents walk and the supersedes check traverse links by
-- target_id; the active unique index above only leads with source_id.
CREATE INDEX document_links_active_tgt_idx
    ON document_links (target_id) WHERE invalid_at IS NULL;
