-- 0002_source_kind: mark how each document entered the system (spec section 11).
-- Every source is normalised to plain text by an adapter; source_kind records origin.

ALTER TABLE documents
    ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'file'
        CHECK (source_kind IN ('file','url'));
