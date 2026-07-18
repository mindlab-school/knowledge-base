-- 0004_pending_links: deferred document->document links.
-- Extraction yields link targets by name/URL. If the target document does not
-- exist yet, the link is remembered here and materialised into document_links
-- when the target is later ingested (spec section 3: "reverse resolution by name").

CREATE TABLE pending_document_links (
    id          SERIAL PRIMARY KEY,
    source_id   INT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_name TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'reference',
    UNIQUE (source_id, target_name, kind)
);
CREATE INDEX pending_document_links_name_idx ON pending_document_links (lower(target_name));
