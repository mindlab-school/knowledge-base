-- 0003_session_activity: track last activity per ingest session.
-- The spec (section 4) autocloses a session after 60 minutes of inactivity, on
-- the user's next contact. That requires a last-activity timestamp, updated on
-- every buffered message or attached document.

ALTER TABLE ingest_sessions
    ADD COLUMN last_active_at TIMESTAMPTZ NOT NULL DEFAULT now();
