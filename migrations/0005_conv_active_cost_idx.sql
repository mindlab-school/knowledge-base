-- 0005_conv_active_cost_idx: enforce one active conversation per user and speed
-- up month-to-date cost aggregation.
--
-- The conversations table shipped without the partial unique index that ingest
-- sessions and facts already have, so a non-atomic SELECT-then-INSERT could race
-- two concurrent first-messages into duplicate active rows. Deactivate any such
-- duplicates (keep the newest per user) before adding the constraint.
UPDATE conversations c
SET active = FALSE
WHERE active
  AND id <> (
      SELECT max(id) FROM conversations dup
      WHERE dup.user_id = c.user_id AND dup.active
  );

CREATE UNIQUE INDEX conversations_one_active ON conversations (user_id) WHERE active;

-- month_to_date_cost() filters messages by `usage ? 'cost'`; without this partial
-- index every /chat full-scans the messages table.
CREATE INDEX messages_cost_idx ON messages (created_at) WHERE usage ? 'cost';
