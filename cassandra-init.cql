-- ============================================================
-- Keyspace
-- NOTE: SimpleStrategy is OK for single-DC dev/testing.
-- For production multi-DC, use NetworkTopologyStrategy.
-- ============================================================

CREATE KEYSPACE IF NOT EXISTS jobrec
WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 3
}
AND durable_writes = true;

USE jobrec;

-- ============================================================
-- User events (time-series per user)
-- ============================================================

CREATE TABLE IF NOT EXISTS user_events (
  user_id text,
  event_time timestamp,
  event_id text,
  event_type text,
  job_id text,
  session_id text,
  device_type text,
  location text,
  payload text,
  PRIMARY KEY ((user_id), event_time, event_id)
) WITH CLUSTERING ORDER BY (event_time DESC, event_id ASC)
  AND gc_grace_seconds = 864000
  AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_size': 1,
    'compaction_window_unit': 'DAYS'
  };

-- ============================================================
-- Job events (time-series per job)
-- ============================================================

CREATE TABLE IF NOT EXISTS job_events (
  job_id text,
  event_time timestamp,
  event_id text,
  event_type text,
  user_id text,
  session_id text,
  PRIMARY KEY ((job_id), event_time, event_id)
) WITH CLUSTERING ORDER BY (event_time DESC, event_id ASC)
  AND gc_grace_seconds = 864000;

-- ============================================================
-- Recommendation history (per user)
-- ============================================================

CREATE TABLE IF NOT EXISTS recommendation_history (
  user_id text,
  rec_time timestamp,
  rec_id text,
  job_ids list<text>,
  scores map<text, double>,
  algorithm text,
  top_k int,
  diversify boolean,
  PRIMARY KEY ((user_id), rec_time, rec_id)
) WITH CLUSTERING ORDER BY (rec_time DESC, rec_id ASC);

-- ============================================================
-- User sessions
-- IMPORTANT: Cassandra counter tables cannot mix counters with
-- non-counter columns. So split into two tables.
-- ============================================================

CREATE TABLE IF NOT EXISTS user_sessions (
  session_id text,
  user_id text,
  start_time timestamp,
  end_time timestamp,
  PRIMARY KEY (session_id)
);

CREATE TABLE IF NOT EXISTS user_session_counters (
  session_id text,
  events_count counter,
  PRIMARY KEY (session_id)
);

-- ============================================================
-- Job popularity by hour (counters are OK here: only counters + PK)
-- ============================================================

CREATE TABLE IF NOT EXISTS job_popularity_hourly (
  job_id text,
  hour_bucket timestamp,
  views counter,
  clicks counter,
  applies counter,
  PRIMARY KEY ((job_id), hour_bucket)
) WITH CLUSTERING ORDER BY (hour_bucket DESC);

-- ============================================================
-- Secondary indexes (named for compatibility)
-- NOTE: Secondary indexes can be expensive at scale; prefer
-- dedicated tables (like job_events) for global queries.
-- ============================================================

CREATE INDEX IF NOT EXISTS user_events_event_type_idx ON user_events (event_type);
CREATE INDEX IF NOT EXISTS user_events_job_id_idx     ON user_events (job_id);

CREATE INDEX IF NOT EXISTS job_events_event_type_idx  ON job_events (event_type);
CREATE INDEX IF NOT EXISTS job_events_user_id_idx     ON job_events (user_id);

-- ============================================================
-- Materialized view for recent events by type
-- IMPORTANT RULE: MV primary key MUST include the base table's
-- partition key columns as part of the MV partition key.
-- Base partition key = user_id, so user_id must be in MV partition key.
-- ============================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS recent_events_by_type AS
  SELECT user_id, event_time, event_id, event_type, job_id
  FROM user_events
  WHERE user_id IS NOT NULL
    AND event_type IS NOT NULL
    AND event_time IS NOT NULL
    AND event_id IS NOT NULL
  PRIMARY KEY ((event_type, user_id), event_time, event_id)
  WITH CLUSTERING ORDER BY (event_time DESC, event_id ASC);

-- ============================================================
-- Skill interactions (time-series per skill)
-- ============================================================

CREATE TABLE IF NOT EXISTS skill_interactions (
  skill_name text,
  interaction_time timestamp,
  user_id text,
  job_id text,
  interaction_type text,
  PRIMARY KEY ((skill_name), interaction_time)
) WITH CLUSTERING ORDER BY (interaction_time DESC);
