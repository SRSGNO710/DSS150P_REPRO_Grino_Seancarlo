-- 900_reset.sql — DESTRUCTIVE: drops every pipeline schema. Used only by
-- `python -m pipeline.cli db-reset --yes` (see docs/runbook.md §Database reset).
DROP SCHEMA IF EXISTS ml CASCADE;
DROP SCHEMA IF EXISTS curated CASCADE;
DROP SCHEMA IF EXISTS staging CASCADE;
DROP SCHEMA IF EXISTS audit CASCADE;
