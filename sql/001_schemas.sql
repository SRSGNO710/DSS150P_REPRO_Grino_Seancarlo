-- 001_schemas.sql — idempotent; safe to run on every load.
CREATE SCHEMA IF NOT EXISTS staging;   -- conformed readings (one row per valid reading)
CREATE SCHEMA IF NOT EXISTS curated;   -- data product (one row per station-day)
CREATE SCHEMA IF NOT EXISTS ml;        -- predictions and metrics
CREATE SCHEMA IF NOT EXISTS audit;     -- run records, task records, load log, DQ results
