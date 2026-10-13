-- 002_tables.sql — idempotent DDL for the warehouse tables.
-- Rerun safety: every table has a PRIMARY KEY derived from the declared business
-- key, and loads use INSERT ... ON CONFLICT DO UPDATE (see src/pipeline/db.py),
-- so loading the same snapshot twice cannot create duplicate logical records.
-- Audit columns: pipeline_run_id = run that last inserted/changed the row;
-- row_hash = sha256 of the business columns (unchanged rows are not rewritten).

CREATE TABLE IF NOT EXISTS staging.air_quality_readings (
    reading_key        CHAR(32)         PRIMARY KEY,           -- sha256(location|pollutant|ts)[:32]
    country_code       TEXT             NOT NULL,
    country_label      TEXT             NOT NULL,
    city               TEXT             NOT NULL,
    location           TEXT             NOT NULL,
    latitude           DOUBLE PRECISION,
    longitude          DOUBLE PRECISION,
    pollutant          TEXT             NOT NULL CHECK (pollutant IN ('PM2.5','PM10','NO2','O3','SO2','CO')),
    source_name        TEXT             NOT NULL,
    unit               TEXT             NOT NULL,
    value              DOUBLE PRECISION NOT NULL CHECK (value >= 0),
    last_updated_utc   TIMESTAMPTZ      NOT NULL,
    reading_date       DATE             NOT NULL,
    source_row_number  INTEGER          NOT NULL,
    ingest_batch_id    TEXT             NOT NULL,
    row_hash           CHAR(64)         NOT NULL,
    pipeline_run_id    TEXT             NOT NULL,
    loaded_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_readings_business_key UNIQUE (location, pollutant, last_updated_utc)
);

CREATE TABLE IF NOT EXISTS curated.station_day_pollutants (
    station_day_key    CHAR(32)         PRIMARY KEY,           -- sha256(country|city|location|date)[:32]
    country_label      TEXT             NOT NULL,
    city               TEXT             NOT NULL,
    location           TEXT             NOT NULL,
    reading_date       DATE             NOT NULL,
    co                 DOUBLE PRECISION,
    no2                DOUBLE PRECISION,
    o3                 DOUBLE PRECISION,
    pm10               DOUBLE PRECISION,
    pm25               DOUBLE PRECISION,
    so2                DOUBLE PRECISION,
    n_readings         INTEGER          NOT NULL,
    row_hash           CHAR(64)         NOT NULL,
    pipeline_run_id    TEXT             NOT NULL,
    loaded_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_station_day_business_key UNIQUE (country_label, city, location, reading_date)
);
CREATE INDEX IF NOT EXISTS ix_station_day_country ON curated.station_day_pollutants (country_label);

CREATE TABLE IF NOT EXISTS ml.predictions (
    station_day_key    CHAR(32)         NOT NULL,
    model_name         TEXT             NOT NULL,
    country_label      TEXT             NOT NULL,
    location           TEXT             NOT NULL,
    reading_date       DATE             NOT NULL,
    pm25_actual        DOUBLE PRECISION NOT NULL,
    label_actual       TEXT             NOT NULL CHECK (label_actual IN ('Safe','Unhealthy')),
    label_predicted    TEXT             NOT NULL CHECK (label_predicted IN ('Safe','Unhealthy')),
    proba_unhealthy    DOUBLE PRECISION NOT NULL CHECK (proba_unhealthy BETWEEN 0 AND 1),
    is_deployed_model  BOOLEAN          NOT NULL,
    row_hash           CHAR(64)         NOT NULL,
    pipeline_run_id    TEXT             NOT NULL,
    loaded_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (station_day_key, model_name)
);

-- Metrics are run history: one row per (run, model, metric). A rerun adds rows for
-- its own run id; re-executing the SAME run (Airflow retry) upserts the same rows.
CREATE TABLE IF NOT EXISTS ml.model_metrics (
    pipeline_run_id    TEXT             NOT NULL,
    model_name         TEXT             NOT NULL,
    metric_name        TEXT             NOT NULL,
    metric_value       DOUBLE PRECISION NOT NULL,
    baseline_value     DOUBLE PRECISION,
    abs_diff           DOUBLE PRECISION,
    within_tolerance   BOOLEAN,
    recorded_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_run_id, model_name, metric_name)
);

CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    pipeline_run_id    TEXT             PRIMARY KEY,
    status             TEXT             NOT NULL,
    started_at         TIMESTAMPTZ,
    ended_at           TIMESTAMPTZ,
    parameters         JSONB,
    counts             JSONB,
    layer_fingerprints JSONB,
    error              JSONB,
    updated_at         TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit.task_runs (
    pipeline_run_id    TEXT             NOT NULL,
    stage              TEXT             NOT NULL,
    status             TEXT             NOT NULL,
    started_at         TIMESTAMPTZ,
    ended_at           TIMESTAMPTZ,
    duration_s         DOUBLE PRECISION,
    counts             JSONB,
    error              TEXT,
    updated_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_run_id, stage)
);

CREATE TABLE IF NOT EXISTS audit.load_log (
    pipeline_run_id    TEXT             NOT NULL,
    table_name         TEXT             NOT NULL,
    scope              TEXT             NOT NULL DEFAULT 'full',
    rows_in_snapshot   INTEGER          NOT NULL,
    inserted           INTEGER          NOT NULL,
    updated            INTEGER          NOT NULL,
    deleted            INTEGER          NOT NULL,
    unchanged          INTEGER          NOT NULL,
    rows_in_table      INTEGER          NOT NULL,
    loaded_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_run_id, table_name, scope)
);

CREATE TABLE IF NOT EXISTS audit.dq_results (
    pipeline_run_id    TEXT             NOT NULL,
    layer              TEXT             NOT NULL,
    check_id           TEXT             NOT NULL,
    category           TEXT,
    severity           TEXT             NOT NULL,
    status             TEXT             NOT NULL,
    observed           JSONB,
    threshold          JSONB,
    message            TEXT,
    recorded_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_run_id, layer, check_id)
);
