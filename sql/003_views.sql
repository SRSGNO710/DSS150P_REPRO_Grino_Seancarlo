-- 003_views.sql — evidence/inspection views (idempotent).

-- Duplicate business keys per table. Must be 0 after every run (acceptance A15).
CREATE OR REPLACE VIEW audit.v_duplicate_business_keys AS
SELECT 'staging.air_quality_readings' AS table_name,
       count(*) AS row_count,
       count(*) - count(DISTINCT (location, pollutant, last_updated_utc)) AS duplicate_business_keys
FROM staging.air_quality_readings
UNION ALL
SELECT 'curated.station_day_pollutants',
       count(*),
       count(*) - count(DISTINCT (country_label, city, location, reading_date))
FROM curated.station_day_pollutants
UNION ALL
SELECT 'ml.predictions',
       count(*),
       count(*) - count(DISTINCT (station_day_key, model_name))
FROM ml.predictions;

-- Latest runs with their counts, newest first.
CREATE OR REPLACE VIEW audit.v_run_overview AS
SELECT pipeline_run_id, status, started_at, ended_at,
       ended_at - started_at                         AS duration,
       parameters ->> 'mode'                         AS mode,
       (counts ->> 'raw')::int                       AS raw_rows,
       (counts ->> 'staging')::int                   AS staging_rows,
       (counts ->> 'curated')::int                   AS curated_rows,
       (counts ->> 'model_features')::int            AS model_rows,
       (counts ->> 'train')::int                     AS train_rows,
       (counts ->> 'test')::int                      AS test_rows,
       (counts ->> 'predictions')::int               AS prediction_rows,
       error ->> 'stage'                             AS failed_stage
FROM audit.pipeline_runs
ORDER BY started_at DESC NULLS LAST;

-- Rows per country (the partition key used in the partition demo).
CREATE OR REPLACE VIEW curated.v_station_days_by_country AS
SELECT country_label, count(*) AS station_days, count(pm25) AS with_pm25
FROM curated.station_day_pollutants
GROUP BY country_label;
