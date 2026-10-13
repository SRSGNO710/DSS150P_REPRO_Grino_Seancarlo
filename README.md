# Reproducible PM2.5 Exceedance Pipeline

**DSS150P Â· Fundamentals of Data Engineering â€” Reproducibility Laboratory**
Seancarlo Raphael S. Grino

This repository turns the group project *"Predicting PM2.5 Air Quality Exceedances Using Machine Learning: A Co-Pollutant Chemistry Approach"* (a Jupyter notebook) into an engineered, reproducible data pipeline. Starting from the one permitted source file, it verifies the file's identity and lands it in a raw layer. It then builds staging and curated layers, loads PostgreSQL rerun-safely, validates data contracts, retrains the three classifiers, and checks the regenerated metrics against the published baseline under a declared tolerance. All of it is orchestrated by Apache Airflow inside Docker Compose.

**Result:** every row count, the exact test-set membership and 14 of 15 published metrics are reproduced exactly. The 15th (Gradient Boosting CV F1) differs by 0.0004, inside the 0.01 tolerance, and is explained in [`docs/reproducibility_report.md`](docs/reproducibility_report.md).

![Architecture](docs/architecture.png)

---

## Contents
1. [Prerequisites](#1-prerequisites) Â· 2. [Data](#2-data-acquisition-and-placement) Â· 3. [Setup](#3-setup) Â· 4. [Running with Airflow](#4-running-with-airflow-primary) Â· 5. [Running without Airflow](#5-running-without-airflow) Â· 6. [Validation and tests](#6-validation-and-tests) Â· 7. [Expected outputs](#7-expected-outputs) Â· 8. [How it works](#8-how-it-works) Â· 9. [Troubleshooting](#9-troubleshooting) Â· 10. [Acceptance tests](#10-acceptance-test-map-a1a18) Â· 11. [Production changes](#11-what-i-would-change-for-production) Â· 12. [Documentation index](#12-documentation-index)

## 1. Prerequisites

| Need | Version | Notes |
|---|---|---|
| Docker Desktop (with Compose v2) | recent | Windows: WSL 2 backend. About 4 GB RAM free for the stack |
| Git | any | |
| Free ports | 8080 (Airflow UI), 5433 (warehouse) | change `WAREHOUSE_HOST_PORT` in `.env` if 5433 is taken |
| *(optional)* Python | 3.12 | only for running outside Docker (Â§5.2) |

No absolute paths are needed. Everything is relative to the repository root (`$PIPELINE_HOME` inside containers = `/opt/project`).

## 2. Data acquisition and placement

The pipeline uses exactly one input: **`world_air_quality.csv`** (OpenAQ network aggregation, 54,255 rows), published on Kaggle as [World Air Quality](https://www.kaggle.com/datasets/joebeachcapital/world-air-quality).

The Kaggle page states no licence, so the file is **not committed**. It is referenced by checksum in [`contracts/source_manifest.yml`](contracts/source_manifest.yml):

| Property | Value |
|---|---|
| sha256 | `2b2202a0fa1cffdeddf803e54dd7f822812d86de3a6449c03798bb872b272f10` |
| size | 6,472,005 bytes Â· 54,255 data rows Â· UTF-8 with BOM Â· `;` delimited Â· CRLF |

1. Download the dataset from the Kaggle page (or take `world_air_quality.csv` from the course project archive).
2. Place it at **`data/source/world_air_quality.csv`**. Do not open and re-save it in Excel, which would change the checksum.
3. Verify it (Â§3 step 3). The pipeline refuses any file whose checksum does not match.

A profile of the file (schema, nulls, distributions, quirks) is in [`docs/source_profile.md`](docs/source_profile.md).

## 3. Setup

PowerShell (Windows). bash equivalents are identical except `cp` for `Copy-Item`.

```powershell
git clone https://github.com/DSS150P_REPRO_Grino_Seancarlo.git
cd DSS150P_REPRO_Grino_Seancarlo

# 1. local configuration (git-ignored); edit every change_me value
Copy-Item .env.example .env

# 2. build images and start the stack: warehouse + Airflow (init, webserver, scheduler)
docker compose up -d --build
docker compose ps        # airflow-init "exited (0)", the rest healthy/running

# 3. verify the source file against the manifest
docker compose run --rm pipeline python -m pipeline.cli verify-source
```

What gets built:
* **`dss150p-airflow:2.10.5`**: `apache/airflow:2.10.5-python3.12` plus an isolated venv `/opt/pipeline-venv` holding the pinned pipeline dependencies (`requirements.txt`). Airflow's own pandas/numpy are left untouched.
* **`dss150p-pipeline:1.0`**: `python:3.12-slim` with the same pinned dependencies plus pytest. Used for CLI commands, tests and scripts.
* Two PostgreSQL 16 containers: **`warehouse`** (pipeline target) and **`airflow-db`** (Airflow metadata), each with a named volume.

Configuration lives in [`config/pipeline.yml`](config/pipeline.yml) (paths, cleaning rules, model parameters, tolerance). Credentials come only from `.env`; Compose refuses to start if a required one is missing.

## 4. Running with Airflow (primary)

1. Open **<http://localhost:8080>** and log in with `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` from `.env`.
2. Unpause **`air_quality_pm25_pipeline`** and choose **Trigger DAG w/ config**:

   | Param | Default | Meaning |
   |---|---|---|
   | `mode` | `full` | `full` rebuilds raw from the whole file; `incremental` ingests the next batch after the checkpoint |
   | `model_seed` | `42` | seed for split, CV and models (42 reproduces the baseline; other seeds are sensitivity runs) |
   | `batch_rows` | `20000` | source rows per incremental batch |
   | `source_override` | *(empty)* | repo-relative path to an alternative file, **only** for the failure demo |

3. The DAG runs ten tasks in a fixed chain:

   `ingest_source â†’ validate_raw â†’ transform_staging â†’ build_curated â†’ load_postgres â†’ validate_curated â†’ train_model â†’ evaluate_model â†’ reproducibility_check â†’ publish_outputs`

DAG design ([`dags/air_quality_pipeline.py`](dags/air_quality_pipeline.py)):
* **Orchestration only.** Every task is a `BashOperator` running `python -m pipeline.cli run-stage <task>` from the venv. All logic lives in `src/pipeline/`.
* **`schedule=None`, `catchup=False`.** The source is a static, checksum-pinned snapshot; a calendar schedule would only recompute identical results, so runs are triggered manually after code/config changes, for incremental batches, or for the failure demo. `catchup=False` means enabling the DAG never back-fills. `max_active_runs=1` because runs share the local layers and the checkpoint.
* **Retries only where safe.** `ingest_source` (2Ã—; deterministic batch ids overwrite), `load_postgres` (3Ã—, exponential backoff; idempotent upserts) and `publish_outputs` (2Ã—). Validation, transform and model tasks are deterministic, so a retry could not fix them and they have 0 retries.
* **Timeouts.** `train_model` and `evaluate_model` 15 min, `load_postgres` 10 min, others 30 min; the whole DAG run is capped at 2 h.
* **Failure handling.** `on_failure_callback` logs dag/task/run id/try/params/exception and mirrors the failure into `audit.task_runs`. That covers timeouts, where the stage process cannot record its own failure. `on_retry_callback` logs retries.
* **One run id.** Airflow's `run_id` becomes the `pipeline_run_id` (with `:` and `+` replaced by `_`). It ties together the log lines, `data/outputs/runs/<run_id>/` and the `audit.*` rows.

CLI trigger alternative: `docker compose exec airflow-scheduler airflow dags trigger air_quality_pm25_pipeline --conf '{"mode":"full","model_seed":42}'`

## 5. Running without Airflow

### 5.1 In the pipeline container (recommended)
```powershell
docker compose run --rm pipeline python -m pipeline.cli run-all                       # full run, seed 42
docker compose run --rm pipeline python -m pipeline.cli run-all --mode incremental    # next batch
docker compose run --rm pipeline python -m pipeline.cli run-all --run-id <id> --from-stage load_postgres   # resume
docker compose run --rm pipeline python -m pipeline.cli --help                        # all commands
```
`run-all` calls exactly the same stage functions as the DAG, in the same order.

### 5.2 Local virtual environment (optional)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e . --no-deps
docker compose up -d warehouse          # reachable on localhost:5433 via POSTGRES_HOST/PORT in .env
python -m pipeline.cli run-all          # .env is read automatically
python -m pipeline.cli run-all --no-db  # quick smoke run without PostgreSQL
```

### 5.3 Other commands
| Command | Purpose |
|---|---|
| `verify-source` | checksum/size/rows against the manifest |
| `benchmark --repeats 3` | CSV / JSON / Parquet / PostgreSQL storage experiment â†’ `data/benchmark/` |
| `partition --value Japan --load-db` | write country partitions; read and load one partition on its own |
| `db-status` | duplicate-key report, recent runs, load log |
| `db-init` / `db-reset --yes` | create / drop-and-recreate the warehouse schemas |
| `reset-local --yes [--include-outputs]` | delete local layers and the checkpoint (keeps the source) |
| `show-run --run-id <id>` | print a run summary |
| `scripts/failure_injection.py` | four reversible failure scenarios with recovery and evidence files |
| `scripts/notebook_reference.py` | rebuild `expected/baseline_metrics.json` from the archived notebook |
| `scripts/profile_source.py` | regenerate `docs/source_profile.md` |

## 6. Validation and tests

```powershell
docker compose run --rm pipeline python -m pytest -v                    # unit + integration (+ PostgreSQL)
docker compose exec airflow-scheduler python -m pytest /opt/project/tests/integration/test_dag_integrity.py -p no:cacheprovider
```

* **Unit tests** (`tests/unit/`): cleaning rules and their order, implicit-NA handling, pivot and daily means, the WHO label boundary, engineered features, imputation; each DQ check category against an intentionally broken frame; reproducibility tolerance logic; config validation; CSV/Parquet layer equivalence; a YAML regression guard.
* **Integration tests** (`tests/integration/`): ingest â†’ validate â†’ staging â†’ curated in an isolated project; incremental batches plus an idempotent no-op; quarantine without moving the checkpoint; refusal of a changed source; PostgreSQL idempotent and partition-scoped loads; end-to-end baseline reproduction on the real file; DAG structure (DagBag).
* **Data contract** ([`contracts/data_contract.yml`](contracts/data_contract.yml)): 32 machine-readable checks across raw (9), staging (7), curated (6) and model features (10), plus 4 runtime checks (train/test disjointness, feature dimensions, fileâ†”PostgreSQL parity, duplicate keys in PostgreSQL) covering schema, completeness, uniqueness, validity, consistency, freshness (checksum), predictive integrity (feature order, leakage guard, train/test overlap) and reproducibility (exact counts). They are enforced by `src/pipeline/quality.py`. A failed `severity: error` check fails the task, stops everything downstream, and records the result in `dq_report.json` and `audit.dq_results`. A raw batch that fails is quarantined, and the checkpoint does not move.
* **Structured logs**: JSON lines with `timestamp, level, pipeline_run_id, stage, event, rows_in/rows_out, duration_s, error` go to the Airflow task log and to `logs/pipeline_<run_id>.jsonl`.
* **Run record**: `data/outputs/runs/<run_id>/run_summary.json` holds start/end, status, per-stage results, counts (source, raw, staging, curated, model_features, train, test, predictions) and layer fingerprints. It is mirrored to `audit.pipeline_runs` / `audit.task_runs`.

## 7. Expected outputs

| Layer / output | Rows | Location |
|---|---|---|
| Source / raw | 54,255 | `data/source/`, `data/raw/committed/` |
| Staging readings | 30,545 | `data/staging/air_quality_readings.parquet`, `staging.air_quality_readings` |
| Curated station-days (data product) | 11,558 | `data/curated/station_day_pollutants.parquet`, `curated.station_day_pollutants` |
| Model features | 2,628 (1,513 Unhealthy / 1,115 Safe) | `data/features/model_features.parquet` |
| Train / test | 2,102 / 526 | `data/outputs/runs/<id>/split.csv` |
| Predictions | 526 rows Ã— 3 models | `data/outputs/latest/predictions.csv`, `ml.predictions` (1,578 rows) |
| Metrics + comparison | â€” | `data/outputs/latest/metrics.json`, `metrics_comparison.json` (`status: pass`) |

Reproduced metrics (seed 42):

| Model | Test accuracy | Test F1 (weighted) | 5-fold CV F1 |
|---|---|---|---|
| Logistic Regression | 0.6730 | 0.6570 | 0.6728 |
| Random Forest | 0.7167 | 0.7163 | 0.7077 |
| Gradient Boosting | 0.7053 | 0.7044 | 0.7118 (paper 0.7122) |

## 8. How it works

![Lineage](docs/lineage.png)

* **Raw (lossless landing).** Every source field is stored as the exact string read from the file, plus `_source_row_number, _source_sha256, _record_hash, _ingest_batch_id, _pipeline_run_id, _ingested_at`.
* **Full and incremental ingestion.** The dataset has no update timestamp, so incremental arrival is *simulated* with a **source-row high-water mark** tied to the file checksum (`data/checkpoints/ingest_checkpoint.json`). It is never presented as business time. Batch ids are deterministic (`b000001-020000_2b2202a0`), so re-ingesting a batch overwrites instead of duplicating. Ingestion uses a two-phase commit: `ingest_source` writes to `raw/_pending/`, and only `validate_raw` promotes the batch and advances the checkpoint.
* **Staging** applies the notebook's cleaning rules in the notebook's order (the order matters). Pandas' implicit NA strings (`N/A`, `None`, â€¦) are applied explicitly.
* **Curated** holds daily station means pivoted to one row per station-day, sorted exactly as the notebook did (the split depends on row order).
* **PostgreSQL loads** run in one transaction per table: `COPY` into a temp table, `INSERT â€¦ ON CONFLICT (business-key PK) DO UPDATE â€¦ WHERE row_hash changed`, a scoped delete for full sync, and a `load_log` upsert. A second run or an Airflow retry inserts 0 rows. `audit.v_duplicate_business_keys` must show 0.
* **Model** uses a stratified 80/20 split, seed 42, with the same three estimators and parameters as the notebook. The test-set membership fingerprint is checked against the notebook's.
* **Publishing**: outputs are promoted to `data/outputs/latest/` only after the reproducibility check passes. A failed run never overwrites the published outputs.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `required variable â€¦ is missing a value` | Create `.env` from `.env.example` and fill every value |
| `SourceIdentityError: Source file not found` | Place the CSV at `data/source/world_air_quality.csv` |
| `verify-source` reports a sha256 mismatch | The file was modified (often re-saved by Excel) or is a different download. Get a fresh copy |
| Port 8080 or 5433 already in use | Stop the other service, or change `WAREHOUSE_HOST_PORT` / the webserver port mapping |
| DAG not visible in the UI | Wait about 30 s; `docker compose exec airflow-scheduler airflow dags list-import-errors` |
| `load_postgres` failing with `Cannot connect` | `docker compose ps`; `docker compose start warehouse`; then Clear the task |
| Permission errors on `data/` or `logs/airflow` (Linux/macOS hosts) | Set `AIRFLOW_UID` in `.env` to `id -u` |
| Slow first build | The Airflow image installs the scientific stack into the venv; later builds use the cache |

More failure modes and the restart, rerun, reset and rollback procedures are in [`docs/runbook.md`](docs/runbook.md).

## 10. Acceptance-test map (A1â€“A18)

| ID | Where it is demonstrated |
|---|---|
| A1 clone + README | this file; fresh-clone check in `docs/evidence/` |
| A2 checksum/manifest | `verify-source`; check `raw_source_checksum_matches_manifest` |
| A3 no absolute paths | `config/pipeline.yml` (relative); `test_config_paths_are_relative`; run ids sanitised |
| A4 no secrets | `.env` git-ignored; Compose uses `${VAR:?}`; only `.env.example` committed |
| A5 environment from declared files | `requirements*.txt`, `Dockerfile`, `docker/airflow/Dockerfile`, `docker-compose.yml` |
| A6 raw count/checksum | `raw_row_count_matches_manifest` (54,255) + checksum check |
| A7 incremental without duplicates | `test_incremental_batches_then_idempotent_noop`; `docs/evidence/dev-sandbox/incremental_runs.md` |
| A8 deterministic layers | identical layer fingerprints across runs (`run_summary.json` â†’ `layer_fingerprints`) |
| A9 rerun-safe PostgreSQL load | `audit.load_log` (0 inserted on rerun); `test_load_is_idempotent_and_detects_changes` |
| A10 equivalent logical records across formats | benchmark `same_logical_records`; `db_file_parity_curated`; CSV/Parquet round-trip test |
| A11 independent partition | `partition --value Japan --load-db`; `test_partition_scoped_load_leaves_other_partitions` |
| A12 DAG valid + completes | `test_dag_integrity.py`; Airflow Graph view screenshot |
| A13 recoverable failure | `scripts/failure_injection.py`; runbook Â§7 |
| A14 checks detect invalid conditions | `test_checks_detect_invalid_conditions` (6 categories) + failure injection |
| A15 no duplicate business keys after second run | `audit.v_duplicate_business_keys` = 0 |
| A16 metrics regenerated from curated data | `train_model` reads `model_features` built from curated; metrics written by `evaluate_model` |
| A17 tolerance rule | `metrics_comparison.json` (`status: pass`, max \|Î”\| 0.0004 â‰¤ 0.01) |
| A18 docs consistent | `docs/architecture.png`, `docs/lineage.png`, runbook, reproducibility report |

## 11. What I would change for production

**Replace full recomputation and full-snapshot loads with partition-incremental processing.** Today every run rebuilds staging and curated from all committed raw batches in pandas, then COPY-merges whole tables. That is simple and provably deterministic at 54k rows, but it scales linearly with history and locks the whole table per load. In production I would partition raw and curated by ingest date (and country), recompute only the partitions touched by a new batch, merge only those partitions into PostgreSQL (the scoped `load_frame` already supports this), and move the transforms to a columnar engine such as DuckDB or Polars. Other changes: Airflow 3 with a CeleryExecutor or KubernetesExecutor, secrets in a vault, images pinned by digest, and drift monitoring on the eight features.

## 12. Documentation index

| Document | Content |
|---|---|
| [`docs/reproducibility_report.md`](docs/reproducibility_report.md) | prior vs reproduced results, deviations, deterministic controls, limitations |
| [`docs/runbook.md`](docs/runbook.md) | normal operation, failures, restart/rerun, DB reset, rollback |
| [`docs/storage_benchmark.md`](docs/storage_benchmark.md) | storage experiment and partition demo |
| [`docs/source_profile.md`](docs/source_profile.md) | source identity and schema profile |
| [`docs/reflection.md`](docs/reflection.md) | answers to the 15 analysis questions |
| [`docs/architecture.png`](docs/architecture.png), [`docs/lineage.png`](docs/lineage.png) | diagrams (sources in `docs/diagrams/`) |
| [`docs/evidence/`](docs/evidence/README.md) | verification evidence and the screenshot checklist |
| [`contracts/`](contracts/) | source manifest and data contract |
| [`notebooks/original/`](notebooks/original/) | archived original notebook (reference only; not part of the pipeline) |
