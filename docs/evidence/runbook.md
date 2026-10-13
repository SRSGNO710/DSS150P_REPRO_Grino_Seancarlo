# Runbook — PM2.5 Exceedance Pipeline

Operational guide for running, recovering and resetting the pipeline. Commands are shown for **PowerShell (Windows)**. They are identical in bash unless noted.
`dc` below means `docker compose`.

---

## 1. Normal execution

### 1.1 Start the stack
```powershell
Copy-Item .env.example .env        # first time only; then edit the change_me values
dc up -d --build                   # warehouse, airflow-db, airflow-init, webserver, scheduler
dc ps                              # all "healthy"/"running"; airflow-init "exited (0)"
```
Airflow UI: <http://localhost:8080> (user/password from `.env`).

### 1.2 Run through Airflow (primary path)
1. Unpause **`air_quality_pm25_pipeline`**.
2. **Trigger DAG w/ config**: `mode=full`, `model_seed=42`, `batch_rows=20000`, `source_override` empty.
3. Watch the Graph view. All ten tasks should turn green in about 1–3 minutes.
4. Check the result:
   ```powershell
   Get-Content data\outputs\latest\PUBLISHED_RUN.txt
   dc run --rm pipeline python -m pipeline.cli db-status
   ```

### 1.3 Run without Airflow (same code path)
```powershell
dc run --rm pipeline python -m pipeline.cli run-all                 # full mode, seed 42
dc run --rm pipeline python -m pipeline.cli run-all --mode incremental
```

### 1.4 Expected outputs of a successful full run
| Where | What |
|---|---|
| `data/outputs/latest/` | `metrics.json`, `predictions.csv`, `metrics_comparison.json` (status `pass`), `dq_report.json`, `run_summary.json`, `PUBLISHED_RUN.txt` |
| PostgreSQL | `staging.air_quality_readings` 30,545 · `curated.station_day_pollutants` 11,558 · `ml.predictions` 1,578 · audit rows for the run |
| `logs/pipeline_<run_id>.jsonl` | one JSON line per event (stage start/end, DQ checks, loads, metrics) |

## 2. Run identifiers and where to look

Every Airflow run id (e.g. `manual__2026-10-13T08:00:00+00:00`) becomes the **pipeline_run_id** after `:` and `+` are replaced by `_`, so it is safe for Windows file names.

| To find… | Look at |
|---|---|
| How far a run got and why it failed | `data/outputs/runs/<run_id>/run_summary.json` (`status`, `error.stage`, per-stage status/duration/counts) |
| Which data checks failed | `data/outputs/runs/<run_id>/dq_report.json` or `SELECT * FROM audit.dq_results WHERE pipeline_run_id = '…' AND status <> 'passed';` |
| Detailed log | `logs/pipeline_<run_id>.jsonl`; Airflow task log (UI, or `logs/airflow/`) |
| Run history | `SELECT * FROM audit.v_run_overview;` |
| What each load changed | `SELECT * FROM audit.load_log WHERE pipeline_run_id = '…';` |

## 3. Common failures

| Symptom (failed task / message) | Cause | Fix |
|---|---|---|
| `ingest_source`: `SourceIdentityError: Source file not found` | CSV not placed | Put `world_air_quality.csv` in `data/source/` (README §Data) |
| `validate_raw` fails on `raw_source_checksum_matches_manifest` | Different or modified file | Re-download the dataset; compare with `python -m pipeline.cli verify-source`. Never edit the manifest to match a new file without a documented decision |
| `validate_raw` fails on `raw_schema_required_columns` / `raw_value_parseable` | Corrupted file, or a failure-injection copy passed via `source_override` | Batch is already quarantined (`data/quarantine/<run_id>/`) and the checkpoint is unchanged. Fix the input and trigger a new run |
| `ingest_source`: `CheckpointError: source file changed since the last checkpoint` | Incremental run on a different file | Run once with `mode=full` (rebuilds raw and resets the checkpoint) |
| `transform_staging` / `build_curated`: `stg_row_count_baseline` / `cur_row_count_baseline` failed | Cleaning logic or source changed | Compare `row_counts_by_step` in run_summary with README §Expected counts. Do not relax the contract to make it pass |
| `load_postgres`: `DatabaseError: Cannot connect …` (task retries up to 3×, backoff) | Warehouse down or wrong credentials | `dc ps`; `dc start warehouse`; check `.env` `POSTGRES_PASSWORD`. Then clear the task (§4.2) |
| `validate_curated`: `db_file_parity_curated` failed | Someone changed the warehouse table manually, or the load was partial | Re-run `load_postgres` (idempotent), then clear `validate_curated` |
| `reproducibility_check`: `ReproducibilityError: Materially different` | Metric outside tolerance, or count/split mismatch | Open `metrics_comparison.json`: `count_rows` / `split_rows` with `match:false` point to data or split changes; `metric_rows` point to model or dependency changes. Fix the cause; outputs are **not** published until this passes |
| Task killed after `execution_timeout` | Machine too slow or stuck | The failure callback records it in `audit.task_runs`. Clear the task to retry, or raise the timeout in the DAG |
| `ConfigError: storage.layer_format must be one of …` | Invalid config or environment override | Fix `config/pipeline.yml` / `.env` |
| `airflow-init` exits non-zero / webserver not healthy | Missing `.env` value or port 8080 busy | `dc logs airflow-init`; Compose names the missing variable; free port 8080 |
| Permission denied writing `data/` or `logs/airflow` (Linux hosts) | Container UID differs from host owner | Set `AIRFLOW_UID` in `.env` to the output of `id -u`, then `dc up -d` |

## 4. Restart / rerun procedures

### 4.1 Rerun everything
Trigger the DAG again (full mode). All loads are upserts on business keys, so a second run inserts **0** rows when nothing changed (see `audit.load_log`).

### 4.2 Resume a failed run from the failed task (checkpoint recovery)
* **Airflow:** open the failed run → click the failed task → **Clear** (with "Downstream" selected). The task and everything after it re-run with the **same run id**. Upstream outputs on disk are reused.
* **CLI:** `dc run --rm pipeline python -m pipeline.cli run-all --run-id <run_id> --from-stage <stage>`

Why this is safe: each stage reads only committed outputs of earlier stages, the raw checkpoint advances only after `validate_raw` succeeds, and every database write is an idempotent upsert keyed by business keys.

### 4.3 Retry after the database committed but the task did not hear back
No action needed. A retry replays the same snapshot: 0 inserted, 0 updated, 0 deleted, and the same `audit.load_log` row is upserted. Verify with `SELECT * FROM audit.v_duplicate_business_keys;`. All zeros means no duplicates.

### 4.4 Incremental ingestion
```powershell
# trigger with mode=incremental (and batch_rows), or:
dc run --rm pipeline python -m pipeline.cli run-all --mode incremental --batch-rows 20000
Get-Content data\checkpoints\ingest_checkpoint.json     # last_committed_row advances 20000 → 40000 → 54255
```
When `last_committed_row` = 54,255, further incremental runs ingest 0 rows (idempotent no-op).

## 5. Database reset
```powershell
dc run --rm pipeline python -m pipeline.cli db-reset --yes      # drops staging/curated/ml/audit and recreates them empty
# or remove the whole warehouse volume:
dc down; docker volume rm <project>_warehouse-data; dc up -d
```
Then trigger a full run to repopulate.

## 6. Rollback and cleanup

| Situation | Action |
|---|---|
| A bad run was published | Run folders are never deleted. Re-publish the previous good run: `dc run --rm pipeline python -m pipeline.cli run-stage publish_outputs --run-id <previous_run_id>`. This restores `data/outputs/latest/` and syncs `ml.predictions` to that run's predictions (idempotent upsert + scoped delete) |
| Reset local layers (keep source and outputs) | `dc run --rm pipeline python -m pipeline.cli reset-local --yes` |
| Also delete run outputs | add `--include-outputs` |
| Remove quarantined batches after inspection | delete `data/quarantine/<run_id>/` |
| Remove failure-injection copies | `data/tmp/broken_*.csv` (the script deletes them automatically) |
| Remove benchmark tables | `DROP SCHEMA bench CASCADE;` |
| Stop everything | `dc down` (keeps volumes) · `dc down -v` (also deletes databases) |

## 7. Failure / recovery demonstration (for the lab)

**Automated (all four scenarios, with evidence files):**
```powershell
$env:EVIDENCE_LABEL="docker"; dc run --rm -e EVIDENCE_LABEL=docker pipeline python scripts/failure_injection.py
```
This writes `docs/evidence/failure_injection_docker.md` and `.json`.

**Through Airflow (for screenshots):**
1. Create the bad copy: `dc run --rm pipeline python -c "import sys; sys.path.insert(0,'scripts'); import failure_injection as f; print(f.write_variant('broken_missing_pollutant.csv', f.drop_column('Pollutant')))"`
2. Trigger the DAG with `source_override=data/tmp/broken_missing_pollutant.csv`. `validate_raw` fails (red), and downstream tasks show `upstream_failed`.
3. Show that nothing reached outputs: `PUBLISHED_RUN.txt` is unchanged, the batch is in `data/quarantine/<run_id>/`, and `audit.dq_results` has the failed checks.
4. Recover: trigger again with `source_override` empty. The run is all green.
5. Database variant: `dc stop warehouse`, trigger. `load_postgres` retries and fails. Then `dc start warehouse` → **Clear** `load_postgres` (downstream). The same run completes.
