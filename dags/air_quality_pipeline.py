"""Airflow DAG: PM2.5 exceedance pipeline (DSS150P reproducibility lab).

The DAG only ORCHESTRATES. Every task runs one stage of the repository's
pipeline package (`python -m pipeline.cli run-stage <stage>`), executed in a
dedicated virtualenv (/opt/pipeline-venv) whose pinned dependencies are
isolated from Airflow's own. No transformation/model logic lives in this file.

Schedule: None (manual trigger). Rationale: the source is a static, checksum-
pinned snapshot — no new data arrives on a calendar, so a schedule would only
recompute identical results. Runs are triggered manually (UI / CLI / API),
e.g. after changing code or config, for an incremental batch, or for the
failure/recovery demonstration. catchup=False so enabling the DAG never
back-fills historical intervals; max_active_runs=1 because runs share the local
data layers and the ingestion checkpoint.

Run identifier: Airflow's `run_id` is passed to every task and becomes the
pipeline_run_id (sanitised for file names), so logs, files under
data/outputs/runs/<run_id>/ and audit.* rows trace back to one DAG run.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import timedelta

import pendulum

try:  # Airflow 3.x
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.sdk import DAG, Param
except ImportError:  # Airflow 2.x (the version pinned in docker/airflow/Dockerfile)
    from airflow import DAG
    from airflow.models.param import Param
    from airflow.operators.bash import BashOperator

PROJECT_HOME = "/opt/project"
PIPELINE_PYTHON = "/opt/pipeline-venv/bin/python"
log = logging.getLogger(__name__)

STAGE_COMMAND = (
    f"cd {PROJECT_HOME} && {PIPELINE_PYTHON} -m pipeline.cli run-stage {{stage}} "
    '--run-id "{{{{ run_id }}}}" '
    "--mode {{{{ params.mode }}}} "
    "--seed {{{{ params.model_seed }}}} "
    "--batch-rows {{{{ params.batch_rows }}}}"
    '{{{{ (" --source " ~ params.source_override) if params.source_override else "" }}}}'
)


def _failure_callback(context) -> None:
    """Log useful context and mirror the failure into audit.* (covers timeouts/
    SIGTERM, where the stage process itself could not record its failure)."""
    ti = context["task_instance"]
    exc = context.get("exception")
    log.error(
        "TASK FAILED dag=%s task=%s run_id=%s try=%s/%s params=%s exception=%r log_url=%s",
        ti.dag_id, ti.task_id, context["run_id"], ti.try_number, (ti.max_tries or 0) + 1,
        dict(context.get("params", {})), exc, getattr(ti, "log_url", None),
    )
    try:
        subprocess.run(
            [PIPELINE_PYTHON, "-m", "pipeline.cli", "sync-audit", "--run-id", context["run_id"],
             "--mark-failed", ti.task_id, "--error", f"{type(exc).__name__ if exc else 'Failure'}: {exc}"[:500]],
            cwd=PROJECT_HOME, timeout=60, check=False,
        )
    except Exception as cb_exc:  # noqa: BLE001 — a callback must never raise
        log.warning("audit sync from failure callback skipped: %s", cb_exc)


def _retry_callback(context) -> None:
    ti = context["task_instance"]
    log.warning("RETRYING task=%s run_id=%s after try %s: %r", ti.task_id, context["run_id"], ti.try_number,
                context.get("exception"))


default_args = {
    "owner": "bienn.perez",
    "depends_on_past": False,
    "retries": 0,                                   # default: deterministic steps are NOT retried
    "execution_timeout": timedelta(minutes=30),
    "on_failure_callback": _failure_callback,
    "on_retry_callback": _retry_callback,
}

# Retry policy per stage. Only steps that are idempotent AND can fail for
# transient reasons (I/O, database availability) are retried.
RETRY_POLICY = {
    "ingest_source":   {"retries": 2, "retry_delay": timedelta(seconds=20)},   # deterministic batch id -> overwrite
    "load_postgres":   {"retries": 3, "retry_delay": timedelta(seconds=30),    # upsert on business keys
                        "retry_exponential_backoff": True, "max_retry_delay": timedelta(minutes=5)},
    "publish_outputs": {"retries": 2, "retry_delay": timedelta(seconds=30)},   # upserts + atomic folder swap
}
TIMEOUTS = {  # failure control for the expensive steps
    "train_model": timedelta(minutes=15),
    "evaluate_model": timedelta(minutes=15),
    "load_postgres": timedelta(minutes=10),
}
STAGES = [
    "ingest_source", "validate_raw", "transform_staging", "build_curated", "load_postgres",
    "validate_curated", "train_model", "evaluate_model", "reproducibility_check", "publish_outputs",
]
DOCS = {
    "ingest_source": "Verify source identity and land a raw batch (full or incremental) in raw/_pending.",
    "validate_raw": "Run raw data-contract checks; commit batch + advance checkpoint, or quarantine it.",
    "transform_staging": "Clean/type/de-duplicate readings (notebook rules, notebook order).",
    "build_curated": "Pivot to station-day data product + model-ready features; curated DQ gate.",
    "load_postgres": "Rerun-safe upsert of staging + curated into PostgreSQL.",
    "validate_curated": "Model-feature contract checks + PostgreSQL/file parity + duplicate-key check.",
    "train_model": "Deterministic stratified split, leakage/overlap checks, fit 3 classifiers.",
    "evaluate_model": "Held-out metrics, 5-fold CV, predictions.csv, metrics.json.",
    "reproducibility_check": "Compare with expected/baseline_metrics.json under the tolerance rule.",
    "publish_outputs": "Load predictions/metrics to PostgreSQL; promote outputs to data/outputs/latest.",
}

with DAG(
    dag_id="air_quality_pm25_pipeline",
    description="Reproducible PM2.5 exceedance pipeline: raw -> staging -> curated -> PostgreSQL -> model -> outputs",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    dagrun_timeout=timedelta(hours=2),
    tags=["dss150p", "reproducibility", "air-quality"],
    doc_md=__doc__,
    params={
        "mode": Param("full", type="string", enum=["full", "incremental"],
                      description="full = rebuild raw from the whole file; incremental = next batch after the checkpoint"),
        "model_seed": Param(42, type="integer", minimum=0,
                            description="Seed for split/CV/models. 42 reproduces the baseline; other seeds are sensitivity runs"),
        "batch_rows": Param(20000, type="integer", minimum=1, description="Source rows per incremental batch"),
        "source_override": Param("", type="string",
                                 description="Optional repo-relative path to an alternative source file "
                                             "(failure-injection demo only, e.g. data/tmp/broken_missing_pollutant.csv)"),
    },
) as dag:
    tasks = []
    for stage in STAGES:
        kwargs = dict(RETRY_POLICY.get(stage, {}))
        if stage in TIMEOUTS:
            kwargs["execution_timeout"] = TIMEOUTS[stage]
        tasks.append(BashOperator(
            task_id=stage,
            bash_command=STAGE_COMMAND.format(stage=stage),
            env={"PYTHONPATH": f"{PROJECT_HOME}/src", "PIPELINE_HOME": PROJECT_HOME},
            append_env=True,
            doc_md=DOCS[stage],
            **kwargs,
        ))
    # Explicit linear dependency chain, exactly as specified by the lab.
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
