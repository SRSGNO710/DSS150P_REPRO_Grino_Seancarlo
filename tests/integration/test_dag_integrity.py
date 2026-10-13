"""DAG integrity — runs where Airflow is installed (inside the Airflow containers):

    docker compose exec airflow-scheduler python -m pytest /opt/project/tests/integration/test_dag_integrity.py -p no:cacheprovider
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow.models")

DAGS = Path(__file__).resolve().parents[2] / "dags"
EXPECTED = ["ingest_source", "validate_raw", "transform_staging", "build_curated", "load_postgres",
            "validate_curated", "train_model", "evaluate_model", "reproducibility_check", "publish_outputs"]


@pytest.fixture(scope="module")
def dag():
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    return bag.get_dag("air_quality_pm25_pipeline")


def test_linear_dependencies(dag):
    assert [t.task_id for t in dag.topological_sort()] == EXPECTED
    for up, down in zip(EXPECTED, EXPECTED[1:]):
        assert dag.get_task(up).downstream_task_ids == {down}


def test_schedule_params_retries(dag):
    assert dag.catchup is False and dag.max_active_runs == 1
    assert {"mode", "model_seed", "batch_rows", "source_override"} <= set(dag.params.keys())
    assert dag.get_task("load_postgres").retries == 3
    assert dag.get_task("validate_raw").retries == 0
    assert dag.get_task("train_model").execution_timeout.total_seconds() == 900
