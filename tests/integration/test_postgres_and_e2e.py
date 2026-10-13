"""Integration tests across the database boundary and the full pipeline.

* PostgreSQL tests run when POSTGRES_PASSWORD is set and the server is reachable
  (inside `docker compose run --rm pipeline pytest` they always are); otherwise skipped.
* The end-to-end reproduction test needs the real source file in data/source/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO, postgres_available

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_conn(cfg):
    if not postgres_available():
        pytest.skip("PostgreSQL not reachable (set POSTGRES_* / run inside docker compose)")
    from pipeline import db

    conn = db.connect(cfg)
    db.apply_ddl(conn, cfg.root)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM curated.station_day_pollutants WHERE country_label LIKE 'TEST_%'")
    conn.commit()
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM curated.station_day_pollutants WHERE country_label LIKE 'TEST_%'")
        cur.execute("DELETE FROM audit.load_log WHERE pipeline_run_id LIKE 'pytest_%'")
    conn.commit()
    conn.close()


def _curated_rows(country: str, n: int, pm25: float = 10.0) -> pd.DataFrame:
    from pipeline.layer_io import sha256_text_rows

    df = pd.DataFrame({"country_label": country, "city": "c", "location": [f"loc{i}" for i in range(n)],
                       "reading_date": pd.Timestamp("2024-03-10").date(), "co": None, "no2": 1.0, "o3": 2.0,
                       "pm10": None, "pm25": pm25, "so2": None, "n_readings": 3})
    for col in ("co", "pm10", "so2"):
        df[col] = df[col].astype("float64")
    df.insert(0, "station_day_key", sha256_text_rows(df, ["country_label", "city", "location", "reading_date"]).str[:32])
    return df


@pytest.mark.postgres
def test_load_is_idempotent_and_detects_changes(pg_conn):
    from pipeline import db

    df = _curated_rows("TEST_A", 5)
    first = db.load_frame(pg_conn, df, "curated.station_day_pollutants", "pytest_1", full_sync=True,
                          scope_column="country_label", scope_value="TEST_A")
    second = db.load_frame(pg_conn, df, "curated.station_day_pollutants", "pytest_1", full_sync=True,
                           scope_column="country_label", scope_value="TEST_A")  # = Airflow retry after commit
    assert (int(first["inserted"]), int(second["inserted"]), int(second["updated"])) == (5, 0, 0)
    changed = df.copy()
    changed.loc[0, "pm25"] = 99.0
    third = db.load_frame(pg_conn, changed.iloc[:4], "curated.station_day_pollutants", "pytest_2", full_sync=True,
                          scope_column="country_label", scope_value="TEST_A")
    assert (int(third["updated"]), int(third["deleted"]), int(third["unchanged"])) == (1, 1, 3)
    n = db.scalar(pg_conn, "SELECT count(*) - count(DISTINCT (country_label, city, location, reading_date)) "
                           "FROM curated.station_day_pollutants")
    assert int(n) == 0


@pytest.mark.postgres
def test_partition_scoped_load_leaves_other_partitions(pg_conn):
    from pipeline import db

    db.load_frame(pg_conn, _curated_rows("TEST_A", 3), "curated.station_day_pollutants", "pytest_3",
                  scope_column="country_label", scope_value="TEST_A")
    db.load_frame(pg_conn, _curated_rows("TEST_B", 2), "curated.station_day_pollutants", "pytest_3",
                  scope_column="country_label", scope_value="TEST_B")
    count = lambda c: int(db.scalar(pg_conn, "SELECT count(*) FROM curated.station_day_pollutants "  # noqa: E731
                                             "WHERE country_label = %s", (c,)))
    assert (count("TEST_A"), count("TEST_B")) == (3, 2)


@pytest.mark.skipif(not (REPO / "data" / "source" / "world_air_quality.csv").exists(),
                    reason="real source file not present in data/source/")
def test_end_to_end_reproduces_baseline(tmp_path):
    """Full run on the REAL source in a scratch copy of the repo; must pass the tolerance rule."""
    import shutil

    root = tmp_path / "e2e"
    for d in ("config", "contracts", "sql", "expected", "src"):
        shutil.copytree(REPO / d, root / d)
    (root / "data" / "source").mkdir(parents=True)
    os.symlink(REPO / "data" / "source" / "world_air_quality.csv", root / "data" / "source" / "world_air_quality.csv") \
        if hasattr(os, "symlink") and os.name != "nt" else \
        shutil.copy(REPO / "data" / "source" / "world_air_quality.csv", root / "data" / "source")
    env = dict(os.environ, PIPELINE_HOME=str(root),
               PYTHONPATH=os.pathsep.join(filter(None, [str(root / "src"), os.environ.get("PYTHONPATH")])))
    env.setdefault("PIPELINE_LAYER_FORMAT", os.environ.get("TEST_LAYER_FORMAT", "parquet"))
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        env["PIPELINE_LAYER_FORMAT"] = "csv"
    proc = subprocess.run([sys.executable, "-m", "pipeline.cli", "run-all", "--no-db", "--run-id", "pytest_e2e"],
                          cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    comp = json.loads((root / "data" / "outputs" / "runs" / "pytest_e2e" / "metrics_comparison.json").read_text())
    assert comp["status"] == "pass"
    assert comp["summary"]["counts_exact"] and comp["summary"]["split_membership_exact"]
    assert comp["summary"]["max_abs_diff"] <= 0.01
