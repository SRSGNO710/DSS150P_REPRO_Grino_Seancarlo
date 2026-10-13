"""PostgreSQL access: idempotent DDL, rerun-safe loads, audit synchronisation.

Rerun safety (lab questions 7/8)
--------------------------------
Each table load runs in ONE transaction:
  1. COPY the snapshot into a temp table,
  2. INSERT ... ON CONFLICT (business-key PK) DO UPDATE ... WHERE row_hash differs,
  3. (full sync) DELETE target rows whose key is absent from the snapshot,
  4. upsert one audit.load_log row keyed by (run_id, table, scope),
  5. COMMIT.
If Airflow retries after the COMMIT succeeded but before the task heard back,
the retry replays the same snapshot: step 2 finds every key present with the
same row_hash (0 inserted, 0 updated), step 3 deletes nothing, step 4 upserts
the same log row. The end state is identical — the load is idempotent. If the
failure happened before COMMIT, PostgreSQL rolled everything back atomically.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import Config
from .errors import DatabaseError
from .layer_io import sha256_text_rows

SQL_FILES = ["001_schemas.sql", "002_tables.sql", "003_views.sql"]

# table -> (primary-key columns, business columns hashed into row_hash)
TABLES: dict[str, dict[str, Any]] = {
    "staging.air_quality_readings": {
        "pk": ["reading_key"],
        "columns": ["reading_key", "country_code", "country_label", "city", "location", "latitude", "longitude",
                    "pollutant", "source_name", "unit", "value", "last_updated_utc", "reading_date",
                    "source_row_number", "ingest_batch_id"],
        "hash_exclude": ["ingest_batch_id"],
    },
    "curated.station_day_pollutants": {
        "pk": ["station_day_key"],
        "columns": ["station_day_key", "country_label", "city", "location", "reading_date",
                    "co", "no2", "o3", "pm10", "pm25", "so2", "n_readings"],
        "hash_exclude": [],
    },
    "ml.predictions": {
        "pk": ["station_day_key", "model_name"],
        "columns": ["station_day_key", "model_name", "country_label", "location", "reading_date", "pm25_actual",
                    "label_actual", "label_predicted", "proba_unhealthy", "is_deployed_model"],
        "hash_exclude": [],
    },
}


def _import_driver():
    try:
        import psycopg2  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise DatabaseError("psycopg2 is not installed (pip install -r requirements.txt)") from exc
    return psycopg2


def connect(cfg: Config):
    psycopg2 = _import_driver()
    pg = cfg["postgres"]
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise DatabaseError("POSTGRES_PASSWORD is not set (copy .env.example to .env, or export it)")
    try:
        return psycopg2.connect(host=pg["host"], port=int(pg["port"]), dbname=pg["dbname"], user=pg["user"],
                                password=password, connect_timeout=int(pg.get("connect_timeout_s", 10)),
                                application_name="airquality-pipeline")
    except psycopg2.OperationalError as exc:
        raise DatabaseError(f"Cannot connect to PostgreSQL at {pg['host']}:{pg['port']}/{pg['dbname']}: "
                            f"{str(exc).strip()}") from exc


def apply_ddl(conn, root: Path, files: Iterable[str] = SQL_FILES) -> None:
    with conn.cursor() as cur:
        for name in files:
            cur.execute((root / "sql" / name).read_text(encoding="utf-8"))
    conn.commit()


def _to_copy_csv(df: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map({True: "true", False: "false"})
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: r"\N" if pd.isna(v) else repr(float(v)))
    out.to_csv(buf, index=False, header=False, na_rep=r"\N", lineterminator="\n")
    buf.seek(0)
    return buf


def add_row_hash(df: pd.DataFrame, table: str) -> pd.DataFrame:
    spec = TABLES[table]
    hashed = [c for c in spec["columns"] if c not in spec["hash_exclude"]]
    view = df[hashed].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else repr(float(v)))
        elif isinstance(view[col].dtype, pd.DatetimeTZDtype):
            view[col] = view[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = df.copy()
    out["row_hash"] = sha256_text_rows(view, hashed)
    return out


def load_frame(conn, df: pd.DataFrame, table: str, run_id: str, *, full_sync: bool = True,
               scope_column: str | None = None, scope_value: Any = None) -> dict[str, int]:
    """Rerun-safe merge of `df` (already in table column names) into `table`.

    full_sync=True makes the table equal to the snapshot (rows missing from the
    snapshot are deleted). With scope_column/scope_value only that slice of the
    table is synchronised (used to (re)load one partition independently).
    """
    spec = TABLES[table]
    cols = spec["columns"] + ["row_hash", "pipeline_run_id"]
    frame = add_row_hash(df[spec["columns"]], table)
    frame["pipeline_run_id"] = run_id
    frame = frame[cols]
    pk = spec["pk"]
    tmp = "_load_" + table.replace(".", "_")
    col_list = ", ".join(cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk)
    pk_list = ", ".join(pk)
    join = " AND ".join(f"s.{c} = t.{c}" for c in pk)
    scope = f"t.{scope_column} = %s" if scope_column else None
    stats: dict[str, int] = {"rows_in_snapshot": len(frame)}
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE TEMP TABLE {tmp} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
            cur.copy_expert(f"COPY {tmp} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", _to_copy_csv(frame))
            cur.execute(
                f"INSERT INTO {table} AS t ({col_list}) SELECT {col_list} FROM {tmp} "
                f"ON CONFLICT ({pk_list}) DO UPDATE SET {updates}, updated_at = now() "
                f"WHERE t.row_hash IS DISTINCT FROM EXCLUDED.row_hash "
                f"RETURNING CASE WHEN xmax = 0 THEN 1 ELSE 0 END"
            )
            flags = [int(r[0]) for r in cur.fetchall()]
            stats["inserted"] = sum(flags)
            stats["updated"] = len(flags) - stats["inserted"]
            stats["deleted"] = 0
            if full_sync:
                where = f"NOT EXISTS (SELECT 1 FROM {tmp} s WHERE {join})"
                params: tuple = ()
                if scope:
                    where += f" AND {scope}"
                    params = (scope_value,)
                cur.execute(f"DELETE FROM {table} t WHERE {where} RETURNING 1", params)
                stats["deleted"] = len(cur.fetchall())
            stats["unchanged"] = stats["rows_in_snapshot"] - stats["inserted"] - stats["updated"]
            cur.execute(f"SELECT count(*) FROM {table}")
            stats["rows_in_table"] = int(cur.fetchone()[0])
            scope_label = f"{scope_column}={scope_value}" if scope_column else "full"
            cur.execute(
                "INSERT INTO audit.load_log (pipeline_run_id, table_name, scope, rows_in_snapshot, inserted, updated, "
                "deleted, unchanged, rows_in_table) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (pipeline_run_id, table_name, scope) DO UPDATE SET rows_in_snapshot = EXCLUDED.rows_in_snapshot, "
                "inserted = EXCLUDED.inserted, updated = EXCLUDED.updated, deleted = EXCLUDED.deleted, "
                "unchanged = EXCLUDED.unchanged, rows_in_table = EXCLUDED.rows_in_table, loaded_at = now()",
                (run_id, table, scope_label, stats["rows_in_snapshot"], stats["inserted"], stats["updated"],
                 stats["deleted"], stats["unchanged"], stats["rows_in_table"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return stats


def read_frame(conn, sql: str, dtypes: dict[str, str] | None = None, params: tuple = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        names = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=names)
    for col, kind in (dtypes or {}).items():
        if kind == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif kind == "int":
            df[col] = pd.to_numeric(df[col]).astype("int64")
        elif kind == "date":
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y-%m-%d").dt.date
        elif kind == "str":
            df[col] = df[col].astype(object).map(lambda v: None if v is None else str(v))
    return df


def scalar(conn, sql: str, params: tuple = ()) -> Any:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return None if row is None else row[0]


def duplicate_key_report(conn) -> list[dict[str, Any]]:
    df = read_frame(conn, "SELECT table_name, row_count, duplicate_business_keys FROM audit.v_duplicate_business_keys",
                    {"row_count": "int", "duplicate_business_keys": "int"})
    return df.to_dict(orient="records")


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str, ensure_ascii=False)


def sync_audit(conn, run_doc: dict[str, Any], dq_results: list[dict[str, Any]] | None = None) -> None:
    """Mirror the run record (and DQ results) into audit.* — idempotent upserts."""
    rid = run_doc["pipeline_run_id"]
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit.pipeline_runs (pipeline_run_id, status, started_at, ended_at, parameters, counts, "
                "layer_fingerprints, error) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb) "
                "ON CONFLICT (pipeline_run_id) DO UPDATE SET status = EXCLUDED.status, started_at = EXCLUDED.started_at, "
                "ended_at = EXCLUDED.ended_at, parameters = EXCLUDED.parameters, counts = EXCLUDED.counts, "
                "layer_fingerprints = EXCLUDED.layer_fingerprints, error = EXCLUDED.error, updated_at = now()",
                (rid, run_doc["status"], run_doc.get("started_at"), run_doc.get("ended_at"),
                 _json(run_doc.get("parameters")), _json(run_doc.get("counts")),
                 _json(run_doc.get("layer_fingerprints")), _json(run_doc.get("error"))),
            )
            for stage, st in run_doc.get("stages", {}).items():
                cur.execute(
                    "INSERT INTO audit.task_runs (pipeline_run_id, stage, status, started_at, ended_at, duration_s, "
                    "counts, error) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s) "
                    "ON CONFLICT (pipeline_run_id, stage) DO UPDATE SET status = EXCLUDED.status, "
                    "started_at = EXCLUDED.started_at, ended_at = EXCLUDED.ended_at, duration_s = EXCLUDED.duration_s, "
                    "counts = EXCLUDED.counts, error = EXCLUDED.error, updated_at = now()",
                    (rid, stage, st.get("status"), st.get("started_at"), st.get("ended_at"), st.get("duration_s"),
                     _json(st.get("counts")), st.get("error")),
                )
            for r in dq_results or []:
                cur.execute(
                    "INSERT INTO audit.dq_results (pipeline_run_id, layer, check_id, category, severity, status, "
                    "observed, threshold, message) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                    "ON CONFLICT (pipeline_run_id, layer, check_id) DO UPDATE SET status = EXCLUDED.status, "
                    "observed = EXCLUDED.observed, threshold = EXCLUDED.threshold, message = EXCLUDED.message, "
                    "recorded_at = now()",
                    (rid, r["layer"], r["check_id"], r.get("category"), r["severity"], r["status"],
                     _json(r.get("observed")), _json(r.get("threshold")), r.get("message")),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def load_metrics(conn, run_id: str, comparison: dict[str, Any], metrics: dict[str, Any]) -> int:
    rows = []
    by_key = {(r["model"], r["metric"]): r for r in comparison.get("metric_rows", [])}
    for model, m in metrics["models"].items():
        for section, name in (("test", "accuracy"), ("test", "precision_weighted"), ("test", "recall_weighted"),
                              ("test", "f1_weighted"), ("cv_f1_weighted", "mean"), ("cv_f1_weighted", "std")):
            metric = f"{section}.{name}"
            ref = by_key.get((model, metric), {})
            rows.append((run_id, model, metric, float(m[section][name]), ref.get("baseline"), ref.get("abs_diff"),
                         ref.get("within_tolerance")))
    try:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    "INSERT INTO ml.model_metrics (pipeline_run_id, model_name, metric_name, metric_value, baseline_value, "
                    "abs_diff, within_tolerance) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (pipeline_run_id, model_name, metric_name) DO UPDATE SET metric_value = EXCLUDED.metric_value, "
                    "baseline_value = EXCLUDED.baseline_value, abs_diff = EXCLUDED.abs_diff, "
                    "within_tolerance = EXCLUDED.within_tolerance, recorded_at = now()",
                    row,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)
