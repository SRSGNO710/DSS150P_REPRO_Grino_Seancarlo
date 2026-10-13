"""Storage experiment: CSV vs JSON (lines) vs Parquet vs PostgreSQL on the SAME logical records.

For each format and dataset (staging readings, curated station-days) we measure:
write time, full-read time, on-disk size, and one analytical query
(mean PM2.5 per country, Japan only), each as the median of N repeats. After
every read-back the content fingerprint is compared with the in-memory frame, so
a format only "wins" if it returns exactly the same logical records (A10).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import db as dbm
from .config import Config
from .layer_io import content_fingerprint

DATASETS = {
    "staging_readings": {"key": ["reading_key"], "fp_cols": ["reading_key", "country_label", "city", "location",
                                                             "pollutant", "value", "last_updated_utc", "reading_date"]},
    "curated_station_day": {"key": ["station_day_key"], "fp_cols": ["station_day_key", "country_label", "city",
                                                                     "location", "reading_date", "co", "no2", "o3",
                                                                     "pm10", "pm25", "so2", "n_readings"]},
}


def _median_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return round(statistics.median(times), 4), result


def _restore(df: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame read from a text format back to the reference dtypes."""
    out = df.copy()
    for col in like.columns:
        if col not in out.columns:
            continue
        ref = like[col]
        if isinstance(ref.dtype, pd.DatetimeTZDtype):
            out[col] = pd.to_datetime(out[col], utc=True, format="ISO8601")
        elif ref.dtype == object and len(ref.dropna()) and hasattr(ref.dropna().iloc[0], "isoformat"):
            out[col] = pd.to_datetime(out[col].astype(str).str[:10], format="%Y-%m-%d").dt.date
        elif pd.api.types.is_float_dtype(ref):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif pd.api.types.is_integer_dtype(ref):
            out[col] = pd.to_numeric(out[col]).astype("int64")
        else:
            out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out


def _max_float_error(ref: pd.DataFrame, back: pd.DataFrame, key: list[str]) -> float:
    """Largest absolute difference over all float cells after aligning rows by key."""
    a = ref.set_index(key).sort_index()
    b = back.set_index(key).reindex(a.index)
    worst = 0.0
    for col in a.columns:
        if pd.api.types.is_float_dtype(a[col]):
            diff = (a[col] - pd.to_numeric(b[col], errors="coerce")).abs().max()
            worst = max(worst, 0.0 if pd.isna(diff) else float(diff))
    return worst


def _query(df: pd.DataFrame, dataset: str) -> float:
    if dataset == "staging_readings":
        sub = df[(df["country_label"] == "Japan") & (df["pollutant"] == "PM2.5")]
        return float(sub["value"].mean())
    return float(df.loc[df["country_label"] == "Japan", "pm25"].mean())


def _file_formats(base: Path) -> dict[str, dict[str, Callable]]:
    return {
        "csv": {
            "path": base.with_suffix(".csv"),
            "write": lambda df, p: df.to_csv(p, index=False),
            "read": lambda p: pd.read_csv(p, keep_default_na=False, na_values=[""]),
        },
        "json_lines": {
            "path": base.with_suffix(".jsonl"),
            # NB: pandas' default double_precision=10 silently rounds floats (e.g. daily means);
            # 15 is the maximum and is needed for the same logical records.
            "write": lambda df, p: df.to_json(p, orient="records", lines=True, date_format="iso", force_ascii=False,
                                              double_precision=15),
            "read": lambda p: pd.read_json(p, orient="records", lines=True, convert_dates=False, dtype=False),
        },
        "parquet": {
            "path": base.with_suffix(".parquet"),
            "write": lambda df, p: df.to_parquet(p, index=False, engine="pyarrow"),
            "read": lambda p: pd.read_parquet(p, engine="pyarrow"),
        },
    }


def run_benchmark(cfg: Config, frames: dict[str, pd.DataFrame], repeats: int = 3, use_db: bool = True,
                  log=None) -> dict[str, Any]:
    out_dir = cfg.path("benchmark_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for dataset, df in frames.items():
        spec = DATASETS[dataset]
        df = df[[c for c in df.columns if not c.startswith("_")]].reset_index(drop=True)
        ref_fp = content_fingerprint(df, spec["fp_cols"], spec["key"])
        ref_query = _query(df, dataset)
        for fmt, f in _file_formats(out_dir / dataset).items():
            row: dict[str, Any] = {"dataset": dataset, "format": fmt, "rows": len(df)}
            try:
                path = f["path"]
                row["write_s"], _ = _median_time(lambda: f["write"](df, path), repeats)
                row["read_s"], back = _median_time(lambda: f["read"](path), repeats)
                back = _restore(back, df)
                row["size_bytes"] = path.stat().st_size
                # query = load the file + filter + aggregate (what an analyst would do)
                row["query_s"], q = _median_time(lambda: _query(f["read"](path), dataset), repeats)
                row["rows_read_back"] = len(back)
                row["same_logical_records"] = content_fingerprint(back, spec["fp_cols"], spec["key"]) == ref_fp
                row["max_float_abs_error"] = _max_float_error(df, back, spec["key"])
                row["query_result_matches"] = abs(q - ref_query) < 1e-9
                row["status"] = "ok"
            except ImportError as exc:
                row.update({"status": "skipped", "reason": f"{exc}"})
            results.append(row)
            if log:
                log.info("benchmark %s/%s", dataset, fmt, extra={"event": "benchmark", **row})
        if use_db:
            results.append(_bench_postgres(cfg, dataset, df, spec, ref_fp, ref_query, repeats, log))
    return {"repeats": repeats, "results": results}


def _bench_postgres(cfg, dataset, df, spec, ref_fp, ref_query, repeats, log) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset, "format": "postgresql", "rows": len(df)}
    table = f"bench.{dataset}"
    cols = list(df.columns)
    try:
        conn = dbm.connect(cfg)
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "skipped", "reason": str(exc)})
        return row
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS bench")
            src = "staging.air_quality_readings" if dataset == "staging_readings" else "curated.station_day_pollutants"
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(f"CREATE TABLE {table} AS SELECT {', '.join(cols)} FROM {src} WITH NO DATA")
        conn.commit()

        def write():
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {table}")
                cur.copy_expert(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
                                dbm._to_copy_csv(df[cols]))
            conn.commit()

        dtypes = {c: ("float" if pd.api.types.is_float_dtype(df[c]) else "int" if pd.api.types.is_integer_dtype(df[c])
                      else "str") for c in cols}
        for c in cols:
            if df[c].dtype == object and len(df[c].dropna()) and hasattr(df[c].dropna().iloc[0], "isoformat"):
                dtypes[c] = "date"
        tz_cols = [c for c in cols if isinstance(df[c].dtype, pd.DatetimeTZDtype)]
        for c in tz_cols:
            dtypes.pop(c)
        select = ", ".join(f"to_char({c} AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS {c}" if c in tz_cols
                           else c for c in cols)

        def read():
            back = dbm.read_frame(conn, f"SELECT {select} FROM {table}", dtypes)
            for c in tz_cols:
                back[c] = pd.to_datetime(back[c], utc=True, format="ISO8601")
            return back

        if dataset == "staging_readings":
            q = f"SELECT avg(value) FROM {table} WHERE country_label = 'Japan' AND pollutant = 'PM2.5'"
        else:
            q = f"SELECT avg(pm25) FROM {table} WHERE country_label = 'Japan'"
        row["write_s"], _ = _median_time(write, repeats)
        row["read_s"], back = _median_time(read, repeats)
        row["size_bytes"] = int(dbm.scalar(conn, "SELECT pg_total_relation_size(%s)", (table,)))
        row["query_s"], qv = _median_time(lambda: float(dbm.scalar(conn, q)), repeats)
        row["rows_read_back"] = len(back)
        row["same_logical_records"] = content_fingerprint(back, spec["fp_cols"], spec["key"]) == ref_fp
        row["max_float_abs_error"] = _max_float_error(df, back, spec["key"])
        row["query_result_matches"] = abs(qv - ref_query) < 1e-6
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()
    if log:
        log.info("benchmark %s/postgresql", dataset, extra={"event": "benchmark", **row})
    return row


def render_markdown(report: dict[str, Any], environment: dict[str, str]) -> str:
    lines = ["| Dataset | Format | Rows | Write (s) | Read (s) | Query (s) | Size (KB) | Same records | Max float err | Status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        size = "" if "size_bytes" not in r else f"{r['size_bytes'] / 1024:,.0f}"
        lines.append(f"| {r['dataset']} | {r['format']} | {r['rows']:,} | {r.get('write_s', '')} | {r.get('read_s', '')} | "
                     f"{r.get('query_s', '')} | {size} | {r.get('same_logical_records', '')} | "
                     f"{'' if 'max_float_abs_error' not in r else format(r['max_float_abs_error'], '.1e')} | {r['status']}"
                     f"{' (' + r['reason'][:60] + ')' if r.get('reason') else ''} |")
    env = ", ".join(f"{k} {v}" for k, v in environment.items())
    return "\n".join(lines) + f"\n\n_Median of {report['repeats']} repeats. Environment: {env}._\n"


def save_report(cfg: Config, report: dict[str, Any], environment: dict[str, str]) -> Path:
    out = cfg.path("benchmark_dir") / "storage_benchmark.json"
    out.write_text(json.dumps({"environment": environment, **report}, indent=2, default=str), encoding="utf-8")
    (out.with_suffix(".md")).write_text(render_markdown(report, environment), encoding="utf-8")
    return out
