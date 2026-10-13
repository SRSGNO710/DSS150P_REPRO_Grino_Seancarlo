"""The ten pipeline stages. Airflow tasks and the local `run-all` command both call
exactly these functions (via src/pipeline/cli.py), so there is one implementation.

Every stage takes (cfg, run_id, log, **params) and returns a StageResult with
row counts / details / fingerprints, which the CLI merges into the run record.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import db as dbm
from . import ingest as ing
from .config import Config, load_yaml
from .errors import DataQualityError, ReproducibilityError
from .features import build_model_features
from .layer_io import content_fingerprint, read_table, write_table
from .manifest import load_manifest
from .model import design_matrix, evaluate, fit_models, stratified_split
from .quality import enforce, load_contract, run_checks, split_integrity_checks, write_report
from .reproducibility import compare
from .run_record import RunRecord
from .transform import CURATED_INDEX, CURATED_POLLUTANTS, clean_readings, pivot_station_day

STAGES = [
    "ingest_source", "validate_raw", "transform_staging", "build_curated", "load_postgres",
    "validate_curated", "train_model", "evaluate_model", "reproducibility_check", "publish_outputs",
]
DB_STAGES = {"load_postgres"}

STAGING_FP_COLS = ["reading_key", "country_label", "city", "location", "pollutant", "unit", "value",
                   "last_updated_utc", "reading_date"]
CURATED_FP_COLS = ["station_day_key", *CURATED_INDEX, *CURATED_POLLUTANTS, "n_readings"]


@dataclass
class StageResult:
    counts: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ paths --
def staging_base(cfg: Config) -> Path:
    return cfg.path("staging_dir") / "air_quality_readings"


def curated_base(cfg: Config) -> Path:
    return cfg.path("curated_dir") / "station_day_pollutants"


def features_base(cfg: Config) -> Path:
    return cfg.path("features_dir") / "model_features"


def source_complete(cfg: Config) -> bool:
    ck = ing.load_checkpoint(cfg)
    manifest = load_manifest(cfg)
    return bool(ck) and ck["committed_rows"] == manifest["file"]["data_rows"] \
        and ck["source_sha256"] == manifest["file"]["sha256"]


def _dq(cfg: Config, run_id: str, results: list[dict], layer: str, log) -> None:
    write_report(results, cfg.run_dir(run_id) / "dq_report.json")
    enforce(results, layer, log)


# ---------------------------------------------------------- 1 ingest_source --
def ingest_source(cfg: Config, run_id: str, log, mode: str = "full", source_path: Path | None = None,
                  batch_rows: int | None = None, **_) -> StageResult:
    summary = ing.ingest(cfg, run_id, mode, source_path=source_path, batch_rows=batch_rows, log=log)
    log.info("ingested %s new raw rows (mode=%s, batch=%s)", summary["new_rows"], mode, summary["batch_id"],
             extra={"event": "ingest_summary", "rows_out": summary["new_rows"], **summary})
    return StageResult(counts={"source": summary["source_rows"], "raw_new": summary["new_rows"]}, details=summary)


# ----------------------------------------------------------- 2 validate_raw --
def validate_raw(cfg: Config, run_id: str, log, **_) -> StageResult:
    contract, manifest = load_contract(cfg), load_manifest(cfg)
    pending_meta = ing.pending_batches(cfg)
    pending = ing.read_pending(cfg)
    full = any(m["mode"] == "full" for m in pending_meta)
    ck = ing.load_checkpoint(cfg)
    committed = ing.read_committed_raw(cfg) if (ck and ck.get("batches") and not full) else pd.DataFrame()
    union = pd.concat([committed, pending], ignore_index=True) if len(pending) else committed
    if union.empty:
        raise DataQualityError("raw", [{"check_id": "raw_has_rows", "message": "no raw rows to validate"}])
    complete = len(union) == manifest["file"]["data_rows"]
    ctx = {"manifest": manifest, "source_complete": complete}
    results = run_checks(union, "raw", contract, ctx)
    try:
        _dq(cfg, run_id, results, "raw", log)
    except DataQualityError as exc:
        moved = ing.quarantine_pending(cfg, run_id, reason=str(exc))
        log.error("raw batch quarantined; checkpoint NOT advanced", extra={"event": "batch_quarantined",
                                                                          "files": moved, "error": str(exc)})
        raise
    commit = ing.commit_pending(cfg, run_id)
    ck = commit["checkpoint"]
    log.info("raw validated and committed", extra={"event": "raw_committed", "batches": commit["committed_batches"],
                                                   "last_committed_row": ck["last_committed_row"]})
    return StageResult(counts={"raw": ck["committed_rows"]},
                       details={"committed_batches": commit["committed_batches"],
                                "last_committed_row": ck["last_committed_row"], "source_complete": complete,
                                "checks_passed": sum(r["status"] == "passed" for r in results),
                                "checks_total": len(results)})


# ------------------------------------------------------ 3 transform_staging --
def transform_staging(cfg: Config, run_id: str, log, **_) -> StageResult:
    raw = ing.read_committed_raw(cfg)
    staging, steps = clean_readings(raw, cfg["transform"])
    results = run_checks(staging, "staging", load_contract(cfg), {"source_complete": source_complete(cfg)})
    _dq(cfg, run_id, results, "staging", log)
    staging["_pipeline_run_id"] = run_id
    write_table(staging, staging_base(cfg), cfg.layer_format)
    fp = content_fingerprint(staging, STAGING_FP_COLS, ["reading_key"])
    log.info("staging written", extra={"event": "layer_written", "rows_in": len(raw), "rows_out": len(staging),
                                       "steps": steps})
    return StageResult(counts={"staging": len(staging)}, details={"row_counts_by_step": steps},
                       fingerprints={"staging": fp})


# ---------------------------------------------------------- 4 build_curated --
def build_curated(cfg: Config, run_id: str, log, **_) -> StageResult:
    staging = read_table(staging_base(cfg), cfg.layer_format)
    curated = pivot_station_day(staging)
    contract = load_contract(cfg)
    results = run_checks(curated, "curated", contract, {"source_complete": source_complete(cfg)})
    _dq(cfg, run_id, results, "curated", log)  # gate BEFORE anything reaches PostgreSQL
    curated["_pipeline_run_id"] = run_id
    write_table(curated, curated_base(cfg), cfg.layer_format)
    mf, info = build_model_features(curated, cfg["model"])
    mf["_pipeline_run_id"] = run_id
    write_table(mf, features_base(cfg), cfg.layer_format)
    fps = {
        "curated": content_fingerprint(curated, CURATED_FP_COLS, ["station_day_key"]),
        "model_features": content_fingerprint(mf, ["station_day_key", *cfg["model"]["feature_columns"], "PM2.5",
                                                   "Air_Quality"], ["station_day_key"]),
    }
    log.info("curated + model features written", extra={"event": "layer_written", "rows_in": len(staging),
                                                        "rows_out": len(curated), "model_rows": len(mf)})
    return StageResult(counts={"curated": len(curated), "model_features": len(mf)}, details=info, fingerprints=fps)


# ---------------------------------------------------------- 5 load_postgres --
def _staging_for_db(staging: pd.DataFrame) -> pd.DataFrame:
    return staging.rename(columns={"_source_row_number": "source_row_number", "_ingest_batch_id": "ingest_batch_id"})


def load_postgres(cfg: Config, run_id: str, log, **_) -> StageResult:
    staging = _staging_for_db(read_table(staging_base(cfg), cfg.layer_format))
    curated = read_table(curated_base(cfg), cfg.layer_format)
    conn = dbm.connect(cfg)
    try:
        dbm.apply_ddl(conn, cfg.root)
        stats = {
            "staging.air_quality_readings": dbm.load_frame(conn, staging, "staging.air_quality_readings", run_id),
            "curated.station_day_pollutants": dbm.load_frame(conn, curated, "curated.station_day_pollutants", run_id),
        }
        for table, st in stats.items():
            log.info("loaded %s", table, extra={"event": "table_loaded", "table": table, **st})
        dups = dbm.duplicate_key_report(conn)
    finally:
        conn.close()
    return StageResult(counts={"postgres_staging": stats["staging.air_quality_readings"]["rows_in_table"],
                               "postgres_curated": stats["curated.station_day_pollutants"]["rows_in_table"]},
                       details={"load_stats": stats, "duplicate_business_keys": dups})


# ------------------------------------------------------- 6 validate_curated --
CURATED_DB_DTYPES = {"station_day_key": "str", "country_label": "str", "city": "str", "location": "str",
                     "reading_date": "date", "co": "float", "no2": "float", "o3": "float", "pm10": "float",
                     "pm25": "float", "so2": "float", "n_readings": "int"}


def validate_curated(cfg: Config, run_id: str, log, use_db: bool = True, **_) -> StageResult:
    contract = load_contract(cfg)
    complete = source_complete(cfg)
    mf = read_table(features_base(cfg), cfg.layer_format)
    curated = read_table(curated_base(cfg), cfg.layer_format)
    results = run_checks(mf, "model_features", contract,
                         {"source_complete": complete, "feature_columns": cfg["model"]["feature_columns"]})
    details: dict[str, Any] = {}
    if use_db:
        conn = dbm.connect(cfg)
        try:
            db_cur = dbm.read_frame(conn, f"SELECT {', '.join(CURATED_DB_DTYPES)} FROM curated.station_day_pollutants",
                                    CURATED_DB_DTYPES)
            dups = dbm.duplicate_key_report(conn)
        finally:
            conn.close()
        results += run_checks(db_cur, "curated", contract, {"source_complete": complete}, report_layer="curated@postgres")
        file_fp = content_fingerprint(curated, CURATED_FP_COLS, ["station_day_key"])
        db_fp = content_fingerprint(db_cur, CURATED_FP_COLS, ["station_day_key"])
        dup_total = sum(d["duplicate_business_keys"] for d in dups)
        results += [
            {"check_id": "db_file_parity_curated", "layer": "curated@postgres", "category": "consistency",
             "type": "custom", "severity": "error", "status": "passed" if file_fp == db_fp else "failed",
             "observed": {"file_rows": len(curated), "db_rows": len(db_cur), "file_fp": file_fp[:16], "db_fp": db_fp[:16]},
             "threshold": "identical content", "message": "Curated file and PostgreSQL table hold the same logical records",
             "examples": []},
            {"check_id": "db_no_duplicate_business_keys", "layer": "curated@postgres", "category": "uniqueness",
             "type": "custom", "severity": "error", "status": "passed" if dup_total == 0 else "failed",
             "observed": dups, "threshold": 0, "message": "", "examples": []},
        ]
        details["duplicate_business_keys"] = dups
        details["db_rows"] = len(db_cur)
    _dq(cfg, run_id, results, "model_features", log)
    details["checks"] = {r["check_id"]: r["status"] for r in results}
    return StageResult(counts={"model_features": len(mf)}, details=details)


# ------------------------------------------------------------ 7 train_model --
def train_model(cfg: Config, run_id: str, log, seed: int | None = None, **_) -> StageResult:
    mcfg = dict(cfg["model"])
    seed = mcfg["seed"] if seed is None else seed
    mf = read_table(features_base(cfg), cfg.layer_format)
    X = design_matrix(mf, mcfg)
    idx_train, idx_test = stratified_split(mf, mcfg, seed)
    checks = split_integrity_checks(mf.loc[idx_train, "station_day_key"], mf.loc[idx_test, "station_day_key"],
                                    len(mcfg["feature_columns"]), list(X.loc[idx_train].columns),
                                    list(X.loc[idx_test].columns))
    _dq(cfg, run_id, checks, "model_split", log)
    models = fit_models(X.loc[idx_train], mf.loc[idx_train, "Air_Quality"], mcfg, seed)
    run_dir = cfg.run_dir(run_id)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, run_dir / "models" / f"{name}.joblib")
    split = pd.DataFrame({"station_day_key": mf["station_day_key"],
                          "split": mf.index.isin(idx_test)})
    split["split"] = split["split"].map({True: "test", False: "train"})
    split.to_csv(run_dir / "split.csv", index=False)
    (run_dir / "train_config.json").write_text(json.dumps({"seed": seed, **{k: mcfg[k] for k in (
        "test_size", "cv_folds", "who_threshold", "imputation_fit_scope", "feature_columns", "models")}}, indent=2))
    log.info("models trained", extra={"event": "models_trained", "models": list(models), "seed": seed,
                                      "rows_in": len(mf), "train_rows": len(idx_train), "test_rows": len(idx_test)})
    return StageResult(counts={"train": len(idx_train), "test": len(idx_test)},
                       details={"seed": seed, "models": list(models)})


# --------------------------------------------------------- 8 evaluate_model --
def evaluate_model(cfg: Config, run_id: str, log, seed: int | None = None, **_) -> StageResult:
    mcfg = dict(cfg["model"])
    seed = mcfg["seed"] if seed is None else seed
    run_dir = cfg.run_dir(run_id)
    mf = read_table(features_base(cfg), cfg.layer_format)
    split = pd.read_csv(run_dir / "split.csv")
    if not (split["station_day_key"].to_numpy() == mf["station_day_key"].to_numpy()).all():
        raise DataQualityError("model_split", [{"check_id": "split_matches_features",
                                                "message": "model_features changed after train_model; retrain"}])
    is_test = split["split"].eq("test").to_numpy()
    idx_test, idx_train = mf.index[is_test], mf.index[~is_test]
    models = {name: joblib.load(run_dir / "models" / f"{name}.joblib") for name in mcfg["models"]}
    X = design_matrix(mf, mcfg)
    metrics, preds = evaluate(models, mf, X, idx_train, idx_test, mcfg, seed)
    metrics = {"pipeline_run_id": run_id, **metrics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preds.insert(0, "pipeline_run_id", run_id)
    preds.to_csv(run_dir / "predictions.csv", index=False)
    summary = {n: {"test_accuracy": round(m["test"]["accuracy"], 6), "test_f1": round(m["test"]["f1_weighted"], 6),
                   "cv_f1": round(m["cv_f1_weighted"]["mean"], 6)} for n, m in metrics["models"].items()}
    log.info("evaluation complete", extra={"event": "metrics", "metrics": summary, "rows_out": len(preds)})
    return StageResult(counts={"predictions": len(preds)}, details={"metrics": summary,
                                                                    "best_model_by_cv_f1": metrics["best_model_by_cv_f1"]})


# --------------------------------------------------- 9 reproducibility_check --
def reproducibility_check(cfg: Config, run_id: str, log, **_) -> StageResult:
    run_dir = cfg.run_dir(run_id)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline = load_yaml(cfg.path("baseline_metrics"))  # JSON is valid YAML
    counts = RunRecord(run_dir, run_id).load()["counts"]
    comparison = compare(metrics, baseline, counts, float(cfg.get("reproducibility", "abs_tolerance")),
                         source_complete=source_complete(cfg))
    comparison["pipeline_run_id"] = run_id
    comparison["layer_fingerprints"] = RunRecord(run_dir, run_id).load()["layer_fingerprints"]
    (run_dir / "metrics_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    log.info("reproducibility: %s — %s", comparison["status"], comparison["outcome"],
             extra={"event": "reproducibility", "status": comparison["status"], "summary": comparison["summary"]})
    if comparison["status"] == "fail":
        raise ReproducibilityError(comparison["outcome"] + f" — see {run_dir.name}/metrics_comparison.json")
    return StageResult(details={"status": comparison["status"], "outcome": comparison["outcome"],
                                "max_abs_diff": comparison["summary"]["max_abs_diff"]})


# -------------------------------------------------------- 10 publish_outputs --
PUBLISHED_FILES = ["metrics.json", "predictions.csv", "metrics_comparison.json", "dq_report.json", "run_summary.json"]


def _predictions_long(preds: pd.DataFrame, models: list[str], deployed: str) -> pd.DataFrame:
    frames = []
    for name in models:
        frames.append(pd.DataFrame({
            "station_day_key": preds["station_day_key"], "model_name": name,
            "country_label": preds["country_label"], "location": preds["location"],
            "reading_date": pd.to_datetime(preds["reading_date"]).dt.date,
            "pm25_actual": preds["pm25_actual"].astype(float), "label_actual": preds["label_actual"],
            "label_predicted": preds[f"pred_{name}"], "proba_unhealthy": preds[f"proba_unhealthy_{name}"].astype(float),
            "is_deployed_model": name == deployed,
        }))
    return pd.concat(frames, ignore_index=True)


def publish_outputs(cfg: Config, run_id: str, log, use_db: bool = True, **_) -> StageResult:
    run_dir = cfg.run_dir(run_id)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads((run_dir / "metrics_comparison.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(run_dir / "predictions.csv")
    details: dict[str, Any] = {}
    if use_db:
        long = _predictions_long(preds, list(metrics["models"]), metrics["deployed_model"])
        conn = dbm.connect(cfg)
        try:
            details["predictions_load"] = dbm.load_frame(conn, long, "ml.predictions", run_id)
            details["metrics_rows"] = dbm.load_metrics(conn, run_id, comparison, metrics)
            details["duplicate_business_keys"] = dbm.duplicate_key_report(conn)
        finally:
            conn.close()
    latest = cfg.path("outputs_dir") / "latest"
    staging_dir = latest.with_name(".latest_tmp")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for name in PUBLISHED_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, staging_dir / name)
    (staging_dir / "PUBLISHED_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    if latest.exists():
        shutil.rmtree(latest)
    staging_dir.rename(latest)  # promote only after every upstream check passed
    log.info("outputs published", extra={"event": "published", "run_dir": run_dir.name, "files": PUBLISHED_FILES})
    details["published_to"] = "data/outputs/latest"
    return StageResult(details=details)


STAGE_FUNCS = {name: globals()[name] for name in STAGES}
