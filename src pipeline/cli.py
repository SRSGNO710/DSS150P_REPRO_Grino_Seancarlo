"""Command-line entry point: `python -m pipeline.cli <command> ...`

Airflow tasks call `run-stage <stage>`; `run-all` runs the same ten stages in
order for a local (non-Airflow) execution. See README §Running.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .errors import PipelineError
from .logging_utils import get_logger, log_stage
from .run_record import RunRecord
from .stages import DB_STAGES, STAGE_FUNCS, STAGES

EXIT_PIPELINE_ERROR = 2


def sanitize_run_id(raw: str) -> str:
    """Airflow run ids contain ':' and '+', which are invalid in Windows file names
    (the data folder is bind-mounted from the host). Map them deterministically."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def default_run_id(prefix: str = "local") -> str:
    return f"{prefix}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _best_effort_audit(cfg: Config, run_id: str, log) -> None:
    """Mirror the run record to PostgreSQL; never masks the original error."""
    try:
        from . import db as dbm

        rec = RunRecord(cfg.run_dir(run_id), run_id).load()
        report = cfg.run_dir(run_id) / "dq_report.json"
        dq = json.loads(report.read_text(encoding="utf-8")) if report.exists() else []
        conn = dbm.connect(cfg)
        try:
            dbm.apply_ddl(conn, cfg.root)
            dbm.sync_audit(conn, rec, dq)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("audit sync to PostgreSQL skipped: %s", exc, extra={"event": "audit_sync_skipped"})


def execute_stage(cfg: Config, name: str, run_id: str, *, use_db: bool = True, **params: Any):
    log = get_logger(run_id, name, cfg.path("logs_dir"))
    rec = RunRecord(cfg.run_dir(run_id), run_id)
    rec.stage_started(name)
    if name == "ingest_source":
        from .model import library_versions

        rec.set_parameters(mode=params.get("mode"), seed=params.get("seed") or cfg["model"]["seed"],
                           layer_format=cfg.layer_format, imputation_fit_scope=cfg["model"]["imputation_fit_scope"],
                           source_override=str(params["source_path"].name) if params.get("source_path") else None,
                           use_db=use_db, library_versions=library_versions())
    t0 = time.perf_counter()
    try:
        with log_stage(log) as info:
            result = STAGE_FUNCS[name](cfg, run_id, log, use_db=use_db, **params)
            info.update({"counts": result.counts})
    except Exception as exc:
        rec.stage_failed(name, f"{type(exc).__name__}: {exc}", round(time.perf_counter() - t0, 3))
        if use_db:
            _best_effort_audit(cfg, run_id, log)
        raise
    rec.stage_finished(name, counts=result.counts, details=result.details, fingerprints=result.fingerprints,
                       duration_s=round(time.perf_counter() - t0, 3))
    stages_now = rec.load()["stages"]
    if name != "publish_outputs" and stages_now.get("publish_outputs", {}).get("status") == "success" \
            and all(st.get("status") == "success" for st in stages_now.values()):
        rec.finalize("success")  # a re-executed stage (e.g. Airflow retry/clear) of an already-published run
    if name == "publish_outputs":
        doc = rec.finalize("success")
        latest = cfg.path("outputs_dir") / "latest"
        if latest.exists():
            shutil.copy2(rec.path, latest / "run_summary.json")
        log.info("pipeline run succeeded", extra={"event": "run_succeeded", "counts": doc["counts"]})
    if use_db and (name in DB_STAGES or name == "publish_outputs"):
        _best_effort_audit(cfg, run_id, log)
    return result


def _cfg_from_args(args) -> Config:
    cfg = load_config(getattr(args, "config", None))
    return cfg.with_overrides(**{
        "model.seed": getattr(args, "seed", None),
        "model.imputation_fit_scope": getattr(args, "imputation_scope", None),
        "storage.layer_format": getattr(args, "layer_format", None),
    })


def _stage_params(args, cfg: Config) -> dict[str, Any]:
    return {
        "mode": args.mode or cfg.get("ingestion", "default_mode"),
        "source_path": cfg.resolve(args.source) if args.source else None,
        "batch_rows": args.batch_rows,
        "seed": args.seed,
    }


# ----------------------------------------------------------------- commands --
def cmd_run_stage(args) -> int:
    cfg = _cfg_from_args(args)
    run_id = sanitize_run_id(args.run_id)
    execute_stage(cfg, args.stage, run_id, use_db=not args.no_db, **_stage_params(args, cfg))
    return 0


def cmd_run_all(args) -> int:
    cfg = _cfg_from_args(args)
    run_id = sanitize_run_id(args.run_id or default_run_id())
    params = _stage_params(args, cfg)
    stages = STAGES[STAGES.index(args.from_stage):] if args.from_stage else STAGES
    print(f"pipeline_run_id={run_id}", file=sys.stderr)
    for name in stages:
        if args.no_db and name in DB_STAGES:
            continue
        execute_stage(cfg, name, run_id, use_db=not args.no_db, **params)
    rec = RunRecord(cfg.run_dir(run_id), run_id).load()
    print(json.dumps({"pipeline_run_id": run_id, "status": rec["status"], "counts": rec["counts"]}, indent=2))
    return 0


def cmd_verify_source(args) -> int:
    from .manifest import compare_to_manifest, inspect_source, load_manifest

    cfg = load_config(args.config)
    path = cfg.resolve(args.source) if args.source else cfg.path("source_file")
    result = compare_to_manifest(inspect_source(path), load_manifest(cfg))
    print(json.dumps(result, indent=2))
    return 0 if result["matches"] else EXIT_PIPELINE_ERROR


def cmd_db_init(args) -> int:
    from . import db as dbm

    cfg = load_config(args.config)
    conn = dbm.connect(cfg)
    try:
        dbm.apply_ddl(conn, cfg.root)
    finally:
        conn.close()
    print("schemas, tables and views are in place")
    return 0


def cmd_db_reset(args) -> int:
    from . import db as dbm

    if not args.yes:
        print("Refusing to drop schemas without --yes", file=sys.stderr)
        return EXIT_PIPELINE_ERROR
    cfg = load_config(args.config)
    conn = dbm.connect(cfg)
    try:
        dbm.apply_ddl(conn, cfg.root, ["900_reset.sql"])
        dbm.apply_ddl(conn, cfg.root)
    finally:
        conn.close()
    print("database reset: schemas dropped and recreated empty")
    return 0


def cmd_db_status(args) -> int:
    from . import db as dbm

    cfg = load_config(args.config)
    conn = dbm.connect(cfg)
    try:
        out = {
            "duplicate_business_keys": dbm.duplicate_key_report(conn),
            "recent_runs": dbm.read_frame(conn, "SELECT pipeline_run_id, status, mode, staging_rows, curated_rows, "
                                                "test_rows, prediction_rows, failed_stage FROM audit.v_run_overview "
                                                "LIMIT 10").to_dict(orient="records"),
            "load_log": dbm.read_frame(conn, "SELECT pipeline_run_id, table_name, scope, inserted, updated, deleted, "
                                             "unchanged, rows_in_table FROM audit.load_log ORDER BY loaded_at DESC "
                                             "LIMIT 12").to_dict(orient="records"),
        }
    finally:
        conn.close()
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_reset_local(args) -> int:
    from .ingest import reset_raw

    if not args.yes:
        print("Refusing to delete local data layers without --yes", file=sys.stderr)
        return EXIT_PIPELINE_ERROR
    cfg = load_config(args.config)
    reset_raw(cfg)
    for key in ("staging_dir", "curated_dir", "features_dir", "quarantine_dir", "benchmark_dir"):
        d = cfg.path(key)
        for child in d.glob("*"):
            if child.name != ".gitkeep":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    if args.include_outputs:
        d = cfg.path("outputs_dir")
        for child in d.glob("*"):
            if child.name != ".gitkeep":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    print("local layers removed (source file and, unless --include-outputs, run outputs kept)")
    return 0


def cmd_benchmark(args) -> int:
    from .layer_io import read_table
    from .model import library_versions
    from .stages import curated_base, staging_base
    from .storage_benchmark import save_report, run_benchmark

    cfg = _cfg_from_args(args)
    run_id = sanitize_run_id(args.run_id or default_run_id("bench"))
    log = get_logger(run_id, "storage_benchmark", cfg.path("logs_dir"))
    frames = {"staging_readings": read_table(staging_base(cfg), cfg.layer_format),
              "curated_station_day": read_table(curated_base(cfg), cfg.layer_format)}
    with log_stage(log):
        report = run_benchmark(cfg, frames, repeats=args.repeats, use_db=not args.no_db, log=log)
    path = save_report(cfg, report, library_versions())
    print(path.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


def cmd_partition(args) -> int:
    from . import db as dbm
    from .layer_io import read_table
    from .partition import read_partition, verify_partition, write_partitions
    from .stages import CURATED_FP_COLS, curated_base

    cfg = _cfg_from_args(args)
    run_id = sanitize_run_id(args.run_id or default_run_id("partition"))
    log = get_logger(run_id, "partition_demo", cfg.path("logs_dir"))
    col = cfg.get("storage", "partition_column")
    curated = read_table(curated_base(cfg), cfg.layer_format)
    out: dict[str, Any] = {"pipeline_run_id": run_id}
    with log_stage(log):
        out["write"] = write_partitions(cfg, curated, col)
        part, meta = read_partition(cfg, args.value)
        out["read"] = meta
        out["verify"] = verify_partition(curated, part, col, meta["value"], CURATED_FP_COLS)
        if args.load_db:
            conn = dbm.connect(cfg)
            try:
                dbm.apply_ddl(conn, cfg.root)
                out["db_load"] = dbm.load_frame(conn, part, "curated.station_day_pollutants", run_id,
                                                full_sync=True, scope_column=col, scope_value=meta["value"])
                out["db_rows_for_partition"] = int(dbm.scalar(
                    conn, f"SELECT count(*) FROM curated.station_day_pollutants WHERE {col} = %s", (meta["value"],)))
            finally:
                conn.close()
        log.info("partition demo", extra={"event": "partition_demo", **out})
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["verify"]["identical"] else EXIT_PIPELINE_ERROR


def cmd_show_run(args) -> int:
    cfg = load_config(args.config)
    run_id = sanitize_run_id(args.run_id)
    print(json.dumps(RunRecord(cfg.run_dir(run_id), run_id).load(), indent=2))
    return 0


def cmd_sync_audit(args) -> int:
    cfg = load_config(args.config)
    run_id = sanitize_run_id(args.run_id)
    if args.mark_failed:
        rec = RunRecord(cfg.run_dir(run_id), run_id)
        doc = rec.load()
        if doc["status"] != "failed":
            rec.stage_failed(args.mark_failed, args.error or "task failed (reported by Airflow callback)")
    _best_effort_audit(cfg, run_id, get_logger(run_id, "sync_audit", cfg.path("logs_dir")))
    return 0


# ------------------------------------------------------------------- parser --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description="PM2.5 exceedance data pipeline")
    p.add_argument("--config", default=None, help="config file (default config/pipeline.yml)")
    sub = p.add_subparsers(dest="command", required=True)

    def stage_opts(sp):
        sp.add_argument("--mode", choices=["full", "incremental"], default=None)
        sp.add_argument("--source", default=None, help="override source file (failure-injection demos)")
        sp.add_argument("--batch-rows", type=int, default=None, help="incremental batch size in source rows")
        sp.add_argument("--seed", type=int, default=None, help="model seed (baseline = 42)")
        sp.add_argument("--imputation-scope", choices=["full", "train"], default=None)
        sp.add_argument("--layer-format", choices=["parquet", "csv"], default=None)
        sp.add_argument("--no-db", action="store_true", help="skip PostgreSQL (local smoke runs only)")

    sp = sub.add_parser("run-stage", help="run one stage (used by Airflow)")
    sp.add_argument("stage", choices=STAGES)
    sp.add_argument("--run-id", required=True)
    stage_opts(sp)
    sp.set_defaults(func=cmd_run_stage)

    sp = sub.add_parser("run-all", help="run all ten stages locally, in DAG order")
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--from-stage", choices=STAGES, default=None, help="resume a failed run from this stage")
    stage_opts(sp)
    sp.set_defaults(func=cmd_run_all)

    sp = sub.add_parser("verify-source", help="check the source file against the manifest")
    sp.add_argument("--source", default=None)
    sp.set_defaults(func=cmd_verify_source)

    sub.add_parser("db-init", help="create schemas/tables/views").set_defaults(func=cmd_db_init)
    sp = sub.add_parser("db-reset", help="DROP and recreate all pipeline schemas")
    sp.add_argument("--yes", action="store_true")
    sp.set_defaults(func=cmd_db_reset)
    sub.add_parser("db-status", help="duplicate-key report, recent runs, load log").set_defaults(func=cmd_db_status)

    sp = sub.add_parser("reset-local", help="delete local raw/staging/curated layers and checkpoint")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--include-outputs", action="store_true")
    sp.set_defaults(func=cmd_reset_local)

    sp = sub.add_parser("benchmark", help="CSV / JSON / Parquet / PostgreSQL storage experiment")
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--repeats", type=int, default=3)
    sp.add_argument("--no-db", action="store_true")
    sp.add_argument("--layer-format", choices=["parquet", "csv"], default=None)
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("partition", help="write country partitions; read/load one independently")
    sp.add_argument("--value", default="Japan", help="partition value (country label)")
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--load-db", action="store_true", help="also merge just this partition into PostgreSQL")
    sp.add_argument("--layer-format", choices=["parquet", "csv"], default=None)
    sp.set_defaults(func=cmd_partition)

    sp = sub.add_parser("show-run", help="print a run summary")
    sp.add_argument("--run-id", required=True)
    sp.set_defaults(func=cmd_show_run)

    sp = sub.add_parser("sync-audit", help="mirror a run record into audit.* (Airflow failure callback)")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--mark-failed", default=None, metavar="STAGE")
    sp.add_argument("--error", default=None)
    sp.set_defaults(func=cmd_sync_audit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as exc:
        print(f"PIPELINE ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return EXIT_PIPELINE_ERROR


if __name__ == "__main__":
    sys.exit(main())
