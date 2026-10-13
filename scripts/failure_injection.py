"""Safe, reversible failure injection + recovery, with evidence capture.

Every scenario works on a TEMPORARY copy or a temporary setting — the real source
file, config and database are never modified. For each scenario the script:
  1. records the "before" state (published run, checkpoint, PostgreSQL counts),
  2. injects the fault and runs the pipeline -> expects a failure at a known stage,
  3. proves bad data did not reach outputs (latest/ unchanged, no metrics, DB unchanged,
     checkpoint not advanced, batch quarantined),
  4. removes the fault and re-runs / resumes -> expects success.
Results are written to docs/evidence/failure_injection_<env>.json and .md.

Usage (repo root):
    python scripts/failure_injection.py                 # all scenarios
    python scripts/failure_injection.py --only schema   # one scenario
    python scripts/failure_injection.py --no-db         # skip DB scenarios
Inside Docker:  docker compose run --rm pipeline python scripts/failure_injection.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source" / "world_air_quality.csv"
TMP = ROOT / "data" / "tmp"
LATEST = ROOT / "data" / "outputs" / "latest" / "PUBLISHED_RUN.txt"
CHECKPOINT = ROOT / "data" / "checkpoints" / "ingest_checkpoint.json"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cli(*args: str, env: dict | None = None) -> tuple[int, str]:
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), full_env.get("PYTHONPATH")]))
    full_env.update(env or {})
    proc = subprocess.run([sys.executable, "-m", "pipeline.cli", *args], cwd=ROOT, env=full_env,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def failed_stage(run_id: str) -> dict | None:
    path = ROOT / "data" / "outputs" / "runs" / run_id / "run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("error")


def db_counts(use_db: bool) -> dict | None:
    if not use_db:
        return None
    code, out = cli("db-status")
    if code != 0:
        return {"error": out.strip().splitlines()[-1] if out.strip() else "db-status failed"}
    data = json.loads(out)
    return {r["table_name"]: int(r["row_count"]) for r in data["duplicate_business_keys"]}


def snapshot(use_db: bool) -> dict:
    ck = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else None
    return {
        "published_run": LATEST.read_text(encoding="utf-8").strip() if LATEST.exists() else None,
        "checkpoint_last_row": None if ck is None else ck["last_committed_row"],
        "checkpoint_sha256": None if ck is None else ck["source_sha256"][:12],
        "db_rows": db_counts(use_db),
    }


def write_variant(name: str, mutate) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / name
    with open(SOURCE, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    rows = mutate(rows)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh, delimiter=";", lineterminator="\r\n").writerows(rows)
    return out


def drop_column(col: str):
    def f(rows):
        i = rows[0].index(col)
        return [r[:i] + r[i + 1:] for r in rows]
    return f


def corrupt_value(row_no: int, value: str):
    def f(rows):
        i = rows[0].index("Value")
        rows[row_no][i] = value
        return rows
    return f


def scenario(name, description, inject, recover, expect_stage, use_db):
    before = snapshot(use_db)
    run_fail = f"fi_{name}_{stamp()}"
    code, out = inject(run_fail)
    err = failed_stage(run_fail)
    after_fail = snapshot(use_db)
    quarantine = ROOT / "data" / "quarantine" / run_fail
    no_outputs = not (ROOT / "data" / "outputs" / "runs" / run_fail / "metrics.json").exists()
    protected = {
        "failed_as_expected": code != 0 and (expect_stage is None or (err or {}).get("stage") == expect_stage
                                             or expect_stage in out),
        "published_run_unchanged": after_fail["published_run"] == before["published_run"],
        "checkpoint_not_advanced": after_fail["checkpoint_last_row"] == before["checkpoint_last_row"]
        and after_fail["checkpoint_sha256"] == before["checkpoint_sha256"],
        "no_model_outputs_for_failed_run": no_outputs,
        "db_unchanged": after_fail["db_rows"] == before["db_rows"],
        "batch_quarantined": quarantine.exists(),
    }
    run_ok, code2, out2 = recover(run_fail)
    after_recover = snapshot(use_db)
    rec_doc = ROOT / "data" / "outputs" / "runs" / run_ok / "run_summary.json"
    rec_status = json.loads(rec_doc.read_text(encoding="utf-8"))["status"] if rec_doc.exists() else None
    tail = [l for l in out.strip().splitlines() if "PIPELINE ERROR" in l or '"stage_failed"' in l]
    result = {
        "scenario": name, "description": description, "failed_run_id": run_fail, "exit_code": code,
        "expected_failed_stage": expect_stage, "observed_error": err,
        "error_line": (tail[-1][:400] if tail else out.strip().splitlines()[-1][:400] if out.strip() else ""),
        "protection": protected, "before": before, "after_failure": after_fail,
        "recovery_run_id": run_ok, "recovery_exit_code": code2, "recovery_status": rec_status,
        "after_recovery": after_recover,
    }
    ok = protected["failed_as_expected"] and protected["published_run_unchanged"] and \
        protected["no_model_outputs_for_failed_run"] and code2 == 0 and rec_status == "success"
    result["verdict"] = "PASS" if ok else "CHECK"
    print(f"[{result['verdict']}] {name}: failed_at={(err or {}).get('stage')} recovered={rec_status}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()
    use_db = not args.no_db
    db_flag = ["--no-db"] if args.no_db else []
    if not SOURCE.exists():
        print("source file missing; see README §Data", file=sys.stderr)
        return 2

    # Make sure there is a successful published baseline run to protect.
    base_run = f"fi_baseline_{stamp()}"
    code, out = cli("run-all", "--run-id", base_run, *db_flag)
    if code != 0:
        print(out[-3000:], file=sys.stderr)
        return code

    def rerun_good(_failed):
        rid = f"fi_recovery_{stamp()}"
        c, o = cli("run-all", "--run-id", rid, *db_flag)
        return rid, c, o

    scenarios = {
        "schema": dict(
            description="Temporary copy of the source with the required column 'Pollutant' removed.",
            inject=lambda rid: cli("run-all", "--run-id", rid, "--source",
                                   str(write_variant("broken_missing_pollutant.csv", drop_column("Pollutant")).relative_to(ROOT)),
                                   *db_flag),
            recover=rerun_good, expect_stage="validate_raw"),
        "invalid_row": dict(
            description="Temporary copy with one unparseable concentration (row 101: Value='not_a_number').",
            inject=lambda rid: cli("run-all", "--run-id", rid, "--source",
                                   str(write_variant("broken_invalid_value.csv", corrupt_value(101, "not_a_number")).relative_to(ROOT)),
                                   *db_flag),
            recover=rerun_good, expect_stage="validate_raw"),
        "invalid_config": dict(
            description="Invalid configuration value PIPELINE_LAYER_FORMAT=xlsx (environment only).",
            inject=lambda rid: cli("run-all", "--run-id", rid, *db_flag, env={"PIPELINE_LAYER_FORMAT": "xlsx"}),
            recover=rerun_good, expect_stage="ConfigError"),
    }
    if use_db:
        def inject_db(rid):
            return cli("run-all", "--run-id", rid, env={"POSTGRES_PORT": "5999"})

        def resume_db(rid):
            c, o = cli("run-all", "--run-id", rid, "--from-stage", "load_postgres")
            return rid, c, o

        scenarios["db_unavailable"] = dict(
            description="PostgreSQL unreachable (POSTGRES_PORT pointed at a closed port); then the SAME run is "
                        "resumed from load_postgres once the database is reachable again.",
            inject=inject_db, recover=resume_db, expect_stage="load_postgres")

    results = []
    for name, spec in scenarios.items():
        if args.only and name != args.only:
            continue
        results.append(scenario(name, spec["description"], spec["inject"], spec["recover"], spec["expect_stage"], use_db))
    for p in TMP.glob("broken_*.csv"):
        p.unlink()  # clean up the temporary copies

    env_label = os.environ.get("EVIDENCE_LABEL", "local")
    out_dir = ROOT / "docs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"generated_at": stamp(), "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                                    "label": env_label}, "baseline_run": base_run, "results": results}
    (out_dir / f"failure_injection_{env_label}.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    lines = [f"# Failure injection & recovery evidence ({env_label})", "",
             f"_Generated {doc['generated_at']} by `scripts/failure_injection.py`; raw data in "
             f"`failure_injection_{env_label}.json`._", "",
             "| Scenario | Failed at | Published output unchanged | Checkpoint not advanced | DB unchanged | "
             "Quarantined | Recovery | Verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        p = r["protection"]
        stage = (r["observed_error"] or {}).get("stage") or ("config load" if r["scenario"] == "invalid_config" else "?")
        lines.append(f"| {r['scenario']} | {stage} | {p['published_run_unchanged']} | {p['checkpoint_not_advanced']} | "
                     f"{p['db_unchanged']} | {p['batch_quarantined']} | {r['recovery_status']} (`{r['recovery_run_id']}`) | "
                     f"**{r['verdict']}** |")
    lines += ["", "## Error messages captured", ""]
    for r in results:
        lines += [f"**{r['scenario']}** — {r['description']}", "", "```", r["error_line"], "```", ""]
    (out_dir / f"failure_injection_{env_label}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"evidence written to docs/evidence/failure_injection_{env_label}.md")
    return 0 if all(r["verdict"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
