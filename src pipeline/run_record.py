"""Pipeline run record (run summary).

One JSON document per pipeline_run_id at data/outputs/runs/<run_id>/run_summary.json.
Each stage merges its own status, duration and row counts into it, so the file
always shows how far a run got, its start/end status, and the counts for raw,
staging, curated, train, test and prediction outputs. The same content is
mirrored to PostgreSQL (audit.pipeline_runs / audit.task_runs) by db.sync_audit.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .logging_utils import utc_now_iso

COUNT_KEYS = ("source", "raw", "staging", "curated", "model_features", "train", "test", "predictions")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str, ensure_ascii=False)
    os.replace(tmp, path)


class RunRecord:
    def __init__(self, run_dir: Path, run_id: str):
        self.path = run_dir / "run_summary.json"
        self.run_id = run_id

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "pipeline_run_id": self.run_id,
            "status": "running",
            "started_at": utc_now_iso(),
            "ended_at": None,
            "parameters": {},
            "counts": {k: None for k in COUNT_KEYS},
            "stages": {},
            "layer_fingerprints": {},
            "error": None,
        }

    def save(self, doc: dict[str, Any]) -> None:
        _atomic_write_json(self.path, doc)

    def set_parameters(self, **params: Any) -> None:
        doc = self.load()
        doc["parameters"].update({k: v for k, v in params.items() if v is not None})
        self.save(doc)

    def stage_started(self, stage: str) -> None:
        doc = self.load()
        doc["status"] = "running"
        doc["error"] = None
        doc["ended_at"] = None
        doc["stages"][stage] = {"status": "running", "started_at": utc_now_iso()}
        self.save(doc)

    def stage_finished(
        self,
        stage: str,
        *,
        counts: dict[str, int] | None = None,
        details: dict[str, Any] | None = None,
        fingerprints: dict[str, str] | None = None,
        duration_s: float | None = None,
    ) -> None:
        doc = self.load()
        st = doc["stages"].setdefault(stage, {})
        st.update({"status": "success", "ended_at": utc_now_iso(), "duration_s": duration_s})
        if details:
            st["details"] = details
        if counts:
            doc["counts"].update(counts)
            st["counts"] = counts
        if fingerprints:
            doc["layer_fingerprints"].update(fingerprints)
        self.save(doc)

    def stage_failed(self, stage: str, error: str, duration_s: float | None = None) -> None:
        doc = self.load()
        st = doc["stages"].setdefault(stage, {})
        st.update({"status": "failed", "ended_at": utc_now_iso(), "duration_s": duration_s, "error": error})
        doc["status"] = "failed"
        doc["ended_at"] = utc_now_iso()
        doc["error"] = {"stage": stage, "message": error}
        self.save(doc)

    def finalize(self, status: str = "success") -> dict[str, Any]:
        doc = self.load()
        doc["status"] = status
        doc["ended_at"] = utc_now_iso()
        self.save(doc)
        return doc
