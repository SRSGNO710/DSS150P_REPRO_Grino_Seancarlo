"""Raw ingestion: full and incremental (checkpointed) landing of the source file.

Design
------
* The raw layer stores every source field as the exact string read from the file
  (no NA conversion, no casting) plus lineage/audit columns, so it is a lossless,
  queryable copy of the original.
* The dataset has no load/update timestamp, so incremental arrival is SIMULATED:
  the checkpoint is a **source-row high-water mark** (`last_committed_row`) tied
  to the source file's sha256. It is never presented as a business timestamp.
* Batch ids are deterministic (`b<start>-<end>_<sha8>`), so re-ingesting the same
  rows overwrites the same file instead of appending a duplicate.
* Two-phase commit: `ingest_source` writes the batch to raw/_pending/; only
  `validate_raw` promotes it to raw/committed/ and advances the checkpoint. A
  batch that fails validation is moved to data/quarantine/ and the checkpoint
  is NOT advanced, so bad data never becomes visible downstream.
* The checkpoint file is the single source of truth for which raw batches are
  committed; files not listed there are ignored and cleaned up.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .errors import CheckpointError
from .layer_io import layer_path, read_table, sha256_text_rows, write_table
from .logging_utils import utc_now_iso
from .manifest import compare_to_manifest, inspect_source, load_manifest

SOURCE_COLUMNS = [
    "Country Code", "City", "Location", "Coordinates", "Pollutant",
    "Source Name", "Unit", "Value", "Last Updated", "Country Label",
]
RAW_AUDIT_COLUMNS = [
    "_source_row_number", "_source_file", "_source_sha256", "_record_hash",
    "_ingest_batch_id", "_pipeline_run_id", "_ingested_at",
]


# ------------------------------------------------------------------ helpers --
def read_source_strings(path: Path) -> pd.DataFrame:
    """Read the source verbatim: every field as str, empty stays ''."""
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    df = df.astype(object)
    df.insert(0, "_source_row_number", range(1, len(df) + 1))
    df["_source_row_number"] = df["_source_row_number"].astype("int64")
    return df


def _raw_dirs(cfg: Config) -> tuple[Path, Path, Path]:
    raw = cfg.path("raw_dir")
    return raw / "committed", raw / "_pending", cfg.path("quarantine_dir")


def batch_id_for(start: int, end: int, sha256: str) -> str:
    return f"b{start:06d}-{end:06d}_{sha256[:8]}"


def load_checkpoint(cfg: Config) -> dict[str, Any] | None:
    path = cfg.path("checkpoint_file")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(cfg: Config, checkpoint: dict[str, Any]) -> None:
    path = cfg.path("checkpoint_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.name}")
    tmp.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: this is the commit point of a raw batch


def _relative(cfg: Config, path: Path) -> str:
    try:
        return path.resolve().relative_to(cfg.root).as_posix()
    except ValueError:
        return path.name  # never record machine-specific absolute paths


# ------------------------------------------------------------------- ingest --
def ingest(cfg: Config, run_id: str, mode: str, source_path: Path | None = None,
           batch_rows: int | None = None, log=None) -> dict[str, Any]:
    if mode not in ("full", "incremental"):
        raise CheckpointError(f"Unknown ingestion mode {mode!r}")
    committed_dir, pending_dir, _ = _raw_dirs(cfg)
    src = source_path or cfg.path("source_file")
    manifest = load_manifest(cfg)
    observed = inspect_source(src)
    identity = compare_to_manifest(observed, manifest)
    if log and not identity["matches"]:
        log.warning("source does not match manifest; validate_raw will reject this batch",
                    extra={"event": "source_identity_mismatch", "mismatches": identity["mismatches"]})

    # Leftovers from an interrupted run are re-derivable: discard them.
    if pending_dir.exists():
        stale = list(pending_dir.iterdir())
        if stale and log:
            log.warning("discarding %d stale pending file(s) from an earlier run", len(stale),
                        extra={"event": "stale_pending_removed"})
        shutil.rmtree(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)

    df = read_source_strings(src)
    total = len(df)
    checkpoint = load_checkpoint(cfg)

    if mode == "full":
        start, end = 1, total
    else:
        if checkpoint and checkpoint["source_sha256"] != observed["sha256"]:
            raise CheckpointError(
                "The source file changed since the last checkpoint "
                f"({checkpoint['source_sha256'][:12]} -> {observed['sha256'][:12]}). "
                "Incremental mode cannot continue on a different file; run with mode=full."
            )
        size = batch_rows or cfg.get("ingestion", "incremental_batch_rows")
        start = (checkpoint["last_committed_row"] + 1) if checkpoint else 1
        end = min(start + size - 1, total)

    summary: dict[str, Any] = {
        "mode": mode, "source_file": _relative(cfg, src), "source_sha256": observed["sha256"],
        "source_rows": total, "identity_matches_manifest": identity["matches"],
        "identity_mismatches": identity["mismatches"],
        "checkpoint_before": None if not checkpoint else checkpoint["last_committed_row"],
    }
    if start > end:
        summary.update({"new_rows": 0, "batch_id": None, "row_start": None, "row_end": None,
                        "note": "No new source rows beyond the checkpoint — nothing to ingest (idempotent rerun)."})
        return summary

    batch = df.iloc[start - 1:end].copy()
    present = [c for c in SOURCE_COLUMNS if c in batch.columns]
    batch_id = batch_id_for(start, end, observed["sha256"])
    batch["_source_file"] = _relative(cfg, src)
    batch["_source_sha256"] = observed["sha256"]
    batch["_record_hash"] = sha256_text_rows(batch, present)
    batch["_ingest_batch_id"] = batch_id
    batch["_pipeline_run_id"] = run_id
    batch["_ingested_at"] = utc_now_iso()
    extra_cols = [c for c in batch.columns if c not in SOURCE_COLUMNS and c not in RAW_AUDIT_COLUMNS]
    batch = batch[present + extra_cols + RAW_AUDIT_COLUMNS]

    write_table(batch, pending_dir / batch_id, cfg.layer_format)
    meta = {
        "batch_id": batch_id, "pipeline_run_id": run_id, "mode": mode, "row_start": start, "row_end": end,
        "rows": len(batch), "source_sha256": observed["sha256"], "source_file": _relative(cfg, src),
        "ingested_at": batch["_ingested_at"].iloc[0], "format": cfg.layer_format,
    }
    (pending_dir / f"{batch_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary.update({"new_rows": len(batch), "batch_id": batch_id, "row_start": start, "row_end": end})
    return summary


# ----------------------------------------------------------- batch registry --
def pending_batches(cfg: Config) -> list[dict[str, Any]]:
    _, pending_dir, _ = _raw_dirs(cfg)
    if not pending_dir.exists():
        return []
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pending_dir.glob("*.meta.json"))]
    return sorted(metas, key=lambda m: m["row_start"])


def _read_batch(cfg: Config, directory: Path, batch_id: str, fmt: str) -> pd.DataFrame:
    return read_table(directory / batch_id, fmt, preserve_empty_strings=True)


def read_pending(cfg: Config) -> pd.DataFrame:
    _, pending_dir, _ = _raw_dirs(cfg)
    frames = [_read_batch(cfg, pending_dir, m["batch_id"], m["format"]) for m in pending_batches(cfg)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_committed_raw(cfg: Config) -> pd.DataFrame:
    """All committed raw rows (only batches listed in the checkpoint), in source order."""
    checkpoint = load_checkpoint(cfg)
    if not checkpoint or not checkpoint.get("batches"):
        raise CheckpointError("No committed raw data. Run ingest_source + validate_raw first.")
    committed_dir, _, _ = _raw_dirs(cfg)
    frames = [_read_batch(cfg, committed_dir, b["batch_id"], b["format"]) for b in checkpoint["batches"]]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("_source_row_number", kind="mergesort").reset_index(drop=True)


def commit_pending(cfg: Config, run_id: str, mode_hint: str | None = None) -> dict[str, Any]:
    """Promote pending batches to committed and advance the checkpoint (atomic)."""
    committed_dir, pending_dir, _ = _raw_dirs(cfg)
    metas = pending_batches(cfg)
    checkpoint = load_checkpoint(cfg)
    if not metas:
        return {"committed_batches": [], "checkpoint": checkpoint}
    committed_dir.mkdir(parents=True, exist_ok=True)
    full = any(m["mode"] == "full" for m in metas)
    batches = [] if (full or checkpoint is None) else list(checkpoint["batches"])
    for m in metas:
        for path in (layer_path(pending_dir / m["batch_id"], m["format"]),
                     pending_dir / f"{m['batch_id']}.dtypes.json"):
            if path.exists():
                os.replace(path, committed_dir / path.name)
        batches = [b for b in batches if b["batch_id"] != m["batch_id"]]
        batches.append({k: m[k] for k in ("batch_id", "row_start", "row_end", "rows", "format", "pipeline_run_id", "mode")}
                       | {"committed_at": utc_now_iso()})
    batches.sort(key=lambda b: b["row_start"])
    new_checkpoint = {
        "checkpoint_type": "source_row_high_water_mark",
        "note": "Simulated incremental ingestion of a static file; NOT a business timestamp.",
        "source_sha256": metas[-1]["source_sha256"],
        "last_committed_row": max(b["row_end"] for b in batches),
        "committed_rows": sum(b["rows"] for b in batches),
        "batches": batches,
        "updated_by_run": run_id,
        "updated_at": utc_now_iso(),
    }
    save_checkpoint(cfg, new_checkpoint)  # <- commit point
    # Clean-up: anything in committed/ that the checkpoint does not reference.
    keep = {b["batch_id"] for b in batches}
    for path in committed_dir.iterdir():
        if path.name.split(".")[0] not in keep:
            path.unlink()
    for path in pending_dir.glob("*.meta.json"):
        path.unlink()
    return {"committed_batches": [m["batch_id"] for m in metas], "checkpoint": new_checkpoint}


def quarantine_pending(cfg: Config, run_id: str, reason: str) -> list[str]:
    _, pending_dir, quarantine_dir = _raw_dirs(cfg)
    target = quarantine_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in sorted(pending_dir.glob("*")):
        os.replace(path, target / path.name)
        moved.append(path.name)
    (target / "REASON.txt").write_text(reason + "\n", encoding="utf-8")
    return moved


def reset_raw(cfg: Config) -> None:
    """Delete raw layer + checkpoint (used by `pipeline reset`)."""
    committed_dir, pending_dir, _ = _raw_dirs(cfg)
    for d in (committed_dir, pending_dir):
        if d.exists():
            shutil.rmtree(d)
    ck = cfg.path("checkpoint_file")
    if ck.exists():
        ck.unlink()
