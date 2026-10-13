"""Integration tests: span ingest -> quality -> transform modules and the file
boundary (raw/staging/curated layers, checkpoint, quarantine)."""
from __future__ import annotations

import json

import pytest

from conftest import SYNTHETIC_ROWS, write_source
from pipeline import ingest as ing
from pipeline.errors import CheckpointError, DataQualityError
from pipeline.layer_io import read_table
from pipeline.stages import build_curated, curated_base, features_base, transform_staging, validate_raw

pytestmark = pytest.mark.integration


def _full_cycle(cfg, log, run_id, **kw):
    s1 = ing.ingest(cfg, run_id, kw.pop("mode", "full"), log=log, **kw)
    validate_raw(cfg, run_id, log)
    return s1


def test_full_ingest_to_curated(cfg, log):
    _full_cycle(cfg, log, "it_full")
    transform_staging(cfg, "it_full", log)
    res = build_curated(cfg, "it_full", log)
    curated = read_table(curated_base(cfg), cfg.layer_format)
    mf = read_table(features_base(cfg), cfg.layer_format)
    assert res.counts == {"curated": len(curated), "model_features": len(mf)}
    assert curated["station_day_key"].is_unique
    assert len(mf) == 1 and mf["Air_Quality"].iloc[0] == "Safe"     # Station B daily PM2.5 mean = 12.0
    ck = ing.load_checkpoint(cfg)
    assert ck["last_committed_row"] == len(SYNTHETIC_ROWS) and ck["checkpoint_type"] == "source_row_high_water_mark"


def test_rerun_full_mode_does_not_duplicate_raw(cfg, log):
    _full_cycle(cfg, log, "it_a")
    _full_cycle(cfg, log, "it_b")
    raw = ing.read_committed_raw(cfg)
    assert len(raw) == len(SYNTHETIC_ROWS)
    assert raw["_source_row_number"].is_unique


def test_incremental_batches_then_idempotent_noop(cfg, log):
    seen = []
    for i in range(5):
        s = _full_cycle(cfg, log, f"it_inc{i}", mode="incremental", batch_rows=5)
        seen.append(s["new_rows"])
    assert seen == [5, 5, 2, 0, 0]
    raw = ing.read_committed_raw(cfg)
    assert len(raw) == len(SYNTHETIC_ROWS) and raw["_source_row_number"].is_unique
    ck = ing.load_checkpoint(cfg)
    assert [b["row_start"] for b in ck["batches"]] == [1, 6, 11]


def test_bad_batch_is_quarantined_and_checkpoint_not_advanced(cfg, log, project):
    _full_cycle(cfg, log, "it_good")
    before = ing.load_checkpoint(cfg)
    broken = project / "data" / "tmp" / "broken.csv"
    header = ["Country Code", "City", "Location", "Coordinates", "Source Name", "Unit", "Value", "Last Updated",
              "Country Label"]  # 'Pollutant' removed
    rows = [r[:4] + r[5:] for r in SYNTHETIC_ROWS]
    write_source(broken, rows, header)
    ing.ingest(cfg, "it_bad", "full", source_path=broken, log=log)
    with pytest.raises(DataQualityError) as exc:
        validate_raw(cfg, "it_bad", log)
    assert "raw_schema_required_columns" in str(exc.value)
    assert ing.load_checkpoint(cfg) == before, "checkpoint must not move on a failed batch"
    assert (project / "data" / "quarantine" / "it_bad" / "REASON.txt").exists()
    assert len(ing.read_committed_raw(cfg)) == len(SYNTHETIC_ROWS), "previous committed raw still intact"
    report = json.loads((cfg.run_dir("it_bad") / "dq_report.json").read_text(encoding="utf-8"))
    assert any(r["check_id"] == "raw_schema_required_columns" and r["status"] == "failed" for r in report)


def test_incremental_refuses_changed_source(cfg, log, project):
    _full_cycle(cfg, log, "it_x", mode="incremental", batch_rows=5)
    changed = project / "data" / "tmp" / "changed.csv"
    write_source(changed, SYNTHETIC_ROWS[:-1])
    with pytest.raises(CheckpointError):
        ing.ingest(cfg, "it_y", "incremental", source_path=changed, log=log)
