"""Unit tests: data-quality checks detect intentionally invalid conditions (A14),
reproducibility tolerance logic, config validation, layer I/O equivalence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config, load_yaml
from pipeline.errors import ConfigError, DataQualityError
from pipeline.layer_io import content_fingerprint, read_table, write_table
from pipeline.quality import enforce, run_checks, split_integrity_checks
from pipeline.reproducibility import compare

REPO = Path(__file__).resolve().parents[2]
CONTRACT = load_yaml(REPO / "contracts" / "data_contract.yml")
FEATS = CONTRACT["layers"]["model_features"]["feature_columns"]


def _model_frame(n=10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pm25 = np.linspace(5, 30, n)
    df = pd.DataFrame({
        "station_day_key": [f"k{i}" for i in range(n)],
        "NO2": rng.uniform(1, 50, n), "O3": rng.uniform(1, 90, n), "SO2": rng.uniform(1, 9, n),
        "CO": rng.uniform(100, 900, n), "SO2_missing": 0, "CO_missing": 0, "PM2.5": pm25,
    })
    df["Ox_total_oxidant"] = df["NO2"] + df["O3"]
    df["NO2_O3_ratio"] = df["NO2"] / (df["O3"] + 1)
    df["Air_Quality"] = np.where(df["PM2.5"] > 15, "Unhealthy", "Safe")
    return df


def _status(results, check_id):
    return next(r["status"] for r in results if r["check_id"] == check_id)


def test_clean_model_frame_passes_all_checks():
    res = run_checks(_model_frame(), "model_features", CONTRACT, {"source_complete": False, "feature_columns": FEATS})
    assert all(r["status"] in ("passed", "skipped") for r in res), [r for r in res if r["status"] == "failed"]


@pytest.mark.parametrize("mutate, check_id", [
    (lambda d: d.drop(columns=["O3"]), "mf_schema_feature_order"),                      # schema
    (lambda d: d.assign(NO2=[None] + list(d["NO2"][1:])), "mf_no_nulls"),               # completeness
    (lambda d: pd.concat([d, d.iloc[[0]]]), "mf_key_unique"),                            # uniqueness
    (lambda d: d.assign(Air_Quality=["Bad"] + list(d["Air_Quality"][1:])), "mf_label_values"),  # validity
    (lambda d: d.assign(Ox_total_oxidant=d["Ox_total_oxidant"] + 1), "mf_ox_equals_sum"),       # consistency
    (lambda d: d.assign(Air_Quality="Safe"), "mf_label_matches_who_rule"),              # business rule
])
def test_checks_detect_invalid_conditions(mutate, check_id):
    res = run_checks(mutate(_model_frame()), "model_features", CONTRACT,
                     {"source_complete": False, "feature_columns": FEATS})
    assert _status(res, check_id) == "failed"
    with pytest.raises(DataQualityError):
        enforce(res, "model_features")


def test_leakage_guard_rejects_pm10_feature():
    res = run_checks(_model_frame(), "model_features", CONTRACT,
                     {"source_complete": False, "feature_columns": FEATS + ["PM10"]})
    assert _status(res, "mf_no_leakage_features") == "failed"


def test_raw_schema_and_parse_checks():
    raw = pd.DataFrame({"Country Code": ["JP"], "City": [""], "Location": ["x"], "Coordinates": [""],
                        "Source Name": ["s"], "Unit": ["µg/m³"], "Value": ["abc"], "Last Updated": ["not-a-date"],
                        "Country Label": ["Japan"], "_source_sha256": ["0" * 64], "_source_row_number": [1]})
    manifest = {"file": {"sha256": "f" * 64, "data_rows": 1}}
    res = run_checks(raw, "raw", CONTRACT, {"manifest": manifest, "source_complete": True})
    assert _status(res, "raw_schema_required_columns") == "failed"      # Pollutant missing
    assert _status(res, "raw_value_parseable") == "failed"
    assert _status(res, "raw_timestamp_parseable") == "failed"
    assert _status(res, "raw_source_checksum_matches_manifest") == "failed"


def test_split_integrity_detects_overlap():
    res = split_integrity_checks(pd.Series(["a", "b"]), pd.Series(["b", "c"]), 2, ["x", "y"], ["x", "y"])
    assert _status(res, "split_train_test_disjoint") == "failed"


# ------------------------------------------------------------ reproducibility --
BASELINE = json.loads((REPO / "expected" / "baseline_metrics.json").read_text(encoding="utf-8"))


def _metrics_like_baseline():
    hist = BASELINE["historical"]
    models = {}
    for name, m in hist["models"].items():
        models[name] = {"test": dict(m["test"]), "cv_f1_weighted": dict(m["cv_f1_weighted"])}
        models[name]["test"]["confusion_matrix"] = {"matrix": [[BASELINE["paper_reported"]["true_safe_correct"][name], 0], [0, 0]]}
    return {"seed": 42, "imputation_fit_scope": "full", "models": models,
            "counts": {"train": hist["counts"]["train"], "test": hist["counts"]["test"]},
            "class_balance": dict(hist["class_balance"]), "split": dict(BASELINE["split_fingerprint"]),
            "best_model_by_cv_f1": hist["best_model_by_cv_f1"]}


COUNTS = {k: v for k, v in BASELINE["historical"]["counts"].items()}


def test_repro_exact_match_passes():
    out = compare(_metrics_like_baseline(), BASELINE, COUNTS, 0.01)
    assert out["status"] == "pass" and out["summary"]["metrics_exact"]


def test_repro_small_difference_within_tolerance():
    m = _metrics_like_baseline()
    m["models"]["gradient_boosting"]["cv_f1_weighted"]["mean"] -= 0.0004
    out = compare(m, BASELINE, COUNTS, 0.01)
    assert out["status"] == "pass" and out["summary"]["n_nonzero_differences"] == 1
    row = next(r for r in out["metric_rows"] if r["classification"] != "exact_match")
    assert row["classification"] == "worse_within_tolerance"


def test_repro_material_difference_or_count_change_fails():
    m = copy.deepcopy(_metrics_like_baseline())
    m["models"]["random_forest"]["test"]["f1_weighted"] += 0.05
    assert compare(m, BASELINE, COUNTS, 0.01)["status"] == "fail"
    m2 = _metrics_like_baseline()
    assert compare(m2, BASELINE, {**COUNTS, "curated": COUNTS["curated"] - 1}, 0.01)["status"] == "fail"
    m3 = _metrics_like_baseline()
    m3["split"]["test_keys_sha256"] = "0" * 64
    assert compare(m3, BASELINE, COUNTS, 0.01)["status"] == "fail"


def test_repro_other_seed_is_not_applicable():
    m = _metrics_like_baseline()
    m["seed"] = 7
    assert compare(m, BASELINE, COUNTS, 0.01)["status"] == "not_applicable"


# ------------------------------------------------------------------- config --
def test_invalid_config_value_rejected(monkeypatch):
    monkeypatch.setenv("PIPELINE_LAYER_FORMAT", "xlsx")
    with pytest.raises(ConfigError):
        load_config()


def test_config_paths_are_relative():
    data = load_yaml(REPO / "config" / "pipeline.yml")
    assert all(not Path(v).is_absolute() for v in data["paths"].values())


# ----------------------------------------------------------------- layer I/O --
def _typed_frame():
    return pd.DataFrame({
        "k": ["a", "b"], "s": ["x", None], "f": [1.5, np.nan], "i": [1, 2],
        "ts": pd.to_datetime(["2024-03-10T08:00:00Z", "2024-03-11T09:30:00Z"], utc=True),
        "d": [pd.Timestamp("2024-03-10").date(), pd.Timestamp("2024-03-11").date()],
    })


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_layer_roundtrip_preserves_logical_records(tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    df = _typed_frame()
    write_table(df, tmp_path / "t", fmt)
    back = read_table(tmp_path / "t", fmt)
    cols = list(df.columns)
    assert content_fingerprint(back, cols, ["k"]) == content_fingerprint(df, cols, ["k"])
    assert back["i"].dtype == "int64" and str(back["ts"].dt.tz) == "UTC"


def test_contract_value_lists_have_no_yaml_booleans():
    """Regression: unquoted NO (nitric oxide) is parsed by YAML 1.1 as False."""
    for layer in CONTRACT["layers"].values():
        for check in layer["checks"]:
            if check["type"] == "accepted_values":
                assert not any(isinstance(v, bool) for v in check["values"]), check["id"]
    assert "NO" in next(c for c in CONTRACT["layers"]["raw"]["checks"] if c["id"] == "raw_pollutant_codes_known")["values"]
