"""Data-quality engine that executes the checks declared in contracts/data_contract.yml.

Each check returns a result dict:
    check_id, layer, category, type, severity, status (passed|failed|skipped),
    observed, threshold, message, examples
`enforce()` raises DataQualityError when any severity=error check failed, which
fails the calling task and therefore blocks every downstream task.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import Config, load_yaml
from .errors import DataQualityError

POLLUTANT_COLS = ["co", "no2", "o3", "pm10", "pm25", "so2"]


def load_contract(cfg: Config) -> dict[str, Any]:
    return load_yaml(cfg.path("data_contract"))


def _is_null(s: pd.Series) -> pd.Series:
    null = s.isna()
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        null = null | (s.astype(str) == "")
    return null


def _examples(df: pd.DataFrame, mask: pd.Series, cols: list[str] | None = None, n: int = 5) -> list[dict]:
    if not mask.any():
        return []
    sub = df.loc[mask, cols] if cols else df.loc[mask]
    return json.loads(sub.head(n).to_json(orient="records", date_format="iso", force_ascii=False))


def _result(check: dict, layer: str, status: str, observed: Any = None, threshold: Any = None,
            message: str = "", examples: list | None = None) -> dict[str, Any]:
    return {
        "check_id": check["id"],
        "layer": layer,
        "category": check.get("category"),
        "type": check["type"],
        "severity": check.get("severity", "error"),
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "message": message,
        "examples": examples or [],
    }


# ----------------------------------------------------------------- built-ins --
def _required_columns(df, check, layer, ctx):
    missing = [c for c in check["columns"] if c not in df.columns]
    if missing:
        return _result(check, layer, "failed", observed={"missing": missing}, threshold="all present",
                       message=f"Missing required columns: {missing}")
    return _result(check, layer, "passed", observed="all present", threshold="all present")


def _parseable(df, check, layer, ctx):
    col = check["column"]
    if col not in df.columns:
        return _result(check, layer, "failed", message=f"Column {col!r} absent; cannot test parseability")
    s = df[col]
    present = ~_is_null(s)
    if check["parse_as"] == "float":
        parsed = pd.to_numeric(s.where(present), errors="coerce")
    else:
        parsed = pd.to_datetime(s.where(present), errors="coerce", utc=True, format="ISO8601")
    bad = present & parsed.isna()
    frac = float(bad.mean()) if len(s) else 0.0
    limit = float(check.get("max_failure_fraction", 0.0))
    status = "passed" if frac <= limit else "failed"
    return _result(check, layer, status, observed={"unparseable_rows": int(bad.sum()), "fraction": frac},
                   threshold=limit, examples=_examples(df, bad, [col]))


def _not_null(df, check, layer, ctx):
    cols = check.get("columns") or [check["column"]]
    limit = float(check.get("max_null_fraction", 0.0))
    observed, failing = {}, []
    for col in cols:
        if col not in df.columns:
            observed[col] = "column missing"
            failing.append(col)
            continue
        frac = float(_is_null(df[col]).mean()) if len(df) else 0.0
        observed[col] = round(frac, 6)
        if frac > limit:
            failing.append(col)
    status = "failed" if failing else "passed"
    return _result(check, layer, status, observed=observed, threshold=limit,
                   message=f"Null fraction above threshold in {failing}" if failing else "")


def _unique(df, check, layer, ctx):
    cols = check["columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return _result(check, layer, "failed", message=f"Key columns missing: {missing}")
    dup = df.duplicated(subset=cols, keep=False)
    n_dup_keys = int(df.loc[dup, cols].drop_duplicates().shape[0])
    status = "passed" if n_dup_keys == 0 else "failed"
    return _result(check, layer, status, observed={"duplicate_keys": n_dup_keys, "rows_involved": int(dup.sum())},
                   threshold=0, examples=_examples(df, dup, cols))


def _accepted_values(df, check, layer, ctx):
    col = check["column"]
    if col not in df.columns:
        return _result(check, layer, "failed", message=f"Column {col!r} absent")
    allowed = set(check["values"])
    s = df[col]
    bad = ~s.isin(allowed) & s.notna()
    status = "passed" if not bad.any() else "failed"
    unexpected = sorted(map(str, s[bad].unique()))[:20]
    return _result(check, layer, status, observed={"unexpected_values": unexpected, "rows": int(bad.sum())},
                   threshold=sorted(map(str, allowed)), examples=_examples(df, bad, [col]))


def _range(df, check, layer, ctx):
    col = check["column"]
    if col not in df.columns:
        return _result(check, layer, "failed", message=f"Column {col!r} absent")
    s = pd.to_numeric(df[col], errors="coerce")
    bad = pd.Series(False, index=df.index)
    if "min" in check:
        bad |= s < check["min"]
    if "max" in check:
        bad |= s > check["max"]
    status = "passed" if not bad.any() else "failed"
    return _result(check, layer, status,
                   observed={"violations": int(bad.sum()), "min": None if s.dropna().empty else float(s.min()),
                             "max": None if s.dropna().empty else float(s.max())},
                   threshold={k: check[k] for k in ("min", "max") if k in check},
                   examples=_examples(df, bad, [col]))


def _row_count(df, check, layer, ctx):
    if check.get("applies_when") == "complete_source" and not ctx.get("source_complete", True):
        return _result(check, layer, "skipped", observed=len(df),
                       message="Raw layer does not yet hold the complete source (incremental mode)")
    expected = ctx["manifest"]["file"]["data_rows"] if check.get("equals_manifest") else check["equals"]
    status = "passed" if len(df) == expected else "failed"
    return _result(check, layer, status, observed=len(df), threshold=expected)


def _source_checksum(df, check, layer, ctx):
    expected = ctx["manifest"]["file"]["sha256"]
    if "_source_sha256" not in df.columns:
        return _result(check, layer, "failed", message="_source_sha256 lineage column missing")
    seen = sorted(df["_source_sha256"].dropna().unique().tolist())
    status = "passed" if seen == [expected] or (not seen and len(df) == 0) else "failed"
    return _result(check, layer, status, observed=seen, threshold=expected,
                   message="" if status == "passed" else "Raw rows were ingested from a file that does not match the manifest")


# -------------------------------------------------------------- custom rules --
def _rule_at_least_one_pollutant(df, check, layer, ctx):
    cols = [c for c in POLLUTANT_COLS if c in df.columns]
    bad = df[cols].isna().all(axis=1) if cols else pd.Series(True, index=df.index)
    return _result(check, layer, "passed" if not bad.any() else "failed",
                   observed={"rows_without_any_pollutant": int(bad.sum())}, threshold=0)


def _rule_pm25_le_pm10(df, check, layer, ctx):
    both = df["pm25"].notna() & df["pm10"].notna()
    bad = both & (df["pm25"] > df["pm10"])
    frac = float(bad.sum() / both.sum()) if both.any() else 0.0
    limit = float(check.get("max_violation_fraction", 0.0))
    return _result(check, layer, "passed" if frac <= limit else "failed",
                   observed={"rows_with_both": int(both.sum()), "pm25_gt_pm10": int(bad.sum()), "fraction": round(frac, 6)},
                   threshold=limit, message="PM2.5 is physically a subset of PM10; violations indicate sensor/network mismatch",
                   examples=_examples(df, bad, ["location", "reading_date", "pm25", "pm10"]))


def _rule_feature_columns_and_order(df, check, layer, ctx):
    contract_feats = ctx["contract_layer"]["feature_columns"]
    config_feats = ctx.get("feature_columns", contract_feats)
    missing = [c for c in contract_feats if c not in df.columns]
    ok = (config_feats == contract_feats) and not missing
    return _result(check, layer, "passed" if ok else "failed",
                   observed={"configured": config_feats, "missing_in_data": missing},
                   threshold=contract_feats,
                   message="" if ok else "Feature list/order differs from the data contract (model input dimension drift)")


def _rule_ox_total(df, check, layer, ctx):
    diff = (df["Ox_total_oxidant"] - (df["NO2"] + df["O3"])).abs()
    bad = diff > 1e-9
    return _result(check, layer, "passed" if not bad.any() else "failed",
                   observed={"violations": int(bad.sum())}, threshold=0,
                   examples=_examples(df, bad, ["station_day_key", "NO2", "O3", "Ox_total_oxidant"]))


def _rule_label_matches_who(df, check, layer, ctx):
    rule = ctx["contract_layer"]["label_rule"]
    expected = np.where(df[ctx["contract_layer"]["target_source_column"]] > rule["threshold"], rule["above"], rule["at_or_below"])
    bad = pd.Series(expected != df[ctx["contract_layer"]["label_column"]].to_numpy(), index=df.index)
    return _result(check, layer, "passed" if not bad.any() else "failed",
                   observed={"mislabelled_rows": int(bad.sum())}, threshold=0,
                   message=f"Air_Quality must be '{rule['above']}' iff PM2.5 > {rule['threshold']} µg/m³ (WHO 24-h guideline)",
                   examples=_examples(df, bad, ["station_day_key", "PM2.5", "Air_Quality"]))


def _rule_no_forbidden_features(df, check, layer, ctx):
    forbidden = set(ctx["contract_layer"]["forbidden_feature_columns"])
    used = set(ctx.get("feature_columns", ctx["contract_layer"]["feature_columns"]))
    leak = sorted(forbidden & used)
    return _result(check, layer, "passed" if not leak else "failed", observed={"leaking_features": leak},
                   threshold=[], message="" if not leak else "Target or target-derived columns used as features (leakage)")


def _rule_class_balance(df, check, layer, ctx):
    shares = df[ctx["contract_layer"]["label_column"]].value_counts(normalize=True)
    minority = float(shares.min()) if len(shares) else 0.0
    limit = float(check.get("min_minority_fraction", 0.0))
    return _result(check, layer, "passed" if minority >= limit else "failed",
                   observed={k: round(float(v), 4) for k, v in shares.items()}, threshold={"min_minority_fraction": limit})


CHECK_TYPES: dict[str, Callable] = {
    "required_columns": _required_columns,
    "parseable": _parseable,
    "not_null": _not_null,
    "unique": _unique,
    "accepted_values": _accepted_values,
    "range": _range,
    "row_count": _row_count,
    "source_checksum": _source_checksum,
}

CUSTOM_RULES: dict[str, Callable] = {
    "at_least_one_pollutant": _rule_at_least_one_pollutant,
    "pm25_not_greater_than_pm10": _rule_pm25_le_pm10,
    "feature_columns_and_order": _rule_feature_columns_and_order,
    "ox_total_equals_no2_plus_o3": _rule_ox_total,
    "label_matches_who_threshold": _rule_label_matches_who,
    "no_forbidden_features": _rule_no_forbidden_features,
    "class_balance_within_bounds": _rule_class_balance,
}


def run_checks(df: pd.DataFrame, layer: str, contract: dict[str, Any], context: dict[str, Any] | None = None,
               only: list[str] | None = None, report_layer: str | None = None) -> list[dict[str, Any]]:
    """Run the contract checks of `layer` on df. `report_layer` relabels results
    (e.g. 'curated@postgres' when re-checking the copy read back from the DB)."""
    ctx = dict(context or {})
    layer_spec = contract["layers"][layer]
    ctx["contract_layer"] = layer_spec
    results = []
    for check in layer_spec.get("checks", []):
        if only and check["id"] not in only:
            continue
        try:
            if check["type"] == "custom":
                fn = CUSTOM_RULES[check["rule"]]
            else:
                fn = CHECK_TYPES[check["type"]]
            results.append(fn(df, check, layer, ctx))
        except KeyError as exc:
            results.append(_result(check, layer, "failed", message=f"Check could not run: missing {exc}"))
    if report_layer:
        for r in results:
            r["layer"] = report_layer
    return results


def split_integrity_checks(train_keys: pd.Series, test_keys: pd.Series, n_features_expected: int,
                           X_train_cols: list[str], X_test_cols: list[str]) -> list[dict[str, Any]]:
    """Predictive-integrity checks evaluated at training time."""
    overlap = sorted(set(train_keys) & set(test_keys))
    out = [
        {"check_id": "split_train_test_disjoint", "layer": "model_split", "category": "predictive_integrity",
         "type": "custom", "severity": "error", "status": "passed" if not overlap else "failed",
         "observed": {"overlapping_keys": len(overlap)}, "threshold": 0, "message": "", "examples": overlap[:5]},
        {"check_id": "split_feature_dimensions_stable", "layer": "model_split", "category": "predictive_integrity",
         "type": "custom", "severity": "error",
         "status": "passed" if (X_train_cols == X_test_cols and len(X_train_cols) == n_features_expected) else "failed",
         "observed": {"train": X_train_cols, "test": X_test_cols}, "threshold": n_features_expected,
         "message": "", "examples": []},
    ]
    return out


def enforce(results: list[dict[str, Any]], layer: str, log=None) -> None:
    failed_errors = [r for r in results if r["status"] == "failed" and r["severity"] == "error"]
    for r in results:
        if log is None:
            continue
        extra = {"event": "dq_check", "check_id": r["check_id"], "check_status": r["status"],
                 "severity": r["severity"], "observed": r["observed"], "threshold": r["threshold"]}
        if r["status"] == "failed":
            (log.error if r["severity"] == "error" else log.warning)("DQ check failed: %s", r["check_id"], extra=extra)
        else:
            log.info("DQ check %s: %s", r["status"], r["check_id"], extra=extra)
    if failed_errors:
        raise DataQualityError(layer, failed_errors)


def write_report(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    layers = {r["layer"] for r in results}
    merged = [r for r in existing if r["layer"] not in layers] + results
    path.write_text(json.dumps(merged, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
