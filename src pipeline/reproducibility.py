"""Machine-readable comparison of reproduced vs baseline results.

Rules (declared in config/pipeline.yml -> reproducibility, and in the report):
* Row counts at every layer, class balance and train/test sizes must match EXACTLY.
* The test-set membership fingerprint must match EXACTLY (same logical rows).
* Every model metric must satisfy |reproduced - baseline| <= abs_tolerance (0.01).
  Primary metric: held-out weighted F1. Secondary: accuracy, weighted
  precision/recall, 5-fold CV F1 mean.
Outcome classes follow the lab's interpretation table.
"""
from __future__ import annotations

from typing import Any

METRIC_PATHS = [
    ("test", "f1_weighted", "primary"),
    ("test", "accuracy", "secondary"),
    ("test", "precision_weighted", "secondary"),
    ("test", "recall_weighted", "secondary"),
    ("cv_f1_weighted", "mean", "secondary"),
]
# Baseline values were printed with 6 decimals; differences below this are rounding.
PRINT_PRECISION = 5e-7


def _classify(diff: float, tol: float) -> str:
    if abs(diff) <= PRINT_PRECISION:
        return "exact_match"
    if abs(diff) > tol:
        return "materially_different"
    return "better_within_tolerance" if diff > 0 else "worse_within_tolerance"


def compare(metrics: dict[str, Any], baseline: dict[str, Any], counts: dict[str, Any], tol: float,
            source_complete: bool = True) -> dict[str, Any]:
    hist = baseline["historical"]
    comparison: dict[str, Any] = {
        "tolerance_rule": f"|reproduced - baseline| <= {tol} for every model metric; counts and split membership exact",
        "abs_tolerance": tol,
        "seed": metrics["seed"],
        "imputation_fit_scope": metrics["imputation_fit_scope"],
        "metric_rows": [],
        "count_rows": [],
        "split_rows": [],
    }
    same_config = metrics["seed"] == baseline["seed"] and metrics["imputation_fit_scope"] == baseline["imputation_fit_scope"]
    applicable = same_config and source_complete
    comparison["applicable"] = applicable
    comparison["source_complete"] = source_complete
    if not source_complete:
        comparison["status"] = "not_applicable"
        comparison["outcome"] = ("Raw layer holds only part of the source (incremental batch in progress); the "
                                 "baseline describes the complete file, so the comparison is informational only.")
    elif not same_config:
        comparison["status"] = "not_applicable"
        comparison["outcome"] = ("Run configuration differs from the baseline (seed/imputation scope); "
                                 "metrics are reported for sensitivity analysis only.")

    # ---- counts (exact)
    for key, expected in hist["counts"].items():
        observed = counts.get(key) if key not in ("train", "test") else metrics["counts"].get(key)
        comparison["count_rows"].append({"item": key, "baseline": expected, "reproduced": observed,
                                         "match": observed == expected})
    for label, expected in hist["class_balance"].items():
        observed = metrics["class_balance"].get(label)
        comparison["count_rows"].append({"item": f"class_{label}", "baseline": expected, "reproduced": observed,
                                         "match": observed == expected})

    # ---- split membership (exact)
    for key in ("train_keys_sha256", "test_keys_sha256"):
        exp, obs = baseline["split_fingerprint"][key], metrics["split"][key]
        comparison["split_rows"].append({"item": key, "baseline": exp, "reproduced": obs, "match": exp == obs})

    # ---- metrics (tolerance)
    for model, hm in hist["models"].items():
        rm = metrics["models"].get(model)
        for section, name, role in METRIC_PATHS:
            b = hm.get(section, {}).get(name)
            if b is None:
                continue
            r = None if rm is None else rm[section][name]
            diff = None if r is None else r - b
            comparison["metric_rows"].append({
                "model": model, "metric": f"{section}.{name}", "role": role,
                "baseline": b, "reproduced": None if r is None else round(r, 6),
                "abs_diff": None if diff is None else round(abs(diff), 6),
                "within_tolerance": diff is not None and abs(diff) <= tol,
                "classification": "missing" if diff is None else _classify(diff, tol),
            })
    # paper's confusion-matrix statement: correctly predicted Safe cases
    for model, expected in baseline.get("paper_reported", {}).get("true_safe_correct", {}).items():
        observed = metrics["models"][model]["test"]["confusion_matrix"]["matrix"][0][0]
        comparison["count_rows"].append({"item": f"{model}_true_safe_correct", "baseline": expected,
                                         "reproduced": observed, "match": observed == expected})
    # best model selection
    comparison["best_model_by_cv_f1"] = {"baseline": hist.get("best_model_by_cv_f1"),
                                        "reproduced": metrics.get("best_model_by_cv_f1"),
                                        "match": hist.get("best_model_by_cv_f1") == metrics.get("best_model_by_cv_f1")}

    counts_ok = all(r["match"] for r in comparison["count_rows"])
    split_ok = all(r["match"] for r in comparison["split_rows"])
    metrics_ok = all(r["within_tolerance"] for r in comparison["metric_rows"])
    nonzero = [r for r in comparison["metric_rows"] if r["classification"] != "exact_match"]
    comparison["summary"] = {
        "counts_exact": counts_ok, "split_membership_exact": split_ok, "metrics_within_tolerance": metrics_ok,
        "metrics_exact": not nonzero, "n_metrics": len(comparison["metric_rows"]),
        "n_nonzero_differences": len(nonzero),
        "max_abs_diff": max((r["abs_diff"] or 0) for r in comparison["metric_rows"]) if comparison["metric_rows"] else None,
        "nonzero_differences": [{k: r[k] for k in ("model", "metric", "baseline", "reproduced", "abs_diff")} for r in nonzero],
    }
    if applicable:
        if counts_ok and split_ok and metrics_ok:
            comparison["status"] = "pass"
            comparison["outcome"] = ("Matches within tolerance (all metrics exact)" if not nonzero else
                                     f"Matches within tolerance ({len(nonzero)} non-zero difference(s), explained in "
                                     "docs/reproducibility_report.md)")
        else:
            comparison["status"] = "fail"
            comparison["outcome"] = "Materially different — reproduction failed until explained/corrected"
    return comparison
