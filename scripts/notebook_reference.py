"""Build expected/baseline_metrics.json from the ARCHIVED original notebook.

Two independent sources, both derived programmatically (nothing is retyped):

1. `historical`  — parsed from the outputs SAVED in
   notebooks/original/Predictive_Final_Project_code.ipynb (what the group
   project actually printed and published in the paper).
2. `reference_execution` — the notebook's code cells executed VERBATIM, headless,
   in the current environment. This yields the split fingerprint (which exact
   station-days formed the 526-row test set), which the saved outputs cannot.

Usage:
    python scripts/notebook_reference.py            # writes expected/baseline_metrics.json
Requires: the source CSV at data/source/ and seaborn (requirements-dev.txt).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "original" / "Predictive_Final_Project_code.ipynb"
SOURCE = ROOT / "data" / "source" / "world_air_quality.csv"
OUT = ROOT / "expected" / "baseline_metrics.json"
MODEL_KEYS = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest",
              "Gradient Boosting": "gradient_boosting"}


def _outputs_text(cell: dict) -> str:
    chunks = []
    for out in cell.get("outputs", []):
        text = out.get("text") or out.get("data", {}).get("text/plain") or ""
        chunks.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(chunks)


def extract_historical(nb: dict) -> dict:
    text = "\n".join(_outputs_text(c) for c in nb["cells"] if c["cell_type"] == "code")

    def grab(pattern: str, cast=int):
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"pattern not found in saved outputs: {pattern}")
        return cast(m.group(1).replace(",", ""))

    hist = {
        "counts": {
            "source": grab(r"\((\d+), 10\)"),
            "staging": grab(r"Final Cleaned Rows: (\d+)"),
            "curated": grab(r"Shape after pivoting: (\d+) rows"),
            "model_features": grab(r"Total rows available for machine learning: (\d+)"),
            "train": grab(r"Training Data: (\d+) rows"),
            "test": grab(r"Testing Data: (\d+) rows"),
        },
        "class_balance": {"Unhealthy": grab(r"Unhealthy\s+(\d+)\n"), "Safe": grab(r"Safe\s+(\d+)\n")},
        "models": {},
    }
    num = r"([0-9.]+)"
    for display, key in MODEL_KEYS.items():
        m = re.search(rf"{display}\s+{num}\s+{num}\s+{num}\s+{num}\n", text)
        acc, prec, rec, f1 = (float(x) for x in m.groups())
        cv = re.search(rf"{display}\s+{num}\s+{num}\s*(?:\n|$)", text.split("5-FOLD CROSS-VALIDATION")[1])
        hist["models"][key] = {
            "display_name": display,
            "test": {"accuracy": acc, "precision_weighted": prec, "recall_weighted": rec, "f1_weighted": f1},
            "cv_f1_weighted": {"mean": float(cv.group(1)), "std": float(cv.group(2))},
        }
    imp_block = text.split("KEY FEATURES INFLUENCING CLASSIFICATION (Random Forest)")[1]
    hist["models"]["random_forest"]["feature_importances"] = {
        f: float(v) for f, v in re.findall(r"^\s*([A-Za-z0-9_]+)\s+([0-9.]+)\s*$", imp_block, flags=re.M)
    }
    best = re.search(r"Best model by 5-fold CV F1-weighted: ([A-Za-z ]+) \(", text).group(1).strip()
    hist["best_model_by_cv_f1"] = MODEL_KEYS[best]
    hist["python_version_recorded_in_notebook"] = nb.get("metadata", {}).get("language_info", {}).get("version")
    return hist


def execute_verbatim(nb: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *a, **k: None  # headless
    workdir = Path(tempfile.mkdtemp(prefix="nbref_"))
    shutil.copy(SOURCE, workdir / "world_air_quality.csv")
    cwd = os.getcwd()
    ns: dict = {}
    try:
        os.chdir(workdir)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), ns)  # noqa: S102
                    plt.close("all")
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)

    mr = ns["df_model_ready"]

    def key(row) -> str:
        s = "\x1f".join(str(row[c]) for c in ("Country Label", "City", "Location", "Date"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

    def fp(index) -> str:
        keys = sorted(mr.loc[index].apply(key, axis=1))
        return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()

    res = ns["results_df"].set_index("Model")
    cvd = ns["cv_df"].set_index("Model")
    import numpy as np
    import pandas as pd
    import sklearn

    return {
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                        "scikit-learn": sklearn.__version__},
        "split": {"train_keys_sha256": fp(ns["X_train_c"].index), "test_keys_sha256": fp(ns["X_test_c"].index)},
        "models": {
            MODEL_KEYS[m]: {
                "test": {"accuracy": float(res.loc[m, "Accuracy"]), "f1_weighted": float(res.loc[m, "F1-Score"])},
                "cv_f1_weighted": {"mean": float(cvd.loc[m, "CV F1 Mean"]), "std": float(cvd.loc[m, "CV F1 Std"])},
            }
            for m in MODEL_KEYS
        },
    }


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    hist = extract_historical(nb)
    ref = execute_verbatim(nb)
    baseline = {
        "description": "Baseline for reproducibility_check. Values under `historical` are parsed from the outputs "
                       "saved in the archived notebook (= the paper's Tables I and II). The split fingerprint comes "
                       "from executing the archived notebook verbatim (scripts/notebook_reference.py).",
        "source_notebook": NOTEBOOK.relative_to(ROOT).as_posix(),
        "source_notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "seed": 42,
        "imputation_fit_scope": "full",
        "primary_metric": "test.f1_weighted",
        "secondary_metrics": ["test.accuracy", "test.precision_weighted", "test.recall_weighted", "cv_f1_weighted.mean"],
        "paper_reported": {
            "true_safe_correct": {"logistic_regression": 98, "random_forest": 146, "gradient_boosting": 141},
            "note": "Paper §IV-B: correctly predicted Safe cases (confusion-matrix cell [Safe, Safe]) out of 223.",
        },
        "historical": hist,
        "split_fingerprint": ref["split"],
        "reference_execution": ref,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(json.dumps({"split": ref["split"], "env": ref["environment"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
