"""Deterministic model training and evaluation (notebook §4–§5, made reusable).

Every source of randomness is pinned to `model.seed` (default 42 = the
notebook's random_state): the stratified split, RandomForest, GradientBoosting
and the CV fold shuffling. LogisticRegression (lbfgs) is deterministic.
"""
from __future__ import annotations

import hashlib
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["Safe", "Unhealthy"]
MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


class ColumnMedianImputer(BaseEstimator, TransformerMixin):
    """Median-impute selected columns using medians learned in fit() only.

    Used when model.imputation_fit_scope = 'train' so that imputation
    statistics never see test (or held-out CV fold) rows.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X[self.columns].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.columns] = X[self.columns].fillna(self.medians_)
        return X


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__}


def keys_fingerprint(keys: pd.Series) -> str:
    """Order-independent fingerprint of a set of station-day keys."""
    return hashlib.sha256("\n".join(sorted(map(str, keys))).encode("utf-8")).hexdigest()


def build_estimators(mcfg: dict, seed: int) -> dict[str, Any]:
    estimators = {}
    for name, spec in mcfg["models"].items():
        params = dict(spec.get("params", {}))
        if spec.get("seeded"):
            params["random_state"] = seed
        est = MODEL_CLASSES[spec["class"]](**params)
        if mcfg["imputation_fit_scope"] == "train":
            est = Pipeline([("impute", ColumnMedianImputer(columns=list(mcfg["impute_columns"]))), ("model", est)])
        estimators[name] = est
    return estimators


def design_matrix(mf: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    """Feature matrix in contract order. With train-scope imputation, SO2/CO are
    taken un-imputed (NaN) and imputed inside each fitted pipeline."""
    X = mf[mcfg["feature_columns"]].copy()
    if mcfg["imputation_fit_scope"] == "train":
        for col in mcfg["impute_columns"]:
            X[col] = mf[f"{col}_raw"]
    return X


def stratified_split(mf: pd.DataFrame, mcfg: dict, seed: int) -> tuple[pd.Index, pd.Index]:
    idx_train, idx_test = train_test_split(
        mf.index, test_size=mcfg["test_size"], random_state=seed, stratify=mf["Air_Quality"], shuffle=True,
    )
    return pd.Index(idx_train), pd.Index(idx_test)


def fit_models(X_train: pd.DataFrame, y_train: pd.Series, mcfg: dict, seed: int) -> dict[str, Any]:
    fitted = {}
    for name, est in build_estimators(mcfg, seed).items():
        fitted[name] = est.fit(X_train, y_train)
    return fitted


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {"labels": LABELS, "rows_are": "actual", "matrix": cm.tolist()},
    }


def cross_validate(estimator, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(estimator), X, y, cv=cv, scoring="f1_weighted")
    return {"mean": float(scores.mean()), "std": float(scores.std()), "folds": [float(s) for s in scores]}


def feature_importances(model, features: list[str]) -> dict[str, float] | None:
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if not hasattr(est, "feature_importances_"):
        return None
    pairs = sorted(zip(features, est.feature_importances_), key=lambda p: -p[1])
    return {f: float(v) for f, v in pairs}


def evaluate(models: dict[str, Any], mf: pd.DataFrame, X: pd.DataFrame, idx_train: pd.Index,
             idx_test: pd.Index, mcfg: dict, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (metrics document, predictions frame)."""
    y = mf["Air_Quality"]
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]
    per_model: dict[str, Any] = {}
    preds = mf.loc[idx_test, ["station_day_key", "country_label", "city", "location", "reading_date", "PM2.5", "Air_Quality"]].copy()
    preds = preds.rename(columns={"PM2.5": "pm25_actual", "Air_Quality": "label_actual"})
    for name, model in models.items():
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, list(model.classes_).index("Unhealthy")]
        per_model[name] = {
            "display_name": DISPLAY_NAMES.get(name, name),
            "test": classification_metrics(y_test, y_pred),
            "cv_f1_weighted": cross_validate(model, X, y, mcfg["cv_folds"], seed),
            "feature_importances": feature_importances(model, list(X.columns)),
        }
        preds[f"pred_{name}"] = y_pred
        preds[f"proba_unhealthy_{name}"] = np.round(proba, 6)
    best = max(per_model, key=lambda n: per_model[n]["cv_f1_weighted"]["mean"])
    deployed = mcfg.get("deployed_model", best)
    preds["pred_deployed"] = preds[f"pred_{deployed}"]
    preds["split"] = "test"
    metrics = {
        "seed": seed,
        "imputation_fit_scope": mcfg["imputation_fit_scope"],
        "features": list(X.columns),
        "counts": {"model_features": int(len(mf)), "train": int(len(idx_train)), "test": int(len(idx_test)),
                   "predictions": int(len(preds))},
        "class_balance": {k: int(v) for k, v in y.value_counts().sort_index().items()},
        "split": {
            "test_size": mcfg["test_size"], "stratified": True, "shuffle": True,
            "train_keys_sha256": keys_fingerprint(mf.loc[idx_train, "station_day_key"]),
            "test_keys_sha256": keys_fingerprint(mf.loc[idx_test, "station_day_key"]),
        },
        "models": per_model,
        "best_model_by_cv_f1": best,
        "deployed_model": deployed,
        "library_versions": library_versions(),
    }
    return metrics, preds.reset_index(drop=True)
