"""Model-ready feature construction (notebook §3.5 and §4, made reusable)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TO_MODEL_NAMES = {"no2": "NO2", "o3": "O3", "so2": "SO2", "co": "CO", "pm25": "PM2.5"}
ID_COLUMNS = ["station_day_key", "country_label", "city", "location", "reading_date"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Total oxidant (NO2 + O3) and the NO2/O3 traffic-vs-regional-ozone ratio."""
    out = df.copy()
    out["Ox_total_oxidant"] = out["NO2"] + out["O3"]
    out["NO2_O3_ratio"] = out["NO2"] / (out["O3"] + 1.0)
    return out


def who_label(pm25: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(np.where(pm25 > threshold, "Unhealthy", "Safe"), index=pm25.index)


def impute_with_flags(df: pd.DataFrame, columns: list[str], medians: dict[str, float] | None = None
                      ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Median-impute `columns` and add `<col>_missing` flags. If `medians` is None
    they are computed from df (the notebook's behaviour)."""
    out = df.copy()
    used: dict[str, float] = {}
    for col in columns:
        out[f"{col}_missing"] = out[col].isna().astype(int)
        value = float(out[col].median()) if medians is None else float(medians[col])
        out[col] = out[col].fillna(value)
        used[col] = value
    return out, used


def build_model_features(curated: pd.DataFrame, mcfg: dict) -> tuple[pd.DataFrame, dict]:
    """Curated station-days -> model-ready dataset.

    Keeps curated row order (= notebook order), which the stratified split
    depends on. SO2_raw / CO_raw keep the un-imputed values so a stricter
    train-only imputation can be applied later (model.imputation_fit_scope=train).
    """
    required = [c.lower().replace(".", "") for c in mcfg["required_predictors"]]  # PM2.5 -> pm25
    m = curated.dropna(subset=required).copy()
    m = m.rename(columns=TO_MODEL_NAMES)
    for col in mcfg["impute_columns"]:
        m[f"{col}_raw"] = m[col]
    m, medians = impute_with_flags(m, mcfg["impute_columns"])
    m = add_engineered_features(m)
    m["Air_Quality"] = who_label(m["PM2.5"], mcfg["who_threshold"])
    feats = mcfg["feature_columns"]
    raw_cols = [f"{c}_raw" for c in mcfg["impute_columns"]]
    m = m[ID_COLUMNS + feats + ["PM2.5", "Air_Quality"] + raw_cols].reset_index(drop=True)
    info = {"full_data_medians": medians, "rows": len(m),
            "class_counts": {k: int(v) for k, v in m["Air_Quality"].value_counts().sort_index().items()}}
    return m, info
