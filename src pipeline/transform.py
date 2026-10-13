"""Staging and curated transformations (pure functions + thin stage wrappers).

STAGING  = row-level conformance: NA semantics, scope filter, typing, de-dup,
           unit standardisation, invalid-value removal, label imputation.
           One row per valid reading; grain = (location, pollutant, last_updated_utc).
CURATED  = business-level reshaping: daily station aggregation and pivot to one
           row per station-day — the reusable data product.

The rules and their ORDER replicate notebook §3.2–§3.3 exactly (e.g. de-dup
happens before the unit filter), because the order changes which rows survive.
"""
from __future__ import annotations

import pandas as pd

from .layer_io import sha256_text_rows

POLLUTANT_RENAME = {"CO": "co", "NO2": "no2", "O3": "o3", "PM10": "pm10", "PM2.5": "pm25", "SO2": "so2"}
CURATED_INDEX = ["country_label", "city", "location", "reading_date"]
CURATED_POLLUTANTS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
STAGING_COLUMNS = [
    "reading_key", "country_code", "country_label", "city", "location", "latitude", "longitude",
    "pollutant", "source_name", "unit", "value", "last_updated_utc", "reading_date",
    "_source_row_number", "_ingest_batch_id",
]


def apply_implicit_na(df: pd.DataFrame, columns: list[str], na_strings: list[str]) -> pd.DataFrame:
    """Turn the strings pandas' read_csv treats as missing by default into NaN.

    The raw layer keeps them verbatim; making the conversion explicit here keeps
    the notebook's behaviour (e.g. City == 'N/A' -> missing -> 'Unknown') while
    documenting it and making it independent of pandas defaults.
    """
    out = df.copy()
    na = set(na_strings)
    for col in columns:
        if col in out.columns:
            s = out[col]
            out[col] = s.where(~(s.isna() | s.isin(na)), None)
    return out


def parse_coordinates(coords: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = coords.fillna("").astype(str).str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        parts[1] = None
    lat = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    lon = pd.to_numeric(parts[1].str.strip(), errors="coerce")
    return lat, lon


def clean_readings(raw: pd.DataFrame, tcfg: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Raw (verbatim strings) -> staging readings. Returns (frame, step row counts)."""
    steps: dict[str, int] = {"raw_rows": len(raw)}
    df = raw.sort_values("_source_row_number", kind="mergesort").copy()
    text_cols = ["Country Code", "City", "Location", "Coordinates", "Pollutant", "Source Name",
                 "Unit", "Last Updated", "Country Label"]
    df = apply_implicit_na(df, text_cols, tcfg["implicit_na_strings"])
    df["Value"] = pd.to_numeric(df["Value"], errors="raise").astype("float64")

    # 1. scope: the six study pollutants
    df = df[df["Pollutant"].isin(tcfg["pollutants"])].copy()
    steps["after_pollutant_filter"] = len(df)
    # 2. typing: timestamps to UTC
    df["last_updated_utc"] = pd.to_datetime(df["Last Updated"], utc=True, format="ISO8601")
    # 3. de-duplicate on the sensor key, keeping the first reading in source order
    key = ["Location", "Pollutant", "last_updated_utc"]
    df = df.drop_duplicates(subset=key, keep=tcfg.get("dedup_keep", "first"))
    steps["after_dedup"] = len(df)
    # 4. unit standardisation (µg/m³ only)
    df = df[df["Unit"] == tcfg["allowed_unit"]]
    steps["after_unit_filter"] = len(df)
    # 5. invalid (negative) concentrations
    df = df[df["Value"] >= tcfg["min_value"]].copy()
    steps["after_value_filter"] = len(df)
    # 6. impute missing geographic labels (before any pivot, which would drop NaN keys)
    fill = tcfg["geo_fill_value"]
    for col in ("City", "Location", "Country Label"):
        df[col] = df[col].fillna(fill)

    lat, lon = parse_coordinates(df["Coordinates"])
    out = pd.DataFrame({
        "country_code": df["Country Code"],
        "country_label": df["Country Label"],
        "city": df["City"],
        "location": df["Location"],
        "latitude": lat,
        "longitude": lon,
        "pollutant": df["Pollutant"],
        "source_name": df["Source Name"],
        "unit": df["Unit"],
        "value": df["Value"],
        "last_updated_utc": df["last_updated_utc"],
        "reading_date": df["last_updated_utc"].dt.date,
        "_source_row_number": df["_source_row_number"].astype("int64"),
        "_ingest_batch_id": df["_ingest_batch_id"],
    })
    ts_iso = out["last_updated_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.insert(0, "reading_key", sha256_text_rows(out.assign(_ts=ts_iso), ["location", "pollutant", "_ts"]).str[:32])
    out = out.reset_index(drop=True)
    steps["staging_rows"] = len(out)
    return out[STAGING_COLUMNS], steps


def pivot_station_day(staging: pd.DataFrame) -> pd.DataFrame:
    """Staging readings -> curated station-day table (daily mean per pollutant)."""
    wide = staging.pivot_table(index=CURATED_INDEX, columns="pollutant", values="value", aggfunc="mean").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns=POLLUTANT_RENAME)
    for col in CURATED_POLLUTANTS:
        if col not in wide.columns:
            wide[col] = float("nan")
    counts = staging.groupby(CURATED_INDEX).size().rename("n_readings").reset_index()
    wide = wide.merge(counts, on=CURATED_INDEX, how="left", validate="one_to_one")
    # Same row order as the notebook: sort by (Location, Date); pandas' multi-key
    # sort is stable, so ties keep the pivot's (Country Label, City) order.
    wide = wide.sort_values(["location", "reading_date", "country_label", "city"], kind="mergesort").reset_index(drop=True)
    wide.insert(0, "station_day_key", sha256_text_rows(wide, CURATED_INDEX).str[:32])
    wide["n_readings"] = wide["n_readings"].astype("int64")
    return wide[["station_day_key", *CURATED_INDEX, *CURATED_POLLUTANTS, "n_readings"]]
