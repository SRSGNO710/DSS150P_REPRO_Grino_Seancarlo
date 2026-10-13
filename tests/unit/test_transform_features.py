"""Unit tests for reusable transformation / feature functions."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from conftest import HEADER, SYNTHETIC_ROWS
from pipeline.config import load_config
from pipeline.features import add_engineered_features, build_model_features, impute_with_flags, who_label
from pipeline.transform import apply_implicit_na, clean_readings, parse_coordinates, pivot_station_day


def _raw_frame(rows=SYNTHETIC_ROWS) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=HEADER).astype(object)
    df.insert(0, "_source_row_number", range(1, len(df) + 1))
    df["_ingest_batch_id"] = "b-test"
    return df


@pytest.fixture(scope="module")
def tcfg():
    return load_config()["transform"]


def test_implicit_na_matches_pandas_defaults():
    df = pd.DataFrame({"City": ["N/A", "None", "NA", "", "Tokyo", "n/a", "Nancy"]})
    out = apply_implicit_na(df, ["City"], load_config()["transform"]["implicit_na_strings"])
    assert out["City"].isna().tolist() == [True, True, True, True, False, True, False]


def test_dedup_happens_before_unit_filter_and_keeps_first(tcfg):
    staging, steps = clean_readings(_raw_frame(), tcfg)
    a_pm25 = staging[(staging["location"] == "Station A") & (staging["pollutant"] == "PM2.5")]
    # The ppm row came first, so the µg/m³ twin is removed by de-dup and the ppm row by the unit filter:
    # exactly the notebook's behaviour (no PM2.5 survives for Station A).
    assert a_pm25.empty
    assert steps["after_pollutant_filter"] == 11 and steps["after_dedup"] == 10


def test_cleaning_rules(tcfg):
    staging, _ = clean_readings(_raw_frame(), tcfg)
    assert (staging["value"] >= 0).all(), "negative readings must be removed"
    assert set(staging["pollutant"]) <= {"PM2.5", "PM10", "NO2", "O3", "SO2", "CO"}
    assert set(staging["unit"]) == {"µg/m³"}
    assert "Unknown" in set(staging["country_label"]), "missing country label imputed"
    assert set(staging.loc[staging["location"] == "Station B", "city"]) == {"Unknown"}, "'N/A' city -> Unknown"
    assert staging["reading_key"].is_unique
    assert str(staging["last_updated_utc"].dt.tz) == "UTC"


def test_parse_coordinates_handles_missing():
    lat, lon = parse_coordinates(pd.Series(["33.88, 130.87", "", None]))
    assert lat.iloc[0] == pytest.approx(33.88) and lon.iloc[0] == pytest.approx(130.87)
    assert math.isnan(lat.iloc[1]) and math.isnan(lon.iloc[2])


def test_pivot_station_day_daily_mean_and_order(tcfg):
    staging, _ = clean_readings(_raw_frame(), tcfg)
    cur = pivot_station_day(staging)
    b = cur[cur["location"] == "Station B"].iloc[0]
    assert b["pm25"] == pytest.approx(12.0) and b["n_readings"] == 5
    assert list(cur["location"]) == sorted(cur["location"]), "sorted by location like the notebook"
    assert cur["station_day_key"].is_unique
    assert list(cur.columns[:5]) == ["station_day_key", "country_label", "city", "location", "reading_date"]


def test_who_label_boundary_is_exclusive():
    labels = who_label(pd.Series([14.99, 15.0, 15.01]), 15.0)
    assert labels.tolist() == ["Safe", "Safe", "Unhealthy"]


def test_engineered_features():
    out = add_engineered_features(pd.DataFrame({"NO2": [30.0], "O3": [59.0]}))
    assert out["Ox_total_oxidant"].iloc[0] == 89.0
    assert out["NO2_O3_ratio"].iloc[0] == pytest.approx(0.5)


def test_impute_with_flags_uses_given_or_own_median():
    df = pd.DataFrame({"SO2": [1.0, None, 5.0]})
    out, med = impute_with_flags(df, ["SO2"])
    assert med["SO2"] == 3.0 and out["SO2"].tolist() == [1.0, 3.0, 5.0] and out["SO2_missing"].tolist() == [0, 1, 0]
    out2, _ = impute_with_flags(df, ["SO2"], medians={"SO2": 9.0})
    assert out2["SO2"].tolist() == [1.0, 9.0, 5.0]


def test_model_features_exclude_pm10_and_follow_contract_order(tcfg):
    mcfg = load_config()["model"]
    staging, _ = clean_readings(_raw_frame(), tcfg)
    mf, info = build_model_features(pivot_station_day(staging), mcfg)
    assert "PM10" not in mf.columns and "pm10" not in mf.columns
    assert list(mf.columns[5:5 + len(mcfg["feature_columns"])]) == mcfg["feature_columns"]
    assert info["rows"] == 1  # only Station B has PM2.5, NO2 and O3 together
