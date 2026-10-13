"""Reading/writing data layers in a format-agnostic, type-preserving way.

Parquet (default) preserves types natively. The CSV fallback writes a small
`<name>.dtypes.json` sidecar so that reading a layer back yields exactly the
same logical frame (same dtypes, NA semantics and values) in either format.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _kind(series: pd.Series) -> str:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return "datetime_utc"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    non_null = series.dropna()
    if len(non_null) and isinstance(non_null.iloc[0], dt.date) and not isinstance(non_null.iloc[0], dt.datetime):
        return "date"
    return "string"


def layer_path(base: Path, fmt: str) -> Path:
    return base.with_suffix(".parquet" if fmt == "parquet" else ".csv")


def write_table(df: pd.DataFrame, base: Path, fmt: str) -> Path:
    """Write df to `base` + extension atomically. Returns the data file path."""
    base.parent.mkdir(parents=True, exist_ok=True)
    target = layer_path(base, fmt)
    tmp = target.with_name(f".tmp_{target.name}")
    if fmt == "parquet":
        df.to_parquet(tmp, index=False, engine="pyarrow")
    elif fmt == "csv":
        df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
        kinds = {c: _kind(df[c]) for c in df.columns}
        sidecar = base.with_suffix(".dtypes.json")
        sidecar.write_text(json.dumps(kinds, indent=1, ensure_ascii=False), encoding="utf-8")
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported layer format {fmt!r}")
    os.replace(tmp, target)
    return target


def read_table(base: Path, fmt: str, preserve_empty_strings: bool = False) -> pd.DataFrame:
    """Read a layer back. `preserve_empty_strings=True` keeps '' as '' in string
    columns (used for the raw layer, which stores source fields verbatim)."""
    target = layer_path(base, fmt)
    if not target.exists():
        raise FileNotFoundError(f"Layer file not found: {target}")
    if fmt == "parquet":
        return pd.read_parquet(target, engine="pyarrow")
    kinds: dict[str, str] = json.loads(base.with_suffix(".dtypes.json").read_text(encoding="utf-8"))
    string_cols = [c for c, k in kinds.items() if k == "string"]
    df = pd.read_csv(
        target,
        dtype={c: object for c in string_cols},
        keep_default_na=False,
        na_values={c: [""] for c in kinds if kinds[c] != "string"},
        encoding="utf-8",
    )
    for col, kind in kinds.items():
        if kind == "string":
            if not preserve_empty_strings:
                # Empty string in a string column is a NULL written by to_csv.
                df[col] = df[col].astype(object).where(df[col] != "", None)
        elif kind == "datetime_utc":
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
        elif kind == "datetime":
            df[col] = pd.to_datetime(df[col], format="ISO8601")
        elif kind == "date":
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d").dt.date
        elif kind == "int":
            df[col] = df[col].astype("int64")
        elif kind == "float":
            df[col] = df[col].astype("float64")
        elif kind == "bool":
            df[col] = df[col].astype(bool)
    return df


def content_fingerprint(df: pd.DataFrame, columns: list[str], sort_by: list[str]) -> str:
    """sha256 of the business content of a frame, independent of audit columns,
    row order and storage format. Floats are rendered with 10 significant digits."""
    view = df[columns].sort_values(sort_by, kind="mergesort").reset_index(drop=True).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else format(float(v), ".10g"))
        elif isinstance(view[col].dtype, pd.DatetimeTZDtype):
            view[col] = view[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            view[col] = view[col].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    buf = io.StringIO()
    view.to_csv(buf, index=False, lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def sha256_text_rows(frame: pd.DataFrame, columns: list[str], sep: str = "\x1f") -> pd.Series:
    """Row-wise sha256 over string-joined columns (used for record/row hashes and keys)."""
    if not len(frame):
        return pd.Series([], dtype=object, index=frame.index)
    parts = [frame[c].astype(str) for c in columns]
    joined = parts[0].str.cat(parts[1:], sep=sep) if len(parts) > 1 else parts[0]
    return joined.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
