"""Partitioned curated layer + independent read/load of one partition (A11).

The curated station-day table is written as Hive-style partitions
    data/curated/partitions/country_label=<slug>/part-0.<ext>
with an index file mapping slug -> country label and row count. A single
partition can then be read without touching any other file, and loaded into
PostgreSQL with a scope-limited merge (only that country's rows are synchronised).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .layer_io import content_fingerprint, read_table, write_table


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return slug or "unknown"


def partitions_root(cfg: Config) -> Path:
    return cfg.path("curated_dir") / "partitions"


def write_partitions(cfg: Config, curated: pd.DataFrame, column: str) -> dict[str, Any]:
    root = partitions_root(cfg)
    tmp = root.with_name(".partitions_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    index: dict[str, Any] = {"partition_column": column, "partitions": {}}
    for value, part in curated.groupby(column, sort=True):
        slug = slugify(value)
        if slug in index["partitions"]:  # disambiguate slug collisions deterministically
            slug = f"{slug}_{len(index['partitions'])}"
        write_table(part.reset_index(drop=True), tmp / f"{column}={slug}" / "part-0", cfg.layer_format)
        index["partitions"][slug] = {"value": value, "rows": int(len(part))}
    (tmp / "_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
    return {"partitions": len(index["partitions"]), "rows": int(len(curated)), "root": "data/curated/partitions"}


def read_partition(cfg: Config, value: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = partitions_root(cfg)
    index = json.loads((root / "_index.json").read_text(encoding="utf-8"))
    col = index["partition_column"]
    match = [s for s, meta in index["partitions"].items() if meta["value"] == value or s == value]
    if not match:
        raise KeyError(f"No partition for {col}={value!r}. Available: {sorted(m['value'] for m in index['partitions'].values())[:15]}...")
    slug = match[0]
    df = read_table(root / f"{col}={slug}" / "part-0", cfg.layer_format)
    return df, {"partition_column": col, "slug": slug, "value": index["partitions"][slug]["value"],
                "files_read": 1, "rows": len(df), "total_partitions": len(index["partitions"])}


def verify_partition(full: pd.DataFrame, part: pd.DataFrame, column: str, value: str, fp_cols: list[str]) -> dict[str, Any]:
    expected = full[full[column] == value]
    fp_full = content_fingerprint(expected, fp_cols, ["station_day_key"])
    fp_part = content_fingerprint(part, fp_cols, ["station_day_key"])
    return {"rows_expected": int(len(expected)), "rows_in_partition": int(len(part)), "identical": fp_full == fp_part}
