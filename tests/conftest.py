"""Shared fixtures. Integration tests run in an ISOLATED temporary project root
(PIPELINE_HOME=tmp) with a small synthetic source file, so they never touch the
real data/ folder and need no download."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HEADER = ["Country Code", "City", "Location", "Coordinates", "Pollutant", "Source Name", "Unit", "Value",
          "Last Updated", "Country Label"]

# Hand-built rows covering every cleaning rule. Columns follow HEADER.
SYNTHETIC_ROWS = [
    # station A (Japan): PM2.5 reported twice for the same timestamp: ppm FIRST, then µg/m³
    ["JP", "Tokyo", "Station A", "35.68, 139.69", "PM2.5", "net-a", "ppm", "0.02", "2024-03-10T13:30:00+05:30", "Japan"],
    ["JP", "Tokyo", "Station A", "35.68, 139.69", "PM2.5", "net-a", "µg/m³", "22.0", "2024-03-10T13:30:00+05:30", "Japan"],
    ["JP", "Tokyo", "Station A", "35.68, 139.69", "NO2", "net-a", "µg/m³", "30.0", "2024-03-10T13:30:00+05:30", "Japan"],
    ["JP", "Tokyo", "Station A", "35.68, 139.69", "O3", "net-a", "µg/m³", "40.0", "2024-03-10T13:30:00+05:30", "Japan"],
    # station B (Japan): two PM2.5 readings on the same UTC day -> daily mean 12.0
    ["JP", "N/A", "Station B", "35.00, 139.00", "PM2.5", "net-a", "µg/m³", "10.0", "2024-03-10T08:00:00+05:30", "Japan"],
    ["JP", "N/A", "Station B", "35.00, 139.00", "PM2.5", "net-a", "µg/m³", "14.0", "2024-03-10T12:00:00+05:30", "Japan"],
    ["JP", "N/A", "Station B", "35.00, 139.00", "NO2", "net-a", "µg/m³", "5.0", "2024-03-10T08:00:00+05:30", "Japan"],
    ["JP", "N/A", "Station B", "35.00, 139.00", "O3", "net-a", "µg/m³", "60.0", "2024-03-10T08:00:00+05:30", "Japan"],
    ["JP", "N/A", "Station B", "35.00, 139.00", "SO2", "net-a", "µg/m³", "3.0", "2024-03-10T08:00:00+05:30", "Japan"],
    # station C (Chile): negative reading (dropped), out-of-scope pollutant (dropped), missing country label
    ["CL", "Santiago", "Station C", "", "PM2.5", "net-c", "µg/m³", "-4.0", "2024-03-11T10:00:00+00:00", ""],
    ["CL", "Santiago", "Station C", "", "TEMPERATURE", "net-c", "c", "21.0", "2024-03-11T10:00:00+00:00", ""],
    ["CL", "Santiago", "Station C", "", "PM10", "net-c", "µg/m³", "50.0", "2024-03-11T10:00:00+00:00", ""],
]


def write_source(path: Path, rows: list[list[str]], header: list[str] = HEADER) -> str:
    text = "\r\n".join(";".join(r) for r in [header, *rows]) + "\r\n"
    data = b"\xef\xbb\xbf" + text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Isolated project root with synthetic source + matching manifest/contract."""
    root = tmp_path / "proj"
    for d in ("config", "contracts", "sql", "expected"):
        shutil.copytree(REPO / d, root / d)
    src = root / "data" / "source" / "world_air_quality.csv"
    sha = write_source(src, SYNTHETIC_ROWS)
    manifest = yaml.safe_load((root / "contracts" / "source_manifest.yml").read_text(encoding="utf-8"))
    manifest["file"].update({"sha256": sha, "size_bytes": src.stat().st_size, "data_rows": len(SYNTHETIC_ROWS)})
    (root / "contracts" / "source_manifest.yml").write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    contract = yaml.safe_load((root / "contracts" / "data_contract.yml").read_text(encoding="utf-8"))
    for layer in contract["layers"].values():  # baseline row counts do not apply to synthetic data
        layer["checks"] = [c for c in layer["checks"] if not (c["type"] == "row_count" and "equals" in c)]
    (root / "contracts" / "data_contract.yml").write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("PIPELINE_HOME", str(root))
    monkeypatch.setenv("PIPELINE_LAYER_FORMAT", os.environ.get("TEST_LAYER_FORMAT", "csv"))
    for var in ("PIPELINE_SOURCE_PATH", "PIPELINE_BATCH_ROWS", "PIPELINE_MODEL_SEED"):
        monkeypatch.delenv(var, raising=False)
    return root


@pytest.fixture
def cfg(project):
    from pipeline.config import load_config

    return load_config()


@pytest.fixture
def log():
    from pipeline.logging_utils import get_logger

    return get_logger("pytest", "test")


def postgres_available() -> bool:
    from pipeline.config import load_dotenv

    load_dotenv(REPO)  # the isolated test project has no .env; use the repository's
    if not os.environ.get("POSTGRES_PASSWORD"):
        return False
    try:
        from pipeline.config import load_config
        from pipeline import db

        conn = db.connect(load_config())
        conn.close()
        return True
    except Exception:  # noqa: BLE001
        return False
