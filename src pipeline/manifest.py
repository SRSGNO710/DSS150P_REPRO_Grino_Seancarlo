"""Source identity verification against contracts/source_manifest.yml."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import Config, load_yaml
from .errors import SourceIdentityError


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(cfg: Config) -> dict[str, Any]:
    return load_yaml(cfg.path("source_manifest"))


def inspect_source(path: Path) -> dict[str, Any]:
    """Return observed identity facts about a candidate source file."""
    if not path.exists():
        raise SourceIdentityError(
            f"Source file not found at {path}. Obtain world_air_quality.csv (see README §Data) "
            f"and place it at data/source/world_air_quality.csv"
        )
    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.count(b"\n")
    if data and not data.endswith(b"\n"):
        lines += 1
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "data_rows": max(lines - 1, 0),  # minus header
        "has_bom": data.startswith(b"\xef\xbb\xbf"),
    }


def compare_to_manifest(observed: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest["file"]
    checks = {
        "sha256": (observed["sha256"], expected["sha256"]),
        "size_bytes": (observed["size_bytes"], expected["size_bytes"]),
        "data_rows": (observed["data_rows"], expected["data_rows"]),
    }
    mismatches = {k: {"observed": o, "expected": e} for k, (o, e) in checks.items() if o != e}
    return {"matches": not mismatches, "mismatches": mismatches, "observed": observed}
