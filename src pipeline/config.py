"""Configuration loading.

All file paths in config/pipeline.yml are relative to the project root, so the
repository can be cloned anywhere (acceptance test A3: no machine-specific
absolute paths). The project root is $PIPELINE_HOME when set (containers use
/opt/project), otherwise the directory two levels above this file.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

# env var -> (section, key, caster)
ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "PIPELINE_SOURCE_PATH": ("paths", "source_file", str),
    "PIPELINE_LAYER_FORMAT": ("storage", "layer_format", str),
    "PIPELINE_BATCH_ROWS": ("ingestion", "incremental_batch_rows", int),
    "PIPELINE_MODEL_SEED": ("model", "seed", int),
    "POSTGRES_HOST": ("postgres", "host", str),
    "POSTGRES_PORT": ("postgres", "port", int),
    "POSTGRES_DB": ("postgres", "dbname", str),
    "POSTGRES_USER": ("postgres", "user", str),
}

VALID_LAYER_FORMATS = {"parquet", "csv"}
VALID_MODES = {"full", "incremental"}
VALID_IMPUTATION_SCOPES = {"full", "train"}


def project_root() -> Path:
    env = os.environ.get("PIPELINE_HOME")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


class Config:
    """Thin wrapper around the YAML dict with path resolution helpers."""

    def __init__(self, data: dict[str, Any], root: Path):
        self.data = data
        self.root = root

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.data.get(section, {}).get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a key from the `paths` section against the project root."""
        value = self.data["paths"].get(key)
        if value is None:
            raise ConfigError(f"paths.{key} is not configured")
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def resolve(self, value: str | os.PathLike) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def run_dir(self, run_id: str) -> Path:
        return self.path("outputs_dir") / "runs" / run_id

    @property
    def layer_format(self) -> str:
        return self.data["storage"]["layer_format"]

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with dotted-key overrides, e.g. {'model.seed': 7}."""
        data = copy.deepcopy(self.data)
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, key = dotted.split(".", 1)
            data.setdefault(section, {})[key] = value
        cfg = Config(data, self.root)
        validate_config(cfg)
        return cfg


def validate_config(cfg: Config) -> None:
    fmt = cfg.get("storage", "layer_format")
    if fmt not in VALID_LAYER_FORMATS:
        raise ConfigError(f"storage.layer_format must be one of {sorted(VALID_LAYER_FORMATS)}, got {fmt!r}")
    mode = cfg.get("ingestion", "default_mode")
    if mode not in VALID_MODES:
        raise ConfigError(f"ingestion.default_mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    rows = cfg.get("ingestion", "incremental_batch_rows")
    if not isinstance(rows, int) or rows <= 0:
        raise ConfigError(f"ingestion.incremental_batch_rows must be a positive integer, got {rows!r}")
    scope = cfg.get("model", "imputation_fit_scope")
    if scope not in VALID_IMPUTATION_SCOPES:
        raise ConfigError(f"model.imputation_fit_scope must be one of {sorted(VALID_IMPUTATION_SCOPES)}, got {scope!r}")
    test_size = cfg.get("model", "test_size")
    if not (isinstance(test_size, (int, float)) and 0 < test_size < 1):
        raise ConfigError(f"model.test_size must be in (0, 1), got {test_size!r}")
    seed = cfg.get("model", "seed")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"model.seed must be a non-negative integer, got {seed!r}")
    tol = cfg.get("reproducibility", "abs_tolerance")
    if not (isinstance(tol, (int, float)) and 0 <= tol < 1):
        raise ConfigError(f"reproducibility.abs_tolerance must be in [0, 1), got {tol!r}")


def load_dotenv(root: Path) -> None:
    """Load KEY=VALUE pairs from <root>/.env into os.environ WITHOUT overriding
    variables that are already set (containers get theirs from docker-compose)."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | os.PathLike | None = None) -> Config:
    root = project_root()
    load_dotenv(root)
    cfg_path = Path(path) if path else Path(os.environ.get("PIPELINE_CONFIG", root / "config" / "pipeline.yml"))
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for env, (section, key, caster) in ENV_OVERRIDES.items():
        raw = os.environ.get(env)
        if raw not in (None, ""):
            try:
                data.setdefault(section, {})[key] = caster(raw)
            except ValueError as exc:
                raise ConfigError(f"Environment variable {env}={raw!r} is not a valid {caster.__name__}") from exc
    cfg = Config(data, root)
    validate_config(cfg)
    return cfg


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
