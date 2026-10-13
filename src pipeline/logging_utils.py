"""Structured (JSON-lines) logging.

Every record carries: timestamp (UTC ISO-8601), level, pipeline_run_id, stage,
event, message and — where relevant — rows_in / rows_out / duration_s / error.
Records go to stderr (captured by Airflow task logs; stdout is kept for command results) and to
logs/pipeline_<run_id>.jsonl so one DAG run can be traced end to end.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_STD_ATTRS = set(logging.LogRecord("x", 0, "x", 0, "x", None, None).__dict__) | {"message", "asctime"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "pipeline_run_id": getattr(record, "pipeline_run_id", None),
            "stage": getattr(record, "stage", None),
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["error"] = payload.get("error") or str(record.exc_info[1])
            payload["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class _RunAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = dict(self.extra)
        for key, value in (kwargs.pop("extra", {}) or {}).items():
            # LogRecord reserves names like 'message'/'args'; never crash a stage over a field name.
            extra[f"field_{key}" if key in _STD_ATTRS else key] = value
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(run_id: str, stage: str, logs_dir: Path | None = None) -> logging.LoggerAdapter:
    logger = logging.getLogger(f"pipeline.{run_id}")
    if not getattr(logger, "_configured", False):
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream = logging.StreamHandler(sys.stderr)  # stdout is reserved for command results
        stream.setFormatter(JsonFormatter())
        logger.addHandler(stream)
        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logs_dir / f"pipeline_{run_id}.jsonl", encoding="utf-8")
            fh.setFormatter(JsonFormatter())
            logger.addHandler(fh)
        logger._configured = True  # type: ignore[attr-defined]
    return _RunAdapter(logger, {"pipeline_run_id": run_id, "stage": stage})


@contextmanager
def log_stage(log: logging.LoggerAdapter, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log stage_start / stage_end (or stage_failed) with duration.

    The yielded dict can be filled with row counts by the caller; its content is
    added to the final record.
    """
    info: dict[str, Any] = {}
    start = time.perf_counter()
    log.info("stage started", extra={"event": "stage_start", **fields})
    try:
        yield info
    except Exception as exc:
        info["duration_s"] = round(time.perf_counter() - start, 3)
        log.error(
            "stage failed: %s", exc,
            exc_info=True,
            extra={"event": "stage_failed", "error": f"{type(exc).__name__}: {exc}", **fields, **info},
        )
        raise
    else:
        info["duration_s"] = round(time.perf_counter() - start, 3)
        log.info("stage finished", extra={"event": "stage_end", **fields, **info})
