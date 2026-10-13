"""Exception types. Each maps to a distinct, documented failure mode (docs/runbook.md)."""


class PipelineError(Exception):
    """Base class for all expected pipeline failures."""


class ConfigError(PipelineError):
    """Invalid or missing configuration value."""


class SourceIdentityError(PipelineError):
    """The source file is missing or does not match contracts/source_manifest.yml."""


class CheckpointError(PipelineError):
    """The ingestion checkpoint is inconsistent with the source or the raw layer."""


class DataQualityError(PipelineError):
    """One or more severity=error data-contract checks failed."""

    def __init__(self, layer: str, failed_checks: list[dict]):
        self.layer = layer
        self.failed_checks = failed_checks
        ids = ", ".join(c["check_id"] for c in failed_checks)
        super().__init__(f"Data quality failure in layer '{layer}': {ids}")


class ReproducibilityError(PipelineError):
    """Reproduced metrics/counts fall outside the declared tolerance."""


class DatabaseError(PipelineError):
    """PostgreSQL is unreachable or a load statement failed."""
