"""death_detection — two-mode death QC (per-snip flags + per-animal death_event)."""

from .contract import (
    DEATH_DETECTION_QC_TABLE_COLUMNS,
    DEATH_EVENT_TABLE_COLUMNS,
    validate_death_detection_qc,
    validate_death_event,
)
from .entrypoint import run_death_detection

__all__ = [
    "DEATH_DETECTION_QC_TABLE_COLUMNS",
    "DEATH_EVENT_TABLE_COLUMNS",
    "validate_death_detection_qc",
    "validate_death_event",
    "run_death_detection",
]
