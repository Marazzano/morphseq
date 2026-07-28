"""focus_qc — interior structural-edge-content QC heuristic (ghost/structureless embryo detection)."""

from .contract import FOCUS_QC_TABLE_COLUMNS, validate_focus_qc
from .entrypoint import run_focus_qc

__all__ = [
    "FOCUS_QC_TABLE_COLUMNS",
    "validate_focus_qc",
    "run_focus_qc",
]
