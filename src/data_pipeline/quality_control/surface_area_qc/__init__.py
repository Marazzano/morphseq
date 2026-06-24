"""surface_area_qc — stage-binned two-sided surface-area outlier QC (one sa_outlier_flag per snip)."""

from .contract import SURFACE_AREA_QC_REQUIRED_COLUMNS, validate_surface_area_qc
from .entrypoint import run_surface_area_qc

__all__ = [
    "SURFACE_AREA_QC_REQUIRED_COLUMNS",
    "validate_surface_area_qc",
    "run_surface_area_qc",
]
