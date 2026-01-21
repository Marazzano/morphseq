"""Analysis-specific code for optimal transport morphometrics on embryo masks.

This module provides embryo-specific I/O, preprocessing, visualization, and
pipeline orchestration for morphological dynamics analysis using optimal transport.

Reusable optimal transport utilities are in src.analyze.utils.optimal_transport.
"""

from src.analyze.utils.optimal_transport import (
    UOTConfig,
    UOTFrame,
    UOTFramePair,
    UOTSupport,
    UOTProblem,
    UOTResult,
    SamplingMode,
    MassMode,
    POTBackend,
)

from .frame_mask_io import (
    load_mask_from_csv,
    load_mask_pair_from_csv,
    load_mask_series_from_csv,
    load_mask_from_png,
)
from .preprocess import preprocess_pair
from .run_transport import run_uot_pair, build_problem
from .run_timeseries import run_timeseries_from_csv

__all__ = [
    # Re-exported from utils for convenience
    "UOTConfig",
    "UOTFrame",
    "UOTFramePair",
    "UOTSupport",
    "UOTProblem",
    "UOTResult",
    "SamplingMode",
    "MassMode",
    "POTBackend",
    # Embryo-specific I/O
    "load_mask_from_csv",
    "load_mask_pair_from_csv",
    "load_mask_series_from_csv",
    "load_mask_from_png",
    # Embryo-specific preprocessing
    "preprocess_pair",
    # Pipeline orchestration
    "run_uot_pair",
    "build_problem",
    "run_timeseries_from_csv",
]
