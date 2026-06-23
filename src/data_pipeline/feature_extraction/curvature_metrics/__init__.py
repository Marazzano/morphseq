"""curvature_metrics feature product — centerline length + curvature summaries per snip."""

from .compute import compute_curvature_features, compute_curvature_for_mask
from .contract import (
    CURVATURE_FEATURES_REQUIRED_COLUMNS,
    validate_curvature_features,
)

__all__ = [
    "CURVATURE_FEATURES_REQUIRED_COLUMNS",
    "validate_curvature_features",
    "compute_curvature_features",
    "compute_curvature_for_mask",
]
