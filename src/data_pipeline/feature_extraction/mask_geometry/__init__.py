"""mask_geometry feature product — micron-aware geometry per snip.

Public surface: the contract (required columns + validator), the compute functions
(pure per-mask + per-snip batch), and the runnable entrypoint.
"""

from .compute import compute_mask_geometry_features, compute_mask_geometry_for_mask
from .contract import (
    MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS,
    validate_mask_geometry_features,
)

__all__ = [
    "MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS",
    "validate_mask_geometry_features",
    "compute_mask_geometry_features",
    "compute_mask_geometry_for_mask",
]
