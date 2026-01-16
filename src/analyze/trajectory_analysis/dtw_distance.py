"""
DEPRECATED: This module has been moved to trajectory_analysis.distance.dtw_distance

Please update your imports:
    OLD: from trajectory_analysis.dtw_distance import compute_dtw_distance_matrix
    NEW: from trajectory_analysis.distance import compute_dtw_distance_matrix

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.dtw_distance is deprecated. "
    "Import from trajectory_analysis.distance instead: "
    "from trajectory_analysis.distance import compute_dtw_distance_matrix",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from .distance.dtw_distance import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
    compute_trajectory_distances,
)

__all__ = [
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'prepare_multivariate_array',
    'compute_md_dtw_distance_matrix',
    'compute_trajectory_distances',
]
