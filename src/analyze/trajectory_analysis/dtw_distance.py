"""
DEPRECATED: This module has been moved to src.analyze.utils.timeseries.dtw

Please update your imports:
    OLD: from trajectory_analysis.dtw_distance import compute_dtw_distance_matrix
    NEW: from src.analyze.utils.timeseries import compute_dtw_distance_matrix

For backward compatibility, you can also use:
    from trajectory_analysis.distance import compute_dtw_distance_matrix

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.dtw_distance is deprecated. "
    "Import from src.analyze.utils.timeseries instead: "
    "from src.analyze.utils.timeseries import compute_dtw_distance_matrix",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from src.analyze.utils.timeseries.dtw import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
    compute_md_dtw_distance_matrix,
)

__all__ = [
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'compute_md_dtw_distance_matrix',
]
