"""
DEPRECATED: This module has moved to trajectory_analysis.qc.quality_control

This file remains for backward compatibility but will be removed in a future version.
Please update your imports:

Old:
    from trajectory_analysis.outliers import identify_outliers

New:
    from trajectory_analysis.qc import identify_outliers

Or directly:
    from trajectory_analysis.qc.quality_control import identify_outliers
"""

import warnings

warnings.warn(
    "trajectory_analysis.outliers is deprecated. "
    "Use trajectory_analysis.qc instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all functions from new location for backward compatibility
from .qc.quality_control import (
    identify_outliers,
    remove_outliers_from_distance_matrix,
)

__all__ = [
    'identify_outliers',
    'remove_outliers_from_distance_matrix',
]
