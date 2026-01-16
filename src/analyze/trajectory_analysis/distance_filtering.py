"""
DEPRECATED: This module has moved to trajectory_analysis.qc.quality_control

This file remains for backward compatibility but will be removed in a future version.
Please update your imports:

Old:
    from trajectory_analysis.distance_filtering import identify_embryo_outliers_iqr

New:
    from trajectory_analysis.qc import identify_embryo_outliers_iqr

Or directly:
    from trajectory_analysis.qc.quality_control import identify_embryo_outliers_iqr
"""

import warnings

warnings.warn(
    "trajectory_analysis.distance_filtering is deprecated. "
    "Use trajectory_analysis.qc instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all functions from new location for backward compatibility
from .qc.quality_control import (
    identify_embryo_outliers_iqr,
    filter_data_and_ids,
    identify_cluster_outliers_combined,
)

__all__ = [
    'identify_embryo_outliers_iqr',
    'filter_data_and_ids',
    'identify_cluster_outliers_combined',
]
