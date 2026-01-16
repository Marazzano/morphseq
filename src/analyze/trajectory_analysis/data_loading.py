"""
DEPRECATED: This module has been moved to trajectory_analysis.io.data_loading

Please update your imports:
    OLD: from trajectory_analysis.data_loading import compute_dtw_distance_from_df
    NEW: from trajectory_analysis.io import compute_dtw_distance_from_df

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.data_loading is deprecated. "
    "Import from trajectory_analysis.io instead: "
    "from trajectory_analysis.io import compute_dtw_distance_from_df",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from .io.data_loading import (
    load_experiment_dataframe,
    extract_trajectory_dataframe,
    dataframe_to_trajectories,
    interpolate_trajectories,
    compute_dtw_distance_from_df,
)

__all__ = [
    'load_experiment_dataframe',
    'extract_trajectory_dataframe',
    'dataframe_to_trajectories',
    'interpolate_trajectories',
    'compute_dtw_distance_from_df',
]
