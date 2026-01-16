"""
DEPRECATED: This module has been moved to trajectory_analysis.utilities.trajectory_utils

Please update your imports:
    OLD: from trajectory_analysis.trajectory_utils import extract_trajectories_df
    NEW: from trajectory_analysis.utilities import extract_trajectories_df

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.trajectory_utils is deprecated. "
    "Import from trajectory_analysis.utilities instead: "
    "from trajectory_analysis.utilities import extract_trajectories_df",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from .utilities.trajectory_utils import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    interpolate_to_common_grid_multi_df,
    df_to_trajectories,
    extract_early_late_means,
    compute_trend_line,
    # Legacy API
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    pad_trajectories_for_plotting,
)

__all__ = [
    'extract_trajectories_df',
    'interpolate_to_common_grid_df',
    'interpolate_to_common_grid_multi_df',
    'df_to_trajectories',
    'extract_early_late_means',
    'compute_trend_line',
    'extract_trajectories',
    'interpolate_trajectories',
    'interpolate_to_common_grid',
    'pad_trajectories_for_plotting',
]
