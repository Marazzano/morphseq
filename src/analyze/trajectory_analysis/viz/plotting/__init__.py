"""
Plotting subpackage for trajectory analysis visualization.

Contains:
- core: Main plotting functions (cluster trajectories, membership, heatmaps, scatter)
- faceted: Faceted trajectory plots with grouping

NOTE: 3D plotting has been moved to src.analyze.viz.plotting.plotting_3d (generic location).
The import is re-exported here for backward compatibility.
"""

# Core plotting functions
from .core import (
    # DataFrame-first API (recommended)
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_membership_vs_k,
    # Legacy API (deprecated but kept for backward compatibility)
    plot_cluster_trajectories,
    plot_membership_trajectories,
)

# Faceted plotting (now imports from modular faceted/ subdirectory)
from .faceted import (
    # Main plotting functions
    plot_trajectories_faceted,
    plot_multimetric_trajectories,
    plot_proportion_grid,
    plot_proportion_faceted,
    # IR dataclasses (for advanced users)
    TraceData,
    SubplotData,
    FigureData,
    # Color utilities
    STANDARD_PALETTE,
    # New names
    create_color_lookup,
    create_color_lookup_from_column,
    create_color_state,
    get_color_from_state,
    # Deprecated aliases
    make_color_lookup,
    build_color_lookup_for_column,
    build_color_state,
    resolve_color_value,
    # Error band utilities
    validate_error_type,
    compute_error_band,
    compute_linear_fit,
)

# 3D plotting - re-exported from generic location for backward compatibility
from src.analyze.viz.plotting import plot_3d_scatter

__all__ = [
    # Core plotting
    'plot_cluster_trajectories_df',
    'plot_membership_trajectories_df',
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_membership_vs_k',
    'plot_cluster_trajectories',  # deprecated
    'plot_membership_trajectories',  # deprecated
    # Faceted plotting (main functions)
    'plot_trajectories_faceted',
    'plot_multimetric_trajectories',
    'plot_proportion_grid',
    'plot_proportion_faceted',
    # Faceted plotting (IR dataclasses)
    'TraceData',
    'SubplotData',
    'FigureData',
    # Faceted plotting (color utilities)
    'STANDARD_PALETTE',
    # New names
    'create_color_lookup',
    'create_color_lookup_from_column',
    'create_color_state',
    'get_color_from_state',
    # Deprecated aliases
    'make_color_lookup',
    'build_color_lookup_for_column',
    'build_color_state',
    'resolve_color_value',
    # Faceted plotting (error band utilities)
    'validate_error_type',
    'compute_error_band',
    'compute_linear_fit',
    # 3D plotting (re-exported from viz.plotting)
    'plot_3d_scatter',
]
