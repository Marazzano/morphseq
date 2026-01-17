"""
DEPRECATED: This module has moved to trajectory_analysis.viz.plotting.faceted

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.viz.plotting import (
        plot_trajectories_faceted,
        plot_multimetric_trajectories,
        plot_proportion_grid,
        plot_proportion_faceted,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.facetted_plotting is deprecated. "
    "Import from trajectory_analysis.viz.plotting instead.",
    DeprecationWarning,
    stacklevel=2
)

from .viz.plotting.faceted import (
    plot_trajectories_faceted,
    plot_multimetric_trajectories,
    plot_proportion_grid,
    plot_proportion_faceted,
    FigureData,
    make_color_lookup,
    build_color_lookup_for_column,
    build_color_state,
    resolve_color_value,
    validate_error_type,
    compute_error_band,
    compute_linear_fit,
)

__all__ = [
    'plot_trajectories_faceted',
    'plot_multimetric_trajectories',
    'plot_proportion_grid',
    'plot_proportion_faceted',
    'FigureData',
    'make_color_lookup',
    'build_color_lookup_for_column',
    'build_color_state',
    'resolve_color_value',
    'validate_error_type',
    'compute_error_band',
    'compute_linear_fit',
]
