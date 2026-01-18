"""
Faceted plotting utilities for trajectory analysis.

This module provides generic faceted plots that can group by ANY column.
Supports both Plotly (interactive HTML) and Matplotlib (static PNG).
Uses an Intermediate Representation (IR) pattern for backend-agnostic plotting.

Public API:
-----------
Trajectory Plots:
    - plot_trajectories_faceted: Standard faceted plot for single metric
    - plot_multimetric_trajectories: Multi-metric plot (rows=metrics, cols=groups)

Proportion Plots:
    - plot_proportion_faceted: Faceted proportion/bar charts
    - plot_proportion_grid: DEPRECATED - Use plot_proportion_faceted instead

Shared Components:
    - TraceData, SubplotData, FigureData: IR dataclasses
    - STANDARD_PALETTE: Default color palette
    - validate_error_type: Error type validation helper
"""

# Public trajectory plotting API
from .trajectories import (
    plot_trajectories_faceted,
    plot_multimetric_trajectories,
)

# Public proportion plotting API
from .proportions import (
    plot_proportion_faceted,
    plot_proportion_grid,  # Deprecated but kept for compatibility
)

# Shared utilities (exposed for advanced users)
from .shared import (
    # IR dataclasses
    TraceData,
    SubplotData,
    FigureData,
    # Constants
    STANDARD_PALETTE,
    VALID_ERROR_TYPES,
    # Validation
    validate_error_type,
    # Color helpers (new names)
    create_color_lookup,
    create_color_lookup_from_column,
    create_color_state,
    get_color_from_state,
    # Color helpers (deprecated aliases)
    make_color_lookup,
    build_color_lookup_for_column,
    build_color_state,
    resolve_color_value,
    # Computation
    compute_error_band,
    compute_linear_fit,
)

__all__ = [
    # Trajectory plots
    'plot_trajectories_faceted',
    'plot_multimetric_trajectories',
    # Proportion plots
    'plot_proportion_faceted',
    'plot_proportion_grid',
    # IR dataclasses
    'TraceData',
    'SubplotData',
    'FigureData',
    # Constants
    'STANDARD_PALETTE',
    'VALID_ERROR_TYPES',
    # Validation
    'validate_error_type',
    # Color helpers (new names)
    'create_color_lookup',
    'create_color_lookup_from_column',
    'create_color_state',
    'get_color_from_state',
    # Color helpers (deprecated)
    'make_color_lookup',
    'build_color_lookup_for_column',
    'build_color_state',
    'resolve_color_value',
    # Computation
    'compute_error_band',
    'compute_linear_fit',
]
