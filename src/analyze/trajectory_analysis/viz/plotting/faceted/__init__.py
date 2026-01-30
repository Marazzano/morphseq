"""
Faceted plotting utilities for trajectory analysis.

This module provides generic faceted plots that can group by ANY column.
Supports both Plotly (interactive HTML) and Matplotlib (static PNG).
Uses an Intermediate Representation (IR) pattern for backend-agnostic plotting.

Public API:
-----------
Proportion Plots:
    - plot_proportions: Proportion/bar charts (preferred)

Shared Components:
    - TraceData, SubplotData, FigureData: IR dataclasses
    - STANDARD_PALETTE: Default color palette
    - validate_error_type: Error type validation helper
"""

# Public proportion plotting API
from .proportions import (
    plot_proportions,
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
    # Proportion plots
    'plot_proportions',
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
