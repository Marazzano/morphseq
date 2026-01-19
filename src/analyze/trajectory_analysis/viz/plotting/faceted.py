"""
DEPRECATED: This module has been split into submodules for better organization.

The faceted plotting functionality is now organized as:
- faceted/shared.py: IR dataclasses, color helpers, error band functions, renderers
- faceted/trajectories.py: Line/trajectory plotting functions
- faceted/proportions.py: Bar/proportion plotting functions
- faceted/__init__.py: Public API exports

For new code, use:
    from .faceted import plot_trajectories_faceted, plot_proportion_faceted

This backward compatibility shim will be removed in a future release.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'trajectory_analysis.viz.plotting.faceted' (as a module) is deprecated. "
    "The module has been split into submodules: faceted/shared.py, faceted/trajectories.py, "
    "faceted/proportions.py. Please update imports to: "
    "'from trajectory_analysis.viz.plotting.faceted import <function_name>'. "
    "This backward compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all public API from the new submodule structure
from .faceted import *

# Explicitly list what's available for import *
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
    # Color helpers
    'make_color_lookup',
    'build_color_lookup_for_column',
    'build_color_state',
    'resolve_color_value',
    # Computation
    'compute_error_band',
    'compute_linear_fit',
]
