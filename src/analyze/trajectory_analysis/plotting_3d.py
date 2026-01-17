"""
DEPRECATED: This module has moved to trajectory_analysis.viz.plotting.plotting_3d

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.viz.plotting import plot_3d_scatter
"""
import warnings

warnings.warn(
    "trajectory_analysis.plotting_3d is deprecated. "
    "Import from trajectory_analysis.viz.plotting instead.",
    DeprecationWarning,
    stacklevel=2
)

from .viz.plotting.plotting_3d import plot_3d_scatter

__all__ = ['plot_3d_scatter']
