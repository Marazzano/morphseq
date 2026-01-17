"""
DEPRECATED: This module has moved to trajectory_analysis.viz.plotting.core

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.viz.plotting import (
        plot_cluster_trajectories_df,
        plot_membership_trajectories_df,
        plot_posterior_heatmap,
        plot_2d_scatter,
        plot_membership_vs_k,
        plot_cluster_trajectories,
        plot_membership_trajectories,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.plotting is deprecated. "
    "Import from trajectory_analysis.viz.plotting instead.",
    DeprecationWarning,
    stacklevel=2
)

from .viz.plotting.core import (
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_membership_vs_k,
    plot_cluster_trajectories,
    plot_membership_trajectories,
)

__all__ = [
    'plot_cluster_trajectories_df',
    'plot_membership_trajectories_df',
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_membership_vs_k',
    'plot_cluster_trajectories',
    'plot_membership_trajectories',
]
