"""
Generic Plotting Utilities

Domain-agnostic plotting functions for time series and 3D visualization.

These functions use generic time-series algorithms from utils.timeseries
(DTW, DBA, interpolation) and have no trajectory_analysis dependencies.

Modules
=======
- feature_over_time : Faceted time series plotting (faceting-engine)
- time_series : Legacy single-panel time series plotting
- faceted/ : Legacy faceted time series plotting
- plotting_3d : 3D scatter plots with trajectory lines

Functions
=========
Time Series Plotting:
- plot_feature_over_time : Plot a feature over time, colored by group
- plot_feature_over_time_faceted : Faceted feature-over-time plot

3D Plotting:
- plot_3d_scatter : 3D scatter plot with optional trajectory/mean lines

For domain-specific trajectory visualizations (genotype styling, phenotype colors),
see: src.analyze.trajectory_analysis.viz.plotting
"""

from .feature_over_time import plot_feature_over_time
from .time_series import (
    plot_time_series_by_group,  # Deprecated
    plot_embryos_metric_over_time,  # Deprecated
    get_membership_category_colors,
)
from .faceted.time_series import (
    plot_feature_over_time_faceted,
    plot_time_series_faceted,  # Deprecated
    plot_embryos_metric_over_time_faceted,  # Deprecated
)
from .plotting_3d import plot_3d_scatter

__all__ = [
    # Generic API
    'plot_feature_over_time',
    'plot_feature_over_time_faceted',
    'plot_3d_scatter',
    'get_membership_category_colors',
    # Backward compat aliases
    'plot_time_series_by_group',
    'plot_time_series_faceted',
    'plot_embryos_metric_over_time',
    'plot_embryos_metric_over_time_faceted',
]
