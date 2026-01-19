"""
Generic Plotting Utilities

Domain-agnostic plotting functions for time series visualization.

These functions use generic time-series algorithms from utils.timeseries
(DTW, DBA, interpolation) and have no trajectory_analysis dependencies.

Modules
=======
- time_series : Single-panel time series plotting
- faceted/ : Faceted (multi-panel) time series plotting

Functions
=========
Time Series Plotting:
- plot_feature_over_time : Plot a feature over time, colored by group
- plot_feature_over_time_faceted : Faceted feature-over-time plot

For domain-specific trajectory visualizations (genotype styling, phenotype colors),
see: src.analyze.trajectory_analysis.viz.plotting
"""

from .time_series import (
    plot_feature_over_time,
    plot_time_series_by_group,  # Deprecated
    plot_embryos_metric_over_time,  # Deprecated
    get_membership_category_colors,
)
from .faceted.time_series import (
    plot_feature_over_time_faceted,
    plot_time_series_faceted,  # Deprecated
    plot_embryos_metric_over_time_faceted,  # Deprecated
)

__all__ = [
    # Generic API
    'plot_feature_over_time',
    'plot_feature_over_time_faceted',
    'get_membership_category_colors',
    # Backward compat aliases
    'plot_time_series_by_group',
    'plot_time_series_faceted',
    'plot_embryos_metric_over_time',
    'plot_embryos_metric_over_time_faceted',
]
