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
- plot_time_series_by_group : Plot time series colored by group
- plot_time_series_faceted : Faceted time series plot

For domain-specific trajectory visualizations (genotype styling, phenotype colors),
see: src.analyze.trajectory_analysis.viz.plotting
"""

from .time_series import (
    plot_time_series_by_group,
    plot_embryos_metric_over_time,  # Alias for backward compat
    get_membership_category_colors,
)
from .faceted.time_series import (
    plot_time_series_faceted,
    plot_embryos_metric_over_time_faceted,  # Alias for backward compat
)

__all__ = [
    # Generic API
    'plot_time_series_by_group',
    'plot_time_series_faceted',
    'get_membership_category_colors',
    # Backward compat aliases
    'plot_embryos_metric_over_time',
    'plot_embryos_metric_over_time_faceted',
]
