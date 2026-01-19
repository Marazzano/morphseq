"""
Faceted Plotting Utilities

Multi-panel (faceted) plotting functions for comparing time series
across experiments, conditions, or other groupings.

Functions
=========
- plot_feature_over_time_faceted : Create faceted grid of time series plots
- plot_time_series_faceted : Deprecated wrapper
- plot_embryos_metric_over_time_faceted : Deprecated wrapper
"""

from .time_series import (
    plot_feature_over_time_faceted,
    plot_time_series_faceted,
    plot_embryos_metric_over_time_faceted,
)

__all__ = [
    'plot_feature_over_time_faceted',
    'plot_time_series_faceted',
    'plot_embryos_metric_over_time_faceted',
]
