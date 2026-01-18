"""
Faceted Plotting Utilities

Multi-panel (faceted) plotting functions for comparing time series
across experiments, conditions, or other groupings.

Functions
=========
- plot_time_series_faceted : Create faceted grid of time series plots
- plot_embryos_metric_over_time_faceted : Alias for backward compat
"""

from .time_series import (
    plot_time_series_faceted,
    plot_embryos_metric_over_time_faceted,
)

__all__ = [
    'plot_time_series_faceted',
    'plot_embryos_metric_over_time_faceted',
]
