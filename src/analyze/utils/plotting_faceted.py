"""
DEPRECATED: This module has been moved to src.analyze.viz.plotting.faceted.time_series

Please update your imports:
    OLD: from src.analyze.utils.plotting_faceted import plot_embryos_metric_over_time_faceted
    NEW: from src.analyze.viz.plotting import plot_feature_over_time_faceted

For backward compatibility with the old function name:
    from src.analyze.viz.plotting import plot_embryos_metric_over_time_faceted

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "src.analyze.utils.plotting_faceted is deprecated. "
    "Import from src.analyze.viz.plotting instead: "
    "from src.analyze.viz.plotting import plot_feature_over_time_faceted",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from src.analyze.viz.plotting.faceted.time_series import (
    plot_feature_over_time_faceted,
    plot_time_series_faceted,
    plot_embryos_metric_over_time_faceted,
)

__all__ = [
    'plot_feature_over_time_faceted',
    'plot_time_series_faceted',
    'plot_embryos_metric_over_time_faceted',
]
