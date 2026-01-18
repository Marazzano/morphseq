"""
DEPRECATED: This module has been moved to src.analyze.viz.plotting.time_series

Please update your imports:
    OLD: from src.analyze.utils.plotting import plot_embryos_metric_over_time
    NEW: from src.analyze.viz.plotting import plot_time_series_by_group

For backward compatibility with the old function name:
    from src.analyze.viz.plotting import plot_embryos_metric_over_time

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "src.analyze.utils.plotting is deprecated. "
    "Import from src.analyze.viz.plotting instead: "
    "from src.analyze.viz.plotting import plot_time_series_by_group",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from src.analyze.viz.plotting.time_series import (
    plot_time_series_by_group,
    plot_embryos_metric_over_time,
    get_membership_category_colors,
)

__all__ = [
    'plot_time_series_by_group',
    'plot_embryos_metric_over_time',
    'get_membership_category_colors',
]
