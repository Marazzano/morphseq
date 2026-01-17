"""
Deprecated: use distance_metrics instead.

This module remains for backward compatibility with older imports.
"""

import warnings

warnings.warn(
    "difference_detection.statistics is deprecated. "
    "Use difference_detection.distance_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .distance_metrics import (
    compute_energy_distance,
    compute_mmd,
    compute_mean_distance,
    compute_rbf_kernel,
    estimate_bandwidth_median,
)

__all__ = [
    "compute_energy_distance",
    "compute_mmd",
    "compute_mean_distance",
    "compute_rbf_kernel",
    "estimate_bandwidth_median",
]
