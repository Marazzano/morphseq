"""
Deprecated: use difference_detection.compat or the new classification APIs.

This module remains for backward compatibility with older imports.
"""

import warnings

warnings.warn(
    "difference_detection.comparison_multiclass is deprecated. "
    "Use difference_detection.compat.compare_groups_multiclass instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .compat.comparison_multiclass import compare_groups_multiclass

__all__ = [
    "compare_groups_multiclass",
]
