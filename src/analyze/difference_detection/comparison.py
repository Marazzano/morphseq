"""
Deprecated: use difference_detection.compat or the new classification APIs.

This module remains for backward compatibility with older imports.
"""

import warnings

warnings.warn(
    "difference_detection.comparison is deprecated. "
    "Use difference_detection.compat.compare_groups instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .compat.comparison import add_group_column, compare_groups

__all__ = [
    "add_group_column",
    "compare_groups",
]
