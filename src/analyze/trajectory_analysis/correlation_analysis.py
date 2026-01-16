"""
DEPRECATED: This module has been moved to trajectory_analysis.utilities.correlation

Please update your imports:
    OLD: from trajectory_analysis.correlation_analysis import test_anticorrelation
    NEW: from trajectory_analysis.utilities import test_anticorrelation

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.correlation_analysis is deprecated. "
    "Import from trajectory_analysis.utilities instead: "
    "from trajectory_analysis.utilities import test_anticorrelation",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from .utilities.correlation import test_anticorrelation

__all__ = ['test_anticorrelation']
