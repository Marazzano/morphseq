"""
DEPRECATED: This module has been moved to trajectory_analysis.distance.dba

Please update your imports:
    OLD: from trajectory_analysis.dba import dba
    NEW: from trajectory_analysis.distance import dba

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.dba is deprecated. "
    "Import from trajectory_analysis.distance instead: "
    "from trajectory_analysis.distance import dba",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from .distance.dba import dba

__all__ = ['dba']
