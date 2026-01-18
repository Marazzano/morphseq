"""
DEPRECATED: This module has been moved to src.analyze.utils.timeseries.dba

Please update your imports:
    OLD: from trajectory_analysis.dba import dba
    NEW: from src.analyze.utils.timeseries import dba

For backward compatibility, you can also use:
    from trajectory_analysis.distance import dba

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.dba is deprecated. "
    "Import from src.analyze.utils.timeseries instead: "
    "from src.analyze.utils.timeseries import dba",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from src.analyze.utils.timeseries.dba import dba

__all__ = ['dba']
