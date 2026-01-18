"""
DTW Barycenter Averaging (DBA)

This module re-exports the DBA algorithm from the generic utils.timeseries package.

The canonical location is now: src.analyze.utils.timeseries.dba

For direct imports:
    from src.analyze.utils.timeseries import dba
    from src.analyze.utils.timeseries.dba import dba

This module exists for backward compatibility with:
    from trajectory_analysis.distance import dba
    from trajectory_analysis.distance.dba import dba

References
----------
Petitjean, F., Ketterlin, A., & Gancarski, P. (2011). A global averaging method
for dynamic time warping, with applications to clustering. Pattern Recognition.
"""

# Re-export from canonical location
from src.analyze.utils.timeseries.dba import dba

__all__ = ['dba']
