"""Distance Computation Algorithms

This subpackage contains distance computation algorithms for trajectory analysis,
including Dynamic Time Warping (DTW) and DTW Barycenter Averaging (DBA).

Generic DTW/DBA algorithms are re-exported from src.analyze.utils.timeseries.
Domain-specific functions remain in this package.

Functions
=========
DTW Distance (generic from utils.timeseries):
- compute_dtw_distance : Compute DTW distance between two 1D sequences
- compute_dtw_distance_matrix : Compute pairwise DTW distances for multiple 1D sequences
- compute_md_dtw_distance_matrix : Compute pairwise MD-DTW distances

DTW Distance (domain-specific):
- prepare_multivariate_array : Convert DataFrame to 3D array for MD-DTW
- compute_trajectory_distances : High-level function to compute trajectory distances

DBA (from utils.timeseries):
- dba : DTW Barycenter Averaging for computing consensus sequences
"""

from .dtw_distance import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
    compute_trajectory_distances,
)
from .dba import dba

__all__ = [
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'prepare_multivariate_array',
    'compute_md_dtw_distance_matrix',
    'compute_trajectory_distances',
    'dba',
]
