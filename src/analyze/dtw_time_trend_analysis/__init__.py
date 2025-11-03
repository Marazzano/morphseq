"""
DTW Time-Trend Analysis Package

Dynamic Time Warping (DTW) based analysis for temporal trajectories in morphological data.

This package provides utilities for:
1. Computing DTW distances between variable-length temporal sequences
2. Processing and aligning embryo trajectories to common timepoint grids
3. Statistical testing of early/late trajectory patterns (anticorrelation analysis)

Core Modules
============
- dtw_distance : DTW computation with Sakoe-Chiba band constraint
- trajectory_utils : Trajectory extraction, interpolation, and alignment
- trajectory_statistics : Statistical tests for trajectory patterns

Example Usage
=============
    from src.analyze.dtw_time_trend_analysis import (
        compute_dtw_distance_matrix,
        extract_trajectories,
        interpolate_to_common_grid,
        test_anticorrelation
    )

    # Extract per-embryo trajectories from long-format data
    trajectories, embryo_ids, df_long = extract_trajectories(
        df, genotype_filter='cep290_homozygous', metric_name='normalized_baseline_deviation'
    )

    # Interpolate to common timepoint grid
    interpolated_trajs, embryo_ids_interp, _, common_grid = interpolate_to_common_grid(
        df_long, grid_step=0.5
    )

    # Compute pairwise DTW distances
    distance_matrix = compute_dtw_distance_matrix(interpolated_trajs, window=3)

    # Test for early/late anticorrelation patterns
    results = test_anticorrelation(
        cluster_assignments, early_means, late_means, embryo_ids_interp
    )
"""

from .dtw_distance import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
)

from .trajectory_utils import (
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    extract_early_late_means,
)

from .trajectory_statistics import (
    test_anticorrelation,
)

__all__ = [
    # DTW distance computation
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    # Trajectory processing
    'extract_trajectories',
    'interpolate_trajectories',
    'interpolate_to_common_grid',
    'extract_early_late_means',
    # Statistical testing
    'test_anticorrelation',
]

__version__ = '0.1.0'
