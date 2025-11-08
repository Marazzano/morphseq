"""
Trajectory Analysis Package

Comprehensive framework for DTW-based trajectory analysis, bootstrap consensus
clustering, and probabilistic quality assessment.

This package provides a DataFrame-centric API where time information (hpf - hours
post fertilization) travels with trajectory data throughout the entire pipeline,
eliminating time-axis misalignment issues common with array-based approaches.

Core Workflow
=============
1. Extract: Load data and filter by genotype → df_filtered [embryo_id, hpf, metric_value]
2. Align: Interpolate all trajectories to common hpf grid → df_interpolated
3. Convert: Extract arrays for DTW computation via df_to_trajectories() helper
4. Cluster: Compute DTW distances and bootstrap cluster
5. Analyze: Compute posterior probabilities and quality metrics
6. Plot: Visualize trajectories using df_interpolated with hpf column

This package unifies:
- DTW distance computation
- Trajectory processing and alignment (DataFrame-first API)
- Bootstrap consensus clustering
- Posterior probability analysis
- Membership quality classification
- Trajectory visualization

Examples
--------
>>> from src.analyze.trajectory_analysis import (
...     extract_trajectories_df,
...     interpolate_to_common_grid_df,
...     df_to_trajectories,
...     compute_dtw_distance_matrix,
...     run_bootstrap_hierarchical,
...     analyze_bootstrap_results,
...     classify_membership_2d,
...     plot_cluster_trajectories_df
... )
>>>
>>> # Extract and filter trajectories (DataFrame-centric)
>>> df_filtered = extract_trajectories_df(df, genotype_filter='wildtype')
>>> df_interpolated = interpolate_to_common_grid_df(df_filtered)
>>>
>>> # Convert to arrays for DTW (one-liner helper)
>>> trajectories, ids, grid = df_to_trajectories(df_interpolated)
>>>
>>> # Compute DTW distances
>>> D = compute_dtw_distance_matrix(trajectories, window=5)
>>>
>>> # Bootstrap clustering
>>> bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=ids, n_bootstrap=100)
>>>
>>> # Analyze posteriors
>>> posteriors = analyze_bootstrap_results(bootstrap_results)
>>>
>>> # Classify membership quality
>>> classification = classify_membership_2d(
...     posteriors['max_p'],
...     posteriors['log_odds_gap'],
...     posteriors['modal_cluster'],
...     embryo_ids=posteriors['embryo_ids']
... )
>>>
>>> # Visualize using DataFrame (preserves time alignment)
>>> fig = plot_cluster_trajectories_df(
...     df_interpolated,
...     posteriors['modal_cluster'],
...     embryo_ids=posteriors['embryo_ids'],
...     show_mean=True
... )

Migration from v0.1.x
=====================
The API changed significantly in v0.2.0 to fix time-axis alignment bugs.
Old array-based functions are deprecated but still functional with deprecation
warnings. See README.md for detailed migration guide.
"""

__version__ = '0.2.0'

# DTW & Distance Computation
from .dtw_distance import compute_dtw_distance, compute_dtw_distance_matrix
from .dba import dba

# Trajectory Processing - NEW API (DataFrame-centric, v0.2.0+)
from .trajectory_utils import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    df_to_trajectories
)

# Trajectory Processing - DEPRECATED (v0.1.x, kept for backward compatibility)
from .trajectory_utils import (
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    pad_trajectories_for_plotting,
    extract_early_late_means
)

# Correlation Analysis
from .correlation_analysis import test_anticorrelation

# Bootstrap Clustering
from .bootstrap_clustering import (
    run_bootstrap_hierarchical,
    compute_consensus_labels
)

# Posterior Analysis
from .cluster_posteriors import (
    analyze_bootstrap_results,
    compute_assignment_posteriors,
    compute_quality_metrics,
    align_bootstrap_labels
)

# Classification
from .cluster_classification import (
    classify_membership_2d,
    classify_membership_adaptive,
    get_classification_summary
)

# Plotting - NEW API (DataFrame-centric, v0.2.0+)
from .plotting import (
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df
)

# Plotting - DEPRECATED (v0.1.x, kept for backward compatibility)
from .plotting import (
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_cluster_trajectories,
    plot_membership_trajectories
)

__all__ = [
    # DTW & Distance
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'dba',

    # Trajectory Processing - NEW API (v0.2.0+)
    'extract_trajectories_df',
    'interpolate_to_common_grid_df',
    'df_to_trajectories',

    # Trajectory Processing - DEPRECATED (v0.1.x)
    'extract_trajectories',
    'interpolate_trajectories',
    'interpolate_to_common_grid',
    'pad_trajectories_for_plotting',
    'extract_early_late_means',

    # Correlation analysis
    'test_anticorrelation',

    # Bootstrap clustering
    'run_bootstrap_hierarchical',
    'compute_consensus_labels',

    # Posterior analysis
    'analyze_bootstrap_results',
    'compute_assignment_posteriors',
    'compute_quality_metrics',
    'align_bootstrap_labels',

    # Classification
    'classify_membership_2d',
    'classify_membership_adaptive',
    'get_classification_summary',

    # Plotting - NEW API (v0.2.0+)
    'plot_cluster_trajectories_df',
    'plot_membership_trajectories_df',

    # Plotting - DEPRECATED (v0.1.x)
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_cluster_trajectories',
    'plot_membership_trajectories',
]
