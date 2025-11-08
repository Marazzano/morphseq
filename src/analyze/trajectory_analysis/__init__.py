"""
Trajectory Analysis Package

Comprehensive framework for DTW-based trajectory analysis, bootstrap consensus
clustering, and probabilistic quality assessment.

This package unifies:
- DTW distance computation
- Trajectory processing and alignment
- Bootstrap consensus clustering
- Posterior probability analysis
- Membership quality classification
- Trajectory visualization

Examples
--------
>>> from src.analyze.trajectory_analysis import (
...     extract_trajectories,
...     compute_dtw_distance_matrix,
...     run_bootstrap_hierarchical,
...     analyze_bootstrap_results,
...     classify_membership_2d
... )
>>>
>>> # Load and process trajectories
>>> trajectories, ids, grid = extract_trajectories(df, genotype='wildtype')
>>>
>>> # Compute DTW distances
>>> D = compute_dtw_distance_matrix(trajectories, window=5)
>>>
>>> # Bootstrap clustering
>>> bootstrap_results = run_bootstrap_hierarchical(D, k=3, n_bootstrap=100)
>>>
>>> # Analyze posteriors
>>> posteriors = analyze_bootstrap_results(bootstrap_results)
>>>
>>> # Classify membership quality
>>> classification = classify_membership_2d(
...     posteriors['max_p'],
...     posteriors['log_odds_gap'],
...     posteriors['modal_cluster']
... )
"""

__version__ = '0.2.0'

# DTW & Distance Computation
from .dtw_distance import compute_dtw_distance, compute_dtw_distance_matrix
from .dba import dba

# Trajectory Processing
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

# Plotting
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

    # Trajectory Processing
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

    # Plotting
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_cluster_trajectories',
    'plot_membership_trajectories',
]
