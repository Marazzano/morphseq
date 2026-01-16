"""
DEPRECATED: This module has moved to trajectory_analysis.clustering.bootstrap_clustering

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.clustering import (
        run_bootstrap_hierarchical,
        run_bootstrap_kmedoids,
        compute_consensus_labels,
        get_cluster_assignments,
        compute_coassociation_matrix,
        coassociation_to_distance,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.bootstrap_clustering is deprecated. "
    "Import from trajectory_analysis.clustering instead.",
    DeprecationWarning,
    stacklevel=2
)

from .clustering.bootstrap_clustering import (
    run_bootstrap_hierarchical,
    run_bootstrap_kmedoids,
    compute_consensus_labels,
    get_cluster_assignments,
    compute_coassociation_matrix,
    coassociation_to_distance,
)

__all__ = [
    'run_bootstrap_hierarchical',
    'run_bootstrap_kmedoids',
    'compute_consensus_labels',
    'get_cluster_assignments',
    'compute_coassociation_matrix',
    'coassociation_to_distance',
]
