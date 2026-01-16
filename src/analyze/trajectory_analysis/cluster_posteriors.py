"""
DEPRECATED: This module has moved to trajectory_analysis.clustering.cluster_posteriors

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.clustering import (
        analyze_bootstrap_results,
        compute_assignment_posteriors,
        compute_quality_metrics,
        align_bootstrap_labels,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.cluster_posteriors is deprecated. "
    "Import from trajectory_analysis.clustering instead.",
    DeprecationWarning,
    stacklevel=2
)

from .clustering.cluster_posteriors import (
    analyze_bootstrap_results,
    compute_assignment_posteriors,
    compute_quality_metrics,
    align_bootstrap_labels,
)

__all__ = [
    'analyze_bootstrap_results',
    'compute_assignment_posteriors',
    'compute_quality_metrics',
    'align_bootstrap_labels',
]
