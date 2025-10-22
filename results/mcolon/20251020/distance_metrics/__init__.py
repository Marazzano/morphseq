"""
Distance metrics comparison module for morphological divergence analysis.
"""

from .auroc_comparison import (
    compute_distance_auroc,
    compare_distance_metrics,
    bootstrap_paired_difference,
    compute_all_classification_metrics,
    compute_roc_curves,
    aggregate_across_time_bins
)

__all__ = [
    'compute_distance_auroc',
    'compare_distance_metrics',
    'bootstrap_paired_difference',
    'compute_all_classification_metrics',
    'compute_roc_curves',
    'aggregate_across_time_bins'
]
