"""
Pair analysis utilities for trajectory analysis.

This module provides reusable functions for comparing trajectories across
experimental groups (e.g., genotypes, pairs, treatments).

Level 2: Pair-specific plotting functions that wrap Level 1 generic plotting.
"""

from .data_utils import (
    get_trajectories_for_group,
    compute_binned_mean,
    get_global_axis_ranges,
)

# Lazy import of plotting to avoid circular imports
def __getattr__(name):
    """Lazy loading of plotting functions."""
    if name in ['plot_pairs_overview', 'plot_genotypes_by_pair',
                'plot_single_genotype_across_pairs', 'plot_genotypes_overlaid',
                'plot_all_pairs_overview', 'plot_homozygous_across_pairs']:
        from .plotting import (
            plot_pairs_overview,
            plot_genotypes_by_pair,
            plot_single_genotype_across_pairs,
            plot_genotypes_overlaid,
            plot_all_pairs_overview,
            plot_homozygous_across_pairs,
        )
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'get_trajectories_for_group',
    'compute_binned_mean',
    'get_global_axis_ranges',
    'plot_pairs_overview',
    'plot_genotypes_by_pair',
    'plot_single_genotype_across_pairs',
    # Deprecated
    'plot_genotypes_overlaid',
    'plot_all_pairs_overview',
    'plot_homozygous_across_pairs',
]
