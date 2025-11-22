"""
Pair analysis utilities for trajectory analysis.

This module provides reusable functions for comparing trajectories across
experimental groups (e.g., genotypes, pairs, treatments).
"""

from .data_utils import (
    get_trajectories_for_group,
    compute_binned_mean,
)

from .plotting import (
    plot_genotypes_overlaid,
    plot_faceted_trajectories,
    GENOTYPE_COLORS,
    GENOTYPE_ORDER,
)

__all__ = [
    'get_trajectories_for_group',
    'compute_binned_mean',
    'plot_genotypes_overlaid',
    'plot_faceted_trajectories',
    'GENOTYPE_COLORS',
    'GENOTYPE_ORDER',
]
