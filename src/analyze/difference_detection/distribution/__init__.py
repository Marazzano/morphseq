"""
Distribution-based difference detection methods.

This module provides statistical tests for comparing multivariate distributions
without assuming specific parametric forms.

Available tests:
- Energy distance: Distance metric between distributions
- MMD (Maximum Mean Discrepancy): Kernel-based distribution metric
- Hotelling's T2: Tests for mean differences

Submodules
==========
- distances : Distance metrics from reference distributions
- energy : Energy distance and Hotelling's T2 tests
- mmd : Maximum Mean Discrepancy tests
"""

from .distances import (
    compute_mahalanobis_distance,
    compute_euclidean_distance,
    compute_standardized_distance,
    compute_cosine_distance,
    compute_all_distances,
    detect_outliers_mahalanobis,
)

from .energy import (
    compute_energy_distance,
    permutation_test_energy,
    hotellings_t2_test,
)

from .mmd import (
    compute_rbf_kernel,
    estimate_bandwidth_median,
    compute_mmd,
    permutation_test_mmd,
    mmd_kernel_width_test,
)

__all__ = [
    # Distances
    "compute_mahalanobis_distance",
    "compute_euclidean_distance",
    "compute_standardized_distance",
    "compute_cosine_distance",
    "compute_all_distances",
    "detect_outliers_mahalanobis",
    # Energy
    "compute_energy_distance",
    "permutation_test_energy",
    "hotellings_t2_test",
    # MMD
    "compute_rbf_kernel",
    "estimate_bandwidth_median",
    "compute_mmd",
    "permutation_test_mmd",
    "mmd_kernel_width_test",
]
