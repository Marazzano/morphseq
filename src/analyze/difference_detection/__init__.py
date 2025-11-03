"""
Difference detection package for analyzing genotype effects and temporal patterns.

This package provides tools for:
1. Classification-based tests (logistic regression for discrimination)
2. Distribution-based tests (energy distance, MMD)
3. Horizon plots (time series comparisons)
4. Time matrix analysis (temporal correlation/prediction matrices)

Submodules
==========
- classification : Predictive models and onset detection (placeholder)
- distribution : Statistical tests for distribution differences
- horizon_plots : Heatmap visualization utilities
- time_matrix : Temporal data reshaping and analysis
- pipelines : High-level orchestration workflows
- plotting : General visualization helpers (placeholder)
- metrics : Statistical metrics computation (placeholder)
"""

# Core utilities (newly extracted)
from . import horizon_plots
from . import time_matrix
from . import distribution

# Distribution-based testing (newly implemented)
from . import pipelines

# Classification module
from . import classification
from .classification import (
    predictive_signal_test,
    compute_embryo_penetrance,
    summarize_penetrance,
    get_high_penetrance_embryos,
)

# Expose key functions at package level for convenience
from .horizon_plots import (
    plot_horizon_grid,
    plot_single_horizon,
    plot_best_condition_map,
    compute_shared_colorscale,
)

from .time_matrix import (
    load_time_matrix_results,
    build_metric_matrices,
    align_matrix_times,
    compute_matrix_statistics,
    filter_matrices_by_time_range,
    interpolate_missing_times,
)
from .pipelines import (
    HorizonPlotContext,
    load_and_prepare_time_matrices,
    render_horizon_grid,
    summarise_bundles,
)

# Distribution-based testing functions
from .distribution import (
    compute_energy_distance,
    permutation_test_energy,
    hotellings_t2_test,
    compute_mmd,
    permutation_test_mmd,
    mmd_kernel_width_test,
    compute_mahalanobis_distance,
    compute_euclidean_distance,
)

__all__ = [
    # Submodules
    'horizon_plots',
    'time_matrix',
    'distribution',
    'classification',
    'pipelines',
    # Horizon plots
    'plot_horizon_grid',
    'plot_single_horizon',
    'plot_best_condition_map',
    'compute_shared_colorscale',
    # Time matrix utilities
    'load_time_matrix_results',
    'build_metric_matrices',
    'align_matrix_times',
    'compute_matrix_statistics',
    'filter_matrices_by_time_range',
    'interpolate_missing_times',
    # Pipeline orchestration
    'HorizonPlotContext',
    'load_and_prepare_time_matrices',
    'render_horizon_grid',
    'summarise_bundles',
    # Distribution-based testing
    'compute_energy_distance',
    'permutation_test_energy',
    'hotellings_t2_test',
    'compute_mmd',
    'permutation_test_mmd',
    'mmd_kernel_width_test',
    'compute_mahalanobis_distance',
    'compute_euclidean_distance',
    # Classification
    'predictive_signal_test',
    'compute_embryo_penetrance',
    'summarize_penetrance',
    'get_high_penetrance_embryos',
]
