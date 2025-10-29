"""
Difference detection package for analyzing genotype effects and temporal patterns.

This package provides tools for:
1. Classification-based tests (logistic regression for discrimination)
2. Distribution-based tests (energy distance, MMD)
3. Horizon plots (time series comparisons)
4. Time matrix analysis (temporal correlation/prediction matrices)

Submodules
==========
- classification : Predictive models and onset detection
- distribution : Statistical tests for distribution differences
- horizon_plots : Heatmap visualization utilities
- time_matrix : Temporal data reshaping and analysis
- plotting : General visualization helpers
- metrics : Statistical metrics computation
- pipelines : High-level orchestration workflows
"""

# Core utilities (newly extracted)
from . import horizon_plots
from . import time_matrix

# Existing modules
from . import classification
from . import distribution
from . import plotting
from . import metrics
from . import pipelines

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
)

__all__ = [
    # Submodules
    'horizon_plots',
    'time_matrix',
    'classification',
    'distribution',
    'plotting',
    'metrics',
    'pipelines',
    # Main functions
    'plot_horizon_grid',
    'plot_single_horizon',
    'plot_best_condition_map',
    'compute_shared_colorscale',
    'load_time_matrix_results',
    'build_metric_matrices',
    'align_matrix_times',
    'compute_matrix_statistics',
]
