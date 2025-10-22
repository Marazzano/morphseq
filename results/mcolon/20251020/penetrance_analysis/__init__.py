"""
Incomplete penetrance analysis module.

Quantifies how morphological deviation (distance from WT) relates to
classifier-based mutant probability to identify penetrant vs non-penetrant embryos.
"""

from .correlation import (
    compute_per_embryo_metrics,
    compute_correlation_statistics,
    bootstrap_correlation_ci,
    correlation_by_time_bin
)

from .visualization import (
    plot_distance_vs_probability,
    plot_correlation_summary,
    plot_correlation_over_time,
    plot_regression_fit,
    plot_regression_diagnostics,
    plot_regression_comparison,
    plot_r_squared_evolution,
    plot_slope_evolution,
    plot_temporal_cutoffs,
    plot_scatter_by_timebin,
    plot_distance_time_heatmap,
    plot_interaction_model,
    plot_penetrance_onset,
    plot_dual_prediction_heatmaps,
    plot_trajectory_examples,
    plot_prediction_error_scatter,
    plot_penetrance_distribution
)

from .regression import (
    fit_ols_regression,
    fit_logit_regression,
    compute_regression_metrics,
    compute_predicted_values,
    compute_residual_diagnostics,
    test_normality,
    test_heteroskedasticity,
    compute_penetrance_cutoff,
    bootstrap_regression_ci,
    logit_transform
)

from .temporal_analysis import (
    compute_per_bin_regression,
    compute_sliding_window_regression,
    fit_interaction_model,
    identify_penetrance_onset,
    compute_temporal_cutoffs,
    compare_early_vs_late
)

from .trajectory_prediction import (
    prepare_trajectory_data,
    train_trajectory_model,
    predict_with_model,
    cross_validate_trajectory_model,
    predict_all_trajectories,
    compute_per_embryo_prediction_metrics,
    classify_penetrance_by_dual_models,
    detect_penetrance_onset
)

from .trajectory_loeo import (
    create_trajectory_pairs,
    train_loeo_and_full_model,
    test_model_on_genotype,
    compute_overall_metrics,
    compute_per_embryo_metrics,
    compute_error_vs_horizon,
    classify_penetrance_dual_model
)

from .trajectory_viz_loeo import (
    create_aggregated_heatmap,
    create_per_embryo_heatmaps,
    compute_r2_per_cell,
    plot_aggregated_heatmap,
    plot_per_embryo_grid,
    plot_error_vs_horizon,
    plot_temporal_breakdown,
    plot_per_embryo_error_distribution,
    plot_model_comparison_3x3,
    plot_penetrance_classification
)

__all__ = [
    # Correlation
    'compute_per_embryo_metrics',
    'compute_correlation_statistics',
    'bootstrap_correlation_ci',
    'correlation_by_time_bin',
    # Visualization
    'plot_distance_vs_probability',
    'plot_correlation_summary',
    'plot_correlation_over_time',
    'plot_regression_fit',
    'plot_regression_diagnostics',
    'plot_regression_comparison',
    'plot_r_squared_evolution',
    'plot_slope_evolution',
    'plot_temporal_cutoffs',
    'plot_scatter_by_timebin',
    'plot_distance_time_heatmap',
    'plot_interaction_model',
    'plot_penetrance_onset',
    'plot_dual_prediction_heatmaps',
    'plot_trajectory_examples',
    'plot_prediction_error_scatter',
    'plot_penetrance_distribution',
    # Regression
    'fit_ols_regression',
    'fit_logit_regression',
    'compute_regression_metrics',
    'compute_predicted_values',
    'compute_residual_diagnostics',
    'test_normality',
    'test_heteroskedasticity',
    'compute_penetrance_cutoff',
    'bootstrap_regression_ci',
    'logit_transform',
    # Temporal Analysis
    'compute_per_bin_regression',
    'compute_sliding_window_regression',
    'fit_interaction_model',
    'identify_penetrance_onset',
    'compute_temporal_cutoffs',
    'compare_early_vs_late',
    # Trajectory Prediction
    'prepare_trajectory_data',
    'train_trajectory_model',
    'predict_with_model',
    'cross_validate_trajectory_model',
    'predict_all_trajectories',
    'compute_per_embryo_prediction_metrics',
    'classify_penetrance_by_dual_models',
    'detect_penetrance_onset',
    # Trajectory LOEO
    'create_trajectory_pairs',
    'train_loeo_and_full_model',
    'test_model_on_genotype',
    'compute_overall_metrics',
    'compute_per_embryo_metrics',
    'compute_error_vs_horizon',
    'classify_penetrance_dual_model',
    # Trajectory LOEO Visualization
    'create_aggregated_heatmap',
    'create_per_embryo_heatmaps',
    'compute_r2_per_cell',
    'plot_aggregated_heatmap',
    'plot_per_embryo_grid',
    'plot_error_vs_horizon',
    'plot_temporal_breakdown',
    'plot_per_embryo_error_distribution',
    'plot_model_comparison_3x3',
    'plot_penetrance_classification'
]
