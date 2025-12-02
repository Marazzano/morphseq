"""
Regression utilities for predicting continuous targets from embeddings.

This package provides core utilities for continuous regression analysis with
leave-one-embryo-out cross-validation. Modules can be migrated to
src/analyze/difference_detection/classification/ after validation.

Modules
-------
regression.py
    Core regression models and training (Ridge, GradientBoosting)

evaluation.py
    Metrics, cross-validation, and result aggregation

data_prep.py
    Feature/target preparation and data handling

plotting.py
    Visualization and diagnostic plots
"""

from .regression import (
    get_model,
    train_regression_model_loeo,
    train_regression_model_holdout,
    predict_with_trained_model,
)

from .evaluation import (
    compute_regression_metrics,
    aggregate_loeo_results,
    get_feature_importance,
    compare_multiple_models,
    compute_residual_statistics,
    compute_prediction_error_by_group,
)

from .data_prep import (
    prepare_features_and_target,
    validate_data_completeness,
    filter_by_genotype,
    stratify_by_genotype,
)

from .plotting import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_model_comparison,
    plot_metrics_table,
)

__all__ = [
    'get_model',
    'train_regression_model_loeo',
    'train_regression_model_holdout',
    'predict_with_trained_model',
    'compute_regression_metrics',
    'aggregate_loeo_results',
    'get_feature_importance',
    'compare_multiple_models',
    'compute_residual_statistics',
    'compute_prediction_error_by_group',
    'prepare_features_and_target',
    'validate_data_completeness',
    'filter_by_genotype',
    'stratify_by_genotype',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_metrics_table',
]
