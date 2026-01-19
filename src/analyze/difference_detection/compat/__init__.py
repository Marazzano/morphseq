"""
Backward compatibility layer for old difference_detection API.

This module provides deprecated wrappers for old function names and imports.
All functions emit deprecation warnings pointing users to the new API.

DEPRECATED: This module will be removed in a future version.
Use the new API directly from the parent package.
"""

import warnings

# Emit package-level deprecation warning
warnings.warn(
    "The difference_detection.compat module is deprecated. "
    "Update your imports to use the new API:\n"
    "  - Binary classification: classification_test.run_binary_classification_test()\n"
    "  - Multiclass classification: classification_test_multiclass.run_multiclass_classification_test()\n"
    "  - Group labeling: classification_test.assign_group_labels()\n"
    "  - Divergence: classification_test.compute_timeseries_divergence()\n"
    "  - Distribution tests: distribution_test.permutation_test_distribution()\n"
    "  - Penetrance: penetrance_threshold.run_penetrance_threshold_analysis()\n"
    "See refactor.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export old comparison functions with deprecation
from .comparison import add_group_column, compare_groups, compute_metric_divergence
from .comparison_multiclass import compare_groups_multiclass

__all__ = [
    'add_group_column',
    'compare_groups',
    'compute_metric_divergence',
    'compare_groups_multiclass',
]
