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
    "  - For binary classification: Use classification_test module (when available)\n"
    "  - For multiclass: Use classification_test_multiclass module (when available)\n"
    "  - For distribution tests: Use distribution_test.permutation_test_distribution()\n"
    "  - For penetrance: Use penetrance_threshold.run_penetrance_threshold_analysis()\n"
    "See refactor.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export old comparison functions with deprecation
from .comparison import add_group_column, compare_groups
from .comparison_multiclass import compare_groups_multiclass

__all__ = [
    'add_group_column',
    'compare_groups',
    'compare_groups_multiclass',
]
