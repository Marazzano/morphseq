"""
DEPRECATED: This module has moved to trajectory_analysis.clustering.cluster_classification

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.clustering import (
        classify_membership_2d,
        classify_membership_adaptive,
        get_classification_summary,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.cluster_classification is deprecated. "
    "Import from trajectory_analysis.clustering instead.",
    DeprecationWarning,
    stacklevel=2
)

from .clustering.cluster_classification import (
    classify_membership_2d,
    classify_membership_adaptive,
    get_classification_summary,
)

__all__ = [
    'classify_membership_2d',
    'classify_membership_adaptive',
    'get_classification_summary',
]
