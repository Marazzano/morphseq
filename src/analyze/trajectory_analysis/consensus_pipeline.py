"""
DEPRECATED: This module has moved to trajectory_analysis.clustering.consensus_pipeline

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.clustering import (
        run_consensus_pipeline,
        create_filtering_log,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.consensus_pipeline is deprecated. "
    "Import from trajectory_analysis.clustering instead.",
    DeprecationWarning,
    stacklevel=2
)

from .clustering.consensus_pipeline import (
    run_consensus_pipeline,
    create_filtering_log,
)

__all__ = [
    'run_consensus_pipeline',
    'create_filtering_log',
]
