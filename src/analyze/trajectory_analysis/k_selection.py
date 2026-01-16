"""
DEPRECATED: This module has moved to trajectory_analysis.clustering.k_selection

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.clustering import (
        evaluate_k_range,
        plot_k_selection,
        run_k_selection_pipeline,
        run_two_phase_pipeline,
        run_k_selection_with_plots,
        add_membership_column,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.k_selection is deprecated. "
    "Import from trajectory_analysis.clustering instead.",
    DeprecationWarning,
    stacklevel=2
)

from .clustering.k_selection import (
    evaluate_k_range,
    plot_k_selection,
    run_k_selection_pipeline,
    run_two_phase_pipeline,
    run_k_selection_with_plots,
    add_membership_column,
)

__all__ = [
    'evaluate_k_range',
    'plot_k_selection',
    'run_k_selection_pipeline',
    'run_two_phase_pipeline',
    'run_k_selection_with_plots',
    'add_membership_column',
]
