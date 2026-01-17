"""
DEPRECATED: This module has moved to trajectory_analysis.viz.dendrogram

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.viz import (
        generate_dendrograms,
        plot_dendrogram,
        add_cluster_column,
        plot_dendrogram_with_categories,
        PASTEL_COLORS,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.dendrogram is deprecated. "
    "Import from trajectory_analysis.viz instead.",
    DeprecationWarning,
    stacklevel=2
)

from .viz.dendrogram import (
    generate_dendrograms,
    plot_dendrogram,
    add_cluster_column,
    plot_dendrogram_with_categories,
    PASTEL_COLORS,
)

__all__ = [
    'generate_dendrograms',
    'plot_dendrogram',
    'add_cluster_column',
    'plot_dendrogram_with_categories',
    'PASTEL_COLORS',
]
