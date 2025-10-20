"""
Visualization functions for phenotype emergence analysis.

This package provides plotting functions for:
- Classification: AUROC, trajectories, penetrance
- Distribution: Energy distance, bootstrap, LOO (future)
"""

# Classification visualization
from .classification_plots import (
    plot_auroc_over_time,
    plot_auroc_with_significance,
    summarize_significant_bins
)

from .trajectory_plots import (
    plot_signed_margin_trajectories,
    plot_signed_margin_heatmap
)

from .penetrance_plots import (
    plot_penetrance_distribution,
    plot_penetrance_summary_by_genotype
)

__all__ = [
    # AUROC plots
    'plot_auroc_over_time',
    'plot_auroc_with_significance',
    'summarize_significant_bins',
    # Trajectory plots
    'plot_signed_margin_trajectories',
    'plot_signed_margin_heatmap',
    # Penetrance plots
    'plot_penetrance_distribution',
    'plot_penetrance_summary_by_genotype',
]
