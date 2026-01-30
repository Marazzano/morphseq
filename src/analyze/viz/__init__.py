"""
Visualization Utilities

Generic visualization utilities for data analysis. These functions are
domain-agnostic and can be used across different analysis contexts.

Subpackages
===========
- plotting : Time series plotting utilities

For domain-specific trajectory visualizations (genotype styling, phenotype colors),
see: src.analyze.trajectory_analysis.viz
"""

from . import plotting
from .hpf_coverage import (
    experiment_hpf_coverage,
    longest_interval_where,
    plot_hpf_overlap_quick,
)

__all__ = [
    'plotting',
    'experiment_hpf_coverage',
    'longest_interval_where',
    'plot_hpf_overlap_quick',
]
