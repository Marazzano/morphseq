"""
Classification and penetrance analysis for phenotype emergence.

This package provides:
- Predictive classification with permutation testing
- Embryo-level penetrance metrics
- Class imbalance handling via balanced class weights (default)
"""

from .predictive_test import predictive_signal_test
from .penetrance import (
    compute_embryo_penetrance,
    summarize_penetrance,
    get_high_penetrance_embryos
)

__all__ = [
    'predictive_signal_test',
    'compute_embryo_penetrance',
    'summarize_penetrance',
    'get_high_penetrance_embryos',
]
