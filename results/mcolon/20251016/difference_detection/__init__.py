"""
Unified difference detection module with common interface.

This module provides two approaches for detecting phenotypic differences:
1. Classification-based: Logistic regression with permutation testing
2. Distribution-based: Energy distance and Hotelling's T² tests

Both approaches return compatible output formats for downstream integration.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# Import classification methods
from .classification import (
    predictive_signal_test,
    compute_embryo_penetrance,
    summarize_penetrance,
    get_high_penetrance_embryos
)


def run_classification_test(
    df_binned: pd.DataFrame,
    group1: str,
    group2: str,
    group_col: str = "genotype",
    time_col: str = "time_bin",
    z_cols: Optional[list] = None,
    n_cv_splits: int = 5,
    n_permutations: int = 100,
    use_class_weights: bool = True,
    random_state: int = 42,
    alpha: float = 0.05,
    confidence_threshold: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """
    Run classification-based difference detection.
    
    Uses logistic regression with permutation testing to detect when genotype
    can be predicted from morphological features better than chance.
    
    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data with latent features
    group1 : str
        First group label (e.g., "wildtype")
    group2 : str
        Second group label (e.g., "homozygous")
    group_col : str, default="genotype"
        Column name containing group labels
    time_col : str, default="time_bin"
        Column name containing time bins
    z_cols : list, optional
        Latent feature columns (auto-detected if None)
    n_cv_splits : int, default=5
        Number of cross-validation splits for AUROC
    n_permutations : int, default=100
        Number of permutations for null distribution
        Can override via MORPHSEQ_N_PERMUTATIONS env var
    use_class_weights : bool, default=True
        Use balanced class weights to handle imbalance
    random_state : int, default=42
        Random seed for reproducibility
    alpha : float, default=0.05
        Significance threshold for onset detection
    confidence_threshold : float, default=0.1
        Threshold for "confident" predictions
    **kwargs
        Additional arguments passed to underlying functions
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'time_results': pd.DataFrame with AUROC stats per time bin
        - 'embryo_results': pd.DataFrame with per-embryo penetrance metrics
        - 'onset_info': dict with onset time and significance
        - 'comparison_info': dict with metadata about the comparison
    
    Examples
    --------
    >>> results = run_classification_test(
    ...     df_binned,
    ...     group1="wildtype",
    ...     group2="homozygous"
    ... )
    >>> print(f"Onset at {results['onset_info']['onset_time']} hpf")
    """
    # Check environment variable for permutation override
    n_permutations = int(os.environ.get("MORPHSEQ_N_PERMUTATIONS", n_permutations))
    
    # Filter to just the two groups of interest
    df_subset = df_binned[df_binned[group_col].isin([group1, group2])].copy()
    
    if df_subset.empty:
        raise ValueError(f"No data found for groups {group1} and {group2}")
    
    # Run predictive signal test
    df_time_results, df_embryo_probs = predictive_signal_test(
        df_subset,
        group_col=group_col,
        time_col=time_col,
        z_cols=z_cols,
        n_splits=n_cv_splits,
        n_perm=n_permutations,
        random_state=random_state,
        return_embryo_probs=True,
        use_class_weights=use_class_weights
    )
    
    # Compute embryo-level penetrance metrics
    df_embryo_results = None
    if df_embryo_probs is not None and not df_embryo_probs.empty:
        df_embryo_results = compute_embryo_penetrance(
            df_embryo_probs,
            confidence_threshold=confidence_threshold
        )
    
    # Detect onset time (first significant time bin)
    onset_info = _detect_onset(df_time_results, alpha=alpha)
    
    # Compile comparison metadata
    comparison_info = {
        'group1': group1,
        'group2': group2,
        'method': 'classification',
        'n_permutations': n_permutations,
        'n_cv_splits': n_cv_splits,
        'use_class_weights': use_class_weights,
        'alpha': alpha,
        'random_state': random_state
    }
    
    return {
        'time_results': df_time_results,
        'embryo_results': df_embryo_results,
        'embryo_probs': df_embryo_probs,  # Include raw probabilities too
        'onset_info': onset_info,
        'comparison_info': comparison_info
    }


def run_distribution_test(
    df_binned: pd.DataFrame,
    group1: str,
    group2: str,
    group_col: str = "genotype",
    time_col: str = "time_bin",
    z_cols: Optional[list] = None,
    n_permutations: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """
    Run distribution-based difference detection (energy distance).
    
    Uses energy distance and Hotelling's T² to detect distributional
    differences between groups.
    
    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data with latent features
    group1 : str
        First group label
    group2 : str
        Second group label
    group_col : str, default="genotype"
        Column name containing group labels
    time_col : str, default="time_bin"
        Column name containing time bins
    z_cols : list, optional
        Latent feature columns (auto-detected if None)
    n_permutations : int, default=1000
        Number of permutations for significance testing
    random_state : int, default=42
        Random seed for reproducibility
    alpha : float, default=0.05
        Significance threshold for onset detection
    **kwargs
        Additional arguments
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'time_results': pd.DataFrame with energy distance stats per time bin
        - 'embryo_results': None (not applicable for distribution tests)
        - 'onset_info': dict with onset time and significance
        - 'comparison_info': dict with metadata about the comparison
    
    Notes
    -----
    This method is not yet implemented. Placeholder for future integration.
    """
    raise NotImplementedError(
        "Distribution-based testing not yet implemented. "
        "Use run_classification_test() for now."
    )


def _detect_onset(
    df_time_results: pd.DataFrame,
    alpha: float = 0.05,
    time_col: str = "time_bin"
) -> Dict[str, Any]:
    """
    Detect onset time from time-resolved results.
    
    Parameters
    ----------
    df_time_results : pd.DataFrame
        Time-resolved test results with 'pval' column
    alpha : float
        Significance threshold
    time_col : str
        Name of time column
    
    Returns
    -------
    dict
        Onset information with keys:
        - onset_time: float or None
        - pvalue: float or None
        - is_significant: bool
    """
    if df_time_results.empty:
        return {
            'onset_time': None,
            'pvalue': None,
            'is_significant': False
        }
    
    # Find first significant time bin
    df_sig = df_time_results[df_time_results['pval'] < alpha]
    
    if df_sig.empty:
        return {
            'onset_time': None,
            'pvalue': None,
            'is_significant': False
        }
    
    # Get earliest significant time
    onset_idx = df_sig[time_col].idxmin()
    onset_row = df_sig.loc[onset_idx]
    
    return {
        'onset_time': onset_row[time_col],
        'pvalue': onset_row['pval'],
        'is_significant': True
    }


__all__ = [
    # Main interface
    'run_classification_test',
    'run_distribution_test',
    
    # Classification exports
    'predictive_signal_test',
    'compute_embryo_penetrance',
    'summarize_penetrance',
    'get_high_penetrance_embryos',
]
