"""
Trajectory Statistical Analysis

Functions for statistical testing and analysis of temporal trajectory patterns.

This module provides utilities for testing hypotheses about trajectory behavior,
particularly for detecting anti-correlated or correlated patterns between
early and late temporal windows.

Functions
=========
- test_anticorrelation : Test for anti-correlated early/late patterns with permutation testing
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Optional


def test_anticorrelation(
    cluster_assignments: np.ndarray,
    early_means: np.ndarray,
    late_means: np.ndarray,
    embryo_ids: List[str],
    correlation_threshold: float = 0.3,
    n_permutations: int = 1000,
    verbose: bool = False
) -> Dict[int, Dict[str, any]]:
    """
    Test for anti-correlated patterns between early and late trajectory phases.

    For each cluster, tests whether early and late mean values show significant
    negative (anti-correlated), positive (correlated), or no correlation using
    Pearson correlation with permutation testing.

    Parameters
    ----------
    cluster_assignments : np.ndarray
        Cluster assignment for each embryo (shape: n_embryos)
    early_means : np.ndarray
        Mean metric values in early window for each embryo (shape: n_embryos)
    late_means : np.ndarray
        Mean metric values in late window for each embryo (shape: n_embryos)
    embryo_ids : list of str
        Embryo IDs (should match length of early_means and late_means)
    correlation_threshold : float, default=0.3
        Absolute correlation magnitude threshold for classification:
        - r < -threshold : "Anti-correlated"
        - r > +threshold : "Correlated"
        - else : "Uncorrelated"
    n_permutations : int, default=1000
        Number of permutations for permutation testing
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    results : dict
        Dictionary with cluster IDs as keys. Each value is a dict containing:
        - n_embryos : int
            Number of valid embryo pairs in cluster
        - early_mean : float
            Mean of early values
        - late_mean : float
            Mean of late values
        - pearson_r : float
            Pearson correlation coefficient
        - p_value : float
            Two-tailed p-value from Pearson correlation
        - permutation_p : float
            Empirical p-value from permutation test
        - interpretation : str
            One of "Anti-correlated", "Correlated", or "Uncorrelated"

    Notes
    -----
    Permutation testing:
    - Randomly shuffles late_means and recomputes correlation
    - P-value = proportion of permutations with |r_perm| >= |r_observed|
    - Provides non-parametric test that doesn't assume normality

    Handling of NaN values:
    - Both early_means and late_means must be non-NaN to form a valid pair
    - Clusters with < 3 valid pairs cannot be tested (insufficient sample size)

    Examples
    --------
    >>> results = test_anticorrelation(
    ...     cluster_assignments,
    ...     early_means,
    ...     late_means,
    ...     embryo_ids,
    ...     correlation_threshold=0.3
    ... )
    >>> for cluster_id, stats in results.items():
    ...     print(f"Cluster {cluster_id}: {stats['interpretation']} "
    ...           f"(r={stats['pearson_r']:.3f}, p={stats['p_value']:.4f})")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 8: ANTI-CORRELATION TEST")
        print(f"{'='*80}")

    anticorr_results = {}
    unique_clusters = np.unique(cluster_assignments)

    for cluster_id in unique_clusters:
        cluster_mask = cluster_assignments == cluster_id
        cluster_embryos = np.array(embryo_ids)[cluster_mask]

        early_vals = early_means[cluster_mask]
        late_vals = late_means[cluster_mask]

        # Filter out NaN pairs
        valid_mask = ~(np.isnan(early_vals) | np.isnan(late_vals))
        early_valid = early_vals[valid_mask]
        late_valid = late_vals[valid_mask]

        n_embryos = len(early_valid)

        if verbose:
            print(f"\n  Cluster {cluster_id}: {n_embryos} embryos")

        if n_embryos >= 3:
            # Pearson correlation
            r, p_value = pearsonr(early_valid, late_valid)

            # Permutation test
            permutation_rs = []
            np.random.seed(42)  # For reproducibility

            for _ in range(n_permutations):
                late_shuffled = np.random.permutation(late_valid)
                r_perm, _ = pearsonr(early_valid, late_shuffled)
                permutation_rs.append(r_perm)

            permutation_p = np.mean(np.abs(np.array(permutation_rs)) >= np.abs(r))

            # Classification
            if r < -correlation_threshold:
                interpretation = "Anti-correlated"
            elif r > correlation_threshold:
                interpretation = "Correlated"
            else:
                interpretation = "Uncorrelated"

            anticorr_results[cluster_id] = {
                'n_embryos': n_embryos,
                'early_mean': np.mean(early_valid),
                'late_mean': np.mean(late_valid),
                'pearson_r': r,
                'p_value': p_value,
                'permutation_p': permutation_p,
                'interpretation': interpretation
            }

            if verbose:
                print(f"    Pearson r: {r:.4f} (p={p_value:.4f})")
                print(f"    Permutation p: {permutation_p:.4f}")
                print(f"    Interpretation: {interpretation}")
        else:
            if verbose:
                print(f"    Insufficient valid pairs (need ≥3)")

    return anticorr_results
