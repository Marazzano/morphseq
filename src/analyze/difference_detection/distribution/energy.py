"""
Energy distance and related tests for distribution-based difference detection.

Provides statistical methods for testing whether two multivariate distributions
are significantly different, including energy distance and Hotelling's T2 test.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform


def compute_energy_distance(
    X1: np.ndarray,
    X2: np.ndarray
) -> float:
    """
    Compute energy distance between two multivariate distributions.

    Energy distance is a statistical metric for measuring distance between
    probability distributions. It equals zero if and only if the two
    distributions are identical.

    The energy distance is defined as:
        E(P, Q) = E[|X - Y|] - 0.5*E[|X - X'|] - 0.5*E[|Y - Y'|]
    where:
    - X, X' are iid samples from P
    - Y, Y' are iid samples from Q
    - |.| denotes the Euclidean distance

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution

    Returns
    -------
    float
        Energy distance (non-negative)

    References
    ----------
    Szekely, G. J., & Rizzo, M. L. (2013). Energy statistics: a new approach.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.

    Examples
    --------
    >>> X1 = np.random.randn(100, 10)
    >>> X2 = np.random.randn(100, 10) + 0.5  # Shifted distribution
    >>> energy = compute_energy_distance(X1, X2)
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Compute pairwise distances within X1
    if n1 > 1:
        dist_X1 = squareform(pdist(X1))
        mean_X1_X1 = np.mean(dist_X1[np.triu_indices_from(dist_X1, k=1)])
    else:
        mean_X1_X1 = 0

    # Compute pairwise distances within X2
    if n2 > 1:
        dist_X2 = squareform(pdist(X2))
        mean_X2_X2 = np.mean(dist_X2[np.triu_indices_from(dist_X2, k=1)])
    else:
        mean_X2_X2 = 0

    # Compute pairwise distances between X1 and X2
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    mean_X1_X2 = np.mean(distances)

    # Compute energy distance
    energy = mean_X1_X2 - 0.5 * mean_X1_X1 - 0.5 * mean_X2_X2

    return max(0.0, energy)  # Ensure non-negative


def permutation_test_energy(
    X1: np.ndarray,
    X2: np.ndarray,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Permutation test for energy distance.

    Tests the null hypothesis that X1 and X2 come from the same distribution
    by computing the energy distance on permuted data.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    n_permutations : int, default=1000
        Number of permutations to perform
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - 'energy': float, observed energy distance
        - 'pvalue': float, permutation p-value (two-tailed)
        - 'null_distribution': np.ndarray, energy distances under null

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = permutation_test_energy(X1, X2, n_permutations=100)
    >>> print(f"p-value: {result['pvalue']}")
    """
    rng = np.random.RandomState(random_state)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    # Compute observed energy distance
    observed_energy = compute_energy_distance(X1, X2)

    # Combine data and permute
    combined = np.vstack([X1, X2])
    n1 = X1.shape[0]

    null_distribution = []

    for _ in range(n_permutations):
        # Permute indices
        perm_idx = rng.permutation(combined.shape[0])
        perm_X1 = combined[perm_idx[:n1]]
        perm_X2 = combined[perm_idx[n1:]]

        # Compute energy distance on permuted data
        perm_energy = compute_energy_distance(perm_X1, perm_X2)
        null_distribution.append(perm_energy)

    null_distribution = np.array(null_distribution)

    # Compute p-value (two-tailed)
    pvalue = np.mean(null_distribution >= observed_energy)

    return {
        'energy': observed_energy,
        'pvalue': pvalue,
        'null_distribution': null_distribution
    }


def hotellings_t2_test(
    X1: np.ndarray,
    X2: np.ndarray
) -> Dict[str, Any]:
    """
    Hotelling's T2 test for multivariate mean difference.

    Tests whether two multivariate distributions have the same mean.
    This is the multivariate generalization of the t-test.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution

    Returns
    -------
    dict
        Dictionary with keys:
        - 't2_statistic': float, Hotelling's T2 value
        - 'f_statistic': float, F-distributed test statistic
        - 'pvalue': float, p-value under null hypothesis
        - 'df1': int, numerator degrees of freedom
        - 'df2': int, denominator degrees of freedom
        - 'mean_diff': np.ndarray, difference in means

    Notes
    -----
    Returns NaN for p-value if covariance matrix is singular or if
    degrees of freedom are invalid.

    References
    ----------
    Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979).
    Multivariate Analysis. Academic Press.

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = hotellings_t2_test(X1, X2)
    >>> print(f"p-value: {result['pvalue']}")
    """
    from scipy.stats import f

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    n1, p = X1.shape
    n2 = X2.shape[0]

    # Means and covariance
    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)
    mean_diff = mu1 - mu2

    # Pooled covariance matrix
    S1 = np.cov(X1.T, ddof=1) if n1 > 1 else np.zeros((p, p))
    S2 = np.cov(X2.T, ddof=1) if n2 > 1 else np.zeros((p, p))

    if n1 > 1 and n2 > 1:
        S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    elif n1 > 1:
        S_pooled = S1
    elif n2 > 1:
        S_pooled = S2
    else:
        return {
            't2_statistic': np.nan,
            'f_statistic': np.nan,
            'pvalue': np.nan,
            'df1': p,
            'df2': np.nan,
            'mean_diff': mean_diff
        }

    # Add regularization for numerical stability
    S_pooled += np.eye(p) * 1e-10

    # Compute T2 statistic
    try:
        S_inv = np.linalg.inv(S_pooled)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_pooled)

    t2_stat = (n1 * n2) / (n1 + n2) * mean_diff @ S_inv @ mean_diff

    # Convert to F statistic
    df1 = p
    df2 = n1 + n2 - p - 1

    if df2 > 0:
        f_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * t2_stat
        pvalue = 1 - f.cdf(f_stat, df1, df2)
    else:
        f_stat = np.nan
        pvalue = np.nan

    return {
        't2_statistic': t2_stat,
        'f_statistic': f_stat,
        'pvalue': pvalue,
        'df1': df1,
        'df2': df2,
        'mean_diff': mean_diff
    }
