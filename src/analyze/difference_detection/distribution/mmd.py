"""
Maximum Mean Discrepancy (MMD) for distribution-based testing.

Provides implementations of MMD and related permutation tests for comparing
multivariate distributions without assuming specific functional forms.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


def compute_rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    bandwidth: Optional[float] = None
) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix between two point sets.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        First set of points
    X2 : np.ndarray, shape (n2, p)
        Second set of points
    bandwidth : float, optional
        Kernel bandwidth (sigma). If None, uses median heuristic.

    Returns
    -------
    np.ndarray, shape (n1, n2)
        Kernel matrix K(x, y) = exp(-||x-y||^2/(2*bandwidth^2))
    """
    # Compute pairwise distances
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Squared distances: ||x - y||^2
    sq_norms_1 = np.sum(X1**2, axis=1, keepdims=True)  # (n1, 1)
    sq_norms_2 = np.sum(X2**2, axis=1, keepdims=True)  # (n2, 1)

    sq_distances = sq_norms_1 + sq_norms_2.T - 2 * X1 @ X2.T

    # Ensure non-negative (numerical stability)
    sq_distances = np.maximum(sq_distances, 0)

    # Determine bandwidth if not provided
    if bandwidth is None:
        bandwidth = estimate_bandwidth_median(np.vstack([X1, X2]))

    # Compute kernel
    kernel = np.exp(-sq_distances / (2 * bandwidth**2))

    return kernel


def estimate_bandwidth_median(X: np.ndarray) -> float:
    """
    Estimate kernel bandwidth using median heuristic.

    The median heuristic sets bandwidth to the median pairwise distance
    in the pooled sample.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Data points

    Returns
    -------
    float
        Estimated bandwidth
    """
    n = X.shape[0]

    if n <= 1:
        return 1.0

    # Compute pairwise distances
    sq_distances = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * X @ X.T
    sq_distances = np.maximum(sq_distances, 0)
    distances = np.sqrt(sq_distances)

    # Return median of upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    distances_upper = distances[triu_idx]

    if len(distances_upper) > 0:
        bandwidth = np.median(distances_upper)
    else:
        bandwidth = 1.0

    return max(bandwidth, 1e-6)  # Avoid zero bandwidth


def compute_mmd(
    X1: np.ndarray,
    X2: np.ndarray,
    bandwidth: Optional[float] = None
) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.

    MMD is a distance metric between probability distributions that equals
    zero if and only if the two distributions are identical. It can be
    estimated from finite samples and is computationally efficient.

    The squared MMD is:
        MMD^2(X, Y) = E_x,x'[k(x,x')] + E_y,y'[k(y,y')] - 2*E_x,y[k(x,y)]
    where k is a kernel function (here, RBF).

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    bandwidth : float, optional
        Kernel bandwidth. If None, uses median heuristic.

    Returns
    -------
    float
        Maximum Mean Discrepancy value (non-negative)

    References
    ----------
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
    (2012). A kernel two-sample test. The journal of machine learning research,
    13(1), 723-773.

    Examples
    --------
    >>> X1 = np.random.randn(100, 10)
    >>> X2 = np.random.randn(100, 10) + 0.3
    >>> mmd = compute_mmd(X1, X2)
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Compute kernels
    K11 = compute_rbf_kernel(X1, X1, bandwidth=bandwidth)
    K22 = compute_rbf_kernel(X2, X2, bandwidth=bandwidth)
    K12 = compute_rbf_kernel(X1, X2, bandwidth=bandwidth)

    # Compute MMD^2 (unbiased estimator)
    # E[k(X,X')] using upper triangle (excluding diagonal)
    mean_K11 = np.sum(np.triu(K11, k=1)) / (n1 * (n1 - 1)) if n1 > 1 else 0
    mean_K22 = np.sum(np.triu(K22, k=1)) / (n2 * (n2 - 1)) if n2 > 1 else 0

    # E[k(X,Y)]
    mean_K12 = np.mean(K12)

    # Compute squared MMD
    mmd_squared = mean_K11 + mean_K22 - 2 * mean_K12

    # Return MMD (ensure non-negative)
    mmd = np.sqrt(max(0.0, mmd_squared))

    return mmd


def permutation_test_mmd(
    X1: np.ndarray,
    X2: np.ndarray,
    n_permutations: int = 1000,
    bandwidth: Optional[float] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Permutation test for MMD.

    Tests the null hypothesis that X1 and X2 come from the same distribution
    by computing MMD on permuted data.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    n_permutations : int, default=1000
        Number of permutations to perform
    bandwidth : float, optional
        Kernel bandwidth. If None, uses median heuristic.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mmd': float, observed MMD value
        - 'pvalue': float, permutation p-value (two-tailed)
        - 'null_distribution': np.ndarray, MMD values under null

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = permutation_test_mmd(X1, X2, n_permutations=100)
    >>> print(f"p-value: {result['pvalue']}")
    """
    rng = np.random.RandomState(random_state)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    # Estimate bandwidth from combined data if not provided
    if bandwidth is None:
        combined = np.vstack([X1, X2])
        bandwidth = estimate_bandwidth_median(combined)

    # Compute observed MMD
    observed_mmd = compute_mmd(X1, X2, bandwidth=bandwidth)

    # Combine data and permute
    combined = np.vstack([X1, X2])
    n1 = X1.shape[0]

    null_distribution = []

    for _ in range(n_permutations):
        # Permute indices
        perm_idx = rng.permutation(combined.shape[0])
        perm_X1 = combined[perm_idx[:n1]]
        perm_X2 = combined[perm_idx[n1:]]

        # Compute MMD on permuted data
        perm_mmd = compute_mmd(perm_X1, perm_X2, bandwidth=bandwidth)
        null_distribution.append(perm_mmd)

    null_distribution = np.array(null_distribution)

    # Compute p-value (two-tailed)
    pvalue = np.mean(null_distribution >= observed_mmd)

    return {
        'mmd': observed_mmd,
        'pvalue': pvalue,
        'null_distribution': null_distribution,
        'bandwidth': bandwidth
    }


def mmd_kernel_width_test(
    X1: np.ndarray,
    X2: np.ndarray,
    bandwidths: Optional[np.ndarray] = None,
    n_permutations: int = 100,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test MMD across multiple kernel bandwidths.

    Useful for robust testing across different scales.

    Parameters
    ----------
    X1, X2 : np.ndarray
        Sample arrays
    bandwidths : np.ndarray, optional
        Bandwidth values to test. If None, uses geometric series.
    n_permutations : int, default=100
        Permutations per bandwidth test
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with per-bandwidth results and summary statistics
    """
    rng = np.random.RandomState(random_state)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    # Generate bandwidths if not provided
    if bandwidths is None:
        base_bandwidth = estimate_bandwidth_median(np.vstack([X1, X2]))
        bandwidths = base_bandwidth * np.array([0.5, 1.0, 2.0, 4.0])

    results_per_bandwidth = {}
    pvalues = []

    for bw in bandwidths:
        result = permutation_test_mmd(
            X1, X2,
            n_permutations=n_permutations,
            bandwidth=bw,
            random_state=rng
        )
        results_per_bandwidth[float(bw)] = result
        pvalues.append(result['pvalue'])

    return {
        'per_bandwidth_results': results_per_bandwidth,
        'bandwidth_values': bandwidths,
        'pvalues': np.array(pvalues),
        'min_pvalue': np.min(pvalues),
        'mean_pvalue': np.mean(pvalues),
        'summary': 'Reject null if any or most bandwidths show significance'
    }
