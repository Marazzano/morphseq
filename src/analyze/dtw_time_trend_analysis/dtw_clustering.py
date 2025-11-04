import numpy as np
from scipy.ndimage import gaussian_filter1d

def dba(series_list, dtw_func, weights=None, max_iter=10, smooth_sigma=0.0, verbose=False):
    """
    Dynamic Time Warping Barycenter Averaging (numerically stable)
    --------------------------------------------------------------
    Args:
        series_list : list of np.ndarray
            Each 1-D array represents a trajectory (lengths may differ).
        dtw_func : callable
            Function returning (path, dist) given two series.
        weights : list or np.ndarray, optional
            Per-series weights. Defaults to uniform.
        max_iter : int
            Number of refinement iterations.
        smooth_sigma : float
            Gaussian smoothing σ (0 disables).
        verbose : bool
            Print per-iteration stats.
    Returns:
        np.ndarray : barycenter trajectory.
    """

    n = len(series_list)
    if n == 0:
        raise ValueError("series_list is empty")

    series_list = [np.asarray(s, dtype=np.float64) for s in series_list]
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= np.maximum(weights.sum(), 1e-8)

    bary = series_list[0].copy()  # init (could replace w/ medoid)

    for it in range(max_iter):
        accum = np.zeros_like(bary)
        counts = np.zeros_like(bary)
        total_cost = 0.0

        for s, w in zip(series_list, weights):
            path, dist = dtw_func(s, bary)
            total_cost += dist * w
            for i, j in path:
                if j < len(accum) and i < len(s):
                    accum[j] += w * s[i]
                    counts[j] += w

        counts = np.maximum(counts, 1e-8)
        bary = np.nan_to_num(accum / counts, nan=0.0, posinf=0.0, neginf=0.0)

        if smooth_sigma > 0:
            bary = gaussian_filter1d(bary, sigma=smooth_sigma)

        if verbose:
            print(f"Iter {it+1}/{max_iter} | mean cost: {total_cost/n:.6f}")

    return bary
