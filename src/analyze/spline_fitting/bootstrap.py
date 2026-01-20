"""Bootstrap spline fitting with uncertainty estimation.

This module provides bootstrap-based spline fitting to quantify uncertainty in
trajectory curves. It supports both single-spline and grouped multi-spline fitting.

Example:
    >>> from src.analyze.spline_fitting import spline_fit_wrapper
    >>>
    >>> # Fit single spline (backwards compatible)
    >>> wt_spline = spline_fit_wrapper(wt_df, pca_cols=['PC1', 'PC2', 'PC3'])
    >>>
    >>> # Fit multiple splines by group
    >>> all_splines = spline_fit_wrapper(
    ...     df, group_by='phenotype',
    ...     pca_cols=['PC1', 'PC2', 'PC3']
    ... )
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from .lpc_model import LocalPrincipalCurve


def spline_fit_wrapper(
    df,
    pca_cols=None,
    group_by=None,
    stage_col="predicted_stage_hpf",
    bandwidth=0.5,
    h=None,
    max_iter=2500,
    tol=1e-5,
    angle_penalty_exp=1,
    n_bootstrap=10,
    bootstrap_size=2500,
    n_spline_points=500,
    time_window=2,
    obs_weights=None
):
    """Fit splines with bootstrap uncertainty estimation.

    Supports both single-spline fitting (group_by=None) and grouped multi-spline
    fitting (group_by='column_name'). Returns mean spline coordinates with
    standard error estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with trajectory points.
    pca_cols : list of str, optional
        Column names for coordinates to fit spline through.
        If None, auto-detects columns matching pattern 'PCA_.*_bio'.
    group_by : str, optional
        Column name to group by. If provided, fits one spline per group.
        If None, fits single spline to all data (backwards compatible).
    stage_col : str, default='predicted_stage_hpf'
        Column name for developmental stage/time.
    bandwidth : float, default=0.5
        Bandwidth for LocalPrincipalCurve kernel.
    h : float, optional
        Step size for curve progression. Defaults to bandwidth if not specified.
    max_iter : int, default=2500
        Maximum iterations for LPC fitting.
    tol : float, default=1e-5
        Convergence tolerance for LPC.
    angle_penalty_exp : int, default=1
        Exponent for angle penalty in LPC.
    n_bootstrap : int, default=10
        Number of bootstrap iterations.
    bootstrap_size : int, default=2500
        Number of points per bootstrap sample.
    n_spline_points : int, default=500
        Number of points in final spline output.
    time_window : float, default=2
        Time window for selecting anchor points (early/late stage).
    obs_weights : ndarray, optional
        Per-observation weights for bootstrap sampling. If None, uniform weights.

    Returns
    -------
    spline_df : pd.DataFrame
        Fitted spline(s) with columns:
        - pca_cols: mean spline coordinates
        - {col}_se for col in pca_cols: standard errors
        - group_by column (if group_by was provided)
        - 'spline_point_index': point index along spline

    Notes
    -----
    **Backwards Compatibility**:
    When group_by=None, returns single spline DataFrame (original behavior).

    **New Group-by Feature**:
    When group_by='column_name', fits one spline per group and returns
    combined DataFrame with all splines stacked.

    Examples
    --------
    Fit single spline:

    >>> wt_spline = spline_fit_wrapper(
    ...     wt_df,
    ...     pca_cols=['PC1', 'PC2', 'PC3'],
    ...     n_bootstrap=100
    ... )

    Fit splines by phenotype:

    >>> all_splines = spline_fit_wrapper(
    ...     df,
    ...     group_by='phenotype',
    ...     pca_cols=['PC1', 'PC2', 'PC3'],
    ...     n_bootstrap=100
    ... )
    >>> # Returns DataFrame with 'phenotype' column + coordinates + SE columns
    """
    # Handle group_by case: fit multiple splines
    if group_by is not None:
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found in DataFrame")

        groups = df[group_by].unique()
        spline_results = []

        for group_val in tqdm(groups, desc=f"Fitting splines by {group_by}"):
            group_df = df[df[group_by] == group_val].copy()

            if group_df.empty:
                continue

            # Fit single spline for this group
            group_spline = _fit_single_spline(
                df=group_df,
                pca_cols=pca_cols,
                stage_col=stage_col,
                bandwidth=bandwidth,
                h=h,
                max_iter=max_iter,
                tol=tol,
                angle_penalty_exp=angle_penalty_exp,
                n_bootstrap=n_bootstrap,
                bootstrap_size=bootstrap_size,
                n_spline_points=n_spline_points,
                time_window=time_window,
                obs_weights=obs_weights
            )

            # Add group identifier
            group_spline[group_by] = group_val
            spline_results.append(group_spline)

        # Combine all splines
        if spline_results:
            return pd.concat(spline_results, ignore_index=True)
        else:
            return pd.DataFrame()

    # Single spline case (backwards compatible)
    else:
        return _fit_single_spline(
            df=df,
            pca_cols=pca_cols,
            stage_col=stage_col,
            bandwidth=bandwidth,
            h=h,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp,
            n_bootstrap=n_bootstrap,
            bootstrap_size=bootstrap_size,
            n_spline_points=n_spline_points,
            time_window=time_window,
            obs_weights=obs_weights
        )


def _fit_single_spline(
    df,
    pca_cols=None,
    stage_col="predicted_stage_hpf",
    bandwidth=0.5,
    h=None,
    max_iter=2500,
    tol=1e-5,
    angle_penalty_exp=1,
    n_bootstrap=10,
    bootstrap_size=2500,
    n_spline_points=500,
    time_window=2,
    obs_weights=None
):
    """Fit a single spline with bootstrap uncertainty (internal helper)."""
    # Auto-detect PCA columns if not provided
    if pca_cols is None:
        pattern = r"PCA_.*_bio"
        pca_cols = [col for col in df.columns if re.search(pattern, col)]
        if not pca_cols:
            raise ValueError(
                "No PCA columns found. Either provide pca_cols or ensure columns match pattern 'PCA_.*_bio'"
            )

    # Validate columns exist
    missing_cols = set(pca_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if stage_col not in df.columns:
        raise ValueError(f"stage_col '{stage_col}' not found in DataFrame")

    # Setup observation weights
    if obs_weights is None:
        obs_weights = np.ones(df.shape[0])
    obs_weights = obs_weights / np.sum(obs_weights)

    bootstrap_size = min(df.shape[0], bootstrap_size)

    # Extract coordinate array
    coord_array = df[pca_cols].values

    # Compute anchor points (early and late stage)
    min_time = df[stage_col].min()
    early_mask = (df[stage_col] >= min_time) & (df[stage_col] < min_time + time_window)
    early_points = df.loc[early_mask, pca_cols].values

    max_time = df[stage_col].max()
    late_mask = df[stage_col] >= (max_time - time_window)
    late_points = df.loc[late_mask, pca_cols].values

    if len(early_points) == 0 or len(late_points) == 0:
        raise ValueError(
            f"No anchor points found. Check stage_col range and time_window. "
            f"Stage range: [{min_time}, {max_time}], time_window: {time_window}"
        )

    # Bootstrap iterations
    spline_boot_array = np.zeros((n_spline_points, len(pca_cols), n_bootstrap))
    rng = np.random.RandomState(42)

    for n in tqdm(range(n_bootstrap), desc="Bootstrap iterations", leave=False):
        # Sample data
        subset_indices = rng.choice(len(coord_array), size=bootstrap_size, replace=True, p=obs_weights)
        coord_subset = coord_array[subset_indices, :]

        # Random anchor points
        start_idx = rng.choice(len(early_points))
        stop_idx = rng.choice(len(late_points))
        start_point = early_points[start_idx, :]
        stop_point = late_points[stop_idx, :]

        # Fit LPC
        lpc = LocalPrincipalCurve(
            bandwidth=bandwidth,
            h=h,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp
        )

        lpc.fit(
            coord_subset,
            start_points=start_point[None, :],
            end_point=stop_point[None, :],
            num_points=n_spline_points
        )

        spline_boot_array[:, :, n] = lpc.cubic_splines[0]

    # Compute mean and standard error
    mean_spline = np.mean(spline_boot_array, axis=2)
    se_spline = np.std(spline_boot_array, axis=2)

    # Build output DataFrame
    se_cols = [col + "_se" for col in pca_cols]
    spline_df = pd.DataFrame(mean_spline, columns=pca_cols)
    spline_df[se_cols] = se_spline
    spline_df['spline_point_index'] = range(len(spline_df))

    return spline_df
