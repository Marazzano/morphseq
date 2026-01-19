"""
Faceted time series plotting for multi-experiment analysis.

Extends plot_time_series_by_group with faceting capability to create
subplot grids for comparing multiple experiments/conditions side-by-side.

Uses generic time-series algorithms from utils.timeseries (DTW, DBA, interpolation).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List

# Generic time-series imports (no trajectory_analysis dependencies)
from src.analyze.utils.timeseries.dba import dba
from src.analyze.utils.timeseries.dtw import compute_dtw_distance
from src.analyze.utils.timeseries.interpolation import (
    interpolate_to_common_grid,
    pad_trajectories,
)


def _plot_single_group_on_axis(
    ax: plt.Axes,
    group_df: pd.DataFrame,
    group: str,
    color: tuple,
    metric: str,
    time_col: str,
    id_col: str,
    show_individual: bool,
    show_sd_band: bool,
    trend_method: Optional[str],
    smooth_window: Optional[int],
    alpha_individual: float,
    alpha_trend: float,
    linewidth_individual: float,
    linewidth_trend: float
) -> None:
    """
    Plot a single group's trajectories on a given axis.

    Helper function used by plot_time_series_faceted.
    """
    # Plot individual trajectories
    if show_individual:
        for entity_id in group_df[id_col].unique():
            entity_df = group_df[group_df[id_col] == entity_id].sort_values(time_col)

            # Apply smoothing if requested
            if smooth_window is not None:
                y_values = entity_df[metric].rolling(
                    window=smooth_window,
                    center=True,
                    min_periods=1
                ).mean()
            else:
                y_values = entity_df[metric]

            ax.plot(
                entity_df[time_col],
                y_values,
                color=color,
                alpha=alpha_individual,
                linewidth=linewidth_individual,
                zorder=1
            )

    # Compute trend and std per timepoint
    if trend_method is not None or show_sd_band:
        # Prepare data for DBA or averaging
        # Convert to long format for interpolation
        group_long = (
            group_df[[id_col, time_col, metric]]
            .rename(columns={
                id_col: 'embryo_id',
                time_col: 'hpf',
                metric: 'metric_value'
            })
        )

        # Interpolate to common grid and align trajectories
        try:
            interp_trajs, ids_ordered, orig_lens, common_grid = interpolate_to_common_grid(
                group_long, grid_step=0.5, verbose=False
            )
            if len(interp_trajs) == 0:
                raise ValueError("No trajectories available after interpolation")

            padded_trajs = pad_trajectories(
                trajectories=interp_trajs,
                common_grid=common_grid,
                df_long=group_long,
                ids=ids_ordered,
                verbose=False
            )

            padded_array = np.asarray(padded_trajs, dtype=float)
            if padded_array.ndim == 1:
                padded_array = padded_array[np.newaxis, :]

            valid_mask = ~np.isnan(padded_array).all(axis=0)
            if not valid_mask.any():
                raise ValueError("No valid timepoints after padding")

            padded_array = padded_array[:, valid_mask]
            aligned_grid = common_grid[valid_mask]

            with np.errstate(invalid='ignore'):
                std_values = np.nanstd(padded_array, axis=0)

            trend_values = None
            if trend_method is not None:
                agg_method = trend_method.lower()

                if agg_method == 'dba' and len(interp_trajs) > 1:
                    try:
                        # Convert smooth_window to smooth_sigma for DBA and use as DTW window
                        if smooth_window is not None:
                            smooth_sigma = smooth_window / 2.0
                            dtw_window = smooth_window
                        else:
                            smooth_sigma = 0.0
                            dtw_window = 3  # Fallback to default DTW window

                        # Define DTW function for DBA
                        def dtw_func(seq1, seq2):
                            dist = compute_dtw_distance(seq1, seq2, window=dtw_window)
                            # Create approximate path (diagonal alignment)
                            min_len = min(len(seq1), len(seq2))
                            path = [(i, i) for i in range(min_len)]
                            return path, dist

                        # Compute DBA barycenter
                        barycenter = dba(
                            interp_trajs,
                            dtw_func=dtw_func,
                            weights=None,
                            max_iter=10,
                            smooth_sigma=smooth_sigma,
                            verbose=False
                        )

                        orig_grid = np.linspace(aligned_grid[0], aligned_grid[-1], len(barycenter))
                        trend_values = np.interp(
                            aligned_grid,
                            orig_grid,
                            barycenter,
                            left=np.nan,
                            right=np.nan
                        )

                    except Exception as e:
                        agg_method = 'mean'  # Fall back to mean

                # Compute mean or median if DBA not used or failed
                if trend_values is None:
                    with np.errstate(invalid='ignore'):
                        if agg_method == 'median':
                            trend_values = np.nanmedian(padded_array, axis=0)
                        else:  # mean
                            trend_values = np.nanmean(padded_array, axis=0)

            trend_df = pd.DataFrame({
                time_col: aligned_grid,
                'trend': trend_values if trend_values is not None else np.nan,
                'std': std_values
            }).dropna(subset=['trend', 'std'], how='all')

            if trend_df.empty:
                raise ValueError("Empty statistics dataframe after alignment")

        except Exception as e:
            # Fallback to simple groupby
            agg_funcs = ['std']
            if trend_method is not None:
                if trend_method.lower() == 'median':
                    agg_funcs.append('median')
                else:
                    agg_funcs.append('mean')

            trend_df = (
                group_df
                .groupby(time_col)[metric]
                .agg(agg_funcs)
                .reset_index()
            )

            # Apply smoothing if requested
            if smooth_window is not None:
                if trend_method is not None:
                    col_name = 'median' if trend_method.lower() == 'median' else 'mean'
                    trend_df['trend'] = trend_df[col_name].rolling(
                        window=smooth_window,
                        center=True,
                        min_periods=1
                    ).mean()
                else:
                    trend_df['trend'] = np.nan

                trend_df['std'] = trend_df['std'].rolling(
                    window=smooth_window,
                    center=True,
                    min_periods=1
                ).mean()
            else:
                if trend_method is not None:
                    col_name = 'median' if trend_method.lower() == 'median' else 'mean'
                    trend_df['trend'] = trend_df[col_name]
                else:
                    trend_df['trend'] = np.nan

        # Plot SD band
        if show_sd_band:
            ax.fill_between(
                trend_df[time_col],
                trend_df['trend'] - trend_df['std'],
                trend_df['trend'] + trend_df['std'],
                color=color,
                alpha=0.2,
                zorder=2
            )

        # Plot trend line
        if trend_method is not None:
            ax.plot(
                trend_df[time_col],
                trend_df['trend'],
                color=color,
                alpha=alpha_trend,
                linewidth=linewidth_trend,
                label=str(group),
                zorder=3
            )


def plot_time_series_faceted(
    df: pd.DataFrame,
    metric: str = 'metric_value',
    time_col: str = 'hpf',
    id_col: str = 'id',
    color_by: str = 'group',
    facet_by: str = 'experiment',
    show_individual: bool = True,
    trend_method: Optional[str] = 'mean',
    show_sd_band: bool = False,
    smooth_window: Optional[int] = None,
    alpha_individual: float = 0.3,
    alpha_trend: float = 0.9,
    linewidth_individual: float = 0.8,
    linewidth_trend: float = 2.5,
    figsize_per_panel: Tuple[float, float] = (6, 5),
    facet_ncols: Optional[int] = None,
    facet_sharex: bool = True,
    facet_sharey: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 100,
    palette: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series trajectories with faceting for multi-experiment comparison.

    Creates a grid of subplots, one per unique value in `facet_by` column.
    Within each subplot, trajectories are colored by `color_by` variable.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with time series data.
        Required columns: id_col, time_col, metric, color_by, facet_by
    metric : str, default='metric_value'
        Column name for metric to plot on y-axis
    time_col : str, default='hpf'
        Column name for time (x-axis)
    id_col : str, default='id'
        Column name for trajectory identifiers
    color_by : str, default='group'
        Column name to use for grouping and coloring trajectories
    facet_by : str, default='experiment'
        Column name to use for creating separate subplots (facets).
        Each unique value will get its own panel.
    show_individual : bool, default=True
        If True, show individual trajectories (light lines)
    trend_method : str or None, default='mean'
        Method for computing group consensus trajectory. If None, no trend shown.
        Options:
        - 'dba': DTW Barycenter Averaging (time-warped, most accurate but slow)
        - 'mean': Arithmetic mean per timepoint (fast, standard)
        - 'median': Median per timepoint (robust to outliers)
        - None: Don't show trend line
    show_sd_band : bool, default=False
        If True, show +/-1 standard deviation band around trend
    smooth_window : int, optional, default=None
        Window size for smoothing. If None, no smoothing applied.
        For DBA: converted to Gaussian sigma (smooth_window/2.0).
        For mean/median: used as rolling window size.
    alpha_individual : float, default=0.3
        Transparency for individual trajectory lines
    alpha_trend : float, default=0.9
        Transparency for trend lines
    linewidth_individual : float, default=0.8
        Line width for individual trajectories
    linewidth_trend : float, default=2.5
        Line width for trend lines
    figsize_per_panel : tuple, default=(6, 5)
        Size of each subplot panel (width, height) in inches
    facet_ncols : int, optional
        Number of columns in facet grid. If None, uses sqrt(n_facets).
    facet_sharex : bool, default=True
        Share x-axis across facets
    facet_sharey : bool, default=True
        Share y-axis across facets
    title : str, optional
        Overall figure title
    xlabel : str, optional
        X-axis label. If None, uses time_col
    ylabel : str, optional
        Y-axis label. If None, uses metric
    save_path : str or Path, optional
        Path to save figure
    dpi : int, default=100
        Resolution for saved figure
    palette : str, optional
        Seaborn color palette name

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the faceted plot

    Examples
    --------
    >>> # Compare groups across experiments
    >>> fig = plot_time_series_faceted(
    ...     df_all,
    ...     color_by='group',
    ...     facet_by='experiment',
    ...     facet_ncols=3
    ... )
    """
    # Validate required columns
    required_cols = [metric, time_col, id_col, color_by, facet_by]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to non-null values
    df = df[required_cols].dropna().copy()

    if len(df) == 0:
        raise ValueError("No valid data after filtering NaN values")

    # Get unique groups for coloring
    groups = sorted(df[color_by].unique())
    n_groups = len(groups)

    # Set up colors
    if palette is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_groups]
    else:
        colors = sns.color_palette(palette, n_groups)

    group_colors = dict(zip(groups, colors))

    # Get unique facet values
    facets = sorted(df[facet_by].unique())
    n_facets = len(facets)

    # Determine grid layout
    if facet_ncols is None:
        # Smart default: sqrt(n) rounded up
        facet_ncols = int(np.ceil(np.sqrt(n_facets)))
    facet_nrows = int(np.ceil(n_facets / facet_ncols))

    # Create subplot grid
    total_width = figsize_per_panel[0] * facet_ncols
    total_height = figsize_per_panel[1] * facet_nrows
    fig, axes = plt.subplots(
        facet_nrows, facet_ncols,
        figsize=(total_width, total_height),
        dpi=dpi,
        sharex=facet_sharex,
        sharey=facet_sharey,
        squeeze=False  # Always return 2D array
    )
    axes = axes.flatten()  # Flatten to 1D for easy iteration

    # Hide unused subplots
    for idx in range(n_facets, len(axes)):
        axes[idx].set_visible(False)

    # Plot each facet
    for facet_idx, facet_value in enumerate(facets):
        ax = axes[facet_idx]
        facet_df = df[df[facet_by] == facet_value].copy()

        # Add facet title
        ax.set_title(str(facet_value), fontsize=11, fontweight='bold', pad=8)

        # Plot each group within this facet
        for group in groups:
            group_df = facet_df[facet_df[color_by] == group].copy()

            if len(group_df) == 0:
                continue

            color = group_colors[group]

            _plot_single_group_on_axis(
                ax, group_df, group, color,
                metric, time_col, id_col,
                show_individual, show_sd_band,
                trend_method, smooth_window,
                alpha_individual, alpha_trend,
                linewidth_individual, linewidth_trend
            )

        # Set labels
        if xlabel is None:
            xlabel_text = time_col.replace('_', ' ').title()
        else:
            xlabel_text = xlabel

        if ylabel is None:
            ylabel_text = metric.replace('_', ' ').title()
        else:
            ylabel_text = ylabel

        # Only show x-label on bottom row
        if facet_idx >= len(facets) - facet_ncols:
            ax.set_xlabel(xlabel_text, fontsize=10)

        # Only show y-label on left column
        if facet_idx % facet_ncols == 0:
            ax.set_ylabel(ylabel_text, fontsize=10)

        # Add legend to first panel only
        if trend_method is not None and facet_idx == 0:
            ax.legend(
                title=color_by.replace('_', ' ').title(),
                loc='best',
                fontsize=9,
                title_fontsize=10
            )

        # Add grid
        ax.grid(True, alpha=0.3)

    # Set overall title
    if title is None:
        if ylabel is None:
            ylabel_text = metric.replace('_', ' ').title()
        title = f"{ylabel_text} Across {facet_by.replace('_', ' ').title()}"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved faceted plot: {save_path}")

    return fig


def plot_embryos_metric_over_time_faceted(
    df: pd.DataFrame,
    metric: str = 'normalized_baseline_deviation',
    time_col: str = 'predicted_stage_hpf',
    embryo_col: str = 'embryo_id',
    color_by: str = 'genotype',
    facet_by: str = 'experiment_id',
    show_individual: bool = True,
    trend_method: Optional[str] = 'mean',
    show_sd_band: bool = False,
    smooth_window: Optional[int] = None,
    alpha_individual: float = 0.3,
    alpha_trend: float = 0.9,
    linewidth_individual: float = 0.8,
    linewidth_trend: float = 2.5,
    figsize_per_panel: Tuple[float, float] = (6, 5),
    facet_ncols: Optional[int] = None,
    facet_sharex: bool = True,
    facet_sharey: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 100,
    palette: Optional[str] = None,
) -> plt.Figure:
    """
    Backward-compatible wrapper for plot_time_series_faceted.
    """
    warnings.warn(
        "plot_embryos_metric_over_time_faceted is deprecated. "
        "Use plot_time_series_faceted instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_time_series_faceted(
        df,
        metric=metric,
        time_col=time_col,
        id_col=embryo_col,
        color_by=color_by,
        facet_by=facet_by,
        show_individual=show_individual,
        trend_method=trend_method,
        show_sd_band=show_sd_band,
        smooth_window=smooth_window,
        alpha_individual=alpha_individual,
        alpha_trend=alpha_trend,
        linewidth_individual=linewidth_individual,
        linewidth_trend=linewidth_trend,
        figsize_per_panel=figsize_per_panel,
        facet_ncols=facet_ncols,
        facet_sharex=facet_sharex,
        facet_sharey=facet_sharey,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        dpi=dpi,
        palette=palette,
    )
