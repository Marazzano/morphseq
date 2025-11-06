"""
General plotting utilities for embryo time series analysis.

This module provides flexible plotting functions for visualizing embryo metric
trajectories over time, with support for grouping by any categorical variable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Tuple


def plot_embryos_metric_over_time(
    df: pd.DataFrame,
    metric: str = 'normalized_baseline_deviation',
    time_col: str = 'predicted_stage_hpf',
    embryo_col: str = 'embryo_id',
    color_by: str = 'genotype',
    show_individual: bool = True,
    show_mean: bool = True,
    show_sd_band: bool = False,
    smooth_window: Optional[int] = 5,
    alpha_individual: float = 0.3,
    alpha_mean: float = 0.8,
    linewidth_individual: float = 0.8,
    linewidth_mean: float = 2.5,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 100,
    palette: Optional[str] = None
) -> plt.Figure:
    """
    Plot embryo metric trajectories over time, colored by grouping variable.

    Flexible function for visualizing time series data grouped by any categorical
    variable (genotype, cluster, phenotype, etc.). Shows individual trajectories
    and/or group means with optional standard deviation bands.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with embryo time series data.
        Required columns: embryo_col, time_col, metric, color_by
    metric : str, default='normalized_baseline_deviation'
        Column name for metric to plot on y-axis
    time_col : str, default='predicted_stage_hpf'
        Column name for time/developmental stage (x-axis)
    embryo_col : str, default='embryo_id'
        Column name for embryo identifiers
    color_by : str, default='genotype'
        Column name to use for grouping and coloring trajectories
        (e.g., 'genotype', 'cluster', 'phenotype')
    show_individual : bool, default=True
        If True, show individual embryo trajectories (light/transparent lines)
    show_mean : bool, default=True
        If True, show mean trajectory per group (bold lines)
    show_sd_band : bool, default=False
        If True, show ±1 standard deviation band around mean
    smooth_window : int, optional, default=5
        Rolling window size for smoothing trajectories. If None, no smoothing
        is applied. Uses centered rolling mean to reduce noise.
    alpha_individual : float, default=0.3
        Transparency for individual trajectory lines (0=invisible, 1=opaque)
    alpha_mean : float, default=0.8
        Transparency for mean trajectory lines
    linewidth_individual : float, default=0.8
        Line width for individual trajectories
    linewidth_mean : float, default=2.5
        Line width for mean trajectories
    figsize : tuple, default=(12, 6)
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates based on metric and color_by
    xlabel : str, optional
        X-axis label. If None, uses time_col
    ylabel : str, optional
        Y-axis label. If None, uses metric
    save_path : str or Path, optional
        Path to save figure. If None, figure is not saved
    dpi : int, default=100
        Resolution for saved figure
    palette : str, optional
        Seaborn color palette name. If None, uses default tab10 colors

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot

    Examples
    --------
    >>> # Plot genotypes overlaid
    >>> fig = plot_embryos_metric_over_time(df, color_by='genotype')

    >>> # Plot clusters with SD bands, no individual trajectories
    >>> fig = plot_embryos_metric_over_time(
    ...     df,
    ...     color_by='cluster',
    ...     show_individual=False,
    ...     show_sd_band=True
    ... )

    >>> # Custom styling
    >>> fig = plot_embryos_metric_over_time(
    ...     df,
    ...     metric='arc_length_ratio',
    ...     color_by='phenotype',
    ...     alpha_individual=0.1,
    ...     palette='Set2',
    ...     save_path='outputs/phenotype_overlay.png'
    ... )
    """
    # Validate required columns
    required_cols = [metric, time_col, embryo_col, color_by]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to non-null values
    df = df[[embryo_col, time_col, metric, color_by]].dropna().copy()

    if len(df) == 0:
        raise ValueError("No valid data after filtering NaN values")

    # Get unique groups
    groups = sorted(df[color_by].unique())
    n_groups = len(groups)

    # Set up colors
    if palette is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_groups]
    else:
        colors = sns.color_palette(palette, n_groups)

    group_colors = dict(zip(groups, colors))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot each group
    for group in groups:
        group_df = df[df[color_by] == group].copy()
        color = group_colors[group]

        # Plot individual trajectories
        if show_individual:
            for embryo_id in group_df[embryo_col].unique():
                embryo_df = group_df[group_df[embryo_col] == embryo_id].sort_values(time_col)

                # Apply smoothing if requested
                if smooth_window is not None:
                    y_values = embryo_df[metric].rolling(
                        window=smooth_window,
                        center=True,
                        min_periods=1
                    ).mean()
                else:
                    y_values = embryo_df[metric]

                ax.plot(
                    embryo_df[time_col],
                    y_values,
                    color=color,
                    alpha=alpha_individual,
                    linewidth=linewidth_individual,
                    zorder=1
                )

        # Compute mean and std per timepoint
        if show_mean or show_sd_band:
            mean_df = (
                group_df
                .groupby(time_col)[metric]
                .agg(['mean', 'std'])
                .reset_index()
            )

            # Apply smoothing to mean and std if requested
            if smooth_window is not None:
                mean_df['mean_smoothed'] = mean_df['mean'].rolling(
                    window=smooth_window,
                    center=True,
                    min_periods=1
                ).mean()
                mean_df['std_smoothed'] = mean_df['std'].rolling(
                    window=smooth_window,
                    center=True,
                    min_periods=1
                ).mean()
            else:
                mean_df['mean_smoothed'] = mean_df['mean']
                mean_df['std_smoothed'] = mean_df['std']

            # Plot SD band
            if show_sd_band:
                ax.fill_between(
                    mean_df[time_col],
                    mean_df['mean_smoothed'] - mean_df['std_smoothed'],
                    mean_df['mean_smoothed'] + mean_df['std_smoothed'],
                    color=color,
                    alpha=0.2,
                    zorder=2
                )

            # Plot mean
            if show_mean:
                ax.plot(
                    mean_df[time_col],
                    mean_df['mean_smoothed'],
                    color=color,
                    alpha=alpha_mean,
                    linewidth=linewidth_mean,
                    label=str(group),
                    zorder=3
                )

    # Set labels and title
    if xlabel is None:
        xlabel = time_col.replace('_', ' ').title()
    if ylabel is None:
        ylabel = metric.replace('_', ' ').title()
    if title is None:
        title = f"{ylabel} Over {xlabel.split()[0]} by {color_by.replace('_', ' ').title()}"

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add legend
    if show_mean:
        ax.legend(title=color_by.replace('_', ' ').title(), loc='best', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig
