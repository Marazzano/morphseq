"""
Plotting utilities for pair analysis.

Reusable plot functions for comparing trajectories across groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data_utils import get_trajectories_for_group, compute_binned_mean, get_global_axis_ranges


# Default genotype configuration
GENOTYPE_ORDER = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
GENOTYPE_COLORS = {
    'cep290_wildtype': '#2E7D32',      # Green
    'cep290_heterozygous': '#FFA500',  # Orange
    'cep290_homozygous': '#D32F2F',    # Red
}


def plot_genotypes_overlaid(
    df: pd.DataFrame,
    groups: List[str],
    group_col: str = 'pair',
    genotype_col: str = 'genotype',
    time_col: str = 'predicted_stage_hpf',
    metric_col: str = 'baseline_deviation_normalized',
    embryo_id_col: str = 'embryo_id',
    genotype_order: List[str] = None,
    genotype_colors: Dict[str, str] = None,
    output_path: Optional[Path] = None,
    title: str = 'Curvature Trajectories by Group - All Genotypes Compared',
    figsize: Tuple[int, int] = (15, 4.5),
    bin_width: float = 0.5,
) -> plt.Figure:
    """Create a 1xN plot showing all genotypes overlaid for each group.

    Args:
        df: DataFrame with trajectory data
        groups: List of group values (e.g., ['cep290_pair_1', 'cep290_pair_2', ...])
        group_col: Column name for groups (e.g., 'pair')
        genotype_col: Column name for genotypes
        time_col: Column name for time values
        metric_col: Column name for metric values
        embryo_id_col: Column name for embryo IDs
        genotype_order: List of genotypes in plotting order
        genotype_colors: Dict mapping genotype to color
        output_path: Path to save figure (optional)
        title: Figure title
        figsize: Figure size as (width, height)
        bin_width: Width of bins for mean calculation

    Returns:
        matplotlib Figure object
    """
    if genotype_order is None:
        genotype_order = GENOTYPE_ORDER
    if genotype_colors is None:
        genotype_colors = GENOTYPE_COLORS

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)

    if n_groups == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    # First pass: collect all data for global axis ranges
    all_data = {}
    all_trajectories = []

    for group in groups:
        for genotype in genotype_order:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_group(
                df,
                {group_col: group, genotype_col: genotype},
                time_col=time_col,
                metric_col=metric_col,
                embryo_id_col=embryo_id_col,
            )
            all_data[(group, genotype)] = (trajectories, embryo_ids, n_embryos)
            if trajectories:
                all_trajectories.append(trajectories)

    time_min, time_max, metric_min, metric_max = get_global_axis_ranges(all_trajectories)

    # Second pass: plot with aligned axes
    for col_idx, group in enumerate(groups):
        ax = axes[col_idx]

        for genotype in genotype_order:
            trajectories, embryo_ids, n_embryos = all_data[(group, genotype)]

            if trajectories is None or n_embryos == 0:
                continue

            color = genotype_colors.get(genotype, '#888888')

            # Plot individual trajectories (faded)
            for traj in trajectories:
                ax.plot(traj['times'], traj['metrics'], alpha=0.2, linewidth=0.8, color=color)

            # Plot mean trajectory
            all_times = np.concatenate([t['times'] for t in trajectories])
            all_metrics = np.concatenate([t['metrics'] for t in trajectories])
            bin_times, bin_means = compute_binned_mean(all_times, all_metrics, bin_width)

            if bin_times:
                label = f'{genotype.replace("cep290_", "").title()} (n={n_embryos})'
                ax.plot(bin_times, bin_means, color=color, linewidth=2.5, label=label, zorder=5)

        ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.set_ylabel('Normalized Baseline Deviation', fontsize=10)
        ax.set_title(f'{group}', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(time_min, time_max)
        ax.set_ylim(metric_min, metric_max)
        ax.tick_params(labelsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_faceted_trajectories(
    df: pd.DataFrame,
    row_groups: List[str],
    col_groups: List[str],
    row_col: str = 'pair',
    col_col: str = 'genotype',
    time_col: str = 'predicted_stage_hpf',
    metric_col: str = 'baseline_deviation_normalized',
    embryo_id_col: str = 'embryo_id',
    group_colors: Dict[str, str] = None,
    output_path: Optional[Path] = None,
    title: str = 'Curvature Trajectories - Faceted Overview',
    figsize: Optional[Tuple[int, int]] = None,
    bin_width: float = 0.5,
) -> plt.Figure:
    """Create a faceted plot with rows and columns of subplots.

    Args:
        df: DataFrame with trajectory data
        row_groups: List of row group values (e.g., pairs)
        col_groups: List of column group values (e.g., genotypes)
        row_col: Column name for row groups
        col_col: Column name for column groups
        time_col: Column name for time values
        metric_col: Column name for metric values
        embryo_id_col: Column name for embryo IDs
        group_colors: Dict mapping col_groups to colors
        output_path: Path to save figure (optional)
        title: Figure title
        figsize: Figure size (auto-calculated if None)
        bin_width: Width of bins for mean calculation

    Returns:
        matplotlib Figure object
    """
    if group_colors is None:
        group_colors = GENOTYPE_COLORS

    n_rows = len(row_groups)
    n_cols = len(col_groups)

    if figsize is None:
        figsize = (5 * n_cols, 4.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # First pass: collect all data for global axis ranges
    all_data = {}
    all_trajectories = []

    for row_group in row_groups:
        for col_group in col_groups:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_group(
                df,
                {row_col: row_group, col_col: col_group},
                time_col=time_col,
                metric_col=metric_col,
                embryo_id_col=embryo_id_col,
            )
            all_data[(row_group, col_group)] = (trajectories, embryo_ids, n_embryos)
            if trajectories:
                all_trajectories.append(trajectories)

    time_min, time_max, metric_min, metric_max = get_global_axis_ranges(all_trajectories)

    # Second pass: plot with aligned axes
    for row_idx, row_group in enumerate(row_groups):
        for col_idx, col_group in enumerate(col_groups):
            ax = axes[row_idx, col_idx]

            trajectories, embryo_ids, n_embryos = all_data[(row_group, col_group)]

            if trajectories is None or n_embryos == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='lightgray')
            else:
                color = group_colors.get(col_group, '#888888')

                # Plot individual trajectories
                for traj in trajectories:
                    ax.plot(traj['times'], traj['metrics'], alpha=0.25, linewidth=0.8, color=color)

                # Plot mean trajectory
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metrics = np.concatenate([t['metrics'] for t in trajectories])
                bin_times, bin_means = compute_binned_mean(all_times, all_metrics, bin_width)

                if bin_times:
                    ax.plot(bin_times, bin_means, color=color, linewidth=2.2, label='Mean', zorder=5)

            # Labels and styling
            ax.set_xlabel('Time (hpf)', fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(f'{row_group}\n\nNormalized Baseline Deviation', fontsize=9)
            else:
                ax.set_ylabel('')

            if row_idx == 0:
                col_label = col_group.replace('cep290_', '').title()
                ax.set_title(f'{col_label} (n={n_embryos})', fontweight='bold', fontsize=10)
            else:
                ax.set_title(f'n={n_embryos}', fontsize=9)

            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=8)
            ax.set_xlim(time_min, time_max)
            ax.set_ylim(metric_min, metric_max)

            # Legend only on top-left
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig
