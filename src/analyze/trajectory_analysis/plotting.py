"""
Plotting Functions

Visualization for trajectory analysis, clustering, and quality assessment.

Functions
---------
- plot_posterior_heatmap: Posterior probability heatmap
- plot_2d_scatter: 2D scatter of max_p vs log_odds_gap
- plot_cluster_trajectories: Trajectories colored by cluster
- plot_membership_trajectories: Trajectories colored by membership category
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from .config import DEFAULT_DPI, DEFAULT_FIGSIZE, MEMBERSHIP_COLORS


def plot_posterior_heatmap(
    posterior_analysis: Dict[str, Any],
    embryo_ids: Optional[List[str]] = None,
    *,
    figsize: tuple = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot posterior probability heatmap.

    Rows = embryos, Columns = clusters
    Color intensity = p_i(c)

    Parameters
    ----------
    posterior_analysis : dict
        Output from analyze_bootstrap_results()
    embryo_ids : list of str, optional
        Embryo identifiers for y-axis labels
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    p_matrix = posterior_analysis['p_matrix']
    n_embryos, n_clusters = p_matrix.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(p_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Embryo', fontsize=12)
    ax.set_title('Posterior Probabilities p_i(c)', fontsize=14, fontweight='bold')

    # Ticks
    ax.set_xticks(np.arange(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])

    if embryo_ids is not None and len(embryo_ids) == n_embryos:
        ax.set_yticks(np.arange(n_embryos))
        ax.set_yticklabels(embryo_ids, fontsize=8)
    else:
        ax.set_yticks(np.arange(0, n_embryos, max(1, n_embryos // 20)))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_2d_scatter(
    classification: Dict[str, Any],
    embryo_ids: Optional[List[str]] = None,
    *,
    show_thresholds: bool = True,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot 2D scatter of max_p vs log_odds_gap.

    Points colored by membership category (core/uncertain/outlier).
    Optionally shows threshold lines.

    Parameters
    ----------
    classification : dict
        Output from classify_membership_2d()
    embryo_ids : list of str, optional
        For labeling outlier points
    show_thresholds : bool
        Show threshold lines
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save
    dpi : int
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    max_p = classification['max_p']
    log_odds_gap = classification['log_odds_gap']
    categories = classification['category']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot by category
    for category in ['outlier', 'uncertain', 'core']:
        mask = categories == category
        if np.sum(mask) > 0:
            color = MEMBERSHIP_COLORS.get(category, 'gray')
            ax.scatter(max_p[mask], log_odds_gap[mask], label=category, alpha=0.6, s=50, color=color)

    # Threshold lines
    if show_thresholds:
        thresholds = classification.get('thresholds', {})
        max_p_thresh = thresholds.get('threshold_max_p', 0.8)
        log_odds_thresh = thresholds.get('threshold_log_odds_gap', 0.7)
        outlier_thresh = thresholds.get('threshold_outlier_max_p', 0.5)

        ax.axvline(max_p_thresh, color='red', linestyle='--', alpha=0.5, label=f'max_p = {max_p_thresh}')
        ax.axvline(outlier_thresh, color='orange', linestyle='--', alpha=0.5, label=f'outlier = {outlier_thresh}')
        ax.axhline(log_odds_thresh, color='green', linestyle='--', alpha=0.5, label=f'log_odds = {log_odds_thresh}')

    ax.set_xlabel('Max Posterior Probability', fontsize=12)
    ax.set_ylabel('Log-Odds Gap', fontsize=12)
    ax.set_title('2D Membership Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_cluster_trajectories(
    trajectories: List[np.ndarray],
    common_grid: np.ndarray,
    cluster_labels: np.ndarray,
    embryo_ids: Optional[List[str]] = None,
    *,
    show_mean: bool = True,
    show_individual: bool = True,
    figsize: tuple = DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot trajectories colored by cluster assignment.

    Parameters
    ----------
    trajectories : list of np.ndarray
        Individual trajectories
    common_grid : np.ndarray
        Common time grid
    cluster_labels : np.ndarray
        Cluster assignments
    embryo_ids : list of str, optional
        Embryo identifiers
    show_mean : bool
        Show cluster means
    show_individual : bool
        Show individual trajectories
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save
    dpi : int
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_clusters = int(np.max(cluster_labels)) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot individual trajectories
    if show_individual:
        for i, traj in enumerate(trajectories):
            cluster = cluster_labels[i]
            grid_subset = common_grid[:len(traj)]
            ax.plot(grid_subset, traj, color=colors[cluster], alpha=0.3, linewidth=0.8)

    # Plot cluster means
    if show_mean:
        for c in range(n_clusters):
            mask = cluster_labels == c
            if np.sum(mask) > 0:
                cluster_trajs = [trajectories[i] for i in np.where(mask)[0]]
                # Compute mean (handling variable lengths)
                min_len = min([len(t) for t in cluster_trajs])
                mean_traj = np.mean([t[:min_len] for t in cluster_trajs], axis=0)
                grid_subset = common_grid[:min_len]
                ax.plot(grid_subset, mean_traj, color=colors[c], linewidth=2.5, label=f'Cluster {c}')

    ax.set_xlabel('HPF', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Trajectories by Cluster', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_membership_trajectories(
    trajectories: List[np.ndarray],
    common_grid: np.ndarray,
    classification: Dict[str, Any],
    *,
    per_cluster: bool = True,
    figsize: tuple = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot trajectories colored by membership category (core/uncertain/outlier).

    If per_cluster=True, creates panel grid with one subplot per cluster.

    Parameters
    ----------
    trajectories : list of np.ndarray
        Individual trajectories
    common_grid : np.ndarray
        Common time grid
    classification : dict
        Output from classify_membership_2d()
    per_cluster : bool
        Create per-cluster panels
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save
    dpi : int
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    categories = classification['category']
    clusters = classification['cluster']
    n_clusters = int(np.max(clusters)) + 1

    if per_cluster:
        fig, axes = plt.subplots(1, n_clusters, figsize=figsize, sharey=True)
        if n_clusters == 1:
            axes = [axes]

        for c in range(n_clusters):
            ax = axes[c]
            mask = clusters == c

            for category in ['outlier', 'uncertain', 'core']:
                cat_mask = mask & (categories == category)
                if np.sum(cat_mask) > 0:
                    color = MEMBERSHIP_COLORS.get(category, 'gray')
                    for i in np.where(cat_mask)[0]:
                        traj = trajectories[i]
                        grid_subset = common_grid[:len(traj)]
                        ax.plot(grid_subset, traj, color=color, alpha=0.4, linewidth=0.8)

            ax.set_title(f'Cluster {c}', fontweight='bold')
            ax.set_xlabel('HPF')
            if c == 0:
                ax.set_ylabel('Metric Value')
            ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=MEMBERSHIP_COLORS.get(cat, 'gray'), label=cat.capitalize())
                          for cat in ['core', 'uncertain', 'outlier']]
        fig.legend(handles=legend_elements, loc='upper right')

    else:
        fig, ax = plt.subplots(figsize=figsize)

        for category in ['outlier', 'uncertain', 'core']:
            mask = categories == category
            if np.sum(mask) > 0:
                color = MEMBERSHIP_COLORS.get(category, 'gray')
                for i in np.where(mask)[0]:
                    traj = trajectories[i]
                    grid_subset = common_grid[:len(traj)]
                    ax.plot(grid_subset, traj, color=color, alpha=0.4, linewidth=0.8, label=category if i == np.where(mask)[0][0] else '')

        ax.set_xlabel('HPF', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Trajectories by Membership Category', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
