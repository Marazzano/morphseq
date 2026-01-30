"""
Proportion plotting functions for faceted layouts.

Primary API:
    - plot_proportions: Proportion plots (preferred)

Usage
-----
Faceted mode (row/col are facet variables; colors are category values):
    >>> fig = plot_proportions(
    ...     df,
    ...     color_by_grouping='phenotype',
    ...     row_by='genotype_suffix',
    ...     col_by='pair',
    ... )

Grid mode (row_by is a list of categorical columns):
    >>> fig = plot_proportions(
    ...     df,
    ...     col_by='cluster',
    ...     row_by=['genotype', 'pair', 'experiment_id'],
    ...     count_by='embryo_id',
    ... )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union

from ...styling import get_color_for_genotype
from ....config import (
    HEIGHT_PER_ROW,
    WIDTH_PER_COL,
)

from .shared import (
    STANDARD_PALETTE,
    make_color_lookup,
    _normalize_color,
)


# ==============================================================================
# Proportion Grid Plot (Categorical Breakdown) - Internal Implementation
# ==============================================================================

def _plot_proportions_grid_impl(
    df: pd.DataFrame,
    col_by: str,
    row_by: List[str],
    count_by: str = 'embryo_id',
    col_order: Optional[List] = None,
    row_labels: Optional[Dict[str, str]] = None,
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    color_palette: Optional[Dict[str, Dict[str, str]]] = None,
) -> plt.Figure:
    """
    Plot proportion breakdown of categorical variables across groups.

    Creates a grid where:
    - Each column = one value of col_by (e.g., cluster)
    - Each row = one categorical variable from row_by list
    - Each cell = bar chart showing proportions of that categorical

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical columns.
    col_by : str
        Column defining the grid columns (e.g., 'cluster').
    row_by : List[str]
        List of categorical column names for rows (e.g., ['genotype', 'pair']).
    count_by : str
        Column to count unique values of (default: 'embryo_id').
    col_order : Optional[List]
        Explicit ordering for col_by values.
    row_labels : Optional[Dict[str, str]]
        Display labels for row_by columns (e.g., {'experiment_id': 'Experiment'}).
    height_per_row : int
        Height in pixels per row.
    width_per_col : int
        Width in pixels per column.
    output_path : Optional[Path]
        If provided, saves figure to this path.
    title : Optional[str]
        Figure title.
    normalize : bool
        If True, show proportions (0-1). If False, show raw counts.
    bar_mode : str
        'stacked' for single stacked bar, 'grouped' for side-by-side bars.
    color_palette : Optional[Dict[str, Dict[str, str]]]
        Nested dict of {row_category: {value: color}}. If None, uses defaults.

    Returns
    -------
    plt.Figure
        The matplotlib figure.

    Example
    -------
    >>> fig = plot_proportions(
    ...     df,
    ...     col_by='cluster',
    ...     row_by=['genotype', 'pair', 'experiment_id'],
    ...     count_by='embryo_id',
    ...     bar_mode='grouped',  # or 'stacked'
    ...     title='Cluster Composition Breakdown',
    ... )
    """
    # Determine column values (clusters)
    col_values = sorted(df[col_by].unique())
    if col_order:
        col_values = [c for c in col_order if c in col_values]

    n_rows = len(row_by)
    n_cols = len(col_values)

    # Build color palettes for each categorical
    palettes = {}
    for cat in row_by:
        if color_palette and cat in color_palette:
            palettes[cat] = color_palette[cat]
        elif cat == 'genotype':
            # Use genotype-specific colors
            unique_vals = sorted(df[cat].dropna().unique())
            palettes[cat] = {v: get_color_for_genotype(str(v)) for v in unique_vals}
        else:
            unique_vals = sorted(df[cat].dropna().unique())
            palettes[cat] = {v: STANDARD_PALETTE[i % len(STANDARD_PALETTE)]
                           for i, v in enumerate(unique_vals)}

    # Create figure
    fig_width = max(6, n_cols * (width_per_col / 100))
    fig_height = max(4, n_rows * (height_per_row / 100))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Track legend entries per row (categorical)
    row_legend_entries = {cat: {} for cat in row_by}

    for r_idx, cat in enumerate(row_by):
        cat_values = sorted(df[cat].dropna().unique())

        for c_idx, col_val in enumerate(col_values):
            ax = axes[r_idx, c_idx]

            # Filter to this column value
            subset = df[df[col_by] == col_val]

            # Count unique count_by values per category
            counts = subset.groupby(cat)[count_by].nunique()

            # Reindex to ensure all category values present
            counts = counts.reindex(cat_values, fill_value=0)

            if normalize and counts.sum() > 0:
                proportions = counts / counts.sum()
            else:
                proportions = counts

            # Draw bars based on mode
            n_bars = len(cat_values)

            if bar_mode == 'grouped':
                # Side-by-side bars
                bar_width = 0.8 / n_bars
                x_positions = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, n_bars)

                for i, val in enumerate(cat_values):
                    height = proportions.get(val, 0)
                    color = palettes[cat].get(val, STANDARD_PALETTE[0])
                    ax.bar(x_positions[i], height, width=bar_width,
                           color=color, edgecolor='white', linewidth=0.5)

                    # Track for legend
                    if val not in row_legend_entries[cat]:
                        row_legend_entries[cat][val] = color

                # Y-axis for grouped mode
                max_val = proportions.max() if len(proportions) > 0 else 1
                ax.set_ylim(0, 1.05 if normalize else max_val * 1.1)
            else:
                # Stacked bar (default)
                bottom = 0
                bar_width = 0.6
                for val in cat_values:
                    height = proportions.get(val, 0)
                    if height > 0:
                        color = palettes[cat].get(val, STANDARD_PALETTE[0])
                        ax.bar(0, height, bottom=bottom, width=bar_width,
                               color=color, edgecolor='white', linewidth=0.5)

                        # Track for legend
                        if val not in row_legend_entries[cat]:
                            row_legend_entries[cat][val] = color

                        bottom += height

                ax.set_ylim(0, 1 if normalize else counts.sum() * 1.05)

            # Styling
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])

            # Column titles on top row
            if r_idx == 0:
                ax.set_title(f'{col_by}={col_val}', fontsize=10, fontweight='bold')

            # Y-axis labels on left column only
            if c_idx == 0:
                label = row_labels.get(cat, cat) if row_labels else cat
                ax.set_ylabel(label, fontsize=10, fontweight='bold')
                if normalize:
                    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            else:
                ax.set_yticks([])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    # Add legends for each row on the right side
    for r_idx, cat in enumerate(row_by):
        if row_legend_entries[cat]:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5)
                for val, color in row_legend_entries[cat].items()
            ]
            legend_labels = list(row_legend_entries[cat].keys())

            # Position legend to the right of this row
            ax_right = axes[r_idx, -1]
            ax_right.legend(
                legend_handles, legend_labels,
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),
                fontsize=8,
                frameon=True,
                framealpha=0.9,
            )

    # Title
    final_title = title or f'Proportion Breakdown by {col_by}'
    fig.suptitle(final_title, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')

    return fig


# ==============================================================================
# Proportion Faceted Plot (Consistent API with plot_feature_over_time)
# ==============================================================================

def _plot_proportions_impl(
    df: pd.DataFrame,
    color_by_grouping: str,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    count_by: str = 'embryo_id',
    facet_order: Optional[Dict[str, List]] = None,
    color_order: Optional[List] = None,
    color_palette: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
) -> plt.Figure:
    """
    Plot proportion breakdown with faceted grid structure.

    API is consistent with plot_feature_over_time:
    - row_by/col_by define facet grid by column VALUES (not variable names)
    - color_by_grouping defines the categorical for bar colors

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical columns.
    color_by_grouping : str
        Column name for bar coloring (e.g., 'phenotype').
    row_by : Optional[str]
        Column defining grid rows (e.g., 'genotype'). None = single row.
    col_by : Optional[str]
        Column defining grid columns (e.g., 'pair'). None = single column.
    count_by : str
        Column to count unique values of (default: 'embryo_id').
    facet_order : Optional[Dict[str, List]]
        Ordering for facet values. Keys: 'row_by', 'col_by'.
        Example: {'row_by': ['wildtype', 'het', 'hom'], 'col_by': ['pair_2', 'pair_7']}
    color_order : Optional[List]
        Order of color_by_grouping values (bar order and legend order).
    color_palette : Optional[Dict[str, str]]
        Color mapping {value: hex_color}. Uses STANDARD_PALETTE if not provided.
    normalize : bool
        If True, show proportions (0-1). If False, show raw counts.
    bar_mode : str
        'grouped' for side-by-side bars, 'stacked' for stacked bars.
    height_per_row : int
        Height in pixels per row.
    width_per_col : int
        Width in pixels per column.
    output_path : Optional[Path]
        If provided, saves figure to this path.
    title : Optional[str]
        Figure title.
    show_counts : bool
        If True, show count annotations on bars.

    Returns
    -------
    plt.Figure
        The matplotlib figure.

    Example
    -------
    >>> fig = plot_proportions(
    ...     df,
    ...     color_by_grouping='phenotype',
    ...     row_by='genotype_suffix',
    ...     col_by='pair',
    ...     color_palette={
    ...         'CE': '#9467BD',
    ...         'HTA': '#17BECF',
    ...         'BA_rescue': '#E377C2',
    ...         'non_penetrant': '#7F7F7F',
    ...     },
    ...     facet_order={
    ...         'row_by': ['wildtype', 'heterozygous', 'homozygous'],
    ...         'col_by': ['pair_2', 'pair_7', 'pair_8'],
    ...     },
    ...     title='Phenotype Distribution by Pair and Genotype',
    ... )
    """
    # Determine row and column values
    # When row_by/col_by is None, use a single row/column with a placeholder label
    if row_by is not None:
        row_values = sorted(df[row_by].dropna().unique())
        if facet_order and 'row_by' in facet_order:
            row_values = [v for v in facet_order['row_by'] if v in row_values]
    else:
        row_values = ['_all_']  # Placeholder for single row

    if col_by is not None:
        col_values = sorted(df[col_by].dropna().unique())
        if facet_order and 'col_by' in facet_order:
            col_values = [v for v in facet_order['col_by'] if v in col_values]
    else:
        col_values = ['_all_']  # Placeholder for single column

    # Determine color_by_grouping values
    # Use make_color_lookup for consistent color ordering with trajectory plots
    # (preserves order-of-first-occurrence rather than alphabetical sort)

    # Get color values - respect Categorical ordering if present
    col_series = df[color_by_grouping]
    if hasattr(col_series, 'cat') and col_series.cat.ordered:
        # Use Categorical order, but only for categories that exist in data
        present_cats = set(col_series.dropna().unique())
        color_values = [c for c in col_series.cat.categories if c in present_cats]
    else:
        color_values = list(pd.unique(col_series.dropna()))

    # Handle color_palette: can be None, a dict, or a list
    if color_palette is None:
        color_palette = make_color_lookup(df[color_by_grouping])
    elif isinstance(color_palette, (list, tuple)):
        # Convert list to dict using color_values order
        normalized_palette = [_normalize_color(c) for c in color_palette]
        color_palette = {v: normalized_palette[i % len(normalized_palette)]
                        for i, v in enumerate(color_values)}
    if color_order:
        color_values = [v for v in color_order if v in color_values]

    n_rows = len(row_values)
    n_cols = len(col_values)

    # Figure sizing
    fig_width = max(6, n_cols * (width_per_col / 100))
    fig_height = max(4, n_rows * (height_per_row / 100))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Legend tracking
    legend_handles = []
    legend_labels = []

    for r_idx, row_val in enumerate(row_values):
        for c_idx, col_val in enumerate(col_values):
            ax = axes[r_idx, c_idx]

            # Filter to this cell (skip filtering if using '_all_' placeholder)
            subset = df.copy()
            if row_by is not None and row_val != '_all_':
                subset = subset[subset[row_by] == row_val]
            if col_by is not None and col_val != '_all_':
                subset = subset[subset[col_by] == col_val]

            # Count unique count_by values per color_by_grouping category
            counts = subset.groupby(color_by_grouping, observed=True)[count_by].nunique()
            counts = counts.reindex(color_values, fill_value=0)

            total = counts.sum()
            if normalize and total > 0:
                proportions = counts / total
            else:
                proportions = counts

            # Draw bars
            n_bars = len(color_values)

            if bar_mode == 'grouped':
                bar_width = 0.8 / max(n_bars, 1)
                x_positions = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, n_bars) if n_bars > 0 else []

                for i, val in enumerate(color_values):
                    height = proportions.get(val, 0)
                    color = color_palette.get(val, STANDARD_PALETTE[0])

                    bar = ax.bar(x_positions[i], height, width=bar_width,
                                color=color, edgecolor='white', linewidth=0.5)

                    # Add count annotation if requested
                    if show_counts and counts.get(val, 0) > 0:
                        ax.annotate(f'{int(counts.get(val, 0))}',
                                   xy=(x_positions[i], height),
                                   ha='center', va='bottom',
                                   fontsize=7, color='black')

                    # Track for legend (only once)
                    if r_idx == 0 and c_idx == 0:
                        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                                           facecolor=color,
                                                           edgecolor='white',
                                                           linewidth=0.5))
                        legend_labels.append(val)

                # Y-axis limits for grouped mode
                max_val = proportions.max() if len(proportions) > 0 else 1
                ax.set_ylim(0, 1.15 if normalize else max_val * 1.2)

            else:  # stacked
                bottom = 0
                bar_width = 0.6

                for val in color_values:
                    height = proportions.get(val, 0)
                    if height > 0:
                        color = color_palette.get(val, STANDARD_PALETTE[0])
                        ax.bar(0, height, bottom=bottom, width=bar_width,
                              color=color, edgecolor='white', linewidth=0.5)

                        # Track for legend (only once)
                        if r_idx == 0 and c_idx == 0 and val not in legend_labels:
                            legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                                               facecolor=color,
                                                               edgecolor='white',
                                                               linewidth=0.5))
                            legend_labels.append(val)

                        bottom += height

                ax.set_ylim(0, 1.05 if normalize else total * 1.1)

            # Styling - tighter bar spacing
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])

            # Column titles on top row (skip if single column placeholder)
            if r_idx == 0 and col_by is not None and col_val != '_all_':
                ax.set_title(f'{col_val}', fontsize=10, fontweight='bold')

            # Row labels on left column (skip if single row placeholder)
            if c_idx == 0 and row_by is not None and row_val != '_all_':
                ax.set_ylabel(f'{row_val}', fontsize=10, fontweight='bold')

            # Y-axis formatting - set ticks on ALL facets for grid alignment
            if normalize:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                if c_idx == 0:
                    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
                else:
                    ax.set_yticklabels([])  # Hide labels but keep ticks for grid
            else:
                if c_idx != 0:
                    ax.set_yticklabels([])

            # Add horizontal grid lines across all facets
            ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
            ax.set_axisbelow(True)

            # Clean up spines - keep bottom for baseline
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('gray')

    # Add single legend for all panels
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='center right',
            bbox_to_anchor=(0.98, 0.5),
            fontsize=9,
            frameon=True,
            framealpha=0.9,
            title=color_by_grouping,
            title_fontsize=10,
        )

    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    else:
        parts = []
        if col_by:
            parts.append(f'by {col_by}')
        if row_by:
            parts.append(f'and {row_by}')
        default_title = f'{color_by_grouping} Distribution {" ".join(parts)}'
        fig.suptitle(default_title, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.82, 0.96])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')

    return fig


def plot_proportions(
    df: pd.DataFrame,
    color_by_grouping: Optional[str] = None,
    row_by: Optional[Union[str, List[str]]] = None,
    col_by: Optional[str] = None,
    count_by: str = 'embryo_id',
    facet_order: Optional[Dict[str, List]] = None,
    color_order: Optional[List] = None,
    color_palette: Optional[Dict] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
    row_labels: Optional[Dict[str, str]] = None,
    col_order: Optional[List] = None,
) -> plt.Figure:
    """
    Proportion plotting with two modes:

    1) Faceted mode (default):
       - row_by/col_by are column names whose VALUES define the grid
       - color_by_grouping defines bar categories

    2) Grid mode (when row_by is a list of column names):
       - Each row is a different categorical column
       - col_by defines grid columns
       - color_by_grouping must be None
    """
    if isinstance(row_by, (list, tuple)):
        if color_by_grouping is not None:
            raise ValueError("When row_by is a list, color_by_grouping must be None.")
        if col_by is None:
            raise ValueError("col_by is required when row_by is a list.")
        return _plot_proportions_grid_impl(
            df=df,
            col_by=col_by,
            row_by=list(row_by),
            count_by=count_by,
            col_order=col_order,
            row_labels=row_labels,
            height_per_row=height_per_row,
            width_per_col=width_per_col,
            output_path=output_path,
            title=title,
            normalize=normalize,
            bar_mode=bar_mode,
            color_palette=color_palette,
        )

    if color_by_grouping is None:
        raise ValueError("color_by_grouping is required when row_by is a string or None.")

    return _plot_proportions_impl(
        df=df,
        color_by_grouping=color_by_grouping,
        row_by=row_by,
        col_by=col_by,
        count_by=count_by,
        facet_order=facet_order,
        color_order=color_order,
        color_palette=color_palette,
        normalize=normalize,
        bar_mode=bar_mode,
        height_per_row=height_per_row,
        width_per_col=width_per_col,
        output_path=output_path,
        title=title,
        show_counts=show_counts,
    )
