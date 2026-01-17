"""
Pair-specific plotting utilities (Level 2).

Wraps Level 1 generic faceted_plotting with pair-specific logic and defaults.
Automatically handles missing pair columns, genotype-based pair fallbacks, etc.
"""

import warnings
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..viz.plotting.faceted import plot_trajectories_faceted
from ..viz.styling import build_genotype_style_config, sort_genotypes_by_suffix


def _ensure_pair_column(df: pd.DataFrame, pair_col: str = 'pair', genotype_col: str = 'genotype') -> pd.DataFrame:
    """Ensure pair column exists. Create fallback `{gene_prefix}_unknown_pair` if missing.

    Args:
        df: Input DataFrame
        pair_col: Name of pair column
        genotype_col: Name of genotype column

    Returns:
        DataFrame with pair column (modified if needed)

    Note:
        Uses gene prefix (not full genotype) so all genotypes from the same gene
        share one unknown_pair. E.g., tmem67_homozygous → tmem67_unknown_pair
    """
    from ..viz.styling import extract_genotype_prefix

    df = df.copy()

    if pair_col not in df.columns or df[pair_col].isna().all():
        if genotype_col in df.columns:
            # Create fallback: {gene_prefix}_unknown_pair (not {full_genotype}_unknown_pair)
            # This groups all genotypes from same gene into one pair
            df[pair_col] = df[genotype_col].apply(
                lambda g: f"{extract_genotype_prefix(str(g))}_unknown_pair"
            )
        else:
            # Last resort: all embryos treated as one "unknown" pair
            df[pair_col] = 'unknown_pair'

    return df


def plot_pairs_overview(
    df: pd.DataFrame,
    pairs: Optional[List[str]] = None,
    genotypes: Optional[List[str]] = None,
    pair_col: str = 'pair',
    genotype_col: str = 'genotype',
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    y_label: str = 'Value',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    **kwargs
) -> Any:
    """Create NxM grid: pairs (rows) × genotypes (columns).

    Level 2: Pair-specific wrapper around plot_trajectories_faceted.

    Args:
        df: DataFrame with trajectory data
        pairs: Specific pairs to plot (if None, uses all in df)
        genotypes: Specific genotypes to plot (if None, uses all in df)
        pair_col: Column name for pairs
        genotype_col: Column name for genotypes
        x_col: X-axis column
        y_col: Y-axis column
        line_by: Column defining individual lines
        backend: 'plotly', 'matplotlib', or 'both'
        output_path: Save location
        title: Figure title
        **kwargs: Additional arguments passed to plot_trajectories_faceted

    Returns:
        Figure(s) depending on backend
    """
    # Ensure pair column exists
    df = _ensure_pair_column(df, pair_col, genotype_col)

    # Auto-detect pairs/genotypes if not specified
    if pairs is None:
        pairs = sorted(df[pair_col].unique())
    if genotypes is None:
        genotypes = sorted(df[genotype_col].unique())

    # Build genotype styling
    style_config = build_genotype_style_config(genotypes)
    ordered_genotypes = style_config['order']

    # Set title
    if title is None:
        title = f"All Pairs × Genotypes Overview"

    # Create faceted plot with pair as rows, genotype as columns
    fig = plot_trajectories_faceted(
        df[df[pair_col].isin(pairs)],
        x_col=x_col,
        y_col=y_col,
        line_by=line_by,
        row_by=pair_col,
        col_by=genotype_col,
        color_by_grouping=None,
        facet_order={genotype_col: ordered_genotypes},
        backend=backend,
        output_path=output_path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        bin_width=bin_width,
        smooth_method=smooth_method,
        smooth_params=smooth_params,
        **kwargs
    )

    return fig


def plot_genotypes_by_pair(
    df: pd.DataFrame,
    pairs: Optional[List[str]] = None,
    pair_col: str = 'pair',
    genotype_col: str = 'genotype',
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    y_label: str = 'Value',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    **kwargs
) -> Any:
    """Create 1xN plot with genotypes overlaid per pair.

    Level 2: Pair-specific wrapper around plot_trajectories_faceted.

    Args:
        df: DataFrame with trajectory data
        pairs: Specific pairs to plot (if None, uses all in df)
        pair_col: Column name for pairs
        genotype_col: Column name for genotypes
        x_col: X-axis column
        y_col: Y-axis column
        line_by: Column defining individual lines
        backend: 'plotly', 'matplotlib', or 'both'
        output_path: Save location
        title: Figure title
        **kwargs: Additional arguments passed to plot_trajectories_faceted

    Returns:
        Figure(s) depending on backend
    """
    # Ensure pair column exists
    df = _ensure_pair_column(df, pair_col, genotype_col)

    # Auto-detect pairs if not specified
    if pairs is None:
        pairs = sorted(df[pair_col].unique())

    # Get all unique genotypes and sort by suffix
    all_genotypes = sorted(df[genotype_col].unique())
    style_config = build_genotype_style_config(all_genotypes)
    ordered_genotypes = style_config['order']

    # Set title
    if title is None:
        title = f"Genotypes by Pair (All Overlaid)"

    # Create faceted plot with pairs as columns, genotypes overlaid
    fig = plot_trajectories_faceted(
        df[df[pair_col].isin(pairs)],
        x_col=x_col,
        y_col=y_col,
        line_by=line_by,
        row_by=None,
        col_by=pair_col,
        color_by_grouping=genotype_col,
        facet_order={genotype_col: ordered_genotypes},
        backend=backend,
        output_path=output_path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        bin_width=bin_width,
        smooth_method=smooth_method,
        smooth_params=smooth_params,
        **kwargs
    )

    return fig


def plot_single_genotype_across_pairs(
    df: pd.DataFrame,
    genotype: str,
    pairs: Optional[List[str]] = None,
    pair_col: str = 'pair',
    genotype_col: str = 'genotype',
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    y_label: str = 'Value',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    **kwargs
) -> Any:
    """Create 1xN plot for single genotype across pairs.

    Level 2: Pair-specific wrapper around plot_trajectories_faceted.

    Args:
        df: DataFrame with trajectory data
        genotype: Genotype to plot
        pairs: Specific pairs to plot (if None, uses all in df)
        pair_col: Column name for pairs
        genotype_col: Column name for genotypes
        x_col: X-axis column
        y_col: Y-axis column
        line_by: Column defining individual lines
        backend: 'plotly', 'matplotlib', or 'both'
        output_path: Save location
        title: Figure title
        **kwargs: Additional arguments passed to plot_trajectories_faceted

    Returns:
        Figure(s) depending on backend
    """
    # Ensure pair column exists
    df = _ensure_pair_column(df, pair_col, genotype_col)

    # Filter to single genotype
    df_filtered = df[df[genotype_col] == genotype].copy()

    # Auto-detect pairs if not specified
    if pairs is None:
        pairs = sorted(df_filtered[pair_col].unique())

    # Set title
    if title is None:
        title = f"{genotype} Across Pairs"

    # Create faceted plot with pairs as columns, single genotype
    fig = plot_trajectories_faceted(
        df_filtered[df_filtered[pair_col].isin(pairs)],
        x_col=x_col,
        y_col=y_col,
        line_by=line_by,
        row_by=None,
        col_by=pair_col,
        color_by_grouping=None,
        backend=backend,
        output_path=output_path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        bin_width=bin_width,
        smooth_method=smooth_method,
        smooth_params=smooth_params,
        **kwargs
    )

    return fig


# Backward compatibility: keep old names pointing to new functions
def plot_genotypes_overlaid(*args, **kwargs):
    """Deprecated: Use plot_genotypes_by_pair instead."""
    warnings.warn(
        "plot_genotypes_overlaid is deprecated. Use plot_genotypes_by_pair instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return plot_genotypes_by_pair(*args, **kwargs)


def plot_all_pairs_overview(*args, **kwargs):
    """Deprecated: Use plot_pairs_overview instead."""
    warnings.warn(
        "plot_all_pairs_overview is deprecated. Use plot_pairs_overview instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return plot_pairs_overview(*args, **kwargs)


def plot_homozygous_across_pairs(df: pd.DataFrame, pairs: Optional[List[str]] = None, **kwargs):
    """Deprecated: Use plot_single_genotype_across_pairs instead."""
    warnings.warn(
        "plot_homozygous_across_pairs is deprecated. Use plot_single_genotype_across_pairs instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Try to infer homozygous genotype
    genotype_col = kwargs.get('genotype_col', 'genotype')
    genotypes = df[genotype_col].unique()
    homozygous = [g for g in genotypes if 'homo' in g.lower()]
    if homozygous:
        return plot_single_genotype_across_pairs(df, homozygous[0], pairs=pairs, **kwargs)
    else:
        raise ValueError("No homozygous genotype found in data")
