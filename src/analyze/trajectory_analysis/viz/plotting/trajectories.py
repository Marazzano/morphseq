"""
Project-specific plotting convenience functions.

Thin wrappers that apply project defaults on top of generic viz.plotting.
"""

import pandas as pd
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from src.analyze.viz.plotting.feature_over_time import plot_feature_over_time
from src.analyze.viz.plotting.faceting_engine import FacetSpec, StyleSpec
from src.analyze.utils.styling import build_suffix_color_lookup
from ...config import GENOTYPE_SUFFIX_COLORS, GENOTYPE_SUFFIX_ORDER


def build_genotype_color_lookup(df: pd.DataFrame, color_by: str) -> Dict[Any, str]:
    """Build genotype-aware color lookup using suffix matching.
    
    Applies project genotype config to generic suffix matching utility.
    """
    if color_by not in df.columns:
        return {}
    
    unique_vals = list(df[color_by].dropna().unique())
    return build_suffix_color_lookup(
        unique_vals,
        GENOTYPE_SUFFIX_COLORS,
        GENOTYPE_SUFFIX_ORDER
    )


def plot_feature(
    df: pd.DataFrame,
    feature: str = 'baseline_deviation_normalized',
    time_col: str = 'predicted_stage_hpf',
    id_col: str = 'embryo_id',
    color_by: Optional[str] = None,
    color_palette: Optional[Dict[Any, str]] = None,
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot feature over time with project-specific defaults.
    
    Convenience wrapper around generic plot_feature_over_time.
    Automatically applies genotype suffix colors when color_by='genotype'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature data
    feature : str, default='baseline_deviation_normalized'
        Feature column to plot
    time_col : str, default='predicted_stage_hpf'
        Time column (project default)
    id_col : str, default='embryo_id'
        Trajectory ID column (project default)
    color_by : str, optional
        Column to color by
    color_palette : dict, optional
        Custom color mapping. If None and color_by='genotype',
        automatically uses GENOTYPE_SUFFIX_COLORS.
    facet_row : str, optional
        Facet by rows
    facet_col : str, optional
        Facet by columns
    **kwargs
        Additional arguments passed to plot_feature_over_time
    
    Returns
    -------
    Figure
        Plotly or matplotlib figure
    """
    # Auto-apply genotype colors if requested
    if color_palette is None and color_by == 'genotype':
        color_palette = build_genotype_color_lookup(df, color_by)
    
    # Delegate to generic plotter
    return plot_feature_over_time(
        df=df,
        feature=feature,
        time_col=time_col,
        id_col=id_col,
        color_by=color_by,
        color_lookup=color_palette,
        facet_row=facet_row,
        facet_col=facet_col,
        **kwargs
    )


# Backward compatibility alias
plot_trajectories_faceted = plot_feature
