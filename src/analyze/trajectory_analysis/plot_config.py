"""
DEPRECATED: This module has been merged into config.py

All constants from this module are now available in config.py.
This file is kept for backward compatibility only.

Please update your imports:
    OLD: from .plot_config import CONSTANT_NAME
    NEW: from .config import CONSTANT_NAME

This wrapper will be removed in a future version.
"""
import warnings

# Re-export everything from config
from .config import (
    # Genotype styling
    GENOTYPE_SUFFIX_COLORS,
    GENOTYPE_SUFFIX_ORDER,
    # Phenotype styling
    PHENOTYPE_COLORS,
    PHENOTYPE_ORDER,
    # Matplotlib styling
    INDIVIDUAL_TRACE_ALPHA,
    INDIVIDUAL_TRACE_LINEWIDTH,
    MEAN_TRACE_LINEWIDTH,
    OVERLAY_ALPHA,
    FACETED_ALPHA,
    TITLE_FONTSIZE,
    SUBPLOT_TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    GRID_ALPHA,
    GRID_LINESTYLE,
    GRID_LINEWIDTH,
    # Plotly styling
    DEFAULT_PLOTLY_HEIGHT,
    DEFAULT_PLOTLY_WIDTH,
    HEIGHT_PER_ROW,
    WIDTH_PER_COL,
    HOVER_TEMPLATE_BASE,
    # Faceted sizing
    MIN_FIGSIZE_WIDTH,
    MIN_FIGSIZE_HEIGHT,
    DEFAULT_FIGSIZE_WIDTH_PER_COL,
    DEFAULT_FIGSIZE_HEIGHT_PER_ROW,
)

warnings.warn(
    "plot_config module is deprecated. Import from config instead: "
    "from .config import CONSTANT_NAME",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'GENOTYPE_SUFFIX_COLORS', 'GENOTYPE_SUFFIX_ORDER',
    'PHENOTYPE_COLORS', 'PHENOTYPE_ORDER',
    'INDIVIDUAL_TRACE_ALPHA', 'INDIVIDUAL_TRACE_LINEWIDTH',
    'MEAN_TRACE_LINEWIDTH', 'OVERLAY_ALPHA', 'FACETED_ALPHA',
    'TITLE_FONTSIZE', 'SUBPLOT_TITLE_FONTSIZE', 'AXIS_LABEL_FONTSIZE',
    'TICK_LABEL_FONTSIZE', 'LEGEND_FONTSIZE',
    'GRID_ALPHA', 'GRID_LINESTYLE', 'GRID_LINEWIDTH',
    'DEFAULT_PLOTLY_HEIGHT', 'DEFAULT_PLOTLY_WIDTH',
    'HEIGHT_PER_ROW', 'WIDTH_PER_COL', 'HOVER_TEMPLATE_BASE',
    'MIN_FIGSIZE_WIDTH', 'MIN_FIGSIZE_HEIGHT',
    'DEFAULT_FIGSIZE_WIDTH_PER_COL', 'DEFAULT_FIGSIZE_HEIGHT_PER_ROW',
]
