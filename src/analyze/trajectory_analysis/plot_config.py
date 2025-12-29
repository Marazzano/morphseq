"""
Plotting configuration and style defaults for trajectory analysis.

Contains color mappings, sizing parameters, and other visual defaults
for creating consistent plots across the analysis pipeline.
"""

# ==============================================================================
# Genotype Styling
# ==============================================================================

# Color mapping based on genotype suffix (independent of gene prefix)
GENOTYPE_SUFFIX_COLORS = {
    'wildtype': '#2E7D32',      # Green
    'heterozygous': '#FFA500',  # Orange
    'homozygous': '#D32F2F',    # Red
    'unknown': '#808080',       # Gray
}
"""Standard colors for genotype suffixes. Works with any gene prefix."""

# Standard ordering for genotype suffixes
GENOTYPE_SUFFIX_ORDER = ['wildtype', 'heterozygous', 'homozygous', 'unknown']
"""Default order for displaying genotype suffixes."""

# ==============================================================================
# Phenotype Styling (distinct from genotype colors)
# ==============================================================================

# Color mapping for phenotype categories (purple/cyan/pink spectrum - distinct from genotype)
PHENOTYPE_COLORS = {
    'CE': '#9467BD',           # Purple
    'HTA': '#17BECF',          # Cyan/Teal
    'BA_rescue': '#E377C2',    # Pink
    'non_penetrant': '#7F7F7F', # Grey
}
"""Colors for phenotype categories. Distinct from genotype suffix colors."""

# Standard ordering for phenotypes
PHENOTYPE_ORDER = ['CE', 'HTA', 'BA_rescue', 'non_penetrant']
"""Default order for displaying phenotypes."""

# ==============================================================================
# Matplotlib Styling
# ==============================================================================

# Line and trace styling
INDIVIDUAL_TRACE_ALPHA = 0.2        # Faded individual trajectories
INDIVIDUAL_TRACE_LINEWIDTH = 0.8    # Thin individual lines
MEAN_TRACE_LINEWIDTH = 2.2          # Bold mean trajectory line
OVERLAY_ALPHA = 0.25                # Faded when overlaying multiple groups
FACETED_ALPHA = 0.25                # Faded when faceting

# Font sizes
TITLE_FONTSIZE = 14
SUBPLOT_TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 9

# Grid and styling
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5

# ==============================================================================
# Plotly Styling
# ==============================================================================

# Figure defaults
DEFAULT_PLOTLY_HEIGHT = 500
DEFAULT_PLOTLY_WIDTH = 1400
HEIGHT_PER_ROW = 350              # Height multiplier for row count
WIDTH_PER_COL = 400               # Width multiplier for column count

# Hover template defaults
HOVER_TEMPLATE_BASE = (
    '<b>Embryo:</b> %{customdata[0]}<br>'
    '<b>Time:</b> %{x:.2f} hpf<br>'
    '<b>Value:</b> %{y:.4f}<br>'
    '<extra></extra>'
)

# ==============================================================================
# Faceted Plot Sizing
# ==============================================================================

# Default sizing for dynamic faceted plots
MIN_FIGSIZE_WIDTH = 6
MIN_FIGSIZE_HEIGHT = 4
DEFAULT_FIGSIZE_WIDTH_PER_COL = 5
DEFAULT_FIGSIZE_HEIGHT_PER_ROW = 4.5
