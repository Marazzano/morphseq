"""
Configuration for WT quantile envelope penetrance pipeline.

Design decisions:
- Penetrance is frame-level (embryo summaries are secondary/descriptive)
- Smoothing selected by envelope shape stability, NOT by optimizing WT outside-rate
- WT calibration reported as diagnostic; formal inference deferred to phase 2
- Het embryos serve as additional WT-like reference for calibration comparison
"""

from pathlib import Path

# ============================================================================
# Data Paths
# ============================================================================

DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/final_data")
EMBRYO_DATA_PATH = DATA_DIR / "embryo_data_with_labels.csv"

OUTPUT_DIR = Path(__file__).parent / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# ============================================================================
# Column Names
# ============================================================================

METRIC_NAME = "baseline_deviation_normalized"
TIME_COL = "predicted_stage_hpf"
EMBRYO_COL = "embryo_id"
GENOTYPE_COL = "genotype"
CATEGORY_COL = "cluster_categories"
SUBCATEGORY_COL = "cluster_subcategories"

# ============================================================================
# Genotypes
# ============================================================================

WT_GENOTYPE = "cep290_wildtype"
HET_GENOTYPE = "cep290_heterozygous"

# ============================================================================
# Envelope Parameters
# ============================================================================

TIME_BIN_WIDTH = 2.0  # hpf

QUANTILE_LOW = 0.025
QUANTILE_HIGH = 0.975

# Candidate LOESS fracs swept from smallest to largest; smallest passing
# validity checks is selected independently for lower and upper curves.
LOESS_CANDIDATE_FRACS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Set to a float to force a specific frac for both curves (skips selection).
LOESS_FRAC_OVERRIDE = None

# Used if no candidate frac passes validity checks.
LOESS_FALLBACK_FRAC = 0.10

# Minimum WT frames in a bin to compute quantile (otherwise unsupported=True).
MIN_WT_FRAMES_PER_BIN = 10

# Metric is non-negative (baseline deviation); enforce lower >= 0 in envelope.
METRIC_NONNEG = True

# One-directional penetrance: only flag frames that EXCEED the upper bound.
# Use True for deviation metrics (baseline_deviation_normalized) where "too low"
# is not a meaningful phenotype.  When True, the lower bound is ignored for
# marking penetrant frames and for plotting.
UPPER_BOUND_ONLY = True

# ============================================================================
# Category Definitions
# ============================================================================

BROAD_CATEGORIES = [
    "Not Penetrant",
    "Intermediate",
    "High_to_Low",
    "Low_to_High",
]

SUBCATEGORIES = [
    "Not Penetrant",
    "Intermediate",
    "High_to_Low_A",
    "High_to_Low_B",
    "Low_to_High_A",
    "Low_to_High_B",
]

# ============================================================================
# Color Schemes
# ============================================================================

CATEGORY_COLORS = {
    "Low_to_High": "#E74C3C",
    "High_to_Low": "#3498DB",
    "Intermediate": "#9B59B6",
    "Not Penetrant": "#2ECC71",
}

SUBCATEGORY_COLORS = {
    "Low_to_High_A": "#E74C3C",
    "Low_to_High_B": "#C0392B",
    "High_to_Low_A": "#3498DB",
    "High_to_Low_B": "#2980B9",
    "Intermediate": "#9B59B6",
    "Not Penetrant": "#2ECC71",
}

GENOTYPE_COLORS = {
    "cep290_wildtype": "#1f77b4",
    "cep290_heterozygous": "#ff7f0e",
    "cep290_homozygous": "#d62728",
}

# ============================================================================
# Plotting Parameters
# ============================================================================

KEY_STAGES_HPF = [24, 48, 72, 96]

FIGSIZE_CURVES = (12, 8)
FIGSIZE_HEATMAP = (14, 6)
FIGSIZE_DIAGNOSTIC = (14, 8)
FIGSIZE_BARS = (10, 6)

DPI = 150
