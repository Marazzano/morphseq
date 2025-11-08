"""
Configuration Module

Centralized default parameters for trajectory analysis pipeline.

All default values are defined here for consistency across modules.
Users can override these in function calls.
"""

# ==============================================================================
# Bootstrap Parameters
# ==============================================================================

N_BOOTSTRAP = 100
"""Number of bootstrap iterations for consensus clustering"""

BOOTSTRAP_FRAC = 0.8
"""Fraction of samples to include in each bootstrap iteration (80%)"""

RANDOM_SEED = 42
"""Random seed for reproducibility"""


# ==============================================================================
# DTW Parameters
# ==============================================================================

DTW_WINDOW = 5
"""Sakoe-Chiba band constraint for DTW alignment (max warping distance)"""

GRID_STEP = 0.5
"""Time step for common grid interpolation (in HPF units)"""


# ==============================================================================
# Data Processing
# ==============================================================================

MIN_TIMEPOINTS = 3
"""Minimum number of timepoints required for a valid trajectory"""

DEFAULT_EMBRYO_ID_COL = 'embryo_id'
"""Default column name for embryo identifiers"""

DEFAULT_METRIC_COL = 'normalized_baseline_deviation'
"""Default column name for metric values"""

DEFAULT_TIME_COL = 'predicted_stage_hpf'
"""Default column name for time values"""

DEFAULT_GENOTYPE_COL = 'genotype'
"""Default column name for genotype labels"""


# ==============================================================================
# Classification Thresholds
# ==============================================================================

THRESHOLD_MAX_P = 0.8
"""Minimum max probability for core membership (80% confidence)"""

THRESHOLD_LOG_ODDS_GAP = 0.7
"""Minimum log-odds gap between top clusters for core membership"""

THRESHOLD_OUTLIER_MAX_P = 0.5
"""Maximum max probability before being classified as outlier (50%)"""

ADAPTIVE_PERCENTILE = 0.75
"""Percentile for adaptive per-cluster thresholds (75th percentile)"""


# ==============================================================================
# Plotting Parameters
# ==============================================================================

DEFAULT_DPI = 120
"""Default resolution for saved figures"""

DEFAULT_FIGSIZE = (12, 6)
"""Default figure size (width, height) in inches"""

MEMBERSHIP_COLORS = {
    'core': '#2ecc71',       # Green
    'uncertain': '#f1c40f',  # Yellow
    'outlier': '#e74c3c'     # Red
}
"""Standard colors for membership categories"""


# ==============================================================================
# File Paths (Optional - can be overridden)
# ==============================================================================

# These are examples - actual paths should be provided by users
DEFAULT_CURV_DIR = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/Keyence'
DEFAULT_META_DIR = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/built_metadata_files'
