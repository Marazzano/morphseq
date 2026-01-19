"""
General-purpose utilities for morphseq analysis.

This package provides helper functions for data loading, time binning,
splitting, and file I/O operations.

Submodules
==========
- binning : Time-based aggregation of VAE embeddings
- data_loading : Load and combine experiment data from build06 outputs
- file_utils : Path management and file I/O helpers
- splitting : Group-aware train/test splitting for ML tasks
- timeseries : Generic time-series algorithms (DTW, DBA, interpolation)

Note: Plotting utilities have been moved to src.analyze.viz.plotting
The old imports from utils.plotting and utils.plotting_faceted still work
but are deprecated.
"""

# Data loading
from .data_loading import (
    load_experiment,
    load_experiments,
    filter_by_genotypes,
    get_genotype_summary,
    validate_required_columns,
)

# Binning utilities
from .binning import (
    bin_embryos_by_time,
    filter_binned_data,
)

# File I/O
from .file_utils import (
    make_safe_comparison_name,
    get_plot_path,
    get_data_path,
    save_dataframe,
    extract_genotype_short_names,
)

# Splitting
from .splitting import (
    train_test_split_by_group,
    leave_one_out_by_group,
    get_group_split_masks,
    get_split_info,
)

# Note: Plotting utilities are now at src.analyze.viz.plotting
# To avoid circular imports, they are not imported here.
# Use: from src.analyze.viz.plotting import plot_feature_over_time
# Or for backward compat (deprecated):
#   from src.analyze.utils.plotting import plot_embryos_metric_over_time

__all__ = [
    # Data loading
    "load_experiment",
    "load_experiments",
    "filter_by_genotypes",
    "get_genotype_summary",
    "validate_required_columns",
    # Binning
    "bin_embryos_by_time",
    "filter_binned_data",
    # File I/O
    "make_safe_comparison_name",
    "get_plot_path",
    "get_data_path",
    "save_dataframe",
    "extract_genotype_short_names",
    # Splitting
    "train_test_split_by_group",
    "leave_one_out_by_group",
    "get_group_split_masks",
    "get_split_info",
]
