"""
DEPRECATED: This module has moved to trajectory_analysis.clustering.cluster_extraction

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.clustering import (
        extract_cluster_embryos,
        get_cluster_summary,
        map_clusters_to_phenotypes,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.cluster_extraction is deprecated. "
    "Import from trajectory_analysis.clustering instead.",
    DeprecationWarning,
    stacklevel=2
)

from .clustering.cluster_extraction import (
    extract_cluster_embryos,
    get_cluster_summary,
    map_clusters_to_phenotypes,
)
