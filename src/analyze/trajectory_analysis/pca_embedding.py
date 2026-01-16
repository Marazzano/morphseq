"""
DEPRECATED: This module has been moved to trajectory_analysis.utilities.pca

Please update your imports:
    OLD: from trajectory_analysis.pca_embedding import fit_pca_on_embeddings
    NEW: from trajectory_analysis.utilities import fit_pca_on_embeddings

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "trajectory_analysis.pca_embedding is deprecated. "
    "Import from trajectory_analysis.utilities instead: "
    "from trajectory_analysis.utilities import fit_pca_on_embeddings",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from .utilities.pca import (
    fit_pca_on_embeddings,
    transform_embeddings_to_pca,
    compute_wt_reference_by_time,
    subtract_wt_reference,
    fit_transform_pca,
)

__all__ = [
    'fit_pca_on_embeddings',
    'transform_embeddings_to_pca',
    'compute_wt_reference_by_time',
    'subtract_wt_reference',
    'fit_transform_pca',
]
