"""Utilities for segmentation and tracking.

mask_processing.py moved OUT to ``segmentation/shared/mask_processing.py`` — it is still
live (several feature_extraction metrics use clean_embryo_mask). Re-exported here for
backward compatibility with this archived package's own imports.
"""

from data_pipeline.object_extraction.segmentation.shared.mask_processing import (
    clean_embryo_mask,
    fill_small_holes,
    largest_connected_component,
    normalize_binary_mask,
    remove_small_components,
)

