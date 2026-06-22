"""Boolean convenience wrapper for the generic frame-mask validator."""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.validate_frame_masks import validate_frame_mask_block


def valid_frame_masks(frame_masks: pd.DataFrame) -> bool:
    """Return True when a frame-mask block satisfies the generic row-level contract."""
    try:
        validate_frame_mask_block(frame_masks)
    except ValueError:
        return False
    return True
