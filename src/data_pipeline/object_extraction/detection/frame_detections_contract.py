"""frame_detections contract — the shared, microscope-agnostic detection product table.

Detection is a standalone post-``frame_inventory`` stage. Every detector backend (GroundingDINO,
Detectron2, …) diverges before this table and converges on it: one row per detector candidate after
the backend adapter has translated native model output into the shared shape. See
``specs/detect-seg-track/targets/detection_world.md``.

Two column blocks compose the table:
  - the shared frame-identity header, IMPORTED from the frame_inventory contract owner
    (``DOWNSTREAM_FRAME_IDENTITY_BLOCK``) — detection does not re-declare it, so there is one source
    of truth;
  - the detection-specific block (``REQUIRED_DETECTION_BLOCK``) defined here.

The ``is_kept`` column is the shared seam between backend-specific filtering and downstream
segmentation: filtering logic is backend-specific, the filtering *outcome* is shared. The table is
flag-not-drop — rejected candidates and no-candidate placeholder rows are RETAINED, not filtered out,
so the artifact proves every frame was processed.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    DOWNSTREAM_FRAME_IDENTITY_BLOCK,
)

# ---------------------------------------------------------------------------
# Detection-specific column block
# ---------------------------------------------------------------------------

# The detector-specific minimum added on top of the shared frame-identity header. One row per
# detector candidate after backend adaptation — NOT only accepted detections.
REQUIRED_DETECTION_BLOCK: tuple[str, ...] = (
    "detection_id",        # {image_id}_det{idx:04d}, or {image_id}_det_none placeholder — unique in well
    "detector_backend",    # e.g. "groundingdino" — non-empty
    "detector_model_id",   # e.g. "SwinT_OGC" — non-empty
    "class_label",         # detector class / matched phrase (NA on no-candidate placeholder)
    "confidence",          # finite, in CONFIDENCE_RANGE on kept rows (NA on placeholder)
    "bbox_x_min_px",       # absolute pixel xyxy (NA on placeholder)
    "bbox_y_min_px",
    "bbox_x_max_px",
    "bbox_y_max_px",
    "bbox_format",         # one of ALLOWED_BBOX_FORMATS (NA on placeholder)
    "is_kept",             # boolean — the shared filtering outcome seam
)

# The full frame_detections contract: shared identity header + detection block.
REQUIRED_FRAME_DETECTIONS_COLUMNS: tuple[str, ...] = (
    *DOWNSTREAM_FRAME_IDENTITY_BLOCK,
    *REQUIRED_DETECTION_BLOCK,
)

# The bbox coordinate columns, grouped for range / finiteness checks.
BBOX_COLUMNS: tuple[str, ...] = (
    "bbox_x_min_px",
    "bbox_y_min_px",
    "bbox_x_max_px",
    "bbox_y_max_px",
)

# Allowed values for ``bbox_format``. MVP emits absolute-pixel xyxy only; extensible later.
ALLOWED_BBOX_FORMATS: tuple[str, ...] = ("xyxy_px_abs",)

# Allowed inclusive confidence range for kept detections.
CONFIDENCE_RANGE: tuple[float, float] = (0.0, 1.0)

# detection_id suffix for the no-candidate placeholder row.
NO_CANDIDATE_SUFFIX: str = "_det_none"


# ---------------------------------------------------------------------------
# detection_id helpers
# ---------------------------------------------------------------------------


def detection_id(image_id: str, candidate_index: int) -> str:
    """Return the canonical real-candidate detection id, e.g. ``..._BF_t0000_det0003``.

    Preferred for debugging; the hard requirement is uniqueness within the well.
    """
    return f"{image_id}_det{int(candidate_index):04d}"


def no_candidate_detection_id(image_id: str) -> str:
    """Return the no-candidate placeholder detection id, e.g. ``..._BF_t0000_det_none``."""
    return f"{image_id}{NO_CANDIDATE_SUFFIX}"


def is_no_candidate_id(value: str) -> bool:
    """True if ``value`` is a no-candidate placeholder detection id."""
    return str(value).endswith(NO_CANDIDATE_SUFFIX)


def has_kept_detections(frame_detections: pd.DataFrame) -> bool:
    """True when this well has at least one KEPT detection — i.e. something was found.

    ``False`` is a normal, expected answer: a 96-well plate legitimately contains wells with no
    embryo, and this contract states that outcome explicitly via the ``_det_none`` placeholder row
    (``is_kept=False``), which ``validate_frame_detections`` accepts as a valid shape.

    This predicate lives HERE, beside the vocabulary it reads, because ``is_kept`` and
    ``_det_none`` are minted by THIS contract (P6: a contract lives with the code that mints it).
    Every consumer that must branch on "did this well have anything?" — SAM2 prompting today,
    anything else tomorrow — asks this one question rather than re-deriving it from ``is_kept``,
    which is how a shared seam quietly grows two incompatible readings.
    """
    if frame_detections.empty or "is_kept" not in frame_detections.columns:
        return False
    return bool(frame_detections["is_kept"].astype(bool).any())
