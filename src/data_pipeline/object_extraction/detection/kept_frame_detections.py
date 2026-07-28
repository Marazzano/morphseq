"""Consumer view + CSV reader for the frame_detections product.

Any downstream stage that acts on detections must consume the kept view through
``kept_frame_detections`` rather than hand-rolling row filtering — this keeps the ``is_kept`` seam
the single place where "what survives filtering" is decided. See
``specs/detect-seg-track/targets/detection_world.md`` "Kept Detection View".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .frame_detections_contract import BBOX_COLUMNS, REQUIRED_FRAME_DETECTIONS_COLUMNS
from .validate_frame_detections import (
    validate_frame_detection_block,
    validate_frame_detections,
)

# Columns coerced to float on read (NA-tolerant — placeholder rows carry NA here).
_FLOAT_COLUMNS: tuple[str, ...] = (*BBOX_COLUMNS, "confidence")

# Columns coerced to str on read.
_STR_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "image_id",
    "channel_id",
    "image_path",
    "detection_id",
    "detector_backend",
    "detector_model_id",
    "class_label",
    "bbox_format",
)


def kept_frame_detections(
    df: pd.DataFrame,
    reference_frame_inventory: pd.DataFrame | None = None,
    *,
    context: str = "frame_detections",
) -> pd.DataFrame:
    """Validate then return only the kept detections (``is_kept == True``).

    The detection-block schema layer always runs. When ``reference_frame_inventory`` is supplied the
    full composed validator runs (schema + reference), so the consume boundary can also confirm the
    detections belong to the trusted inventory. Rejected candidates and no-candidate placeholder rows
    are dropped from the returned view.
    """
    if reference_frame_inventory is not None:
        validate_frame_detections(df, reference_frame_inventory, context=context)
    else:
        validate_frame_detection_block(df, context=context)
    return df[df["is_kept"].astype(bool)].copy()


def read_frame_detections_csv(path: str | Path) -> pd.DataFrame:
    """Read a ``frame_detections.csv`` with stable dtypes for the contract columns.

    ``is_kept`` is coerced from its CSV text form back to a real python ``bool`` column; bbox /
    confidence are floats (NA-tolerant); identity / provenance columns are strings. Unknown
    backend-specific audit columns pass through untouched.
    """
    df = pd.read_csv(path)

    for col in _STR_COLUMNS:
        if col in df.columns:
            # Keep NA as NA (placeholder rows), stringify the rest.
            df[col] = df[col].astype("object").where(df[col].notna(), other=pd.NA)
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str)

    for col in _FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_kept" in df.columns:
        df["is_kept"] = df["is_kept"].map(_coerce_bool).astype(bool)

    return df


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no", ""}:
        return False
    raise ValueError(f"frame_detections: cannot coerce is_kept value {value!r} to bool")


# Silence unused-import linters: REQUIRED_FRAME_DETECTIONS_COLUMNS is re-exported via __init__.
__all__ = [
    "kept_frame_detections",
    "read_frame_detections_csv",
    "REQUIRED_FRAME_DETECTIONS_COLUMNS",
]
