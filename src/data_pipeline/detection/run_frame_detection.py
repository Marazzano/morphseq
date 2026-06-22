"""Backend-agnostic frame_detection router — the standalone stage entry.

Reads a validated per-well ``frame_inventory`` (the public microscope-agnostic seam), runs a detector
backend over its BF frames, and produces the shared ``frame_detections`` table (flag-not-drop: kept,
rejected, and no-candidate placeholder rows all retained). The router never knows how a backend
decided ``is_kept``; it only assembles the shared table and validates it against the trusted
inventory.

The model is INJECTED (a router argument), so this slice carries no GPU / weights dependency: callers
(or tests) load and pass the model. Inference itself is reused verbatim inside each backend adapter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.image_materialization.frame_inventory_contract import (
    REQUIRED_CHANNEL,
    derive_well_id,
    frame_inventory_image_ids,
)

from .backends.detectron2 import detect_frame as _detectron2_detect_frame
from .backends.groundingdino import detect_frame as _groundingdino_detect_frame
from .frame_detections_contract import REQUIRED_FRAME_DETECTIONS_COLUMNS
from .validate_frame_detections import validate_frame_detections

# Backend registry: backend name → detect_frame adapter. Adding a backend is a one-line entry here.
BACKENDS = {
    "groundingdino": _groundingdino_detect_frame,
    "detectron2": _detectron2_detect_frame,
}


def _resolve_backend(backend: str):
    key = str(backend).strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"Unsupported detector backend: {backend!r}. Available: {sorted(BACKENDS)}."
        )
    return BACKENDS[key]


def _identity_row_for(inv_row: pd.Series, image_id: str) -> dict:
    """Build the shared frame-identity header for one inventory frame.

    ``z_index`` is synthesized as NA (not a frame_inventory atom on the BF / projection MVP).
    ``well_id`` is derived from atoms (the inventory carries atoms; ids are derived).
    """
    well_id = derive_well_id(inv_row["experiment_id"], inv_row["well_index"])
    return {
        "experiment_id": str(inv_row["experiment_id"]),
        "well_id": well_id,
        "image_id": str(image_id),
        "time_index": int(inv_row["time_index"]),
        "z_index": pd.NA,
        "channel_id": str(inv_row["channel_id"]),
        "source_image_path": str(inv_row["source_image_path"]),
        "image_width_px": int(inv_row["image_width_px"]),
        "image_height_px": int(inv_row["image_height_px"]),
    }


def run_frame_detection_df(
    reference_frame_inventory: pd.DataFrame,
    *,
    backend: str,
    model,
    detector_model_id: str,
    config=None,
    image_root: str | Path | None = None,
) -> pd.DataFrame:
    """Core router: run a backend over the BF frames of an in-memory frame_inventory.

    Returns the validated ``frame_detections`` table. ``reference_frame_inventory`` is the trusted
    read-only inventory; ``model`` is injected; per-frame inference is delegated to the backend
    adapter. ``image_root`` resolves relative ``source_image_path`` values (None → paths used as-is).
    """
    detect_frame = _resolve_backend(backend)

    inv = reference_frame_inventory.copy()
    inv["image_id"] = frame_inventory_image_ids(inv, scope_label="frame_detection").astype(str)

    # Detection runs on the segmentation-timeline channel only.
    bf = inv[inv["channel_id"].astype(str) == REQUIRED_CHANNEL]

    rows: list[dict] = []
    for _, inv_row in bf.iterrows():
        image_id = str(inv_row["image_id"])
        identity = _identity_row_for(inv_row, image_id)
        image_path = _resolve_image_path(inv_row["source_image_path"], image_root)

        det_rows = detect_frame(
            model,
            image_path,
            identity_row=identity,
            detector_model_id=detector_model_id,
            config=config,
        )
        for det in det_rows:
            rows.append({**identity, **det})

    df = pd.DataFrame(rows, columns=list(REQUIRED_FRAME_DETECTIONS_COLUMNS))
    validate_frame_detections(df, reference_frame_inventory, context="frame_detection")
    return df


def run_frame_detection(
    frame_inventory_csv: str | Path,
    output_csv: str | Path,
    *,
    backend: str,
    model,
    detector_model_id: str,
    config=None,
    image_root: str | Path | None = None,
) -> pd.DataFrame:
    """Path wrapper: read a validated frame_inventory CSV, detect, write frame_detections.csv.

    Returns the validated table. The CSV path form is the standalone-stage seam; the DataFrame core
    (``run_frame_detection_df``) is the unit-testable engine.
    """
    reference_frame_inventory = pd.read_csv(frame_inventory_csv)
    df = run_frame_detection_df(
        reference_frame_inventory,
        backend=backend,
        model=model,
        detector_model_id=detector_model_id,
        config=config,
        image_root=image_root,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _resolve_image_path(source_image_path: str, image_root: str | Path | None) -> Path:
    p = Path(str(source_image_path))
    if p.is_absolute() or image_root is None:
        return p
    return Path(image_root) / p
