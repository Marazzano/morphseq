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

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
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


def _projection_bf_rows(frame_inventory: pd.DataFrame) -> pd.DataFrame:
    """Select the BF *projection* frames detection runs on — never a raw z-stack plane.

    Detection / SAM / tracking consume the single focus-stacked BF frame per timepoint, not the
    individual z_stack planes that may also live in the inventory (a z_stack plane is a distinct
    materialized pixel file with its own row). Filtering here keeps a plane from ever reaching a
    detector that assumes one projected frame per timepoint.

    Back-compat: an inventory written before z_stack existed has no ``image_product_type`` column;
    every BF row in that world IS a projection, so the column-absent case keeps all BF rows.
    """
    bf = frame_inventory[frame_inventory["channel_id"].astype(str) == REQUIRED_CHANNEL]
    if "image_product_type" not in bf.columns:
        return bf
    return bf[bf["image_product_type"].astype(str) == "projection"]


def _identity_row_for(inv_row: pd.Series, image_id: str) -> dict:
    """Build the shared frame-identity header for one inventory frame.

    ``z_index`` is NA because detection only ever runs on projection rows (see
    ``_projection_bf_rows``), and a projection frame's z_index IS NA — this is the true value, not a
    lossy synthesis. ``well_id`` is derived from atoms (the inventory carries atoms; ids are derived).
    """
    well_id = derive_well_id(inv_row["experiment_id"], inv_row["well_index"])
    return {
        "experiment_id": str(inv_row["experiment_id"]),
        "well_id": well_id,
        "image_id": str(image_id),
        "time_index": int(inv_row["time_index"]),
        "z_index": pd.NA,
        "channel_id": str(inv_row["channel_id"]),
        "image_path": str(inv_row["image_path"]),
        "image_width_px": int(inv_row["image_width_px"]),
        "image_height_px": int(inv_row["image_height_px"]),
        # Carried so a backend can express a box bound in PHYSICAL units. Boxes are normalized, so
        # width x height x um_per_px^2 gives the frame area a normalized box scales against. Passed
        # through as-is (None when the inventory lacks it); the BACKEND decides whether a missing
        # scale is fatal, which it is exactly when a um^2 bound was actually requested.
        "image_micrometers_per_pixel": inv_row.get("image_micrometers_per_pixel"),
    }


def _is_seahub_frame(inv_row: pd.Series) -> bool:
    return str(inv_row.get("source_scope", "")).strip().casefold() == "seahub"


def _config_for_frame(config, inv_row: pd.Series):
    """Disable the size/coverage bounds on SeaHub frames, which own a narrower rule.

    SeaHub crops are already one reviewed embryo per canonical well, and
    ``_enforce_seahub_single_embryo`` below reduces to exactly one detection there. Those crops are
    also framed tightly on the animal, so an embryo legitimately fills most of the frame — the very
    signature ``max_frame_coverage`` treats as a background grab. Running both rules would let the
    general one reject the reviewed embryo before the SeaHub rule ever sees it.

    Only the SIZE bounds are disabled; IoU/containment grouping still applies, since a nested
    fragment is a fragment on any scope.

    TODO(benchmark): the two rules encode the same intent ("one embryo per well, keep the largest").
    Once the general rule has run on more scopes, compare them on SeaHub data and collapse to one
    if they agree — see PIPELINE_PHILOSOPHY P4a on options that quietly diverge.
    """
    if config is None or not _is_seahub_frame(inv_row):
        return config
    return replace(
        config,
        min_detection_area_um2=math.nan,
        max_detection_area_um2=math.nan,
        max_frame_coverage=math.nan,
    )


def _enforce_seahub_single_embryo(rows: list[dict], inv_row: pd.Series) -> None:
    """Keep only the largest accepted detection in a one-embryo SeaHub frame.

    SeaHub materialization has already cropped exactly one reviewed embryo into each
    canonical well. GroundingDINO can additionally detect nested embryo fragments in
    that crop; retaining those fragments would mint multiple physical embryos for one
    source embryo. Candidate rows remain in the flag-not-drop table for audit.
    """
    if str(inv_row.get("source_scope", "")).strip().casefold() != "seahub":
        return
    kept = [row for row in rows if bool(row["is_kept"])]
    if len(kept) <= 1:
        return
    largest = max(
        kept,
        key=lambda row: (
            (float(row["bbox_x_max_px"]) - float(row["bbox_x_min_px"]))
            * (float(row["bbox_y_max_px"]) - float(row["bbox_y_min_px"])),
            float(row["confidence"]),
        ),
    )
    for row in kept:
        row["is_kept"] = row is largest


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
    adapter. ``image_root`` resolves relative ``image_path`` values (None → paths used as-is).
    """
    detect_frame = _resolve_backend(backend)

    inv = reference_frame_inventory.copy()
    inv["image_id"] = frame_inventory_image_ids(inv, scope_label="frame_detection").astype(str)

    # Detection runs on the BF projection frames only — never a raw z_stack plane.
    bf = _projection_bf_rows(inv)

    rows: list[dict] = []
    for _, inv_row in bf.iterrows():
        image_id = str(inv_row["image_id"])
        identity = _identity_row_for(inv_row, image_id)
        image_path = _resolve_image_path(inv_row["image_path"], image_root)

        det_rows = detect_frame(
            model,
            image_path,
            identity_row=identity,
            detector_model_id=detector_model_id,
            config=_config_for_frame(config, inv_row),
        )
        _enforce_seahub_single_embryo(det_rows, inv_row)
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


def _resolve_image_path(image_path: str, image_root: str | Path | None) -> Path:
    p = Path(str(image_path))
    if p.is_absolute() or image_root is None:
        return p
    return Path(image_root) / p
