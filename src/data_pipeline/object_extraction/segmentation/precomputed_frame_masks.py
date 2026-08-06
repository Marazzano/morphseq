"""Ingest authoritative experiment-level frame masks into per-well shards.

This adapter is deliberately model-free.  It is the canonical handoff for a scope that
already produced ``frame_masks`` upstream: select one well from the experiment-level table,
validate the selected rows against the trusted per-well ``frame_inventory``, and preserve the
rows byte-for-column rather than allowing a downstream segmentation model to replace them.

An optional ``frame_detections`` input is retained as an independent audit.  Its kept boxes are
written through the existing ``prompt_seeds`` sidecar contract, but they are never used to build,
filter, rank, or otherwise mutate the authoritative masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    validate_frame_identity_block,
)
from data_pipeline.object_extraction.detection.validate_frame_detections import (
    validate_frame_detections,
)
from data_pipeline.object_extraction.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
)
from data_pipeline.object_extraction.segmentation.prompt_seeds import (
    PROMPT_SEED_COLUMNS,
    build_prompt_seeds,
    validate_prompt_seeds,
)
from data_pipeline.object_extraction.segmentation.validate_frame_masks import (
    validate_frame_masks,
)


@dataclass(frozen=True)
class PrecomputedFrameMasksWell:
    """Validated per-well masks plus the independent detection-audit sidecar."""

    frame_masks: pd.DataFrame
    detection_audit: pd.DataFrame


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def select_precomputed_frame_masks_for_well(
    precomputed_frame_masks: pd.DataFrame,
    frame_inventory: pd.DataFrame,
    *,
    well_id: str,
) -> pd.DataFrame:
    """Select and validate one canonical per-well shard.

    Extra source columns are allowed for upstream provenance, but the emitted shard contains
    exactly ``FRAME_MASKS_REQUIRED_COLUMNS`` in canonical order.  A missing well is an error: an
    authoritative experiment-level handoff must represent no-mask frames with the canonical
    placeholder row instead of silently omitting the well.
    """

    expected_well_id = str(well_id)
    _require_columns(
        precomputed_frame_masks,
        FRAME_MASKS_REQUIRED_COLUMNS,
        "precomputed_frame_masks",
    )
    _require_columns(frame_inventory, ("well_id",), "frame_inventory")

    inventory_wells = sorted(frame_inventory["well_id"].dropna().astype(str).unique().tolist())
    if inventory_wells != [expected_well_id]:
        raise ValueError(
            "frame_inventory must contain exactly the requested well_id "
            f"{expected_well_id!r}; found {inventory_wells[:5]}"
        )

    selected = precomputed_frame_masks.loc[
        precomputed_frame_masks["well_id"].astype(str) == expected_well_id,
        list(FRAME_MASKS_REQUIRED_COLUMNS),
    ].copy()
    if selected.empty:
        raise ValueError(
            "precomputed_frame_masks contains no rows for requested well_id "
            f"{expected_well_id!r}; provide canonical no-mask placeholder rows when appropriate"
        )

    selected = selected.sort_values(
        ["time_index", "image_id", "mask_id"], kind="mergesort"
    ).reset_index(drop=True)

    # The shared identity validator checks the complete downstream identity header; the
    # frame_masks validator adds mask ids, tracks, geometry, RLE, and frame-bound checks.
    validate_frame_identity_block(
        selected,
        frame_inventory,
        context="precomputed_frame_masks",
    )
    validate_frame_masks(selected, frame_inventory)
    return selected


def prepare_precomputed_frame_masks_for_well(
    precomputed_frame_masks: pd.DataFrame,
    frame_inventory: pd.DataFrame,
    *,
    well_id: str,
    frame_detections: pd.DataFrame | None = None,
    require_detection_audit: bool = True,
) -> PrecomputedFrameMasksWell:
    """Prepare authoritative masks and a non-authoritative detection audit for one well."""

    selected = select_precomputed_frame_masks_for_well(
        precomputed_frame_masks,
        frame_inventory,
        well_id=well_id,
    )

    if not require_detection_audit:
        audit = pd.DataFrame(columns=PROMPT_SEED_COLUMNS)
        return PrecomputedFrameMasksWell(frame_masks=selected, detection_audit=audit)

    if frame_detections is None:
        raise ValueError(
            "require_detection_audit=True requires a per-well frame_detections table"
        )

    validate_frame_detections(
        frame_detections,
        frame_inventory,
        context="precomputed_frame_masks_detection_audit",
    )
    audit = build_prompt_seeds(frame_detections)
    # A valid no-candidate detection placeholder produces an empty audit table.  That is still a
    # completed audit, so only invoke the prompt validator when kept detections exist.
    if not audit.empty:
        validate_prompt_seeds(audit, frame_detections, frame_inventory)

    return PrecomputedFrameMasksWell(frame_masks=selected, detection_audit=audit)


def write_precomputed_frame_masks_for_well(
    *,
    precomputed_frame_masks_csv: Path,
    frame_inventory_csv: Path,
    well_id: str,
    output_csv: Path,
    detection_audit_csv: Path,
    frame_detections_csv: Path | None = None,
    require_detection_audit: bool = True,
) -> PrecomputedFrameMasksWell:
    """Path-level task adapter used by the pipeline orchestrator."""

    precomputed = pd.read_csv(precomputed_frame_masks_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    frame_detections = (
        pd.read_csv(frame_detections_csv) if frame_detections_csv is not None else None
    )
    result = prepare_precomputed_frame_masks_for_well(
        precomputed,
        frame_inventory,
        well_id=well_id,
        frame_detections=frame_detections,
        require_detection_audit=bool(require_detection_audit),
    )

    output_csv = Path(output_csv)
    detection_audit_csv = Path(detection_audit_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    detection_audit_csv.parent.mkdir(parents=True, exist_ok=True)
    result.frame_masks.to_csv(output_csv, index=False)
    result.detection_audit.to_csv(detection_audit_csv, index=False)
    return result
