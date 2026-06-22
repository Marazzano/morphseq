"""Layered validators for the frame_detections product.

Two layers, composed by ``validate_frame_detections``:

  - ``validate_frame_identity_block`` (imported from the frame_inventory contract owner) — the
    REFERENCE layer: "these bones belong to the right body" (image_id ⊆ inventory; carried columns
    agree with the matching inventory row). Needs the trusted ``reference_frame_inventory``.
  - ``validate_frame_detection_block`` — the SCHEMA layer: "this table has the right bones" (columns
    present, ids unique, provenance non-empty, confidence/bbox ranges, bbox_format allowed,
    is_kept boolean). Needs no inventory.

The detection-block checks are conditional on ``is_kept`` and the ``_det_none`` placeholder suffix
(see ``specs/detect-seg-track/targets/detection_world.md`` "Detection Validator"):

    is_kept == True:
        class_label / confidence / bbox fields must be valid
    is_kept == False and detection_id != {image_id}_det_none:
        rejected candidate — bbox / confidence validated if the backend produced them
    is_kept == False and detection_id == {image_id}_det_none:
        no-candidate placeholder — class_label / confidence / bbox fields may be NA
"""

from __future__ import annotations

import math

import pandas as pd

from data_pipeline.image_materialization.frame_inventory_contract import (
    validate_frame_identity_block,
)

from .frame_detections_contract import (
    ALLOWED_BBOX_FORMATS,
    BBOX_COLUMNS,
    CONFIDENCE_RANGE,
    REQUIRED_DETECTION_BLOCK,
    is_no_candidate_id,
)


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sample(df: pd.DataFrame, mask: pd.Series, cols: list[str]) -> list[dict]:
    present = [c for c in cols if c in df.columns]
    return df.loc[mask, present].head(3).to_dict(orient="records")


def validate_frame_detection_block(df: pd.DataFrame, *, context: str = "frame_detections") -> None:
    """Schema-layer validator for the detection block — "this table has the right bones".

    Checks (no reference inventory required):
      - required detection columns present;
      - ``detection_id`` unique within the well;
      - ``detector_backend`` / ``detector_model_id`` non-empty;
      - ``is_kept`` is boolean;
      - conditional on ``is_kept`` + the ``_det_none`` suffix: kept rows have finite confidence in
        ``CONFIDENCE_RANGE``, finite bbox inside image bounds with min < max, and an allowed
        ``bbox_format``; rejected candidate rows validate any bbox/confidence they carry; no-candidate
        placeholder rows may be NA.

    Raises:
        ValueError: prefixed with ``context`` and an offending-row sample, on any failure.
    """
    missing = [c for c in REQUIRED_DETECTION_BLOCK if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] missing required detection columns: {sorted(missing)}")

    if len(df) == 0:
        return

    # detection_id unique within the well (the table is one well; image_materialization guarantees
    # one well_id, enforced by the identity layer — here we just require global uniqueness).
    dup = df.duplicated(subset=["detection_id"], keep=False)
    if dup.any():
        n = int(dup.sum())
        sample = _sample(df, dup, ["image_id", "detection_id"])
        raise ValueError(
            f"[{context}] {n} row(s) have a non-unique detection_id within the well. "
            f"First offenders: {sample}"
        )

    # Provenance columns non-empty on every row.
    for col in ("detector_backend", "detector_model_id"):
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty.any():
            n = int(empty.sum())
            sample = _sample(df, empty, ["detection_id", col])
            raise ValueError(
                f"[{context}] {n} row(s) have an empty '{col}'. First offenders: {sample}"
            )

    # is_kept must be a real boolean (bool dtype, or all-python-bool object column).
    is_kept = df["is_kept"]
    if is_kept.isna().any():
        n = int(is_kept.isna().sum())
        raise ValueError(f"[{context}] {n} row(s) have a null is_kept; it must be boolean.")
    if not all(isinstance(v, (bool,)) for v in is_kept.tolist()):
        raise ValueError(
            f"[{context}] is_kept must be boolean; got dtype={is_kept.dtype} with non-bool values."
        )

    # Row classification for conditional validation.
    placeholder = df["detection_id"].astype(str).map(is_no_candidate_id)
    kept = is_kept.astype(bool)

    # Kept rows: a no-candidate placeholder can never be kept.
    bad_kept_placeholder = kept & placeholder
    if bad_kept_placeholder.any():
        n = int(bad_kept_placeholder.sum())
        sample = _sample(df, bad_kept_placeholder, ["detection_id", "is_kept"])
        raise ValueError(
            f"[{context}] {n} no-candidate placeholder row(s) marked is_kept=True. "
            f"First offenders: {sample}"
        )

    # Rows whose value columns MUST be valid: kept rows, plus rejected candidates that carry values.
    # Placeholder rows are exempt (value columns may be NA).
    for idx, row in df.iterrows():
        is_placeholder = bool(placeholder.loc[idx])
        is_kept_row = bool(kept.loc[idx])

        if is_placeholder:
            # No-candidate placeholder: value columns may be NA, nothing else to check here.
            continue

        has_any_bbox = any(pd.notna(row[c]) for c in BBOX_COLUMNS)
        has_conf = pd.notna(row["confidence"])

        # Rejected candidates may, in principle, omit values — but if present they must be valid.
        # Kept rows must have valid values.
        require_values = is_kept_row

        if require_values and not has_conf:
            raise ValueError(
                f"[{context}] kept detection {row['detection_id']!r} has NA confidence."
            )
        if require_values and not has_any_bbox:
            raise ValueError(
                f"[{context}] kept detection {row['detection_id']!r} has NA bbox."
            )

        # Confidence range / finiteness (when present).
        if has_conf:
            lo, hi = CONFIDENCE_RANGE
            if not _is_finite(row["confidence"]) or not (lo <= float(row["confidence"]) <= hi):
                raise ValueError(
                    f"[{context}] detection {row['detection_id']!r} has confidence "
                    f"{row['confidence']!r} outside finite range [{lo}, {hi}]."
                )

        # bbox finiteness, bounds, ordering, format (when any bbox value present).
        if has_any_bbox:
            for c in BBOX_COLUMNS:
                if not _is_finite(row[c]):
                    raise ValueError(
                        f"[{context}] detection {row['detection_id']!r} has non-finite {c}={row[c]!r}."
                    )
            x_min, y_min = float(row["bbox_x_min_px"]), float(row["bbox_y_min_px"])
            x_max, y_max = float(row["bbox_x_max_px"]), float(row["bbox_y_max_px"])
            width, height = float(row["image_width_px"]), float(row["image_height_px"])

            if not (x_min < x_max):
                raise ValueError(
                    f"[{context}] detection {row['detection_id']!r}: bbox_x_min_px "
                    f"({x_min}) not < bbox_x_max_px ({x_max})."
                )
            if not (y_min < y_max):
                raise ValueError(
                    f"[{context}] detection {row['detection_id']!r}: bbox_y_min_px "
                    f"({y_min}) not < bbox_y_max_px ({y_max})."
                )
            if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                raise ValueError(
                    f"[{context}] detection {row['detection_id']!r}: bbox "
                    f"[{x_min}, {y_min}, {x_max}, {y_max}] outside image bounds "
                    f"({width} x {height})."
                )

            fmt = row["bbox_format"]
            if pd.isna(fmt) or str(fmt) not in ALLOWED_BBOX_FORMATS:
                raise ValueError(
                    f"[{context}] detection {row['detection_id']!r} has bbox_format {fmt!r}; "
                    f"allowed: {ALLOWED_BBOX_FORMATS}."
                )


def validate_frame_detections(
    df: pd.DataFrame,
    reference_frame_inventory: pd.DataFrame,
    *,
    context: str = "frame_detections",
) -> None:
    """Composed validator: identity (reference) layer + detection (schema) layer.

    ``reference_frame_inventory`` is the trusted per-well frame_inventory, passed in as a READ-ONLY
    DataFrame — neither layer reads files.

    Raises:
        ValueError: from whichever layer fails first.
    """
    validate_frame_identity_block(df, reference_frame_inventory, context=context)
    validate_frame_detection_block(df, context=context)
