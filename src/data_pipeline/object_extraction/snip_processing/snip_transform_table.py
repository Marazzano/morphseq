"""The per-well snip transform table: one row per embryo-time, shared by every channel.

WHY THIS EXISTS. The snip transform used to be computed, used to render pixels, and discarded. That
made a snip an assertion with no supporting evidence: you could look at it, but not ask how it was
built, not reproduce it, and not build a DIFFERENT one from the same inputs. Persisting the transform
is what makes later refinement POSSIBLE WITHOUT BEING SCHEDULED — a yolk-aware orientation can be
derived whenever someone wants it, from the record, rather than requiring yolk masks to be wired into
the snip DAG up front.

GRAIN. File grain and row grain are deliberately different:

    file grain   one table per WELL          (matches snip_inventory, and the per-well job shape)
    row grain    one transform per EMBRYO-TIME

A container holding many records does not claim the well has one transform. Every row carries its own
``snip_transform_id`` and is independently validated.

CHANNEL-INDEPENDENT BY CONSTRUCTION. ``snip_transform_id`` is ``physical_embryo_id x time_index``
with no channel segment, because the transform is derived from the segmentation MASK and is a fact
about the animal, not about a channel. ``BF__clahe_blend`` and ``RFP__no_change`` rows of the same
embryo-time reference the SAME row. This is not a storage optimization — it is what makes sibling
misregistration UNREPRESENTABLE. Per-snip copies of identical geometry can drift apart; one
referenced row cannot.

TWO PAYLOADS, TWO GUARANTEES. They are kept in separate columns because they answer different
questions and have different lifetimes:

    resolved_transform_chain_json    REPLAY: the ordered steps that produced THESE pixels. Rebuilds
                                     into a TransformChain and re-renders byte-for-byte.
    transform_derivation_inputs_json RE-DERIVE: the source identity, grid, calibrations and
                                     orientation DECISION. A future yolk-aware pass keeps the grid
                                     and source, REPLACES the orientation block, and derives a NEW
                                     chain — which the resolved chain alone could never support,
                                     since a refined orientation changes the transform itself.

WHY NOT ONE MATRIX. A composite affine is COORDINATE truth (where a point lands). It cannot express
the anti-aliased resize prefilter, requested-vs-realized scale, crop/pad semantics, flip-by-indexing,
or the image-vs-mask interpolation split — all of which change pixels. Persisting one matrix and
calling the snip reproducible would resurrect the exact mistake ``TransformChain`` was built to
prevent. The full ordered chain is serialized.

WHY JSON-IN-A-COLUMN. The payload is strings, enums, optionals and nested structure. Searchable
scalars are real typed columns; the nested evidence rides in JSON text columns, exactly as
``mask_rle`` does in ``frame_masks``.

Import rules: this module may import numpy/pandas and ``image_geometry``. It MUST NOT import the
entrypoint, orchestration, tasks, or Snakemake rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.model_servers.atomic_write import atomic_write_via
from image_geometry import TransformChain
from image_geometry.transforms import GridTransform

# Serialization format version, carried in every row. Bump on an incompatible shape change; a reader
# that does not recognize the version must fail loud rather than guess at a partial understanding.
SNIP_TRANSFORM_SCHEMA_VERSION = 1

# Searchable scalars as real columns; complex payloads as JSON text. Calibration is PER-AXIS: a
# scalar um_per_px would silently bake in an assumption of square pixels that no contract guarantees.
SNIP_TRANSFORM_TABLE_COLUMNS: tuple[str, ...] = (
    "snip_transform_id",
    "physical_embryo_id",
    "time_index",
    "source_image_id",
    "mask_id",
    "source_shape_h",
    "source_shape_w",
    "source_um_per_px_y",
    "source_um_per_px_x",
    "target_um_per_px_y",
    "target_um_per_px_x",
    "snip_shape_h",
    "snip_shape_w",
    "orientation_policy",
    "orientation_source",
    "flip_x",
    "rotation_angle_rad",
    "centering",
    "schema_version",
    "resolved_transform_chain_json",
    "transform_derivation_inputs_json",
)


class SnipTransformTableError(ValueError):
    """A transform table could not be built, validated, read, or replayed."""


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays and tuples into JSON-native forms, recursively."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _step_to_dict(step: GridTransform) -> dict[str, Any]:
    """One raster step, fully described. ``kind`` and ``interp`` are load-bearing for replay."""
    return {
        "kind": step.kind,
        "name": step.name,
        "affine_2x3": np.asarray(step.affine_2x3, dtype=np.float64).tolist(),
        "in_shape_yx": [int(step.in_shape_yx[0]), int(step.in_shape_yx[1])],
        "out_shape_yx": [int(step.out_shape_yx[0]), int(step.out_shape_yx[1])],
        "interp": step.interp,
        "params": _jsonable(step.params),
    }


def _step_from_dict(raw: dict[str, Any]) -> GridTransform:
    return GridTransform(
        kind=raw["kind"],
        name=raw["name"],
        affine_2x3=np.asarray(raw["affine_2x3"], dtype=np.float64),
        in_shape_yx=(int(raw["in_shape_yx"][0]), int(raw["in_shape_yx"][1])),
        out_shape_yx=(int(raw["out_shape_yx"][0]), int(raw["out_shape_yx"][1])),
        interp=raw["interp"],
        params=dict(raw.get("params") or {}),
    )


def build_snip_transform_row(
    *,
    snip_transform_id: str,
    physical_embryo_id: str,
    time_index: int,
    resolved: Any,
    source_image_id: str,
    mask_id: str,
    orientation_policy: str,
    orientation_source: str,
    no_yolk_policy: str,
    flip_x: bool = False,
) -> dict[str, Any]:
    """Assemble one transform row.

    ``resolved`` is a ``ResolvedSnipTransform``; typed as Any to keep this module free of a circular
    import back into ``snip_transform``.
    """
    canonical = resolved.canonical
    chain = resolved.to_chain()

    # REPLAY payload — the ordered steps that produced these pixels, plus the border policy and the
    # realized (not requested) scale every coordinate mapping must use.
    resolved_chain = {
        "schema_version": SNIP_TRANSFORM_SCHEMA_VERSION,
        "product_shape_hw": list(resolved.product_shape_hw),
        "product_um_per_px": float(resolved.product_um_per_px),
        "rescaled_shape_hw": list(resolved.rescaled_shape_hw),
        "output_shape_hw": list(resolved.output_shape_hw),
        "requested_scale": float(resolved.product_um_per_px / canonical.target_um_per_px),
        "realized_scale_yx": [
            float(resolved.rescaled_shape_hw[0] / resolved.product_shape_hw[0]),
            float(resolved.rescaled_shape_hw[1] / resolved.product_shape_hw[1]),
        ],
        "border_mode": resolved.border_mode,
        "border_value": float(resolved.border_value),
        "crop_px": {
            "x0": int(resolved.crop_x0_px), "y0": int(resolved.crop_y0_px),
            "x1": int(resolved.crop_x1_px), "y1": int(resolved.crop_y1_px),
        },
        "crop_um": {
            "x0": float(resolved.crop_x0_um), "y0": float(resolved.crop_y0_um),
            "x1": float(resolved.crop_x1_um), "y1": float(resolved.crop_y1_um),
        },
        "centering": resolved.centering,
        "steps": [_step_to_dict(s) for s in chain.transforms],
        # Recorded for mapping POINTS only. It is NOT sufficient to re-render and must never be used
        # as if it were — the name says so at every read site.
        "composite_affine_2x3_points_only": chain.composite_affine().tolist(),
    }

    # RE-DERIVE payload — what a fresh derivation needs when it REPLACES the orientation decision.
    derivation_inputs = {
        "schema_version": SNIP_TRANSFORM_SCHEMA_VERSION,
        "source_image_id": source_image_id,
        "mask_id": mask_id,
        "source_shape_hw": list(canonical.source_shape_hw),
        "source_um_per_px": float(canonical.source_um_per_px),
        "target_um_per_px": float(canonical.target_um_per_px),
        "snip_frame_shape_hw": list(canonical.snip_frame_shape_hw),
        "crop_center_um_xy": [float(v) for v in canonical.crop_center_um_xy],
        # The orientation block is the part a refinement REPLACES. It is nested rather than flattened
        # so a future yolk-aware source can add its evidence (margins, yolk_mask_id, confidence)
        # without reshaping the row.
        "orientation": {
            "policy": orientation_policy,
            "source": orientation_source,
            "no_yolk_policy": no_yolk_policy,
            "rotation_angle_rad": float(canonical.rotation_angle_rad),
            "flip_x": bool(flip_x),
        },
    }

    return {
        "snip_transform_id": str(snip_transform_id),
        "physical_embryo_id": str(physical_embryo_id),
        "time_index": int(time_index),
        "source_image_id": str(source_image_id),
        "mask_id": str(mask_id),
        "source_shape_h": int(canonical.source_shape_hw[0]),
        "source_shape_w": int(canonical.source_shape_hw[1]),
        # Per-axis calibration. Today the seam resolves one scalar per grid, so y and x carry the
        # same value; the COLUMNS are per-axis so an anisotropic product needs no schema change.
        "source_um_per_px_y": float(canonical.source_um_per_px),
        "source_um_per_px_x": float(canonical.source_um_per_px),
        "target_um_per_px_y": float(canonical.target_um_per_px),
        "target_um_per_px_x": float(canonical.target_um_per_px),
        "snip_shape_h": int(canonical.snip_frame_shape_hw[0]),
        "snip_shape_w": int(canonical.snip_frame_shape_hw[1]),
        "orientation_policy": str(orientation_policy),
        "orientation_source": str(orientation_source),
        "flip_x": bool(flip_x),
        "rotation_angle_rad": float(canonical.rotation_angle_rad),
        "centering": str(resolved.centering),
        "schema_version": int(SNIP_TRANSFORM_SCHEMA_VERSION),
        "resolved_transform_chain_json": json.dumps(resolved_chain, sort_keys=False),
        "transform_derivation_inputs_json": json.dumps(derivation_inputs, sort_keys=False),
    }


def validate_snip_transform_table(
    df: pd.DataFrame, *, scope_label: str = "snip_transform_table"
) -> None:
    """Presence, uniqueness, and version coherence. Every ROW is validated, not just the file."""
    missing = [c for c in SNIP_TRANSFORM_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise SnipTransformTableError(f"{scope_label}: missing required columns: {missing}")

    if df.empty:
        return

    if df["snip_transform_id"].duplicated().any():
        dupes = df.loc[df["snip_transform_id"].duplicated(keep=False), "snip_transform_id"]
        raise SnipTransformTableError(
            f"{scope_label}: snip_transform_id must be unique — one transform per embryo-time. "
            f"Examples: {sorted(set(dupes))[:5]}. Duplicate ids mean two rows claim the same "
            "geometry, which is exactly the sibling-drift this table exists to make impossible."
        )

    bad_version = sorted(set(df.loc[df["schema_version"] != SNIP_TRANSFORM_SCHEMA_VERSION, "schema_version"]))
    if bad_version:
        raise SnipTransformTableError(
            f"{scope_label}: unsupported schema_version(s) {bad_version}; this reader supports "
            f"{SNIP_TRANSFORM_SCHEMA_VERSION}. Refusing to guess at a record written by a different "
            "contract."
        )


def write_snip_transform_table(df: pd.DataFrame, path: Path) -> Path:
    """Validate, then write ATOMICALLY (temp file -> os.replace).

    A failed or interrupted rerun must never leave a half-written table at the final path: downstream
    consumers treat existence as done, so a visible partial write is silent corruption. Validation
    happens BEFORE the rename, so an invalid table never becomes visible at all.

    Per-well rewrite is the natural granularity — snip processing already runs per well — so there is
    no per-row transactional machinery here for a job shape that does not exist.
    """
    path = Path(path)
    ordered = df.reindex(columns=list(SNIP_TRANSFORM_TABLE_COLUMNS))
    validate_snip_transform_table(ordered, scope_label=f"snip_transform_table({path.name})")

    if path.suffix == ".parquet":
        atomic_write_via(path, lambda tmp: ordered.to_parquet(tmp, index=False))
    else:
        atomic_write_via(path, lambda tmp: ordered.to_csv(tmp, index=False))
    return path


def read_snip_transform_table(path: Path) -> pd.DataFrame:
    """Read and validate a transform table."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    validate_snip_transform_table(df, scope_label=f"snip_transform_table({path.name})")
    return df


def chain_from_row(row: Any) -> TransformChain:
    """Rebuild the executable chain from a table row — the REPLAY path.

    The rebuilt chain renders through the same ``image_geometry`` engine the original run used, so
    replay is byte-identical rather than merely close.
    """
    try:
        payload = json.loads(row["resolved_transform_chain_json"])
        steps = payload["steps"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SnipTransformTableError(
            f"chain_from_row: row has no usable resolved_transform_chain_json ({exc}). It cannot "
            "reproduce a snip."
        ) from exc
    return TransformChain([_step_from_dict(s) for s in steps])


def derivation_inputs_from_row(row: Any) -> dict[str, Any]:
    """The RE-DERIVE payload as a dict — source, grid, calibrations, and orientation decision."""
    try:
        return json.loads(row["transform_derivation_inputs_json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnipTransformTableError(
            f"derivation_inputs_from_row: row has no usable transform_derivation_inputs_json "
            f"({exc}). A different transform cannot be derived from it."
        ) from exc
