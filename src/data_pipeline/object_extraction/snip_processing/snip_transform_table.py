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
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.model_servers.atomic_write import atomic_write_via
from data_pipeline.object_extraction.snip_processing.snip_transform import (
    CanonicalSnipTransform,
    SnipGridSpec,
)
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
    # GEOMETRY source, named explicitly. Once an RFP renderer reads this row, a bare
    # "source_image_id" is ambiguous: the RFP frame is the RENDER source, the BF frame is the
    # GEOMETRY source, and they are different images of the same well and timepoint.
    "geometry_source_image_id",
    "geometry_source_mask_id",
    # THE MASK-HASH FREEZE. The gate fixes placement; this fixes the raster the placement was
    # derived FROM. Without it, a frame_masks regeneration between snip_geometry and a render job
    # pairs old geometry with a newer mask -- pixels placed by one segmentation, masked by another,
    # with no error. Hashed over the DECODED mask, never the RLE string: a re-encode changes the
    # string without changing a pixel, which would produce false failures.
    "geometry_source_mask_sha256",
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
    # THE EMBRYO MASK, named by the row that defines it. The mask is the BF segmentation carried
    # through this row's canonical transform -- a property of the embryo-time, not of any product --
    # so snip_geometry writes it once and every consumer reads the path from here rather than
    # rebuilding a filename from a convention it would then be coupled to.
    "embryo_mask_snip_path",
    # RESERVED TRANSITIONAL METADATA. The column exists for schema continuity, but the compiler
    # does not implement a reflection: no code path reads it, and nothing sets it True. A field that
    # LOOKS executable and is ignored is a trapdoor -- an artifact could claim a flip the pixels
    # never got -- so validation rejects True rather than letting the claim stand. Either make it
    # behavior-bearing (a reflection composes into the same destination affine, so it need not cost
    # another resampling) or drop it in a schema migration; do not leave it merely reserved.
    "flip_x",
    "rotation_angle_rad",
    # THE CANONICAL RECIPE AS TYPED FLOAT64 COLUMNS, NOT JSON. `from_row` must reconstruct a
    # CanonicalSnipTransform that compares EXACTLY equal to the original, and JSON does not
    # round-trip float64 exactly unless written at repr-precision -- so an "exact" assert over a
    # JSON payload would be quietly false. Typed columns also make the recipe queryable without
    # parsing. JSON stays for the resolved chain and derivation evidence, neither compared exactly.
    "crop_center_um_x",
    "crop_center_um_y",
    # TRANSITIONAL, nullable, and named at length ON PURPOSE. This is the integer-latched center
    # measured on the legacy EXPANDED, ROTATED canvas AFTER resizing to the target calibration -- it
    # is product-invariant (that grid derives from physical FOV + target calibration) but it is NOT a
    # physical source coordinate. It MUST NOT be converted into or interpreted as crop_center_um_xy:
    # inverting it back through the rescale disagrees by hundreds of micrometers for a rotated
    # embryo, because rotation changed the canvas axes and origin. A short name invites exactly that
    # 'deduplication'. Deleted with CENTERING_LATCHED; see TODO(remove-legacy-latched-centering).
    "legacy_center_on_target_rescaled_rotated_grid_x",
    "legacy_center_on_target_rescaled_rotated_grid_y",
    "centering",
    "schema_version",
    # NO resolved_transform_chain_json HERE, deliberately. This row is channel-independent, and
    # there is no single resolved chain for it: BF native, RFP native, and a 4x z-stack compile the
    # same recipe into DIFFERENT chains -- different source shape, calibration, resize dimensions,
    # realized scale, and affine translation. Storing one on the shared row would make whichever
    # product happened to write it authoritative for every other. The chain belongs on the PRODUCT
    # inventory row, where it describes the raster that product actually rendered.
    "transform_derivation_inputs_json",
)


class SnipTransformTableError(ValueError):
    """A transform table could not be built, validated, read, or replayed."""


def mask_content_fingerprint(mask: np.ndarray) -> str:
    """Hash a DECODED mask, including shape and dtype.

    THE ONE DEFINITION, shared by the writer (snip_geometry) and the verifier (the renderer). Two
    copies of this arithmetic is precisely how a freeze becomes decorative: I wrote exactly that
    bug -- the writer hashed bare tobytes() while the verifier hashed shape+dtype+bytes, so every
    mask "mismatched" on an unchanged well.

    Pixels, not encoding: two RLE strings can serialize one binary mask, so hashing the string would
    fail on a no-op re-encode while catching nothing real. Shape and dtype join the payload because
    identical byte sequences under different shapes would otherwise be ambiguous.
    """
    import hashlib

    payload = f"{mask.shape}|{mask.dtype}|".encode() + np.ascontiguousarray(mask).tobytes()
    return hashlib.sha256(payload).hexdigest()


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


def build_resolved_chain_payload(resolved: Any) -> str:
    """The REPLAY payload for ONE product's rendering, as JSON.

    Belongs on the PRODUCT inventory row, never on the shared transform row. A canonical transform
    is channel-independent and has no single resolved chain: BF native, RFP native, and a
    4x-downsampled z_stack compile the same physical recipe into different chains -- different
    source shape and calibration, different resize dimensions, different realized scale, different
    affine translation. Whichever product wrote it to the shared row would silently become
    authoritative for all of them.

    The full ordered chain is serialized rather than one composite matrix: a matrix is COORDINATE
    truth only and cannot express the anti-aliased resize prefilter, requested-vs-realized scale,
    crop/pad semantics, or the image-vs-mask interpolation split -- all of which change pixels.
    """
    canonical = resolved.canonical
    chain = resolved.to_chain()
    payload = {
        "schema_version": SNIP_TRANSFORM_SCHEMA_VERSION,
        "product_shape_hw": list(resolved.product_shape_hw),
        "product_um_per_px": float(resolved.product_um_per_px),
        "rescaled_shape_hw": list(resolved.rescaled_shape_hw),
        "output_shape_hw": list(resolved.output_shape_hw),
        "requested_scale": float(
            resolved.product_um_per_px / canonical.grid.default_output_um_per_px_yx[0]
        ),
        "realized_scale_yx": [
            float(resolved.rescaled_shape_hw[0] / resolved.product_shape_hw[0]),
            float(resolved.rescaled_shape_hw[1] / resolved.product_shape_hw[1]),
        ],
        "border_mode": resolved.border_mode,
        "border_value": float(resolved.border_value),
        "centering": resolved.centering,
        "steps": [_step_to_dict(s) for s in chain.transforms],
    }
    return json.dumps(payload, sort_keys=False)


def build_snip_transform_row(
    *,
    snip_transform_id: str,
    physical_embryo_id: str,
    time_index: int,
    canonical: Any,
    source_image_id: str,
    mask_id: str,
    orientation_policy: str,
    orientation_source: str,
    no_yolk_policy: str,
    geometry_source_mask_sha256: str = "",
    embryo_mask_snip_path: str = "",
    flip_x: bool = False,
) -> dict[str, Any]:
    """Assemble one transform row.

    ``canonical`` is a ``CanonicalSnipTransform``; typed as Any to keep this module free of a
    circular import back into ``snip_transform``.

    TAKES THE CANONICAL DIRECTLY, not a resolved transform. This row is product-independent, so
    requiring a resolution would force the geometry step to pick some product's grid to resolve
    against -- and whichever it picked would quietly become the reference for every other.
    """
    # RE-DERIVE payload — what a fresh derivation needs when it REPLACES the orientation decision.
    derivation_inputs = {
        "schema_version": SNIP_TRANSFORM_SCHEMA_VERSION,
        "source_image_id": source_image_id,
        "mask_id": mask_id,
        "source_shape_hw": list(canonical.grid.geometry_source_shape_yx),
        "source_um_per_px": float(canonical.grid.geometry_source_um_per_px_yx[0]),
        "target_um_per_px": float(canonical.grid.default_output_um_per_px_yx[0]),
        "snip_frame_shape_hw": list(canonical.grid.default_output_shape_yx),
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
        "geometry_source_image_id": str(source_image_id),
        "geometry_source_mask_id": str(mask_id),
        "geometry_source_mask_sha256": str(geometry_source_mask_sha256),
        "source_shape_h": int(canonical.grid.geometry_source_shape_yx[0]),
        "source_shape_w": int(canonical.grid.geometry_source_shape_yx[1]),
        # Per-axis calibration. Today the seam resolves one scalar per grid, so y and x carry the
        # same value; the COLUMNS are per-axis so an anisotropic product needs no schema change.
        "source_um_per_px_y": float(canonical.grid.geometry_source_um_per_px_yx[0]),
        "source_um_per_px_x": float(canonical.grid.geometry_source_um_per_px_yx[1]),
        "embryo_mask_snip_path": str(embryo_mask_snip_path),
        "target_um_per_px_y": float(canonical.grid.default_output_um_per_px_yx[0]),
        "target_um_per_px_x": float(canonical.grid.default_output_um_per_px_yx[1]),
        "snip_shape_h": int(canonical.grid.default_output_shape_yx[0]),
        "snip_shape_w": int(canonical.grid.default_output_shape_yx[1]),
        "orientation_policy": str(orientation_policy),
        "orientation_source": str(orientation_source),
        "flip_x": bool(flip_x),
        "rotation_angle_rad": float(canonical.rotation_angle_rad),
        "crop_center_um_x": float(canonical.crop_center_um_xy[0]),
        "crop_center_um_y": float(canonical.crop_center_um_xy[1]),
        "legacy_center_on_target_rescaled_rotated_grid_x": (
            None
            if canonical.legacy_center_on_target_rescaled_rotated_grid_xy is None
            else float(canonical.legacy_center_on_target_rescaled_rotated_grid_xy[0])
        ),
        "legacy_center_on_target_rescaled_rotated_grid_y": (
            None
            if canonical.legacy_center_on_target_rescaled_rotated_grid_xy is None
            else float(canonical.legacy_center_on_target_rescaled_rotated_grid_xy[1])
        ),
        "centering": str(canonical.centering_mode) if hasattr(canonical, "centering_mode") else "",
        "schema_version": int(SNIP_TRANSFORM_SCHEMA_VERSION),
        "transform_derivation_inputs_json": json.dumps(derivation_inputs, sort_keys=False),
    }


def validate_snip_transform_table(
    df: pd.DataFrame, *, scope_label: str = "snip_transform_table"
) -> None:
    """Presence, uniqueness, and version coherence. Every ROW is validated, not just the file."""
    missing = [c for c in SNIP_TRANSFORM_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise SnipTransformTableError(f"{scope_label}: missing required columns: {missing}")

    # flip_x=True would be a claim the compiler cannot honor. Descriptive error, not a bare assert:
    # artifact validation runs outside tests, where an AssertionError says nothing useful.
    if "flip_x" in df.columns and df["flip_x"].fillna(False).astype(bool).any():
        offenders = df.loc[df["flip_x"].fillna(False).astype(bool), "snip_transform_id"].head(3)
        raise SnipTransformTableError(
            f"{scope_label}: flip_x=True is not supported by the snip transform compiler -- no "
            f"render path applies a reflection, so these rows claim a flip their pixels never "
            f"received. Offending snip_transform_id(s): {list(offenders)}."
        )

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
    """Rebuild the executable chain from a PRODUCT INVENTORY row — the REPLAY path.

    Takes a product row, NOT a canonical transform row. The canonical row is channel-independent
    and has no single chain: BF native, RFP native, and a 4x z_stack compile the same recipe into
    different chains. The chain describes one product's actual raster, so it lives with that
    product.

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


def _optional_float(value: Any) -> float | None:
    """None for a SQL/pandas null, else float. Distinguishes 'absent' from 0.0."""
    if value is None:
        return None
    # pandas reads a missing float64 cell as NaN, not None, and NaN != NaN would silently break the
    # exact round-trip equality this module promises.
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(as_float) else as_float


def canonical_from_row(row: Any) -> CanonicalSnipTransform:
    """Rebuild the CanonicalSnipTransform from a table row — THE GATE'S READ PATH.

    This is what makes the geometry gate work: ``snip_geometry`` derives once and writes; every
    render job calls this and then ``transform_for_product`` for its own grid. No render job derives
    geometry, and none needs the mask.

    EXACT reconstruction, not approximate. The recipe's floats are stored as typed float64 columns
    rather than inside a JSON payload precisely so this round-trips bit-for-bit:

        derive(mask) == canonical_from_row(build_snip_transform_row(...))

    A tolerance-based comparison would let a slow drift in the writer pass unnoticed until two
    siblings rendered a pixel apart.
    """
    try:
        grid = SnipGridSpec(
            geometry_source_shape_yx=(int(row["source_shape_h"]), int(row["source_shape_w"])),
            geometry_source_um_per_px_yx=(
                float(row["source_um_per_px_y"]),
                float(row["source_um_per_px_x"]),
            ),
            default_output_um_per_px_yx=(
                float(row["target_um_per_px_y"]),
                float(row["target_um_per_px_x"]),
            ),
            default_output_shape_yx=(int(row["snip_shape_h"]), int(row["snip_shape_w"])),
        )
        latched_x = _optional_float(row["legacy_center_on_target_rescaled_rotated_grid_x"])
        latched_y = _optional_float(row["legacy_center_on_target_rescaled_rotated_grid_y"])
        return CanonicalSnipTransform(
            grid=grid,
            rotation_angle_rad=float(row["rotation_angle_rad"]),
            crop_center_um_xy=(
                float(row["crop_center_um_x"]),
                float(row["crop_center_um_y"]),
            ),
            legacy_center_on_target_rescaled_rotated_grid_xy=(
                None if latched_x is None or latched_y is None else (latched_x, latched_y)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnipTransformTableError(
            f"canonical_from_row: row does not carry a reconstructable canonical transform ({exc}). "
            "A render job cannot resolve a product grid without it, so this row cannot be used."
        ) from exc
