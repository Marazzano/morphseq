"""Frame inventory contract — the shared handoff seam between microscope-aware and agnostic stages.

The frame inventory is the first microscope-agnostic artifact: one per-well CSV table that
downstream stages (segmentation onward) consume.  This module owns:
  - the ATOM column list (what producers author),
  - the DERIVED id list (composed by the build step, checked-if-supplied by the validator),
  - ``derive_well_id`` / ``derive_image_id`` helpers (thin wrappers around the shared constructors),
  - the ``assert_derived_ids_consistent`` guard (recomputes from atoms, fails loud on disagreement),
  - small frozen dataclasses: ``StitchedHandoffSpec``, ``FrameInventorySpec``, ``WellHandoff``.

Key design rules (see ``specs/front_end/frame_inventory_handoff_contract.md``):
  - Unique key = RAW ATOMS:
    ``(experiment_id, well_index, channel_id, time_index, z_index)``.
    ``z_index`` is nullable: NA for projection rows, real integer for z-stack planes.
  - ``well_id`` and ``image_id`` are DERIVED compositions of those atoms, never authored by producers.
  - The validator recomputes derived ids from atoms and fails loud on any disagreement.
  - This module is microscope-agnostic: it does not import YX1 / Keyence logic.

Import direction: this module MAY import ``shared/identifiers/``.  It MUST NOT import
stages, Snakemake rules, tasks, stitch backends, or scope-specific metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.image_materialization.image_product_keys import image_product_key_for_frame_row
from data_pipeline.acquisition.image_materialization.materialized_image_write_policy import (
    MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
    MATERIALIZED_IMAGE_WRITE_POLICY_NULLABLE_COLUMNS,
)
from data_pipeline.shared.identifiers.constructors import build_image_id, build_well_id
from data_pipeline.shared.identifiers.validators import validate_well_id

# ---------------------------------------------------------------------------
# Column manifests
# ---------------------------------------------------------------------------

# The atoms + required context columns — what every producer MUST supply.
# ``well_id`` and ``image_id`` are intentionally absent: the build step derives them.
REQUIRED_FRAME_INVENTORY_COLUMNS: tuple[str, ...] = (
    "experiment_id",               # global experiment id — atom
    "well_index",                  # local well label (B01) — atom
    "channel_id",                  # controlled channel token (BF / GFP …) — atom
    "time_index",                  # T dimension, 0-based contiguous — atom
    "z_index",                     # nullable Z dimension atom: NA projection, integer z_stack
    "image_product_type",          # product family: "projection" | "z_stack" — part of frame identity
    "projection_method",           # method WITHIN projection ("focus_stack" …); NA for z_stack
    "elapsed_time_s",              # seconds since this well's first frame — CARRIED from acquisition
    "acquisition_time_s",          # raw per-frame timestamp — CARRIED from acquisition (audit)
    "image_path",                  # TIFF / PNG / JPEG; absolute OR relative to image_root
    "image_micrometers_per_pixel",  # calibration µm/px of the materialized image, required > 0
    "image_width_px",              # on-disk/post-downsample width (self-check against image header)
    "image_height_px",             # on-disk/post-downsample height (self-check against image header)
    *MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,  # writer-owned bytes-on-disk policy fields
)

# The carried-through time block — OWNED by the acquisition inventory (derived there from the
# scope-specific raw atom), CARRIED unchanged by the materializer. Downstream reads ``elapsed_time_s``.
# See specs/acquisition_inventory_schema_policy.md. NOT part of the per-frame unique key (time is not
# identity).
FRAME_INVENTORY_TIME_BLOCK: tuple[str, ...] = (
    "elapsed_time_s",
    "acquisition_time_s",
)

# Composed from atoms by the build step; checked-if-supplied, never trusted blindly.
DERIVED_FRAME_INVENTORY_COLUMNS: tuple[str, ...] = (
    "well_id",   # {experiment_id}_{well_index}
    "image_id",  # {well_id}_{channel_id}_t{time_index:04d}
)

# CONSTRUCTION-PROVENANCE columns: nullable, contract-DECLARED paths that explain HOW the image in
# the same row was produced. They are 1:1 with image_id, co-produced by the same materialization job,
# and L4-validated alongside the image. A provenance path is NOT an image (it never appears in
# image_path), NOT a product, and NOT a second image identity. Declared here so the assembler
# treats them as known nullable columns rather than schema drift.
#
#   focus_index_map_path — the focus_stack focus_index_map .npz (focus_index_map + z_indices arrays):
#     populated for projection/focus_stack rows; NA for z_stack and non-focus_stack projection rows.
CONSTRUCTION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "focus_index_map_path",
)

# Required-to-exist columns whose values may be null by product/format. ``jpeg_quality`` is present
# for every row but only populated for lossy JPEG-encoded images.
FRAME_INVENTORY_NULLABLE_COLUMNS: tuple[str, ...] = (
    "z_index",
    "projection_method",
    *MATERIALIZED_IMAGE_WRITE_POLICY_NULLABLE_COLUMNS,
)

# The per-frame unique key, stated as ATOMS. This names WHICH columns identify a frame; the
# validator does NOT trust this tuple as opaque strings — it routes the atoms through the
# constructors so the effective key is a DERIVED pair: (image_id, product_key). Two frames are the
# SAME frame only if BOTH match.
#
# Why product_key is part of identity (NOT just image_id): `image_id` addresses a frame within ONE
# product family ({well}_{channel}_t{t}[_z{z}]). But "projection" is a family, not a single product
# — `BF__projection__focus_stack` and `BF__projection__max_intensity` are DIFFERENT frames that share
# the same image_id. Anchoring uniqueness to image_id ALONE would falsely collide them. So the
# effective key is (image_id, product_key); see ``frame_inventory_product_aware_keys``.
#
# Enforcement uses that derived pair, NOT df.duplicated(UNIQUE_FRAME_INVENTORY_KEY_COLUMNS) — the
# atom tuple is the honest column list, but z_index is nullable and Pandas drops/compares NA keys
# inconsistently (the NA landmine). Route through the helper; never simplify back to the raw tuple.
UNIQUE_FRAME_INVENTORY_KEY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_index",
    "channel_id",
    "time_index",
    "z_index",
    "image_product_type",
    "projection_method",
)

ALLOWED_IMAGE_SUFFIXES: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# The channel that defines the segmentation timeline — must be present and contiguous.
REQUIRED_CHANNEL: str = "BF"

# ---------------------------------------------------------------------------
# Downstream frame-identity block — the shared header every post-inventory
# frame-level model product (frame_detections, frame_masks, …) carries through.
# ---------------------------------------------------------------------------

# These columns are the DERIVED, consumer-facing view of a validated frame: the
# atom-composed ids (well_id / image_id) plus the per-frame context that downstream
# products copy through unchanged. This is the single source of truth for that header;
# downstream contracts (e.g. detection/frame_detections_contract.py) IMPORT this rather
# than re-declaring the column list. See specs/detect-seg-track/targets/detection_world.md
# "Required Frame Identity Block".
#
# ``z_index`` is intentionally INCLUDED here and is nullable: projection rows carry NA,
# z_stack rows carry the materialized plane index. ``validate_frame_identity_block``
# synthesizes it as NA when older downstream products did not carry it.
DOWNSTREAM_FRAME_IDENTITY_BLOCK: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "image_id",
    "time_index",
    "z_index",
    "channel_id",
    "image_path",
    "image_width_px",
    "image_height_px",
)

# Identity columns that may be null (present-but-NA) on the MVP.
FRAME_IDENTITY_NULLABLE: frozenset[str] = frozenset({"z_index"})

# The identity columns whose per-frame values a downstream product must carry UNCHANGED
# from the matching frame_inventory row (checked by validate_frame_identity_block).
_FRAME_IDENTITY_CARRIED_COLUMNS: tuple[str, ...] = (
    "time_index",
    "channel_id",
    "image_path",
    "image_width_px",
    "image_height_px",
)

# ---------------------------------------------------------------------------
# Derived-id helpers
# ---------------------------------------------------------------------------


def derive_well_id(experiment_id: str, well_index: str) -> str:
    """Compose the global well id from its two atoms."""
    return build_well_id(experiment_id, well_index)


def derive_image_id(
    well_id: str,
    channel_id: str,
    time_index: int,
    *,
    z_index: int | None = None,
) -> str:
    """Compose the image id from the per-frame key atoms (well_id already derived)."""
    return build_image_id(well_id, channel_id, time_index, z_index=z_index)


def _row_z_index_or_none(row: pd.Series) -> int | None:
    if "z_index" not in row.index or pd.isna(row["z_index"]):
        return None
    return int(row["z_index"])


# ---------------------------------------------------------------------------
# Product-column + derived-id consistency guards
# ---------------------------------------------------------------------------


def assert_product_columns_consistent(df: pd.DataFrame, scope_label: str = "frame_inventory") -> None:
    """Fail loud if the product atoms (``image_product_type`` / ``projection_method``) disagree.

    The two columns are atoms but they constrain each other — composing ``product_key`` from them via
    ``build_image_product_key`` (in ``frame_inventory_product_keys``) is what would surface a
    contradiction, so this guard makes that explicit and names the fix per row:

      - ``projection`` rows REQUIRE a non-null ``projection_method`` and a null ``z_index``;
      - ``z_stack``   rows REQUIRE a null ``projection_method`` and a non-null ``z_index``.

    Without this guard a row like ``image_product_type=z_stack, projection_method=focus_stack`` would
    be rejected only deep inside ``build_image_product_key`` with a less locatable message.
    """
    for col in ("image_product_type", "projection_method"):
        if col not in df.columns:
            raise ValueError(
                f"[{scope_label}] required product column {col!r} is absent. Every frame_inventory "
                "row carries image_product_type and projection_method (atoms of the product_key)."
            )

    ptype = df["image_product_type"].astype(str)
    method_null = df["projection_method"].isna()
    z_null = df["z_index"].isna() if "z_index" in df.columns else pd.Series(True, index=df.index)

    is_projection = ptype == "projection"
    bad_proj = is_projection & (method_null | ~z_null)
    if bad_proj.any():
        sample = df.loc[bad_proj, ["image_product_type", "projection_method", "z_index"]].head(3)
        raise ValueError(
            f"[{scope_label}] {int(bad_proj.sum())} projection row(s) are inconsistent: a projection "
            "frame REQUIRES a non-null projection_method and a null z_index. "
            f"First offenders: {sample.to_dict(orient='records')}"
        )

    is_zstack = ptype == "z_stack"
    bad_z = is_zstack & (~method_null | z_null)
    if bad_z.any():
        sample = df.loc[bad_z, ["image_product_type", "projection_method", "z_index"]].head(3)
        raise ValueError(
            f"[{scope_label}] {int(bad_z.sum())} z_stack row(s) are inconsistent: a z_stack frame "
            "REQUIRES a null projection_method and a non-null z_index. "
            f"First offenders: {sample.to_dict(orient='records')}"
        )


def assert_derived_ids_consistent(df: pd.DataFrame, scope_label: str = "frame_inventory") -> None:
    """Fail loud if any supplied ``well_id`` or ``image_id`` disagrees with the atom-recomputed value.

    The validator calls this after the required-column check.  If a producer omitted a derived column,
    this is a no-op for that column (the build step will add it; the validator adds it if missing).
    """
    if "well_id" in df.columns:
        expected_well_id = df.apply(
            lambda r: derive_well_id(r["experiment_id"], r["well_index"]), axis=1
        )
        bad = df["well_id"] != expected_well_id
        if bad.any():
            n = bad.sum()
            sample = df.loc[bad, ["experiment_id", "well_index", "well_id"]].head(3).to_dict(
                orient="records"
            )
            raise ValueError(
                f"[{scope_label}] {n} row(s) have well_id inconsistent with atoms "
                f"(experiment_id + well_index). First offenders: {sample}"
            )

    if "image_id" in df.columns:
        well_id_col = (
            df["well_id"]
            if "well_id" in df.columns
            else df.apply(lambda r: derive_well_id(r["experiment_id"], r["well_index"]), axis=1)
        )
        expected_image_id = df.apply(
            lambda r: derive_image_id(
                well_id_col[r.name],
                r["channel_id"],
                r["time_index"],
                z_index=_row_z_index_or_none(r),
            ),
            axis=1,
        )
        bad = df["image_id"] != expected_image_id
        if bad.any():
            n = bad.sum()
            sample = df.loc[bad, ["well_id", "channel_id", "time_index", "image_id"]].head(3).to_dict(
                orient="records"
            )
            raise ValueError(
                f"[{scope_label}] {n} row(s) have image_id inconsistent with atoms "
                f"(well_id + channel_id + time_index + z_index). First offenders: {sample}"
            )


def frame_inventory_image_ids(df: pd.DataFrame, scope_label: str = "frame_inventory") -> pd.Series:
    """Recompute the DERIVED ``image_id`` for every row by routing the atoms through the grammar.

    ``image_id`` addresses a frame WITHIN one product family
    (``{well}_{channel}_t{t}`` for projection, ``{well}_{channel}_z{z}_t{t}`` for z_stack). It is ONE
    of the two axes of frame identity — the other is ``product_key`` (see
    ``frame_inventory_product_aware_keys``), because "projection" is a family and two projection
    methods share an image_id. This helper composes
    ``build_image_id(build_well_id(experiment_id, well_index), channel_id, time_index, z_index=...)``
    for each row and validates the intermediate ``well_id`` with ``validate_well_id`` (so a leaked
    bare local label or un-promoted id fails loud HERE). Anchoring to the constructors guarantees the
    key can never drift from the canonical id grammar.

    Raises:
        ValueError: if any row's composed ``well_id`` is not a valid global well_id.
    """
    def _image_id_for_row(row: pd.Series) -> str:
        well_id = build_well_id(row["experiment_id"], row["well_index"])
        validate_well_id(well_id)  # fail loud on a leaked local label / un-promoted id
        return build_image_id(
            well_id,
            row["channel_id"],
            row["time_index"],
            z_index=_row_z_index_or_none(row),
        )

    try:
        return df.apply(_image_id_for_row, axis=1)
    except ValueError as exc:
        raise ValueError(f"[{scope_label}] {exc}") from exc


def frame_inventory_product_keys(df: pd.DataFrame, scope_label: str = "frame_inventory") -> pd.Series:
    """Compose the ``product_key`` for every row from its product columns.

    The product family axis of frame identity: ``channel_id + image_product_type + [projection_method]``
    (e.g. ``BF__projection__focus_stack``, ``BF__z_stack``). Routes through the same
    ``build_image_product_key`` grammar the executor and resolver use, so the manifest's product_key
    can never drift from the canonical product-key vocabulary.
    """
    def _product_key_for_row(row: pd.Series) -> str:
        return image_product_key_for_frame_row(
            channel_id=row["channel_id"],
            image_product_type=row["image_product_type"],
            projection_method=row["projection_method"],
        )

    try:
        return df.apply(_product_key_for_row, axis=1)
    except ValueError as exc:
        raise ValueError(f"[{scope_label}] {exc}") from exc


def frame_inventory_product_aware_keys(
    df: pd.DataFrame, scope_label: str = "frame_inventory"
) -> pd.Series:
    """The effective unique key: the per-row pair ``(image_id, product_key)`` as one string.

    Two rows are the SAME frame only if BOTH their image_id AND their product_key match. This is what
    duplicate detection runs on — it prevents the false collision between two projection products that
    share an image_id (``BF__projection__focus_stack`` vs ``BF__projection__max_intensity`` at the
    same well/channel/time), while still catching a true duplicate (same product, same plane). Both
    components are derived strings (never the raw nullable z_index tuple), so the key collides
    correctly by construction — no Pandas NA-key landmine.
    """
    image_ids = frame_inventory_image_ids(df, scope_label=scope_label)
    product_keys = frame_inventory_product_keys(df, scope_label=scope_label)
    return image_ids.astype(str) + " @ " + product_keys.astype(str)


# ---------------------------------------------------------------------------
# THE identity-contract gate — one named entrypoint, composed of the checks above
# ---------------------------------------------------------------------------


def validate_frame_inventory_identity_contract(
    df: pd.DataFrame, scope_label: str = "frame_inventory"
) -> None:
    """The single gate that answers: "are these rows valid, unique frame identities?"

    This is THE foundational identity check. Every path that admits frame_inventory rows into the
    trusted set — the strict validator, the per-well assembler, the experiment merge — calls THIS,
    not the individual helpers. Composing the checks behind one named gate is what keeps frame
    identity enforced *once*, consistently, instead of as ad-hoc tripwires scattered per caller.

    It enforces, in order:
      1. the product atoms are internally coherent (projection⇔method present + z null;
         z_stack⇔method null + z non-null) — ``assert_product_columns_consistent``;
      2. the effective frame key — the derived ``(image_id, product_key)`` pair — is UNIQUE
         (``frame_inventory_product_aware_keys``), so two projection products that share an image_id
         don't falsely collide, and a true duplicate (same product, same plane) is caught;
      3. any producer-supplied ``well_id`` / ``image_id`` agree with the atom-recomputed values
         (``assert_derived_ids_consistent``).

    It does NOT check table shape (required columns, suffixes — that is the schema layer) or
    well/temporal grain (``validate_well_temporal_grain``, which runs AFTER this and assumes identity
    already passed). Identity first; shape and temporal rules compose on top.

    Raises:
        ValueError: fix-named, on the first failing layer.
    """
    assert_product_columns_consistent(df, scope_label=scope_label)

    # Optional first-class physical acquisition context. Import locally to avoid a
    # module cycle: frame_modality composes the identity helpers owned above.
    from data_pipeline.acquisition.image_materialization.frame_modality import (
        validate_frame_modality_block,
    )

    validate_frame_modality_block(df, scope_label=scope_label)

    frame_keys = frame_inventory_product_aware_keys(df, scope_label=scope_label)
    duplicate_mask = frame_keys.duplicated(keep=False)
    if duplicate_mask.any():
        key_cols = [c for c in UNIQUE_FRAME_INVENTORY_KEY_COLUMNS if c in df.columns]
        duplicates = df.loc[duplicate_mask, key_cols]
        raise ValueError(
            f"[{scope_label}] duplicate frame identities detected (by derived image_id + product_key): "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )

    assert_derived_ids_consistent(df, scope_label=scope_label)


# ---------------------------------------------------------------------------
# Downstream frame-identity validator — reusable across frame-level products
# ---------------------------------------------------------------------------


def validate_frame_identity_block(
    df: pd.DataFrame,
    reference_frame_inventory: pd.DataFrame,
    *,
    context: str = "frame_identity",
) -> None:
    """Validate the shared frame-identity header of a downstream frame-level product.

    Reusable for ``frame_detections``, ``frame_masks``, and later frame-level products: it asserts
    the product's per-row identity is consistent with the trusted ``reference_frame_inventory`` it
    derives from. ``reference_frame_inventory`` is the validated per-well inventory table (atoms;
    derived ids optional) passed in as a READ-ONLY DataFrame — this validator does NOT read files.

    Two-layer doctrine (see ``specs/detect-seg-track/targets/detection_world.md`` "Frame Identity
    Validator"):
      - *schema layer* ("right bones"): all ``DOWNSTREAM_FRAME_IDENTITY_BLOCK`` columns present
        (``z_index`` synthesized NA if absent, then treated as nullable); non-nullable identity columns
        are non-null; rows belong to exactly one ``experiment_id`` and one ``well_id``;
      - *reference layer* ("bones belong to the right body"): every ``image_id`` exists in
        ``reference_frame_inventory`` (membership via ``frame_inventory_image_ids``); ``time_index`` /
        ``channel_id`` / ``image_path`` / ``image_width_px`` / ``image_height_px`` agree with the
        matching inventory row.

    Raises:
        ValueError: prefixed with ``context`` and an offending-row sample, on any failure.
    """
    df = df.copy()
    # z_index is synthesized as NA when the product did not carry it (BF / projection MVP).
    if "z_index" not in df.columns:
        df["z_index"] = pd.NA

    missing = [c for c in DOWNSTREAM_FRAME_IDENTITY_BLOCK if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] missing required frame identity columns: {sorted(missing)}")

    # Required identity columns must be non-null except the explicitly-nullable ones.
    for col in DOWNSTREAM_FRAME_IDENTITY_BLOCK:
        if col in FRAME_IDENTITY_NULLABLE:
            continue
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(
                f"[{context}] identity column '{col}' has {n} null value(s); only "
                f"{sorted(FRAME_IDENTITY_NULLABLE)} may be null"
            )

    # One experiment / one well per product table.
    for col in ("experiment_id", "well_id"):
        vals = sorted(df[col].dropna().unique().tolist())
        if len(vals) > 1:
            raise ValueError(
                f"[{context}] rows span multiple {col} values (expected one): {vals[:5]}"
            )

    # image_id membership: every product image_id must be a known inventory frame.
    inventory_image_ids = set(frame_inventory_image_ids(reference_frame_inventory, scope_label=context))
    unknown = sorted(set(df["image_id"].astype(str)) - inventory_image_ids)
    if unknown:
        raise ValueError(
            f"[{context}] {len(unknown)} image_id(s) not present in reference_frame_inventory. "
            f"First offenders: {unknown[:3]}"
        )

    # Per-frame carried-column agreement against the matching inventory row.
    inv = reference_frame_inventory.copy()
    inv["image_id"] = frame_inventory_image_ids(inv, scope_label=context).astype(str)
    inv_by_image = inv.drop_duplicates(subset=["image_id"]).set_index("image_id")
    for col in _FRAME_IDENTITY_CARRIED_COLUMNS:
        if col not in inv_by_image.columns:
            # The inventory does not carry this column to compare against; skip silently.
            continue
        expected = df["image_id"].astype(str).map(inv_by_image[col])
        # Compare as strings to be dtype-robust (CSV round-trips coerce numeric → object).
        bad = df[col].astype(str) != expected.astype(str)
        if bad.any():
            n = int(bad.sum())
            sample = df.loc[bad, ["image_id", col]].head(3).to_dict(orient="records")
            raise ValueError(
                f"[{context}] {n} row(s) have '{col}' disagreeing with reference_frame_inventory. "
                f"First offenders: {sample}"
            )


# ---------------------------------------------------------------------------
# Small frozen dataclasses — clipboards, not mayors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchedHandoffSpec:
    """Dataset-level ingress descriptor for an external drop-in submission."""

    experiment_id: str
    manifest_path: Path  # the dataset-level dropin_frame_inventory.csv (all wells)
    image_root: Path | None = None  # resolves relative image_path; None → must be absolute


@dataclass(frozen=True)
class FrameInventorySpec:
    """What a valid frame inventory requires — the shared gate's contract."""

    required_columns: tuple[str, ...] = REQUIRED_FRAME_INVENTORY_COLUMNS
    allowed_image_suffixes: tuple[str, ...] = ALLOWED_IMAGE_SUFFIXES
    required_channel: str = REQUIRED_CHANNEL


@dataclass(frozen=True)
class WellHandoff:
    """One well's operational unit: input + output paths for the build → validate → consume flow."""

    experiment_id: str
    well_id: str                           # global, derived — never minted here
    image_root: Path | None
    candidate_manifest_path: Path          # {well_id}_frame_inventory.csv (pre-sentinel)
    validated_frame_inventory_path: Path   # same file; sentinel marks it trusted
    report_path: Path                      # {well_id}_frame_inventory.errors.md (FAIL only)
    validated_sentinel_path: Path          # {well_id}_frame_inventory.csv.validated (PASS only)
