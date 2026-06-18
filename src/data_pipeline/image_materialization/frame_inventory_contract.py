"""Frame inventory contract — the shared handoff seam between microscope-aware and agnostic stages.

The frame inventory is the first microscope-agnostic artifact: one per-well CSV table that
downstream stages (segmentation onward) consume.  This module owns:
  - the ATOM column list (what producers author),
  - the DERIVED id list (composed by the build step, checked-if-supplied by the validator),
  - ``derive_well_id`` / ``derive_image_id`` helpers (thin wrappers around the shared constructors),
  - the ``assert_derived_ids_consistent`` guard (recomputes from atoms, fails loud on disagreement),
  - small frozen dataclasses: ``StitchedHandoffSpec``, ``FrameInventorySpec``, ``WellHandoff``.

Key design rules (see ``specs/front_end/frame_inventory_handoff_contract.md``):
  - Unique key = the FOUR RAW ATOMS: ``(experiment_id, well_index, channel_id, time_index)``.
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
    "elapsed_time_s",              # seconds since this well's first frame — CARRIED from acquisition
    "acquisition_time_s",          # raw per-frame timestamp — CARRIED from acquisition (audit)
    "source_image_path",           # TIFF / PNG / JPEG; absolute OR relative to image_root
    "source_micrometers_per_pixel",  # calibration µm/px, required > 0
    "image_width_px",              # declared width (self-check against image header)
    "image_height_px",             # declared height (self-check against image header)
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

# The per-frame unique key, stated as ATOMS. This names WHICH columns identify a frame; the
# validator does NOT trust this tuple as opaque strings — it routes the atoms through the
# identifier constructors (build_well_id → build_image_id) so the effective key is the DERIVED
# image_id. Uniqueness on these atoms ≡ uniqueness on image_id, but anchored to the grammar so
# the key can never drift from the constructors. See ``frame_inventory_image_ids``.
UNIQUE_FRAME_INVENTORY_KEY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_index",
    "channel_id",
    "time_index",
)

ALLOWED_IMAGE_SUFFIXES: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# The channel that defines the segmentation timeline — must be present and contiguous.
REQUIRED_CHANNEL: str = "BF"

# ---------------------------------------------------------------------------
# Derived-id helpers
# ---------------------------------------------------------------------------


def derive_well_id(experiment_id: str, well_index: str) -> str:
    """Compose the global well id from its two atoms."""
    return build_well_id(experiment_id, well_index)


def derive_image_id(well_id: str, channel_id: str, time_index: int) -> str:
    """Compose the image id from the per-frame key atoms (well_id already derived)."""
    return build_image_id(well_id, channel_id, time_index)


# ---------------------------------------------------------------------------
# Derived-id consistency guard
# ---------------------------------------------------------------------------


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
            lambda r: derive_image_id(well_id_col[r.name], r["channel_id"], r["time_index"]),
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
                f"(well_id + channel_id + time_index). First offenders: {sample}"
            )


def frame_inventory_image_ids(df: pd.DataFrame, scope_label: str = "frame_inventory") -> pd.Series:
    """Recompute the DERIVED ``image_id`` for every row by routing the atoms through the grammar.

    This is the identity-anchored unique key. Rather than treating
    ``(experiment_id, well_index, channel_id, time_index)`` as an opaque column tuple, it composes
    ``build_image_id(build_well_id(experiment_id, well_index), channel_id, time_index)`` for each row
    and validates the intermediate ``well_id`` with ``validate_well_id`` (so a leaked bare local
    label or un-promoted id fails loud HERE, at the boundary). The returned Series IS the effective
    unique key — duplicate ``image_id``s mean duplicate frames. Anchoring the key to the
    constructors guarantees it can never drift from the canonical id grammar.

    Raises:
        ValueError: if any row's composed ``well_id`` is not a valid global well_id.
    """
    def _image_id_for_row(row: pd.Series) -> str:
        well_id = build_well_id(row["experiment_id"], row["well_index"])
        validate_well_id(well_id)  # fail loud on a leaked local label / un-promoted id
        return build_image_id(well_id, row["channel_id"], row["time_index"])

    try:
        return df.apply(_image_id_for_row, axis=1)
    except ValueError as exc:
        raise ValueError(f"[{scope_label}] {exc}") from exc


# ---------------------------------------------------------------------------
# Small frozen dataclasses — clipboards, not mayors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchedHandoffSpec:
    """Dataset-level ingress descriptor for an external drop-in submission."""

    experiment_id: str
    manifest_path: Path  # the dataset-level dropin_frame_inventory.csv (all wells)
    image_root: Path | None = None  # resolves relative source_image_path; None → must be absolute


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
