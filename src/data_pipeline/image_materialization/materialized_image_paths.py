"""Pixel-file path constructor for the native materialized image tree.

This module owns ONE question: *where does a materialized frame's pixel file live?*
It answers with a path — no disk I/O, no table reads.

Locked layout (2026-06-17)::

    built_image_data/
      {experiment_id}/
        materialized_images/
          {well_id}/
            projection/      ← one 2D frame per well × channel × time
              {channel_id}/
                {image_id}.{ext}
            z_stack/         ← one 2D frame per well × channel × z × time  (future)
              {channel_id}/
                {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}.{ext}

``projection/`` vs ``z_stack/`` encodes **image product shape**, not method.  The method
(``focus_stack``, ``max_projection``, etc.) lives in the frame_inventory CSV, not the path.
``channel_id`` stays optical/biological (``BF``, ``GFP``) — never encoded as ``BF_z``.

Candidate isolation: ``candidate=True`` inserts ``candidate/`` between
``materialized_images/`` and ``{well_id}/``, making it structurally impossible for a
candidate run to overwrite the live tree.

Import rules: this module imports ``shared/identifiers/`` (ID grammar) and
``frame_inventory_contract`` (allowed suffixes).  It MUST NOT import orchestration, tasks,
Snakemake rules, or scope-specific code.

Root convention: callers pass ``built_image_data_dir`` (= ``DATA_ROOT / "built_image_data"``,
resolved by the Snakefile / ``paths.py``).  This module does not know about ``DATA_ROOT`` or
the stage folder name — that is orchestration territory.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.image_materialization.frame_inventory_contract import (
    ALLOWED_IMAGE_SUFFIXES,
)
from data_pipeline.shared.identifiers.constructors import build_image_id
from data_pipeline.shared.identifiers.validators import validate_well_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_PRODUCT_TYPES: frozenset[str] = frozenset({"projection", "z_stack"})

_MATERIALIZED_IMAGES_SUBDIR = "materialized_images"
_CANDIDATE_SUBDIR = "candidate"


# ---------------------------------------------------------------------------
# Generic constructor
# ---------------------------------------------------------------------------


def materialized_image_path(
    built_image_data_dir: Path,
    *,
    experiment_id: str,
    well_id: str,
    channel_id: str,
    time_index: int,
    image_product_type: str,
    z_index: int | None = None,
    ext: str = "png",
    candidate: bool = False,
) -> Path:
    """Resolve the pixel-file path for one materialized frame.

    Pure path construction — no disk I/O.

    Args:
        built_image_data_dir: stage root (``DATA_ROOT / "built_image_data"``), resolved by
            the caller before this is called.
        experiment_id: global experiment identifier.
        well_id: global well identifier — validated as non-bare-local by ``validate_well_id``.
        channel_id: optical/biological channel token (``BF``, ``GFP``, …).
        time_index: 0-based time point.
        image_product_type: ``"projection"`` or ``"z_stack"``.
        z_index: required for ``z_stack``; must be ``None`` for ``projection``.
        ext: file extension without leading dot; validated against allowed suffixes.
        candidate: if ``True``, inserts ``candidate/`` so the path can never collide with
            the live tree.

    Returns:
        Absolute path (no guarantee the file exists — this is pure path math).

    Raises:
        ValueError: on unknown ``image_product_type``, ``z_index`` / type mismatch, invalid
            ``well_id``, or disallowed ``ext``.
    """
    if image_product_type not in ALLOWED_IMAGE_PRODUCT_TYPES:
        raise ValueError(
            f"Unknown image_product_type {image_product_type!r}. "
            f"Known types: {sorted(ALLOWED_IMAGE_PRODUCT_TYPES)}."
        )

    dot_ext = f".{ext}" if not ext.startswith(".") else ext
    if dot_ext not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Disallowed image extension {dot_ext!r}. "
            f"Allowed: {ALLOWED_IMAGE_SUFFIXES}."
        )

    validate_well_id(well_id)  # raises on bare local label

    if image_product_type == "projection":
        if z_index is not None:
            raise ValueError(
                f"image_product_type='projection' requires z_index=None; got z_index={z_index!r}."
            )
        image_id = build_image_id(well_id, channel_id, time_index)
        filename = f"{image_id}{dot_ext}"
    else:  # z_stack
        if z_index is None:
            raise ValueError(
                "image_product_type='z_stack' requires an integer z_index; got z_index=None."
            )
        filename = f"{well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}{dot_ext}"

    well_subdir = (
        Path(_CANDIDATE_SUBDIR) / well_id if candidate else Path(well_id)
    )

    return (
        built_image_data_dir
        / experiment_id
        / _MATERIALIZED_IMAGES_SUBDIR
        / well_subdir
        / image_product_type
        / channel_id
        / filename
    )


# ---------------------------------------------------------------------------
# Thin wrappers — named for the common call sites
# ---------------------------------------------------------------------------


def projection_frame_path(
    built_image_data_dir: Path,
    *,
    experiment_id: str,
    well_id: str,
    channel_id: str,
    time_index: int,
    ext: str = "png",
    candidate: bool = False,
) -> Path:
    """Resolve the pixel-file path for one projection frame (convenience wrapper).

    Equivalent to ``materialized_image_path(..., image_product_type="projection",
    z_index=None, ...)``.
    """
    return materialized_image_path(
        built_image_data_dir,
        experiment_id=experiment_id,
        well_id=well_id,
        channel_id=channel_id,
        time_index=time_index,
        image_product_type="projection",
        z_index=None,
        ext=ext,
        candidate=candidate,
    )

# z_stack_frame_path is deferred — add when z_index enters the identity grammar.
