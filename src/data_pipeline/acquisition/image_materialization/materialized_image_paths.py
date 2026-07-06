"""Pixel-file path constructor for the native materialized image tree.

This module owns ONE question: *where does a materialized frame's pixel file live?*
It answers with a path — no disk I/O, no table reads.

Locked layout (channel-first, 2026-06-25)::

    built_image_data/
      {experiment_id}/
        materialized_images/
          {well_id}/
            {channel_id}/                       ← optical/biological channel (BF, GFP)
              z_stack/                          ← one 2D frame per channel × z × time
                {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}.{ext}
              projection/                       ← one 2D frame per channel × time
                {projection_method}/            ← focus_stack / max_projection / …
                  {image_id}.{ext}
                  focus_index_map/              ← construction provenance (NOT an image)
                    {image_id}.npz

``channel_id`` comes FIRST (the locked channel-first grammar). ``projection/`` vs ``z_stack/``
encodes **image product shape**; the projection ``{projection_method}`` segment is in the PATH
(it was previously only in the CSV). ``focus_index_map/`` holds the focus-stack provenance
``.npz`` (a ``focus_index_map`` + ``z_indices`` array pair) — provenance, never a primary image
file. ``channel_id`` stays optical/biological (``BF``, ``GFP``) — never encoded as ``BF_z``.

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

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    ALLOWED_IMAGE_SUFFIXES,
)
from data_pipeline.shared.identifiers.constructors import build_image_id
from data_pipeline.shared.identifiers.validators import validate_well_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_PRODUCT_TYPES: frozenset[str] = frozenset({"projection", "z_stack"})

# Suffixes allowed for CONSTRUCTION-PROVENANCE artifacts (e.g. focus_index_map .npz). This is a
# DISTINCT allowlist from ``ALLOWED_IMAGE_SUFFIXES`` (which gates primary image / source_image_path
# files). A provenance artifact is the EXPLANATION of an image, never an image: keeping the two
# allowlists separate prevents ``.npz`` from ever becoming a legal materialized image suffix.
ALLOWED_PROVENANCE_SUFFIXES: tuple[str, ...] = (".npz",)

_MATERIALIZED_IMAGES_SUBDIR = "materialized_images"
_CANDIDATE_SUBDIR = "candidate"
_FOCUS_INDEX_MAP_SUBDIR = "focus_index_map"


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
    projection_method: str | None = None,
    z_index: int | None = None,
    ext: str = "png",
    candidate: bool = False,
) -> Path:
    """Resolve the pixel-file path for one materialized frame.

    Pure path construction — no disk I/O. Uses the locked channel-first grammar::

        {well_id}/{channel_id}/z_stack/{z_image_id}.{ext}
        {well_id}/{channel_id}/projection/{projection_method}/{image_id}.{ext}

    Args:
        built_image_data_dir: stage root (``DATA_ROOT / "built_image_data"``), resolved by
            the caller before this is called.
        experiment_id: global experiment identifier.
        well_id: global well identifier — validated as non-bare-local by ``validate_well_id``.
        channel_id: optical/biological channel token (``BF``, ``GFP``, …).
        time_index: 0-based time point.
        image_product_type: ``"projection"`` or ``"z_stack"``.
        projection_method: REQUIRED for ``projection`` (``"focus_stack"``, ``"max_projection"``, …);
            must be ``None`` for ``z_stack``. The generic constructor keeps no hidden default so the
            grammar stays honest — the named ``projection_frame_path`` wrapper owns the legacy
            ``focus_stack`` default.
        z_index: required for ``z_stack``; must be ``None`` for ``projection``.
        ext: file extension without leading dot; validated against allowed IMAGE suffixes.
        candidate: if ``True``, inserts ``candidate/`` so the path can never collide with
            the live tree.

    Returns:
        Absolute path (no guarantee the file exists — this is pure path math).

    Raises:
        ValueError: on unknown ``image_product_type``, ``z_index`` / ``projection_method`` mismatch,
            invalid ``well_id``, or disallowed ``ext``.
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
        if projection_method is None:
            raise ValueError(
                "image_product_type='projection' requires a projection_method "
                "(e.g. 'focus_stack'); got projection_method=None."
            )
        image_id = build_image_id(well_id, channel_id, time_index)
        product_segment = Path("projection") / projection_method
        filename = f"{image_id}{dot_ext}"
    else:  # z_stack
        if z_index is None:
            raise ValueError(
                "image_product_type='z_stack' requires an integer z_index; got z_index=None."
            )
        if projection_method is not None:
            raise ValueError(
                f"image_product_type='z_stack' requires projection_method=None; "
                f"got projection_method={projection_method!r}."
            )
        image_id = build_image_id(well_id, channel_id, time_index, z_index=z_index)
        product_segment = Path("z_stack")
        filename = f"{image_id}{dot_ext}"

    well_subdir = (
        Path(_CANDIDATE_SUBDIR) / well_id if candidate else Path(well_id)
    )

    return (
        built_image_data_dir
        / experiment_id
        / _MATERIALIZED_IMAGES_SUBDIR
        / well_subdir
        / channel_id
        / product_segment
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
    projection_method: str = "focus_stack",
    ext: str = "png",
    candidate: bool = False,
) -> Path:
    """Resolve the pixel-file path for one projection frame (convenience wrapper).

    Equivalent to ``materialized_image_path(..., image_product_type="projection", ...)``.
    The ``projection_method`` default of ``"focus_stack"`` is LEGACY-COMPAT sugar for the one
    live caller — the generic ``materialized_image_path`` keeps no such default so the grammar
    stays explicit.
    """
    return materialized_image_path(
        built_image_data_dir,
        experiment_id=experiment_id,
        well_id=well_id,
        channel_id=channel_id,
        time_index=time_index,
        image_product_type="projection",
        projection_method=projection_method,
        z_index=None,
        ext=ext,
        candidate=candidate,
    )


def focus_index_map_path(
    built_image_data_dir: Path,
    *,
    experiment_id: str,
    well_id: str,
    channel_id: str,
    time_index: int,
    ext: str = "npz",
    candidate: bool = False,
) -> Path:
    """Resolve the CONSTRUCTION-PROVENANCE path for one projection's focus-index map.

    Lands under ``{channel_id}/projection/focus_stack/focus_index_map/{image_id}.npz`` — a
    sibling of the focus_stack projection image it explains, 1:1 with that projection ``image_id``.
    This is NOT an image path: it is gated by ``ALLOWED_PROVENANCE_SUFFIXES`` (``.npz``), never by
    ``ALLOWED_IMAGE_SUFFIXES``. The ``.npz`` stores ``focus_index_map`` (per-pixel stack-axis
    offsets) + ``z_indices`` (ordered acquisition z labels for those offsets).
    """
    dot_ext = f".{ext}" if not ext.startswith(".") else ext
    if dot_ext not in ALLOWED_PROVENANCE_SUFFIXES:
        raise ValueError(
            f"Disallowed provenance extension {dot_ext!r}. "
            f"Allowed: {ALLOWED_PROVENANCE_SUFFIXES}."
        )
    validate_well_id(well_id)
    image_id = build_image_id(well_id, channel_id, time_index)
    well_subdir = (
        Path(_CANDIDATE_SUBDIR) / well_id if candidate else Path(well_id)
    )
    return (
        built_image_data_dir
        / experiment_id
        / _MATERIALIZED_IMAGES_SUBDIR
        / well_subdir
        / channel_id
        / "projection"
        / "focus_stack"
        / _FOCUS_INDEX_MAP_SUBDIR
        / f"{image_id}{dot_ext}"
    )


def z_stack_frame_path(
    built_image_data_dir: Path,
    *,
    experiment_id: str,
    well_id: str,
    channel_id: str,
    time_index: int,
    z_index: int,
    ext: str = "png",
    candidate: bool = False,
) -> Path:
    """Resolve the pixel-file path for one z-stack plane (convenience wrapper).

    Equivalent to ``materialized_image_path(..., image_product_type="z_stack",
    z_index=z_index, ...)``.
    """
    return materialized_image_path(
        built_image_data_dir,
        experiment_id=experiment_id,
        well_id=well_id,
        channel_id=channel_id,
        time_index=time_index,
        image_product_type="z_stack",
        z_index=z_index,
        ext=ext,
        candidate=candidate,
    )
