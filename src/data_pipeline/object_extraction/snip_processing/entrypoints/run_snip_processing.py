"""Per-well snip processing entrypoint.

Reads a validated frame_masks shard, the matching frame_inventory shard, and the
per-well physical_embryo_registry shard; JOINS physical_embryo_id onto each valid
mask (it is NOT minted here anymore — the registry is the identity-origination
boundary), builds the crop-level snip identifiers, runs the
extraction/rotation/augmentation stack, and writes the per-well snip_inventory
CSV + pixel PNG files.

Masks are stored as RLE in frame_masks — decoded to numpy here before passing
to the core stack. Yolk masks are not yet wired; rotation falls back to the
mass-distribution heuristic and extraction uses a zero yolk mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scipy.ndimage
import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.shared.identifiers.constructors import (
    build_embryo_id,
    build_image_id,
    build_snip_id,
    build_snip_transform_id,
)
from data_pipeline.shared.identifiers.parsers import parse_image_id
from data_pipeline.object_extraction.snip_processing.augmentation import (
    BACKGROUND_NOISE_SEED,
    BACKGROUND_NOISE_TRUNCATION_UPPER_STANDARD_DEVIATIONS,
)
from data_pipeline.object_extraction.snip_processing.inventory_contract import (
    SNIP_INVENTORY_WRITE_COLUMNS,
    validate_snip_inventory,
)
from data_pipeline.object_extraction.snip_processing.legacy_snip_paths import (
    legacy_flat_snip_path,
    link_legacy_flat_path,
)
from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
    channel_id_for_snip_product_key,
    parse_snip_product_key,
)
from data_pipeline.object_extraction.snip_processing.snip_recipes import (
    CLAHE_BLEND,
    assert_source_dtype_is_acceptable,
    recipe_contract,
    recipe_output_dtype,
    render_snip,
)
from data_pipeline.object_extraction.snip_processing.snip_transform import (
    CENTERING_LATCHED,
    output_grid_id,
    IMAGE_INTERPOLATION,
    MASK_INTERPOLATION,
    apply_transform_to_image,
    apply_transform_to_mask,
    transform_for_product,
)
from data_pipeline.object_extraction.snip_processing.snip_transform_table import (
    build_resolved_chain_payload,
    canonical_from_row,
    mask_content_fingerprint,
    read_snip_transform_table,
)
from data_pipeline.object_extraction.snip_processing.provenance import (
    build_rendering_contract,
    observed_mask_sources,
    rendering_sidecar_path,
    update_rendering_sidecar,
)

# The orientation DECISION recorded on every row, so a later yolk-aware pass knows what it replaces.
# These are the CURRENT policy, not a configuration surface: there is exactly one orientation engine
# today. When a yolk-aware source lands it adds a value here rather than silently changing meaning.
ORIENTATION_POLICY = "pca_major_axis_yolk_down"
ORIENTATION_SOURCE_MASS_DISTRIBUTION = "embryo_mass_distribution"
NO_YOLK_POLICY = "fallback_mass_distribution"


# NOTE: the dtype precondition moved to snip_recipes.SNIP_RECIPE_CONTRACTS, where each recipe
# declares what it accepts. It raises SnipRecipeError, which the per-row handler catches and records
# as is_valid_snip=False -- so one bad frame invalidates its own snip rather than taking down a well
# that also holds good rows.

_BACKGROUND_PIXELS_PER_IMAGE = 5000
_BACKGROUND_FRAME_SAMPLE_COUNT = 50
_BACKGROUND_SAMPLE_SEED = 309


def _is_seahub_frame(inventory_row: pd.Series) -> bool:
    """Return whether one frame-inventory row came from the SeaHub adapter."""
    return str(inventory_row.get("source_scope", "")).strip().casefold() == "seahub"


def _canvas_fill_value(inventory_row: pd.Series) -> float:
    """Return the declared SeaHub canvas fill, safely defaulting to black."""
    try:
        value = float(inventory_row.get("canvas_fill_value", 0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _clean_seahub_mask(mask: np.ndarray) -> np.ndarray:
    """Keep one filled SeaHub object as a final defense before snip transforms.

    Frame-mask validation is still the primary contract. This adapter-specific
    cleanup prevents a stray disconnected SAM component from expanding the PCA
    rotation/crop bounds or surviving as a second object in the persisted snip
    mask. Eight-connectivity treats diagonally touching pixels as one component,
    matching ordinary image-mask connectivity.
    """
    binary = np.asarray(mask).astype(bool, copy=False)
    labels, component_count = scipy.ndimage.label(
        binary,
        structure=scipy.ndimage.generate_binary_structure(2, 2),
    )
    if component_count == 0:
        return np.zeros(binary.shape, dtype=np.uint8)

    component_areas = np.bincount(labels.ravel())
    component_areas[0] = 0
    largest = labels == int(np.argmax(component_areas))
    return scipy.ndimage.binary_fill_holes(largest).astype(np.uint8)


def _physical_embryo_id_by_track(
    physical_embryo_registry: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    """Build the ``(well_id, track_id) -> physical_embryo_id`` lookup from the registry.

    The registry is the single source of identity; snip_processing JOINS against it rather than
    re-minting. Returns an exact-match dict keyed by the join columns.
    """
    return {
        (str(row["well_id"]), str(row["track_id"])): str(row["physical_embryo_id"])
        for _, row in physical_embryo_registry.iterrows()
    }


class MaskProvenanceError(RuntimeError):
    """The masks on disk are not the masks snip_geometry certified.

    A JOB-LEVEL contract failure, deliberately NOT a per-row data condition. An empty mask, a
    clipped embryo, a corrupt frame -- those are row conditions: this well contains one bad object.
    A hash mismatch says something categorically different: the renderer is consuming a DIFFERENT
    REVISION of frame_masks than the geometry gate derived from, so the placement and the mask no
    longer describe the same segmentation event.

    Every artifact in that state is individually plausible -- the placement looks reasonable, the
    mask looks reasonable -- which is what makes it dangerous. Marking such rows invalid would
    prevent their use while NORMALIZING a condition that should never occur, and 143 invalid rows in
    an inventory obscures the actual diagnosis: a stale transform table, or mismatched frame_masks.
    """


def _assert_masks_match_geometry(
    valid_masks: pd.DataFrame,
    transform_rows_by_id: dict[str, Any],
    *,
    physical_embryo_id_by_track: dict[tuple[str, str], str],
    scope_label: str,
) -> None:
    """PREFLIGHT. Verify every mask still matches what geometry was derived from.

    Runs BEFORE any pixel is written, and collects EVERY mismatch rather than exploding on the
    first, so the operator sees one intelligible fact about the well instead of an inventory
    carpeted with invalid rows.

    WHY THIS EARNS ITS KEEP DESPITE SNAKEMAKE. A dependency graph asserts "the declared inputs
    appear current"; this asserts "these are actually the same pixels." The two diverge under copied
    artifacts with preserved timestamps, manual file replacement, partial reruns, stale outputs from
    an older code revision, direct entrypoint invocation, and network-filesystem timing oddities.
    """
    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []

    for _, mask_row in valid_masks.iterrows():
        image_id = str(mask_row["image_id"])
        well_id = str(mask_row["well_id"])
        track_id = str(mask_row["track_id"])
        _, _channel, time_index = parse_image_id(image_id)
        physical_embryo_id = physical_embryo_id_by_track.get((well_id, track_id))
        if physical_embryo_id is None:
            continue  # the registry gate below reports this with a better message

        transform_id = build_snip_transform_id(physical_embryo_id, int(time_index))
        row = transform_rows_by_id.get(transform_id)
        if row is None:
            missing.append(f"{transform_id} (mask_id={mask_row['mask_id']})")
            continue

        expected = str(row["geometry_source_mask_sha256"])
        observed = mask_content_fingerprint(
            decode_binary_mask_rle(json.loads(str(mask_row["mask_rle"]))).astype(np.uint8)
        )
        if expected != observed:
            mismatched.append((str(mask_row["mask_id"]), expected, observed))

    if not missing and not mismatched:
        return

    examples = "\n".join(
        f"  mask_id={mid}\n    expected={exp[:16]}...\n    observed={obs[:16]}..."
        for mid, exp, obs in mismatched[:3]
    ) or "\n".join(f"  {m}" for m in missing[:3])

    raise MaskProvenanceError(
        f"Mask provenance validation failed for {scope_label}:\n"
        f"  {len(mismatched)} of {len(valid_masks)} masks differ from those used to derive snip "
        f"geometry.\n"
        f"  Missing from the transform table: {len(missing)}\n"
        f"  Hash mismatches: {len(mismatched)}\n"
        f"Examples:\n{examples}\n"
        "Regenerate snip_geometry from the current frame_masks artifact, or restore the frame_masks "
        "revision the transform table was built from. No product artifacts were written."
    )


def _source_frames_for_product(
    frame_inventory: pd.DataFrame, *, source_image_product_key: str
) -> pd.DataFrame:
    """Rows of ONE image product, indexed by image_id.

    Fails loud when the product is absent rather than returning an empty frame: a job asked to
    render snips of a product that was never materialized should say so, not produce a well full of
    invalid rows that look like segmentation failures.
    """
    from data_pipeline.acquisition.image_materialization.image_product_keys import (
        image_product_key_for_frame_row,
    )

    keys = frame_inventory.apply(
        lambda row: image_product_key_for_frame_row(
            channel_id=row["channel_id"],
            image_product_type=row.get("image_product_type", "projection"),
            projection_method=row.get("projection_method"),
        ),
        axis=1,
    ) if len(frame_inventory) else pd.Series(dtype=str)

    matched = frame_inventory[keys == source_image_product_key] if len(frame_inventory) else frame_inventory
    if len(frame_inventory) and matched.empty:
        available = sorted(set(keys))
        raise ValueError(
            f"snip_processing: no frame_inventory rows for source product "
            f"{source_image_product_key!r}; available: {available}. Materialize that product before "
            "requesting snips of it."
        )
    return matched.set_index("image_id")


def _source_rows_for_masks(
    valid_masks: pd.DataFrame,
    inventory_index: pd.DataFrame,
    *,
    source_channel_id: str,
    source_image_product_key: str,
) -> dict[object, tuple[str, pd.Series, float]]:
    """Join every renderable mask to a calibrated row of its requested product.

    This is a job-level preflight, before background estimation or pixel writes.
    Missing, null, non-numeric, non-finite, and non-positive calibrations are
    contract failures rather than per-snip invalid rows.
    """

    if inventory_index.index.has_duplicates:
        duplicates = sorted(
            {
                str(value)
                for value in inventory_index.index[
                    inventory_index.index.duplicated(keep=False)
                ]
            }
        )
        raise ValueError(
            "snip_processing: source product "
            f"{source_image_product_key!r} has duplicate image_id rows: {duplicates[:10]}"
        )

    resolved: dict[object, tuple[str, pd.Series, float]] = {}
    failures: list[str] = []
    for row_index, mask_row in valid_masks.iterrows():
        well_id = str(mask_row["well_id"])
        mask_image_id = str(mask_row["image_id"])
        _, _mask_channel_id, time_index = parse_image_id(mask_image_id)
        source_image_id = build_image_id(well_id, source_channel_id, int(time_index))
        if source_image_id not in inventory_index.index:
            failures.append(
                f"image_id={source_image_id!r}: no frame_inventory row for source product "
                f"{source_image_product_key!r} (sibling of mask frame {mask_image_id!r})"
            )
            continue

        inventory_row = inventory_index.loc[source_image_id]
        raw_scale = inventory_row.get("image_micrometers_per_pixel", pd.NA)
        try:
            source_scale = float(raw_scale)
        except (TypeError, ValueError):
            source_scale = float("nan")
        if not np.isfinite(source_scale) or source_scale <= 0:
            failures.append(
                f"image_id={source_image_id!r}, source product "
                f"{source_image_product_key!r}: image_micrometers_per_pixel={raw_scale!r}; "
                "expected a finite positive value from frame_inventory"
            )
            continue
        resolved[row_index] = (source_image_id, inventory_row, source_scale)

    if failures:
        raise ValueError(
            "snip_processing source-calibration preflight failed before writing pixels:\n  "
            + "\n  ".join(failures[:10])
        )
    return resolved


def _assert_transform_write_policy(
    transform_rows_by_id: dict[str, Any],
    *,
    expected_target_micrometers_per_pixel: float,
    expected_output_shape: tuple[int, int],
    snip_product_key: str,
) -> None:
    """Ensure the persisted geometry and this renderer declare one output grid."""

    target = float(expected_target_micrometers_per_pixel)
    if not np.isfinite(target) or target <= 0:
        raise ValueError(
            f"snip_processing: target_pixel_size_um for {snip_product_key!r} must be "
            f"finite and positive; got {expected_target_micrometers_per_pixel!r}."
        )
    for transform_id, transform_row in transform_rows_by_id.items():
        canonical = canonical_from_row(transform_row)
        observed_target = float(canonical.grid.default_output_um_per_px_yx[0])
        observed_shape = tuple(canonical.grid.default_output_shape_yx)
        if not np.isclose(observed_target, target, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"snip_processing: {transform_id!r} targets {observed_target} um/px, but "
                f"product {snip_product_key!r} was invoked with {target} um/px. Regenerate "
                "snip_geometry and render with the same output-grid policy."
            )
        if observed_shape != tuple(expected_output_shape):
            raise ValueError(
                f"snip_processing: {transform_id!r} targets output shape {observed_shape}, "
                f"but product {snip_product_key!r} was invoked with "
                f"{tuple(expected_output_shape)}. Regenerate snip_geometry and render with "
                "the same output-grid policy."
            )


def _estimate_background(
    valid_masks: pd.DataFrame,
    inventory_index: pd.DataFrame,
    *,
    source_rows_by_mask: dict[object, tuple[str, pd.Series, float]] | None = None,
    n_samples: int = _BACKGROUND_FRAME_SAMPLE_COUNT,
    seed: int = _BACKGROUND_SAMPLE_SEED,
) -> tuple[float, float]:
    """Sample background pixels (outside embryo mask) to estimate mean/std.

    Non-SeaHub frames preserve the legacy build03A behavior: take the first
    5,000 full-frame pixels outside the embryo mask. SeaHub frames are centered
    on a large constant canvas, so that prefix is almost always padding. For
    SeaHub, remove the declared ``canvas_fill_value`` and draw up to 5,000
    genuine outside-mask pixels uniformly without replacement. When rendering a
    sibling product (for example RFP from a BF mask), ``source_rows_by_mask``
    selects that product's own frame rather than looking up the mask's BF
    ``image_id``. Both frame and pixel sampling are deterministic under ``seed``.
    """
    # Preserve the legacy global seed contract: downstream background-noise
    # generation uses ``np.random`` too, so the full snip run remains
    # reproducible rather than only this sampling helper.
    np.random.seed(seed)
    indices = valid_masks.index.tolist()
    sample_idx = np.random.choice(
        indices,
        size=min(n_samples, len(indices)),
        replace=False,
    )

    bkg_pixels: list[float] = []
    seahub_fill_values: list[float] = []
    saw_seahub_frame = False
    for i in sample_idx:
        row = valid_masks.loc[i]
        try:
            if source_rows_by_mask is None:
                image_id = str(row["image_id"])
                if image_id not in inventory_index.index:
                    continue
                inventory_row = inventory_index.loc[image_id]
            else:
                source_row = source_rows_by_mask.get(i)
                if source_row is None:
                    continue
                _source_image_id, inventory_row, _source_scale = source_row
            src = Path(str(inventory_row["image_path"]))
            img = skio.imread(str(src))
            if img.ndim == 3:
                img = img[:, :, 0]
            rle = json.loads(str(row["mask_rle"]))
            mask = decode_binary_mask_rle(rle).astype(bool)

            if _is_seahub_frame(inventory_row):
                saw_seahub_frame = True
                fill_value = _canvas_fill_value(inventory_row)
                seahub_fill_values.append(fill_value)
                mask = _clean_seahub_mask(mask).astype(bool)
                bkg = img[(~mask) & (img != fill_value)].astype(float)
                if bkg.size > _BACKGROUND_PIXELS_PER_IMAGE:
                    pixel_idx = np.random.choice(
                        bkg.size,
                        size=_BACKGROUND_PIXELS_PER_IMAGE,
                        replace=False,
                    )
                    bkg = bkg[pixel_idx]
            else:
                bkg = img[~mask].astype(float)
                bkg = bkg[:_BACKGROUND_PIXELS_PER_IMAGE]

            if bkg.size > 0:
                bkg_pixels.extend(bkg.tolist())
        except Exception:
            continue

    if not bkg_pixels:
        if saw_seahub_frame:
            # A crop can legitimately contain no non-fill background (for
            # example a mask covering every pasted source pixel). Preserve a
            # safe, constant background rather than falling back to the legacy
            # mid-gray distribution.
            fill_value = seahub_fill_values[0] if seahub_fill_values else 0.0
            return float(fill_value), 0.0
        return 128.0, 30.0
    return float(np.mean(bkg_pixels)), float(np.std(bkg_pixels))


def run_snip_processing(
    *,
    frame_masks_csv: Path,
    frame_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    snips_dir: Path,
    output_root: Path,
    target_pixel_size_um: float = 7.8,
    output_height_px: int = 576,
    output_width_px: int = 256,
    background_noise_scale: float = 0.1,
    blend_radius_um: float = 20.0,
    snip_transform_table_csv: Path | None = None,
    snip_product_key: str = DEFAULT_BF_SNIP_PRODUCT_KEY,
    # Only the clahe_blend recipe reads this; no_change ignores it. Kept from main, where SeaHub's
    # runtime overlay sets it false for source images that are already normalized.
    apply_clahe: bool = True,
) -> None:
    frame_masks = pd.read_csv(frame_masks_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    physical_embryo_registry = pd.read_csv(physical_embryo_registry_csv)

    # ONE JOB RENDERS ONE PRODUCT. The fanout is {well_id} x {snip_product_key}, so each job owns a
    # disjoint output subtree and can fail without taking its siblings down -- BF succeeding while
    # RFP fails is a real and useful state, since everything downstream depends on BF.
    #
    # DERIVED from the key, never passed alongside it: the key already says which recipe this is,
    # and a second parameter could disagree with it.
    _source_image_product_key, snip_recipe = parse_snip_product_key(snip_product_key)
    source_channel_id = channel_id_for_snip_product_key(snip_product_key)

    valid_masks = frame_masks[frame_masks["is_valid_mask"].astype(bool)].copy()
    # PRODUCT-AWARE SOURCE LOOKUP. The effective frame key is (image_id, product_key), not image_id
    # alone -- frame_inventory_contract warns that anchoring on image_id "would falsely collide" two
    # projection products of one channel. Filtering to the requested source product first makes the
    # remaining image_id index unambiguous.
    inventory_index = _source_frames_for_product(
        frame_inventory, source_image_product_key=_source_image_product_key
    )
    source_rows_by_mask = _source_rows_for_masks(
        valid_masks,
        inventory_index,
        source_channel_id=source_channel_id,
        source_image_product_key=_source_image_product_key,
    )

    experiment_ids = sorted(
        {
            str(value)
            for table in (frame_masks, frame_inventory)
            if "experiment_id" in table.columns
            for value in table["experiment_id"].dropna().tolist()
        }
    )
    if len(experiment_ids) != 1:
        raise ValueError(
            "snip_processing is a per-well, single-experiment writer; expected exactly "
            f"one experiment_id for provenance, found {experiment_ids}."
        )
    well_ids = sorted(
        {
            str(value)
            for table in (frame_masks, inventory_index)
            if "well_id" in table.columns
            for value in table["well_id"].dropna().tolist()
        }
    )
    if len(well_ids) != 1:
        raise ValueError(
            "snip_processing is a per-well writer; expected exactly one well_id for "
            f"provenance, found {well_ids}."
        )

    mask_sources = observed_mask_sources(
        valid_masks,
        source_column="segmentation_backend",
        version_column="segmentation_model_id",
    )
    source_scopes = (
        sorted({str(value) for value in inventory_index["source_scope"].dropna()})
        if "source_scope" in inventory_index.columns
        else []
    )

    # Identity is JOINED from the registry, never minted here. A valid mask whose track has no
    # registry row is a contract violation (the registry is built from frame_masks, so it must
    # cover every detected track) — fail loud rather than silently drop a real embryo.
    physical_embryo_id_by_track = _physical_embryo_id_by_track(physical_embryo_registry)

    output_shape = (output_height_px, output_width_px)
    snips_dir = Path(snips_dir)
    output_root = Path(output_root)

    # Every alias created this run, for debugging external breakage without archaeology.
    legacy_alias_manifest: list[dict[str, str]] = []

    _bg_mean, _bg_std = _estimate_background(
        valid_masks,
        inventory_index,
        source_rows_by_mask=source_rows_by_mask,
    )
    background_mean = background_noise_scale * _bg_mean
    background_std = background_noise_scale * _bg_std

    # THE GATE'S INPUT. snip_geometry derived these once for this well; this job only resolves
    # them onto its own product grid. Optional purely so existing callers/tests that predate the
    # gate still work -- when absent, the per-row lookup fails loud rather than silently deriving.
    transform_rows_by_id: dict[str, Any] = {}
    if snip_transform_table_csv is not None:
        _table = read_snip_transform_table(Path(snip_transform_table_csv))
        transform_rows_by_id = {
            str(r["snip_transform_id"]): r for _, r in _table.iterrows()
        }
        # PREFLIGHT, before a single pixel is written. A provenance failure invalidates this
        # PRODUCT job for this well and nothing else -- a sibling product's already-valid artifacts
        # are untouched -- but it must not produce a shard that mixes mask revisions.
        _assert_masks_match_geometry(
            valid_masks,
            transform_rows_by_id,
            physical_embryo_id_by_track=physical_embryo_id_by_track,
            scope_label=f"{Path(frame_masks_csv).stem} / {snip_product_key}",
        )
        _assert_transform_write_policy(
            transform_rows_by_id,
            expected_target_micrometers_per_pixel=float(target_pixel_size_um),
            expected_output_shape=(int(output_height_px), int(output_width_px)),
            snip_product_key=snip_product_key,
        )

    is_clahe_blend = snip_recipe == CLAHE_BLEND
    declared_dtype = recipe_contract(snip_recipe).output_dtype or "source_dtype"
    rendering_contract = build_rendering_contract(
        snip_product_key=snip_product_key,
        source_image_product_key=_source_image_product_key,
        snip_recipe=snip_recipe,
        target_micrometers_per_pixel=float(target_pixel_size_um),
        blend_radius_micrometers=float(blend_radius_um),
        blend_radius_applied=is_clahe_blend,
        frame_shape=(int(output_height_px), int(output_width_px)),
        mask_contract={
            "artifact": "frame_masks",
            "payload_column": "mask_rle",
            "source_column": "segmentation_backend",
            "version_column": "segmentation_model_id",
            "geometry_mask_policy": "decoded binary frame mask, unchanged",
            "render_mask_policy": {
                "default": "decoded binary frame mask",
                "seahub": (
                    "eight-connected largest component followed by binary hole fill"
                ),
            },
        },
        orientation_contract={
            "policy": ORIENTATION_POLICY,
            "evidence_source": ORIENTATION_SOURCE_MASS_DISTRIBUTION,
            "no_yolk_policy": NO_YOLK_POLICY,
            "derivation_stage": "snip_geometry",
        },
        clahe_enabled=is_clahe_blend and bool(apply_clahe),
        background_contract={
            "estimated_for_job": True,
            "applied_to_pixels": is_clahe_blend,
            "model": "per-well source-frame pixels outside the current embryo mask",
            "estimator": (
                "entrypoints.run_snip_processing._estimate_background with SeaHub "
                "canvas-fill exclusion"
            ),
            "frame_sample_count": _BACKGROUND_FRAME_SAMPLE_COUNT,
            "frame_sample_seed": _BACKGROUND_SAMPLE_SEED,
            "maximum_pixels_per_frame": _BACKGROUND_PIXELS_PER_IMAGE,
            "noise_scaling": float(background_noise_scale),
            "noise_distribution": {
                "implementation": "scipy.stats.truncnorm",
                "lower_intensity": 0.0,
                "upper_standard_deviations": (
                    BACKGROUND_NOISE_TRUNCATION_UPPER_STANDARD_DEVIATIONS
                ),
                "seed": BACKGROUND_NOISE_SEED,
            },
        },
        resampling_contract={
            "source_calibration_column": "image_micrometers_per_pixel",
            "calibration_scope": "selected source image product row",
            "chain": [
                {
                    "step": "resize",
                    "implementation": "cv2.resize via image_geometry.TransformChain",
                    "image_downscale_kernel": "INTER_AREA",
                    "image_upscale_or_identity_kernel": "INTER_LINEAR",
                    "mask_kernel": "INTER_NEAREST",
                    "anti_alias": True,
                },
                {
                    "step": "rotate_center",
                    "implementation": "cv2.warpAffine via image_geometry.TransformChain",
                    "image_kernel": "INTER_LINEAR",
                    "mask_kernel": "INTER_NEAREST",
                    "border_mode": "constant",
                    "border_value": 0.0,
                    "centering": CENTERING_LATCHED,
                },
            ],
        },
        file_encoding={
            "processed_snip": {
                "format": "PNG inferred from extension",
                "extension": ".png",
                "dtype": declared_dtype,
                "color_mode": "2-D grayscale",
                "writer": "skimage.io.imsave",
                "writer_options": {"check_contrast": False},
                "encoder_parameters": "library defaults",
            },
            "embryo_mask_reference": {
                "producer": "snip_geometry",
                "format": "PNG inferred from extension",
                "dtype": "uint8",
                "values": [0, 255],
                "writer": "skimage.io.imsave",
                "writer_options": {"check_contrast": False},
            },
        },
    )

    rows: list[dict[str, Any]] = []
    # Collected inside the loop, reconciled after it. Reconciliation is keyed by snip_transform_id so
    # sibling products of one embryo-time collapse onto ONE row — that collapse is the mechanism
    # that makes duplicate (and therefore driftable) geometry unrepresentable.

    for mask_row_index, mask_row in valid_masks.iterrows():
        image_id = str(mask_row["image_id"])
        track_id = str(mask_row["track_id"])
        mask_id = str(mask_row["mask_id"])
        well_id = str(mask_row["well_id"])

        _, channel_id, time_index = parse_image_id(image_id)
        experiment_id = str(mask_row.get("experiment_id", ""))

        physical_embryo_id = physical_embryo_id_by_track.get((well_id, track_id))
        if physical_embryo_id is None:
            raise ValueError(
                f"No physical_embryo_registry entry for (well_id={well_id!r}, "
                f"track_id={track_id!r}) — every valid frame_masks track must be registered. "
                f"Rebuild the physical_embryo_registry shard for {well_id!r} from the SAME "
                f"frame_masks shard before running snip_processing."
            )
        # embryo_id / snip_id are crop-product naming and legitimately stay here.
        embryo_id = build_embryo_id(physical_embryo_id, image_id)
        snip_id = build_snip_id(embryo_id, image_id)
        # Channel-INDEPENDENT: the transform is a fact about the animal at a time, not about a
        # channel, so every sibling product of this embryo-time resolves to this same id.
        snip_transform_id = build_snip_transform_id(physical_embryo_id, int(time_index))
        _source_image_id, inv_row, pixel_size_um = source_rows_by_mask[mask_row_index]

        out: dict[str, Any] = {
            "snip_id": snip_id,
            "embryo_id": embryo_id,
            "physical_embryo_id": physical_embryo_id,
            "experiment_id": experiment_id,
            "well_id": well_id,
            "image_id": image_id,
            "time_index": time_index,
            "channel_id": channel_id,
            "mask_id": mask_id,
            "track_id": track_id,
            "image_path": str(inv_row["image_path"]),
            "processed_snip_path": None,
            # Product identity is known BEFORE rendering, so a failed row still says which product
            # it was trying to be -- otherwise a failure is unattributable once several products
            # share a well. The paths stay null: no bytes, no alias.
            "snip_product_key": snip_product_key,
            "legacy_flat_snip_path": None,
            "source_image_product_key": _source_image_product_key,
            "output_grid_id": None,
            "pixel_dtype": None,
            "resolved_transform_chain_json": None,
            "embryo_mask": None,
            "embryo_mask_snip_path": None,
            "crop_x_min_px": None,
            "crop_y_min_px": None,
            "crop_x_max_px": None,
            "crop_y_max_px": None,
            "crop_width_px": output_width_px,
            "crop_height_px": output_height_px,
            # DECLARED BY snip_identity_contract, so they must be emitted. These are the same two
            # facts as source_um_per_px / target_um_per_px below, under the names the contract (and
            # every existing downstream consumer) uses. Kept rather than dropped in favour of the
            # _um_per_px spelling: renaming a published column is a separate, deliberate migration,
            # and until it happens a column the contract requires must not arrive as all-NaN.
            "source_micrometers_per_pixel": pixel_size_um,
            "snip_micrometers_per_pixel": float(target_pixel_size_um),
            "is_valid_snip": False,
            "error_message": None,
            # Construction provenance — populated from the resolved transform below. A row that
            # failed to render keeps these NA, which is the honest record: no transform existed.
            "orientation_policy": None,
            "orientation_source": None,
            "no_yolk_policy": None,
            "rotation_angle_rad": None,
            "flip_x": None,
            "crop_x_min_um": None,
            "crop_y_min_um": None,
            "crop_x_max_um": None,
            "crop_y_max_um": None,
            "crop_center_um_x": None,
            "crop_center_um_y": None,
            "source_height_px": None,
            "source_width_px": None,
            "source_um_per_px": None,
            "target_um_per_px": float(target_pixel_size_um),
            "output_height_px": output_height_px,
            "output_width_px": output_width_px,
            "border_mode": None,
            "image_interpolation": None,
            "mask_interpolation": None,
            "realized_scale_y": None,
            "realized_scale_x": None,
            "centering": None,
            # FK into the per-well transform table. Not a path and not optional: every snip has a
            # transform, and the id is CHANNEL-INDEPENDENT so sibling products of this embryo-time
            # reference the SAME row rather than each holding a copy that can drift.
            "snip_transform_id": None,
        }

        try:
            image_path = Path(str(inv_row["image_path"]))

            # Decode RLE mask from frame_masks row.
            rle = json.loads(str(mask_row["mask_rle"]))
            embryo_mask = decode_binary_mask_rle(rle).astype(np.uint8)
            # SEAHUB ADAPTER CLEANUP, carried over from main (409d9c83/37aeb639). A stray
            # disconnected SAM component would otherwise widen the rotation/crop bounds or survive
            # as a second object in the persisted mask. Placed BEFORE any geometry is derived from
            # this mask, which is the only position where it can protect the crop it is meant to.
            if _is_seahub_frame(inv_row):
                embryo_mask = _clean_seahub_mask(embryo_mask)

            image = skio.imread(str(image_path))
            if image.ndim == 3:
                image = image[:, :, 0]

            # HAZARD ZERO BARRIER. This used to be:
            #
            #     if image.dtype != np.uint8:
            #         image = rescale_intensity(image, in_range="image", out_range=(0, 255))
            #
            # which used DTYPE AS A PROXY FOR RECIPE: uint8 was assumed to be a display frame, and
            # anything else was silently converted into one. `in_range="image"` is per-frame min/max
            # autoscaling, so every frame got a DIFFERENT affine map keyed to its own extremes --
            # meaning a 2-copy and a 0-copy embryo in different frames land on the same 0-255 span.
            # That is not attenuation of dosage signal, it is exact erasure, and it is silent.
            #
            # The guard `dtype != uint8` meant it never fired on today's 8-bit BF and would ALWAYS
            # fire on a uint16 fluorescence frame -- i.e. it was dormant precisely until the moment
            # it would destroy the measurement this pipeline is being built to make.
            #
            # The rule is now recipe-driven and there is NO generic dtype fallback:
            #
            #     clahe_blend  requires uint8 (apply_clahe documents and returns uint8, and the
            #                  noise blend is tuned for 8-bit BF)
            #     no_change    preserves source dtype and scale        <- arrives with P2
            #     unknown      fails loudly
            #
            # Until the recipe seam is reachable, this entrypoint IS the legacy BF path, so it
            # asserts its own precondition rather than quietly coercing. Refusing is the whole
            # point: a loud failure is recoverable, a silently rescaled uint16 frame is not.
            # The recipe states its own dtype precondition (snip_recipes.SNIP_RECIPE_CONTRACTS),
            # so this is a DECLARATION rather than the dtype-as-proxy-for-recipe inference it
            # replaced. clahe_blend requires uint8; no_change accepts anything and preserves it.
            assert_source_dtype_is_acceptable(
                image, snip_recipe=snip_recipe, source=str(image_path)
            )

            # Render through the transform seam: derive the physical recipe from the MASK alone,
            # resolve it onto this frame's pixel grid, then rasterize image and mask through the
            # SAME resolved transform so they stay registered by construction.
            #
            # centering=CENTERING_LATCHED is deliberate and TRANSITIONAL: it reproduces the legacy
            # int() centering quantizer so that this commit varies only the resample kernel
            # (skimage -> cv2 INTER_AREA) and the render path (direct-to-canvas, no bounds-expanded
            # intermediate + separate crop). Flipping to CENTERING_CONTINUOUS is a separate commit,
            # so the two pixel changes stay independently attributable.
            #
            # THE GATE'S READ PATH. Geometry is NOT derived here — it was derived once per
            # embryo-time by snip_geometry and persisted. This job deserializes that recipe and
            # resolves it onto its OWN product grid, which is the step that speaks this product's
            # pixel dialect. A render job that re-derived would reopen exactly the drift the gate
            # closes, so derive_snip_transform is deliberately not called on this path.
            if snip_transform_id not in transform_rows_by_id:
                raise KeyError(
                    f"snip_transform_id {snip_transform_id!r} not found in the snip transform "
                    f"table. snip_geometry must run for this well before any product renders; a "
                    "render job may not derive its own geometry."
                )
            transform_row = transform_rows_by_id[snip_transform_id]

            # Mask provenance was verified up front for the whole well, so by here the mask
            # and the transform are known to describe the same segmentation event.

            canonical = canonical_from_row(transform_row)
            resolved = transform_for_product(
                canonical,
                product_shape_hw=(int(image.shape[0]), int(image.shape[1])),
                product_um_per_px=pixel_size_um,
                centering=CENTERING_LATCHED,
            )
            # The output dtype is the RECIPE's, not a constant: hardcoding uint8 here truncated
            # uint16 to 8 bits AFTER the read-boundary barrier had already let it through, which is
            # the same erasure one layer down.
            image_cropped = apply_transform_to_image(
                image,
                resolved,
                dtype=recipe_output_dtype(snip_recipe, source_dtype=image.dtype),
            )
            mask_cropped = apply_transform_to_mask(embryo_mask, resolved)

            # Record HOW this snip was built, on the row. Cheap, queryable, and answers "what
            # happened here" without opening the sidecar. The crop window is carried in BOTH
            # product-pixel and physical coordinates on purpose: product pixels alone cannot express
            # one recipe across calibrations, and physical alone is not auditable against the file
            # actually written to disk.
            out["crop_x_min_px"] = int(resolved.crop_x0_px)
            out["crop_y_min_px"] = int(resolved.crop_y0_px)
            out["crop_x_max_px"] = int(resolved.crop_x1_px)
            out["crop_y_max_px"] = int(resolved.crop_y1_px)
            out["crop_x_min_um"] = float(resolved.crop_x0_um)
            out["crop_y_min_um"] = float(resolved.crop_y0_um)
            out["crop_x_max_um"] = float(resolved.crop_x1_um)
            out["crop_y_max_um"] = float(resolved.crop_y1_um)

            out["orientation_policy"] = ORIENTATION_POLICY
            out["orientation_source"] = ORIENTATION_SOURCE_MASS_DISTRIBUTION
            out["no_yolk_policy"] = NO_YOLK_POLICY
            out["rotation_angle_rad"] = float(canonical.rotation_angle_rad)
            out["flip_x"] = False
            out["crop_center_um_x"] = float(canonical.crop_center_um_xy[0])
            out["crop_center_um_y"] = float(canonical.crop_center_um_xy[1])
            out["source_height_px"] = int(canonical.grid.geometry_source_shape_yx[0])
            out["source_width_px"] = int(canonical.grid.geometry_source_shape_yx[1])
            out["source_um_per_px"] = float(canonical.grid.geometry_source_um_per_px_yx[0])
            out["border_mode"] = resolved.border_mode
            out["image_interpolation"] = IMAGE_INTERPOLATION
            out["mask_interpolation"] = MASK_INTERPOLATION
            out["realized_scale_y"] = float(
                resolved.rescaled_shape_hw[0] / resolved.product_shape_hw[0]
            )
            out["realized_scale_x"] = float(
                resolved.rescaled_shape_hw[1] / resolved.product_shape_hw[1]
            )
            out["centering"] = resolved.centering
            # THIS product's compiled chain. The shared transform row holds the product-independent
            # recipe; the raster that was actually rendered is recorded here, beside the pixels.
            out["resolved_transform_chain_json"] = build_resolved_chain_payload(resolved)
            out["source_image_product_key"] = _source_image_product_key
            out["output_grid_id"] = output_grid_id(
                output_shape_yx=resolved.output_shape_hw,
                output_um_per_px_yx=canonical.grid.default_output_um_per_px_yx,
            )

            # Photometry is the recipe's job; geometry already happened above. no_change returns
            # the cropped pixels untouched, so a uint16 source stays uint16 at full scale.
            augmented = render_snip(
                image_cropped,
                snip_recipe=snip_recipe,
                mask=mask_cropped,
                background_mean=background_mean,
                background_std=background_std,
                blend_radius_um=float(blend_radius_um),
                pixel_size_um=float(target_pixel_size_um),
                # Ignored by no_change (it takes **_ignored); only clahe_blend reads it.
                use_clahe=bool(apply_clahe),
            )

            # EMBRYO FIRST, PRODUCT SECOND. The hierarchy states the ontology: this biological
            # object, and these alternative raster products OF it. Product-first
            # ({snip_product_key}/{physical_embryo_id}/) would read as "this rendering, and all the
            # embryos put through it", which inverts what is shared and what varies — the embryo-time
            # geometry is the shared fact, and BF / RFP / z_stack are sibling observations of it.
            # It also makes the common inspection question ("show me every representation of this
            # embryo") a single directory listing.
            #
            # Job ownership does not need to drive the leaf layout: each product job still owns a
            # DISJOINT product subdirectory beneath each embryo, so concurrent
            # mkdir(exist_ok=True) on the shared embryo directory is safe. Inventories stay
            # product-first, because those ARE owned per job.
            #
            # THE PRODUCT DIRECTORY HOLDS THE BYTES. Writing flat and pointing the product path
            # backward would leave the pre-migration layout as the authority and make the product
            # hierarchy decorative; the alias below is a view onto these files, never the reverse.
            embryo_snips_dir = snips_dir / physical_embryo_id / snip_product_key
            embryo_snips_dir.mkdir(parents=True, exist_ok=True)
            processed_path = embryo_snips_dir / f"{snip_id}.png"
            skio.imsave(str(processed_path), augmented, check_contrast=False)
            # FROM THE WRITTEN FILE, not from the plan. The plan says what was intended; the file
            # says what happened, and for a quantitative product that difference is the point -- a
            # uint16 source silently written as uint8 would look correct in every other column.
            out["pixel_dtype"] = str(skio.imread(str(processed_path)).dtype)

            # THE MASK IS NOT WRITTEN HERE. It is the BF segmentation mask under the canonical
            # transform -- a property of the embryo-time, not of any product -- so snip_geometry
            # writes it ONCE and this path only names it. Writing it in this loop produced one
            # byte-identical copy per product and, worse, derived it N times: nothing structurally
            # stopped two products from disagreeing.
            #
            # mask_cropped is still computed above because the snip RENDER uses it; what moved is
            # the persisted artifact, not the crop.
            embryo_mask_path = Path(str(transform_row["embryo_mask_snip_path"]))

            # Compatibility aliases at the pre-migration flat paths, for the DEFAULT BF PRODUCT
            # ONLY. The flat layout has no product dimension, so exactly one product can own it --
            # and it must be the one legacy consumers meant when they wrote those paths. Letting
            # every product try would have them fight over one alias, which the collision guard
            # correctly rejects (that is how this was caught: the RFP job failed rather than
            # silently repointing BF's alias at RFP pixels).
            #
            # Non-default products are reachable only through their product path and the
            # snip_product_key column, which is the point of the migration.
            # TODO(deprecate-legacy-snip-symlinks): drop with the alias layer.
            owns_legacy_alias = snip_product_key == DEFAULT_BF_SNIP_PRODUCT_KEY
            legacy_processed_path = legacy_flat_snip_path(
                snips_dir=snips_dir,
                physical_embryo_id=physical_embryo_id,
                filename=f"{snip_id}.png",
            )
            # Only the SNIP gets a legacy flat alias now. The mask alias pointed at a per-product
            # copy that no longer exists; consumers reach the mask through
            # embryo_mask_snip_path on the inventory row, which is the seam that made this move
            # cheap in the first place.
            for canonical_path, legacy in () if not owns_legacy_alias else (
                (processed_path, legacy_processed_path),
            ):
                link_legacy_flat_path(
                    canonical_path=canonical_path,
                    legacy_path=legacy,
                    log_manifest=legacy_alias_manifest,
                    snip_product_key=snip_product_key,
                )

            # Record the transform ONCE per embryo-time, keyed by a channel-independent id. Two
            # channels of the same embryo-time derive the SAME recipe (the transform reads the mask,
            # never image pixels), so the second one to arrive must find an identical row rather than
            # append a duplicate — and if it ever does NOT match, that is sibling drift and must fail
            # loud instead of silently keeping one of the two.
            out["snip_transform_id"] = snip_transform_id
            # The transform row already exists -- snip_geometry wrote it. Nothing to record here.

            try:
                out["snip_product_key"] = snip_product_key
                out["legacy_flat_snip_path"] = (
                    legacy_processed_path.relative_to(output_root).as_posix()
                    if owns_legacy_alias
                    else None
                )
                out["processed_snip_path"] = processed_path.relative_to(output_root).as_posix()
            except ValueError:
                out["snip_product_key"] = snip_product_key
                out["legacy_flat_snip_path"] = (
                    str(legacy_processed_path) if owns_legacy_alias else None
                )
                out["processed_snip_path"] = str(processed_path)
            try:
                embryo_mask_rel = embryo_mask_path.relative_to(output_root).as_posix()
            except ValueError:
                embryo_mask_rel = str(embryo_mask_path)
            out["embryo_mask"] = embryo_mask_rel
            out["embryo_mask_snip_path"] = embryo_mask_rel

            out["is_valid_snip"] = True

        except Exception as exc:
            out["error_message"] = f"{type(exc).__name__}: {exc}"
            out["is_valid_snip"] = False

        rows.append(out)

    # rows == [] is a legitimate outcome (well had zero valid masks, e.g. empty/dead well) —
    # pd.DataFrame([]) would produce a ZERO-COLUMN frame that crashes downstream pd.read_csv()
    # with EmptyDataError. Force the schema so an empty well still writes a valid, headered,
    # zero-row snip_inventory — the file itself IS the provenance record ("processed, found
    # nothing"); backtrack to that well's frame_masks/frame_detections shards to see why.
    result_df = pd.DataFrame(rows, columns=list(SNIP_INVENTORY_WRITE_COLUMNS))
    validate_snip_inventory(
        result_df,
        require_rendering_provenance=True,
        scope_label=str(output_csv),
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    observation = {
        "mask_sources_and_versions": mask_sources,
        "source_scope_values": source_scopes,
        "source_micrometers_per_pixel_values": sorted(
            {float(value[2]) for value in source_rows_by_mask.values()}
        ),
        "background_source_mean": float(_bg_mean),
        "background_source_std": float(_bg_std),
        "background_noise_mean": float(background_mean),
        "background_noise_std": float(background_std),
        "pixel_dtypes": sorted(
            {
                str(value)
                for value in result_df.loc[
                    result_df["pixel_dtype"].notna(), "pixel_dtype"
                ]
            }
        ),
        "rendered_snip_count": int(result_df["is_valid_snip"].astype(bool).sum()),
        "inventory_row_count": int(len(result_df)),
    }
    update_rendering_sidecar(
        rendering_sidecar_path(output_csv),
        fixed_contract=rendering_contract,
        snip_product_key=snip_product_key,
        well_observations={well_id: observation for well_id in well_ids},
    )
