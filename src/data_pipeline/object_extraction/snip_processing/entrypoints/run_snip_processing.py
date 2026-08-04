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

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_INVENTORY_COLUMNS,
)
from data_pipeline.shared.identifiers.constructors import (
    build_embryo_id,
    build_snip_id,
    build_snip_transform_id,
)
from data_pipeline.shared.identifiers.parsers import parse_image_id
from data_pipeline.object_extraction.snip_processing.augmentation import augment_snip
from data_pipeline.object_extraction.snip_processing.legacy_snip_paths import (
    legacy_flat_snip_path,
    link_legacy_flat_path,
)
from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
)
from data_pipeline.object_extraction.snip_processing.snip_transform import (
    CENTERING_LATCHED,
    IMAGE_INTERPOLATION,
    MASK_INTERPOLATION,
    apply_transform_to_image,
    apply_transform_to_mask,
    derive_snip_transform,
    transform_for_product,
)
from data_pipeline.object_extraction.snip_processing.snip_transform_table import (
    SNIP_TRANSFORM_TABLE_COLUMNS,
    build_snip_transform_row,
    write_snip_transform_table,
)

# The orientation DECISION recorded on every row, so a later yolk-aware pass knows what it replaces.
# These are the CURRENT policy, not a configuration surface: there is exactly one orientation engine
# today. When a yolk-aware source lands it adds a value here rather than silently changing meaning.
ORIENTATION_POLICY = "pca_major_axis_yolk_down"
ORIENTATION_SOURCE_MASS_DISTRIBUTION = "embryo_mass_distribution"
NO_YOLK_POLICY = "fallback_mass_distribution"


class SnipRecipeDtypeError(ValueError):
    """A source frame's dtype does not match what the rendering recipe requires.

    Raised per row rather than per job on purpose: the caller catches Exception per mask and marks
    ``is_valid_snip=False`` with a reason, so one non-uint8 frame invalidates its own snip and is
    reported, instead of taking down a well that also holds perfectly good BF rows.
    """

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


def _estimate_background(
    valid_masks: pd.DataFrame,
    inventory_index: pd.DataFrame,
    n_samples: int = 50,
    seed: int = 309,
) -> tuple[float, float]:
    """Sample background pixels (outside embryo mask) to estimate mean/std.

    Matches the legacy build03A definition: pixels in the full-frame source
    image where the embryo mask == 0.
    """
    np.random.seed(seed)
    indices = valid_masks.index.tolist()
    sample_idx = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

    bkg_pixels: list[float] = []
    for i in sample_idx:
        row = valid_masks.loc[i]
        image_id = str(row["image_id"])
        if image_id not in inventory_index.index:
            continue
        try:
            src = Path(str(inventory_index.loc[image_id]["image_path"]))
            img = skio.imread(str(src))
            if img.ndim == 3:
                img = img[:, :, 0]
            rle = json.loads(str(row["mask_rle"]))
            mask = decode_binary_mask_rle(rle).astype(bool)
            bkg = img[~mask].astype(float)
            if bkg.size > 0:
                bkg_pixels.extend(bkg[:5000].tolist())
        except Exception:
            continue

    if not bkg_pixels:
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
    snip_transform_table_path: Path | None = None,
) -> None:
    frame_masks = pd.read_csv(frame_masks_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    physical_embryo_registry = pd.read_csv(physical_embryo_registry_csv)

    valid_masks = frame_masks[frame_masks["is_valid_mask"].astype(bool)].copy()
    inventory_index = frame_inventory.set_index("image_id")

    # Identity is JOINED from the registry, never minted here. A valid mask whose track has no
    # registry row is a contract violation (the registry is built from frame_masks, so it must
    # cover every detected track) — fail loud rather than silently drop a real embryo.
    physical_embryo_id_by_track = _physical_embryo_id_by_track(physical_embryo_registry)

    output_shape = (output_height_px, output_width_px)
    snips_dir = Path(snips_dir)
    output_root = Path(output_root)

    # The product this entrypoint renders. SINGLE-VALUED TODAY: this path is the historical BF
    # clahe_blend renderer, and naming that explicitly is what lets the path, the inventory column,
    # and the compatibility alias all agree. P2 turns it into a parameter.
    snip_product_key = DEFAULT_BF_SNIP_PRODUCT_KEY
    # Every alias created this run, for debugging external breakage without archaeology.
    legacy_alias_manifest: list[dict[str, str]] = []

    _bg_mean, _bg_std = _estimate_background(valid_masks, inventory_index)
    background_mean = background_noise_scale * _bg_mean
    background_std = background_noise_scale * _bg_std

    rows: list[dict[str, Any]] = []
    # Collected inside the loop, reconciled after it. Reconciliation is keyed by snip_transform_id so
    # sibling products of one embryo-time collapse onto ONE row — that collapse is the mechanism
    # that makes duplicate (and therefore driftable) geometry unrepresentable.
    pending_transform_rows: list[tuple[str, dict[str, Any]]] = []

    for _, mask_row in valid_masks.iterrows():
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
            "image_path": None,
            "processed_snip_path": None,
            # Product identity is known BEFORE rendering, so a failed row still says which product
            # it was trying to be -- otherwise a failure is unattributable once several products
            # share a well. The paths stay null: no bytes, no alias.
            "snip_product_key": snip_product_key,
            "legacy_flat_snip_path": None,
            "embryo_mask": None,
            "embryo_mask_snip_path": None,
            "crop_x_min_px": None,
            "crop_y_min_px": None,
            "crop_x_max_px": None,
            "crop_y_max_px": None,
            "crop_width_px": output_width_px,
            "crop_height_px": output_height_px,
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
            if image_id not in inventory_index.index:
                raise KeyError(f"image_id {image_id!r} not found in frame_inventory")
            inv_row = inventory_index.loc[image_id]

            image_path = Path(str(inv_row["image_path"]))
            pixel_size_um = float(inv_row["image_micrometers_per_pixel"])
            out["image_path"] = str(inv_row["image_path"])

            # Decode RLE mask from frame_masks row.
            rle = json.loads(str(mask_row["mask_rle"]))
            embryo_mask = decode_binary_mask_rle(rle).astype(np.uint8)

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
            if image.dtype != np.uint8:
                raise SnipRecipeDtypeError(
                    f"snip_processing: legacy BF snip rendering requires a uint8 source; got "
                    f"{image.dtype} from {image_path}. Non-uint8 frames must be rendered through a "
                    "dtype-preserving snip recipe (no_change), not coerced here -- the previous "
                    "per-frame rescale_intensity(in_range='image') destroyed absolute intensity "
                    "and therefore any cross-embryo comparison."
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
            # No yolk mask yet — the angle falls back to the mass-distribution heuristic.
            canonical = derive_snip_transform(
                embryo_mask,
                source_um_per_px=pixel_size_um,
                target_um_per_px=target_pixel_size_um,
                snip_frame_shape_hw=output_shape,
                yolk_mask=None,
            )
            resolved = transform_for_product(
                canonical,
                product_shape_hw=(int(image.shape[0]), int(image.shape[1])),
                product_um_per_px=pixel_size_um,
                centering=CENTERING_LATCHED,
            )
            image_cropped = apply_transform_to_image(image, resolved, dtype=np.uint8)
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

            augmented, _ = augment_snip(
                image_cropped,
                mask_cropped,
                background_mean,
                background_std,
                blend_radius_um=float(blend_radius_um),
                pixel_size_um=float(target_pixel_size_um),
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

            # Persist the cropped embryo mask in the SAME snip coordinate space as the snip image
            # (same crop transform, so they are pixel-aligned by construction). This is the embryo
            # mask fraction_alive ANDs against the per-snip via mask — no model, no re-prediction.
            embryo_mask_path = embryo_snips_dir / f"{snip_id}_embryo.png"
            skio.imsave(str(embryo_mask_path), (mask_cropped > 0).astype(np.uint8) * 255, check_contrast=False)

            # Compatibility aliases at the pre-migration flat paths. Both artifacts get one, since a
            # reader holding an old snip path may hold an old mask path too.
            # TODO(deprecate-legacy-snip-symlinks): drop with the alias layer.
            legacy_processed_path = legacy_flat_snip_path(
                snips_dir=snips_dir,
                physical_embryo_id=physical_embryo_id,
                filename=f"{snip_id}.png",
            )
            for canonical, legacy in (
                (processed_path, legacy_processed_path),
                (
                    embryo_mask_path,
                    legacy_flat_snip_path(
                        snips_dir=snips_dir,
                        physical_embryo_id=physical_embryo_id,
                        filename=f"{snip_id}_embryo.png",
                    ),
                ),
            ):
                link_legacy_flat_path(
                    canonical_path=canonical,
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
            transform_row = build_snip_transform_row(
                snip_transform_id=snip_transform_id,
                physical_embryo_id=physical_embryo_id,
                time_index=int(time_index),
                resolved=resolved,
                source_image_id=image_id,
                mask_id=mask_id,
                orientation_policy=ORIENTATION_POLICY,
                orientation_source=ORIENTATION_SOURCE_MASS_DISTRIBUTION,
                no_yolk_policy=NO_YOLK_POLICY,
                flip_x=False,
            )
            # Deferred to after the loop: a sibling-drift disagreement is a CONTRACT violation, not
            # a per-snip render failure, so it must not be caught by this row's except and buried in
            # error_message as if that one snip were merely bad.
            pending_transform_rows.append((snip_transform_id, transform_row))

            try:
                out["snip_product_key"] = snip_product_key
                out["legacy_flat_snip_path"] = legacy_processed_path.relative_to(
                    output_root
                ).as_posix()
                out["processed_snip_path"] = processed_path.relative_to(output_root).as_posix()
            except ValueError:
                out["snip_product_key"] = snip_product_key
                out["legacy_flat_snip_path"] = str(legacy_processed_path)
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
    result_df = pd.DataFrame(rows, columns=list(SNIP_INVENTORY_COLUMNS))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    # Reconcile the per-embryo-time transforms and write the well's table. Sibling products derive
    # the SAME recipe (the transform reads the mask, never image pixels), so a second arrival must
    # match the first exactly; a mismatch means the derivation is not channel-independent and the
    # snips would not be pixel-registerable, which is a contract violation rather than a bad row.
    if snip_transform_table_path is not None:
        transform_rows: dict[str, dict[str, Any]] = {}
        for transform_id, transform_row in pending_transform_rows:
            existing = transform_rows.get(transform_id)
            if existing is None:
                transform_rows[transform_id] = transform_row
                continue
            if existing["resolved_transform_chain_json"] != transform_row["resolved_transform_chain_json"]:
                raise ValueError(
                    f"snip_transform_id {transform_id!r} was derived twice with DIFFERENT geometry. "
                    "Sibling products of one embryo-time must share one transform, so this means "
                    "the derivation is not channel-independent and the snips would not be "
                    "pixel-registerable. Check that both products carry the same mask and "
                    "calibration."
                )

        # A well with zero valid masks still writes a headered, zero-row table: the file itself is
        # the record that the well was processed and had no transforms, exactly as the zero-row
        # snip_inventory above is.
        write_snip_transform_table(
            pd.DataFrame(
                list(transform_rows.values()),
                columns=list(SNIP_TRANSFORM_TABLE_COLUMNS),
            ),
            Path(snip_transform_table_path),
        )
