"""Per-well snip geometry: derive the canonical transform ONCE per embryo-time.

THE GATE. Every snip product of one embryo-time shares one physical crop recipe — the same center,
angle, and orientation decision — because that recipe is a fact about the ANIMAL, derived from its
segmentation mask, not about any channel. This step derives it once and persists it; render jobs
deserialize it and resolve it onto their own pixel grid.

WHY NOT LET EACH PRODUCT DERIVE ITS OWN. An earlier design did, on the grounds that derivation is
deterministic from identical mask inputs so one IMPLEMENTATION sufficed. The hole: determinism is
only as strong as the inputs being identical, and nothing enforced that. Two product jobs straddling
a frame_masks regeneration, a mask revision, or an orientation-policy default change would derive
different geometry and produce SILENTLY UNREGISTERABLE siblings — no error, no failed job, just a
BF and an RFP snip of one embryo that no longer line up, discovered much later in a composite
overlay. The gate removes that failure mode structurally rather than testing for its absence.

NO IMAGE PIXELS ARE READ HERE. The rotation angle comes from the mask's PCA orientation and the crop
center from its extent, so this step needs frame_masks and the registry and nothing else. That is
also why an RFP render can run in PARALLEL with the BF one: neither waits on the other's artifacts,
only on this table.

Fanout is per-well with NO product wildcard, deliberately — one embryo-time has one transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.object_extraction.snip_processing.snip_transform import (
    apply_transform_to_mask,
    derive_snip_transform,
    transform_for_product,
)
from data_pipeline.object_extraction.snip_processing.snip_transform_table import (
    build_snip_transform_row,
    mask_content_fingerprint,
    write_snip_transform_table,
)
from data_pipeline.shared.identifiers.constructors import build_snip_transform_id
from data_pipeline.shared.identifiers.parsers import parse_image_id
from data_pipeline.object_extraction.snip_processing.defaults import (
    DEFAULT_TARGET_PIXEL_SIZE_UM,
)

# The orientation DECISION recorded on every row, so a later yolk-aware pass knows what it replaces.
ORIENTATION_POLICY = "pca_major_axis_yolk_down"
ORIENTATION_SOURCE_MASS_DISTRIBUTION = "embryo_mass_distribution"
NO_YOLK_POLICY = "fallback_mass_distribution"


def _physical_embryo_id_by_track(registry: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (str(row["well_id"]), str(row["track_id"])): str(row["physical_embryo_id"])
        for _, row in registry.iterrows()
    }


def run_snip_geometry(
    *,
    frame_masks_csv: Path,
    frame_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_path: Path,
    target_pixel_size_um: float = DEFAULT_TARGET_PIXEL_SIZE_UM,
    output_height_px: int = 576,
    output_width_px: int = 256,
) -> None:
    """Derive and persist one canonical transform per (physical_embryo_id, time_index)."""
    frame_masks = pd.read_csv(frame_masks_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    valid_masks = frame_masks[frame_masks["is_valid_mask"].astype(bool)].copy()
    inventory_index = frame_inventory.set_index("image_id")
    physical_embryo_id_by_track = _physical_embryo_id_by_track(registry)
    snip_frame_shape_hw = (int(output_height_px), int(output_width_px))
    # ONE PLACE FOR MASKS, beside the transforms that define them and NOT inside any product
    # directory. Mirrors the snip-grain convention the auxiliary families use
    # (per_well/{well}/masks/{mask_type}/), so the embryo mask lands where a reader already looks
    # for masks rather than beside the pixels it happens to be aligned with.
    mask_dir = Path(output_path).parent / "masks" / "embryo_mask"

    rows: dict[str, dict[str, Any]] = {}
    for _, mask_row in valid_masks.iterrows():
        image_id = str(mask_row["image_id"])
        well_id = str(mask_row["well_id"])
        track_id = str(mask_row["track_id"])
        _, _channel_id, time_index = parse_image_id(image_id)

        physical_embryo_id = physical_embryo_id_by_track.get((well_id, track_id))
        if physical_embryo_id is None:
            raise ValueError(
                f"No physical_embryo_registry entry for (well_id={well_id!r}, "
                f"track_id={track_id!r}) — every valid frame_masks track must be registered. "
                f"Rebuild the physical_embryo_registry shard for {well_id!r} from the SAME "
                "frame_masks shard before running snip_geometry."
            )

        snip_transform_id = build_snip_transform_id(physical_embryo_id, int(time_index))
        if snip_transform_id in rows:
            # frame_masks holds BF rows only, so one embryo-time appears once. A repeat means the
            # shard has duplicate masks for one track+time, which would make the transform
            # ambiguous — fail rather than let last-write-win pick a geometry.
            raise ValueError(
                f"snip_geometry: duplicate mask rows for {snip_transform_id!r} in "
                f"{frame_masks_csv}. One embryo-time must have exactly one canonical transform."
            )

        if image_id not in inventory_index.index:
            raise KeyError(f"image_id {image_id!r} not found in frame_inventory")
        pixel_size_um = float(inventory_index.loc[image_id]["image_micrometers_per_pixel"])

        embryo_mask = decode_binary_mask_rle(json.loads(str(mask_row["mask_rle"]))).astype(np.uint8)

        # No yolk mask yet; the angle falls back to the mass-distribution heuristic. The POLICY is
        # recorded on the row so a later yolk-aware pass knows exactly what it is replacing.
        canonical = derive_snip_transform(
            embryo_mask,
            source_um_per_px=pixel_size_um,
            target_um_per_px=float(target_pixel_size_um),
            snip_frame_shape_hw=snip_frame_shape_hw,
        )
        # THE EMBRYO MASK, WRITTEN ONCE, HERE. It is the BF segmentation mask carried through the
        # canonical transform -- derived from the same mask and the same recipe this step already
        # owns, and about the embryo rather than about any channel. Writing it in the render loop
        # (where it used to live) produced one byte-identical copy PER PRODUCT, and worse, derived
        # it N times: nothing structurally stopped two products from disagreeing, which is the exact
        # drift this gate exists to close for placement.
        #
        # No product dimension and no grid policy. The canonical transform is grid-independent, so
        # any consumer wanting another grid re-resolves it -- that is what transform_for_product is
        # for. This writes the mask on the grid the materialization plan asked for, the same grid
        # this step already derives transforms against.
        resolved = transform_for_product(
            canonical,
            product_shape_hw=embryo_mask.shape[:2],
            product_um_per_px=pixel_size_um,
        )
        mask_path = mask_dir / f"{snip_transform_id}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        skio.imsave(
            str(mask_path),
            (apply_transform_to_mask(embryo_mask, resolved) > 0).astype(np.uint8) * 255,
            check_contrast=False,
        )

        rows[snip_transform_id] = build_snip_transform_row(
            snip_transform_id=snip_transform_id,
            physical_embryo_id=physical_embryo_id,
            time_index=int(time_index),
            canonical=canonical,
            source_image_id=image_id,
            mask_id=str(mask_row["mask_id"]),
            orientation_policy=ORIENTATION_POLICY,
            orientation_source=ORIENTATION_SOURCE_MASS_DISTRIBUTION,
            no_yolk_policy=NO_YOLK_POLICY,
            # Over the DECODED mask: a re-encode changes the RLE string without changing a pixel,
            # so hashing the string would fail on a no-op rewrite.
            geometry_source_mask_sha256=mask_content_fingerprint(embryo_mask),
            # The row NAMES its own mask, so a consumer never rebuilds the filename from a
            # convention it would then be coupled to.
            embryo_mask_snip_path=str(mask_path),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # write_snip_transform_table validates BEFORE the atomic rename, so an invalid table never
    # becomes visible — downstream treats existence as done.
    write_snip_transform_table(pd.DataFrame(list(rows.values())), output_path)
