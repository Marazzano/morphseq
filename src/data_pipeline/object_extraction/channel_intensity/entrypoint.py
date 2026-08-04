"""Run native-grid intensity extraction for ONE well and ONE fluorescence source product.

THE I/O HALF. ``extraction.py`` is pure array work with no notion of disk; this module is the part
that knows where the pixels and the masks live and how a shard is written. Keeping them apart is why
the kernel is unit-testable against synthetic masks and why re-estimating a background later needs no
pixel reads at all.

WHAT IT JOINS, AND WHY THAT JOIN IS NOT OBVIOUS. Fluorescence has no masks of its own --
``frame_masks`` carries BF rows only, because detection and segmentation are BF-only. So an RFP frame
finds its mask through the SHARED (well_id, time_index): the BF mask at that timepoint describes the
same physical embryo in the same field of view, and the two products are on the same native grid.
That is asserted, not assumed -- ``extract_embryo_intensity_evidence`` rejects a mask whose shape
disagrees with the image, which is the failure that would occur if the fluorescence product were
materialized at a different downsample factor than BF.

RAW ONLY, by construction: this writes no corrected column and consults no null. Pooling is a
well-grain estimator and lives in ``feature_extraction/channel_intensity``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.channel_intensity.extraction import (
    ChannelIntensityError,
    extract_embryo_intensity_evidence,
)
from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle

# Physical radii for the background annulus. In MICROMETRES: the membership test is physical, so
# these do not need restating when a product's calibration changes.
DEFAULT_INNER_RADIUS_UM = 150.0
DEFAULT_OUTER_RADIUS_UM = 400.0
# Deliberately LARGER than the outer radius: a neighbour's halo reaches past its mask edge, so
# excluding only the neighbour's own footprint would leave its glow in the background estimate.
DEFAULT_EXCLUDE_RADIUS_UM = 500.0


def _load_native_image(path: Path) -> np.ndarray:
    """Read a materialized frame WITHOUT touching its values.

    ``cv2.IMREAD_UNCHANGED`` is the whole point: any other flag silently converts a 16-bit PNG to
    8-bit, which would erase the dosage signal in exactly the same way as the per-frame
    ``rescale_intensity`` this path exists to avoid -- and just as invisibly.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ChannelIntensityError(f"could not read image: {path}")
    if image.ndim == 3:
        raise ChannelIntensityError(
            f"{path} has {image.shape[2]} channels. Intensity must be measured on a single-channel "
            "quantitative raster; a colour read means the product was written for display."
        )
    return image


def run_channel_intensity(
    *,
    frame_masks_csv: Path,
    frame_inventory_csv: Path,
    output_csv: Path,
    source_image_product_key: str,
    inner_radius_um: float = DEFAULT_INNER_RADIUS_UM,
    outer_radius_um: float = DEFAULT_OUTER_RADIUS_UM,
    exclude_radius_um: float = DEFAULT_EXCLUDE_RADIUS_UM,
) -> pd.DataFrame:
    """Measure every embryo-time in one well on the native fluorescence raster."""
    masks = pd.read_csv(frame_masks_csv)
    inventory = pd.read_csv(frame_inventory_csv)

    channel_id, product_type, projection = _parse_source_key(source_image_product_key)
    frames = inventory[
        (inventory["channel_id"].astype(str) == channel_id)
        & (inventory["image_product_type"].astype(str) == product_type)
    ]
    if projection is not None:
        frames = frames[frames["projection_method"].astype(str) == projection]
    if frames.empty:
        raise ChannelIntensityError(
            f"no frame_inventory rows for {source_image_product_key!r} in {frame_inventory_csv}. "
            "The fluorescence product must be materialized before its intensity can be measured."
        )

    rows: list[dict] = []
    for _, frame in frames.iterrows():
        time_index = int(frame["time_index"])
        # dtype_max from the WRITTEN file, never from the plan: saturation is counted against the
        # container the pixels actually live in, and a wrong ceiling would silently report zero
        # clipped pixels on a saturated embryo.
        image = _load_native_image(Path(frame["image_path"]))
        dtype_max = int(np.iinfo(image.dtype).max)

        at_t = masks[masks["time_index"].astype(int) == time_index]
        neighbors = at_t.to_dict("records")

        um_per_px = float(frame["image_micrometers_per_pixel"])
        for mask_row in neighbors:
            # VALIDITY IS ASYMMETRIC BETWEEN THE TWO ROLES A MASK PLAYS, and conflating them is a
            # real bug that this run produced. As a NEIGHBOUR, an is_valid_mask=False row is still
            # kept -- an invalid mask is a fish emitting photons into someone else's background, and
            # filtering it there would admit exactly the contamination QC flagged. As a measurement
            # TARGET it must be skipped: a row whose segmentation failed has no embryo to measure.
            #
            # Measured on B02: mask m0000 is a full-frame 2304x2304 blob, area 4,987,142 px (~100x
            # a real embryo at 47k-116k), mask_confidence 0.0, is_valid_mask False. Measured as a
            # target it produced a dim, well-sized "embryo" at every timepoint -- three rows of pure
            # background dressed as data, sitting in exactly the dim end of the dosage range where
            # they would corrupt any class boundary drawn there.
            if not bool(mask_row.get("is_valid_mask", True)):
                continue
            target = _decode(mask_row)
            if not target.any():
                continue

            evidence = extract_embryo_intensity_evidence(
                image=image,
                target_mask=target,
                target_mask_id=str(mask_row["mask_id"]),
                neighbors=neighbors,
                decode=_decode,
                um_per_px_yx=(um_per_px, um_per_px),
                inner_radius_um=inner_radius_um,
                outer_radius_um=outer_radius_um,
                exclude_radius_um=exclude_radius_um,
                dtype_max=dtype_max,
            )
            evidence.update(
                experiment_id=str(mask_row["experiment_id"]),
                well_id=str(mask_row["well_id"]),
                time_index=time_index,
                track_id=str(mask_row.get("track_id", "")),
                mask_id=str(mask_row["mask_id"]),
                source_image_product_key=source_image_product_key,
                source_image_id=str(frame["image_id"]),
                # Carried so a post-hoc exposure table can be joined. Exposure and gain are NOT in
                # frame_inventory today, so cross-experiment dosage comparison stays unsupported.
                image_micrometers_per_pixel=um_per_px,
                elapsed_time_s=float(frame.get("elapsed_time_s", float("nan"))),
            )
            rows.append(evidence)

    if not rows:
        raise ChannelIntensityError(
            f"measured no embryos for {source_image_product_key!r}. A well with masks and frames "
            "that yields no rows means the join found no overlapping timepoints."
        )

    frame_out = pd.DataFrame(rows)
    # Histograms are lists; CSV would stringify them inconsistently across pandas versions and the
    # pooling step needs them back as exact integer arrays.
    for column in ("annulus_hist_counts", "embryo_hist_counts"):
        if column in frame_out:
            frame_out[column] = frame_out[column].map(lambda v: json.dumps(list(map(int, v))))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    frame_out.to_csv(temp, index=False)
    temp.replace(output_csv)
    return frame_out


def _decode(mask_row) -> np.ndarray:
    """Decode a frame_masks ROW to a bool mask.

    TAKES THE ROW, NOT THE RLE CELL. ``neighbor_union_mask`` hands its ``decode`` callable a whole
    neighbour record -- it already needs the row for the bbox prefilter -- so a decoder that expected
    a bare payload would work for the target and blow up only on wells that actually HAVE a
    neighbour. That is exactly how this surfaced: three single-embryo wells passed and the one
    multi-embryo well failed, because it was the only one to reach the neighbour path at all.

    The payload carries its own shape, so no shape argument is passed. A CSV round-trip turns the RLE
    mapping into a JSON string, so both forms are accepted rather than making every caller track
    which side of the round-trip it is on.
    """
    rle = mask_row["mask_rle"] if hasattr(mask_row, "__getitem__") else mask_row
    if isinstance(rle, str):
        rle = json.loads(rle)
    return decode_binary_mask_rle(rle).astype(bool)


def _parse_source_key(key: str) -> tuple[str, str, str | None]:
    """``RFP__projection__max`` -> ``("RFP", "projection", "max")``; ``BF__z_stack`` -> no method."""
    parts = key.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    raise ChannelIntensityError(
        f"unparseable source_image_product_key {key!r}; expected channel__type[__method]."
    )
