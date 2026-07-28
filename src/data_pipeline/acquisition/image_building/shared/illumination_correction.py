"""Per-well multiplicative illumination correction for Keyence tile strips.

Keyence brightfield wells show a strong intensity gradient across the tile strip (observed tile means
e.g. 65 / 92 / 155). Because the image is ``observed = true_reflectance * illumination_field``, the
difference two overlapping tiles record for the SAME specimen pixels is the ratio of the illumination
at their two positions — a pure multiplicative factor. So we estimate a per-tile gain that brings each
tile onto the CENTER tile's illumination level, using the region where adjacent tiles physically
overlap.

Model (deliberately minimal): per-tile multiplicative gain ``A``, center tile ``A = 1`` (reference).
No additive/offset term and no intra-tile ramp — a single ratio per seam, estimated robustly as a
ratio of medians over the overlap band. The embryo may sit in the overlap; that is fine, because a
multiplicative field scales embryo and background identically, so embryo pixels are valid signal for
the ratio (this is exactly why an additive model would have been fragile and a multiplicative one is
not).

Granularity: ONE correction per well, constant across time. Estimated from tiles across multiple
z-slices (more paired pixels, averages out focus artifacts), then applied to every z-plane tile
pre-stitch — the FF/projection product inherits it through the focus stack.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Minimum overlap width (px) required to trust a seam's ratio; below this we skip that seam.
_MIN_OVERLAP_PX = 16
# Floor on a tile-region median before we trust a ratio (avoid divide-by-near-zero on black regions).
_MIN_MEDIAN = 1.0


def _horizontal_overlap_columns(
    left_offset_x: float,
    right_offset_x: float,
    tile_width: int,
) -> tuple[slice, slice] | None:
    """Column slices of the physical overlap between two horizontally adjacent tiles.

    ``*_offset_x`` are the tiles' x placements (same units as pixels, e.g. from the master map). The
    left tile spans ``[left_offset_x, left_offset_x + tile_width)``; the overlap is where that meets
    the right tile's span. Returns ``(left_tile_cols, right_tile_cols)`` — the overlapping columns in
    EACH tile's own pixel frame — or ``None`` if the tiles do not overlap enough to be usable.
    """
    lo = float(left_offset_x)
    ro = float(right_offset_x)
    if ro < lo:  # normalize so 'left' really is the smaller offset
        lo, ro = ro, lo
    overlap_start_global = ro
    overlap_end_global = lo + tile_width
    overlap_px = overlap_end_global - overlap_start_global
    if overlap_px < _MIN_OVERLAP_PX:
        return None
    # In the LEFT tile's frame, the overlap is its rightmost `overlap_px` columns.
    left_cols = slice(int(round(overlap_start_global - lo)), tile_width)
    # In the RIGHT tile's frame, the overlap is its leftmost `overlap_px` columns.
    right_cols = slice(0, int(round(overlap_end_global - ro)))
    return left_cols, right_cols


def estimate_well_illumination_gains(
    tile_stacks_by_z: list[dict[str, np.ndarray]],
    ordered_tile_ids: list[str],
    tile_offsets_x: dict[str, float],
    tile_width: int,
    *,
    center_tile_id: str | None = None,
) -> dict[str, float]:
    """Estimate per-tile multiplicative gains bringing each tile onto the center tile's level.

    Args:
        tile_stacks_by_z: sampled z-slices; each entry maps ``tile_id -> 2D uint16/uint8 tile image``.
            Multiple slices give more paired overlap pixels for a robust ratio.
        ordered_tile_ids: tile ids left-to-right (raster order).
        tile_offsets_x: each tile's x placement (master/prior frame), same order-defining units as px.
        tile_width: tile width in px.
        center_tile_id: the reference tile (gain fixed at 1.0). Defaults to the middle of
            ``ordered_tile_ids``.

    Returns:
        ``{tile_id: gain}`` with center gain 1.0. A tile whose seam-to-center overlap is unusable
        (too small, or a near-zero median) gets gain 1.0 (no correction) and a warning.
    """
    n = len(ordered_tile_ids)
    if n < 2:
        return {tid: 1.0 for tid in ordered_tile_ids}
    center = center_tile_id or ordered_tile_ids[n // 2]
    center_pos = ordered_tile_ids.index(center)

    gains: dict[str, float] = {center: 1.0}

    # Walk outward from center; each neighbor is matched to its inner neighbor (already center-scaled).
    order = list(range(center_pos - 1, -1, -1)) + list(range(center_pos + 1, n))
    for pos in order:
        tid = ordered_tile_ids[pos]
        inner_pos = pos + 1 if pos < center_pos else pos - 1
        inner_id = ordered_tile_ids[inner_pos]

        cols = _horizontal_overlap_columns(
            tile_offsets_x[ordered_tile_ids[min(pos, inner_pos)]],
            tile_offsets_x[ordered_tile_ids[max(pos, inner_pos)]],
            tile_width,
        )
        if cols is None:
            log.warning("illumination: tiles %s/%s do not overlap enough — gain 1.0", tid, inner_id)
            gains[tid] = 1.0
            continue
        left_is_this = pos < inner_pos
        this_cols = cols[0] if left_is_this else cols[1]
        inner_cols = cols[1] if left_is_this else cols[0]

        this_vals, inner_vals = [], []
        for zslice in tile_stacks_by_z:
            if tid in zslice and inner_id in zslice:
                this_vals.append(np.asarray(zslice[tid])[:, this_cols].ravel())
                inner_vals.append(np.asarray(zslice[inner_id])[:, inner_cols].ravel())
        if not this_vals:
            gains[tid] = 1.0
            continue
        this_med = float(np.median(np.concatenate(this_vals)))
        inner_med = float(np.median(np.concatenate(inner_vals)))
        if this_med < _MIN_MEDIAN or inner_med < _MIN_MEDIAN:
            log.warning("illumination: tile %s overlap median too low — gain 1.0", tid)
            gains[tid] = 1.0
            continue
        # Bring this tile onto the inner neighbour's (already center-referenced) level, then chain
        # through the inner tile's own gain so everything ends up on the center's scale.
        seam_gain = inner_med / this_med
        gains[tid] = seam_gain * gains[inner_id]

    return gains


def apply_gain(tile_image: np.ndarray, gain: float) -> np.ndarray:
    """Multiply a tile by ``gain``, clipping to the input dtype's range (no wraparound)."""
    if gain == 1.0:
        return tile_image
    info = np.iinfo(tile_image.dtype) if np.issubdtype(tile_image.dtype, np.integer) else None
    scaled = tile_image.astype(np.float64) * gain
    if info is not None:
        scaled = np.clip(scaled, info.min, info.max)
    return scaled.astype(tile_image.dtype)
