"""Experiment-grain Keyence stitch-map builder.

Builds ONE experiment-wide tile-offset calibration from the Keyence acquisition inventory and
writes it as a ``master_params.json`` consumable by ``PreComputeStitchParams(master_params_path=...)``.

Feature-based stitch2d alignment fails on most Keyence brightfield wells (low texture) — on a 96-well
plate only ~3% align all 3 tiles. So the builder does NOT trust any single well: it derives a stage
prior, runs alignment across every well keeping whatever placed (partials included), rejects fits that
stray from the prior, and averages the survivors into one calibration (see Stages A–C below). Tiles
that never aligned fall back to the stage prior, so the map is always complete.

The builder is called ONCE per experiment (before any per-well materialization). Per-well
materialization then loads the pre-computed coords via ``run_align=False`` (the "generate once,
reapply many" contract). See ``keyence_wire_through.md`` Stage C for the full design rationale.

Import rules: imports image primitives from ``image_building/``, the Keyence acquisition-inventory
validator, and nothing from orchestration/tasks/Snakemake.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.acquisition.image_building.shared.focus_stack_group import (
    FocusStackConfig,
    focus_stack_group,
)
from data_pipeline.shared.path_roots import resolve_under_input_root
from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    TileSpec,
    raw_stitch2d_align,
)

log = logging.getLogger(__name__)


def build_keyence_stitch_map(
    acquisition_inventory_df: pd.DataFrame,
    *,
    n_samples: int = 50,
    out_path: Path,
    input_root: Path | None = None,
) -> None:
    """Build the experiment-grain Keyence stitch map and write it to ``out_path``.

    Runs the A–C calibration over every well (focus-stacking each tile's Z planes, aligning, filtering
    fits against the stage prior, averaging survivors) and writes the resulting per-tile-index
    ``[y, x]`` offsets as ``{"metadata": {...}, "coords": {index: [y, x], ...}}`` JSON. The map always
    covers all tiles — indices with no trusted fit fall back to the stage prior — so unlike the prior
    feature-only builder this never fails for lack of a fully-aligned sample. A companion
    ``*__inferred_shifts.csv`` records every successful per-well fit for post-hoc calibration review.

    ``n_samples`` is accepted for signature/CLI compatibility but no longer caps the sweep.
    """
    if acquisition_inventory_df.empty:
        raise ValueError("acquisition_inventory_df is empty — nothing to sample.")

    orientation_raw = str(
        acquisition_inventory_df["orientation"].mode().iloc[0]
    ).lower()
    # Modern Keyence exports often have no explicit orientation in TIFF metadata; legacy behavior
    # treats unknown/non-vertical layouts as horizontal strips.
    orientation = "vertical" if orientation_raw == "vertical" else "horizontal"

    # Hierarchical calibration (Stages A–C). Feature-based stitch2d alignment fails on most Keyence
    # brightfield wells (low texture), so a single well is never trusted to produce the whole map.
    # Instead: derive the per-tile stage prior (A), run alignment on every well keeping whatever
    # placed and re-anchoring to the center tile (B), then reject fits that stray from the prior and
    # average the survivors into one experiment-wide calibration (C). ``n_samples`` is retained for
    # signature compatibility but no longer caps the sweep — every well feeds the calibration.
    prior = stage_prior_offsets(acquisition_inventory_df)
    fits = collect_center_anchored_fits(
        acquisition_inventory_df,
        orientation=orientation,
        input_root=input_root,
        record_path=(out_path.with_name(out_path.stem + "__inferred_shifts.csv")),
    )
    calibrated = calibrate_tile_offsets(fits, prior)  # complete: index -> [y, x], center at 0

    # Stage D: the calibration already spans every tile index (averaged fit where trusted, stage
    # prior where not), so it IS the gap-filled per-tile map. Assemble it into the stitch2d coord
    # array in tile-index order.
    #
    # Frame convention: calibration is CENTER-anchored (center tile -> [0, 0]), but stitch2d
    # re-normalizes any loaded params to a MIN-origin frame (subtracts the per-axis minimum so the
    # leftmost/topmost tile sits at 0). Downstream materialization QC compares the stitched
    # transforms against THIS written map, so the map must already be in that min-origin frame — else
    # every well reads as ~one-tile-pitch off ("deviates_from_master") even when stitched perfectly.
    # Shift each axis by its minimum here so the written map matches stitch2d's normalization.
    n_tiles = len(prior)
    stacked = np.stack([calibrated[idx] for idx in range(n_tiles)], axis=0)  # (n_tiles, 2) [y, x]
    stacked = stacked - stacked.min(axis=0, keepdims=True)  # -> min-origin frame per axis
    coords: dict[str, list[float]] = {
        str(idx): [float(stacked[idx, 0]), float(stacked[idx, 1])]
        for idx in range(n_tiles)
    }

    # tile_shape for metadata: build one sample frame's tiles (any well) to read the tile pixel
    # dimensions the coords are expressed against.
    tile_shape: list[int] = []
    first_pair = (
        acquisition_inventory_df[["well_id", "time_index"]].drop_duplicates().iloc[0]
    )
    try:
        specs = _build_tile_specs(
            acquisition_inventory_df[
                (acquisition_inventory_df["well_id"] == first_pair["well_id"])
                & (acquisition_inventory_df["time_index"] == first_pair["time_index"])
            ],
            input_root=input_root,
        )
        tile_shape = list(specs[0].image.shape[:2])
    except Exception as exc:  # non-fatal: metadata-only field
        log.warning("build_keyence_stitch_map: could not read tile_shape: %s", exc)

    # Stage E: write in the exact schema materialization consumes (metadata + index-keyed [y,x]).
    shape = [n_tiles, 1] if orientation == "vertical" else [1, n_tiles]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "shape": shape,
                    "size": n_tiles,
                    "tile_shape": tile_shape,
                },
                "coords": coords,
            }
        )
    )
    log.info(
        "build_keyence_stitch_map: wrote %d tile coords to %s (calibrated from %d wells)",
        len(coords), out_path, len(fits),
    )


# ======================================================================================
# Hierarchical stitch calibration (Stages A–C).
#
# Feature-based stitch2d alignment fails on most Keyence brightfield wells (low texture): on a
# 96-well plate only ~3% align all 3 tiles, ~80% align exactly 2, ~17% align none. But the tiles
# that DO align place consistently, and the stage records the true mosaic geometry exactly. So we:
#   (A) derive the per-tile prior offset (relative to the center tile) from stage coordinates,
#   (B) run alignment on every well, keeping whatever placed (partials included), re-anchored to
#       the center tile so all wells share one coordinate frame,
#   (C) reject any placed tile whose offset deviates > threshold from the stage prior, then average
#       the survivors per tile index -> an experiment-wide calibration.
# (Stages D/E — applying the calibration to fill gaps and writing the map — are separate.)
#
# Coordinate convention throughout: stitch2d [y, x], keyed by 0-based tile INDEX (the order
# _build_tile_specs returns, i.e. tiles sorted by tile_id). The center is the middle index.
# ======================================================================================

# Max pixel deviation (in EITHER axis) a fit may differ from the stage prior before it is rejected
# from the calibration average. Flat 50 px for all tiles; tunable — revise after a test run.
DEFAULT_CALIBRATION_THRESHOLD_PX = 10.0


def _center_index(n_tiles: int) -> int:
    """The middle tile index (the anchor that aligns most reliably)."""
    return n_tiles // 2


def stage_prior_offsets(
    sample_rows: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Stage A: per-tile prior offset in pixels [y, x], relative to the center tile.

    The stage position is constant across Z/channel/time within a tile, so one (stage_x, stage_y)
    per tile_id fully specifies the mosaic geometry. Converted nm -> um -> px via
    ``micrometers_per_pixel``. Returned keyed by 0-based tile index (tiles sorted by tile_id),
    center index -> [0, 0].
    """
    per_tile = (
        sample_rows.groupby("tile_id")[["stage_x_nm", "stage_y_nm", "micrometers_per_pixel"]]
        .first()
        .sort_index()
    )
    umpp = float(per_tile["micrometers_per_pixel"].iloc[0])
    if umpp <= 0:
        raise ValueError(f"micrometers_per_pixel must be positive, got {umpp!r}.")
    # nm -> um -> px; stitch2d uses [y, x], stage columns are (x, y).
    # Sign flip: stitch2d's placed coords run OPPOSITE to increasing stage position — verified
    # empirically, the raw fits are the exact negative of the stage offset (fit x ~ -678 where
    # stage prior x = +672). Negate so the prior shares stitch2d's convention and fits pass the
    # deviation filter.
    xy_px = -(per_tile[["stage_x_nm", "stage_y_nm"]].to_numpy() / 1000.0 / umpp)
    yx_px = xy_px[:, ::-1]  # -> [y, x]
    center = yx_px[_center_index(len(yx_px))]
    return {i: (yx_px[i] - center) for i in range(len(yx_px))}


def collect_center_anchored_fits(
    acquisition_inventory_df: pd.DataFrame,
    *,
    orientation: str,
    input_root: Path | None = None,
    record_path: Path | None = None,
) -> list[dict[int, np.ndarray]]:
    """Stage B: run alignment on every well; return each well's placed tiles as center-anchored
    offsets [y, x] keyed by tile index.

    Keeps partial alignments (2/3, 1/3) as-is — the whole point is to harvest whatever placed.
    A well is usable only if its center tile placed (otherwise there is no shared anchor); wells
    where the center did not place are dropped. Offsets are expressed relative to the center tile
    so all wells live in one coordinate frame for averaging in Stage C.

    ``record_path`` (optional, non-invasive): if given, every placed tile's inferred center-anchored
    shift is appended to a CSV at that path (columns: well_id, time_index, tile_index, dy, dx,
    n_tiles_placed). Purely a diagnostic side-channel — it does not affect the returned fits or any
    downstream behavior; omit it and nothing is written.
    """
    pairs = (
        acquisition_inventory_df[["well_id", "time_index"]].drop_duplicates().reset_index(drop=True)
    )
    fits: list[dict[int, np.ndarray]] = []
    records: list[dict] = []
    for _, row in pairs.iterrows():
        sample_rows = acquisition_inventory_df[
            (acquisition_inventory_df["well_id"] == row["well_id"])
            & (acquisition_inventory_df["time_index"] == row["time_index"])
        ]
        try:
            tile_specs = _build_tile_specs(sample_rows, input_root=input_root)
            coords_raw = raw_stitch2d_align(tile_specs, orientation=orientation)
        except Exception as exc:
            log.debug("well=%s skipped (build/align): %s", row["well_id"], exc)
            continue
        center = _center_index(len(tile_specs))
        if center not in {int(k) for k in coords_raw}:
            log.debug("well=%s center tile did not place — skipped", row["well_id"])
            continue
        c = np.asarray(coords_raw[center], dtype=float)
        fit = {int(k): np.asarray(v, dtype=float) - c for k, v in coords_raw.items()}
        fits.append(fit)
        if record_path is not None:
            for tidx, off in fit.items():
                records.append({
                    "well_id": row["well_id"], "time_index": row["time_index"],
                    "tile_index": tidx, "dy": float(off[0]), "dx": float(off[1]),
                    "n_tiles_placed": len(fit),
                })
    log.info("collect_center_anchored_fits: %d/%d wells contributed usable fits", len(fits), len(pairs))
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(record_path, index=False)
        log.info("collect_center_anchored_fits: recorded %d inferred shifts to %s", len(records), record_path)
    return fits


def calibrate_tile_offsets(
    fits: list[dict[int, np.ndarray]],
    prior: dict[int, np.ndarray],
    *,
    threshold_px: float = DEFAULT_CALIBRATION_THRESHOLD_PX,
) -> dict[int, np.ndarray]:
    """Stage C: filter fits against the stage prior and average the survivors per tile index.

    For each tile index, a fit is rejected if its center-anchored offset deviates from the prior by
    more than ``threshold_px`` pixels in EITHER axis (y or x). Survivors are averaged to an
    experiment-wide calibration offset [y, x] per tile index.

    Returns the calibration keyed by tile index. A tile index with zero surviving fits falls back to
    the stage prior for that tile (so the calibration is always complete).
    """
    calibrated: dict[int, np.ndarray] = {}
    for idx, prior_off in prior.items():
        kept = [
            fit[idx] for fit in fits
            if idx in fit and np.all(np.abs(fit[idx] - prior_off) <= threshold_px)
        ]
        if kept:
            calibrated[idx] = np.mean(np.stack(kept, axis=0), axis=0)
            log.info("tile idx %d: averaged %d fits (of %d) within %.0f px of prior",
                     idx, len(kept), len(fits), threshold_px)
        else:
            calibrated[idx] = prior_off.copy()
            log.warning("tile idx %d: no fits within threshold — falling back to stage prior", idx)
    return calibrated


def _build_tile_specs(
    sample_rows: pd.DataFrame, *, input_root: Path | None = None
) -> list[TileSpec]:
    """Focus-stack a sample frame's tiles (ONE shared-bounds group) and return TileSpecs.

    Uses the same shared focus-projection route as ``materialize_well_keyence`` so the coords
    are computed on the exact tile pixels production will stitch — one 0.1–99.9% intensity
    range across all tiles/Z planes, no per-tile normalization.
    """
    ordered_tile_ids: list[str] = []
    raw_stacks: list[np.ndarray] = []
    for tile_id, tile_rows in sample_rows.groupby("tile_id"):
        z_paths = (
            tile_rows.sort_values("z_index")["source_tiff_path"]
            .map(lambda p: resolve_under_input_root(
                p, input_root=input_root, scope_label="Keyence acquisition inventory",
                full_root_fallback=True,
            ))
            .tolist()
        )
        if not z_paths:
            raise ValueError(f"No z-plane TIFFs for tile_id={tile_id!r}.")
        planes = [skio.imread(str(p)) for p in z_paths]
        ordered_tile_ids.append(str(tile_id))
        raw_stacks.append(np.stack(planes, axis=0))

    group = focus_stack_group(raw_stacks, config=FocusStackConfig(), device="cpu")
    return [
        TileSpec(tile_id=tid, image=group.tiles[i].projection_u8)
        for i, tid in enumerate(ordered_tile_ids)
    ]
