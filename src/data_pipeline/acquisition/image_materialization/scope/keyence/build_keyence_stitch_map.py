"""Experiment-grain Keyence stitch-map builder.

Samples up to ``n_samples`` ``(well_id, time_index)`` pairs from the Keyence acquisition
inventory, aligns each set of tiles, takes the **median** per-tile transform, and writes a
``master_params.json`` consumable by ``PreComputeStitchParams(master_params_path=...)``.

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

from data_pipeline.acquisition.image_building.shared.log_focus import im_rescale
from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    TileSpec,
    raw_stitch2d_align,
)
from data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1 import (
    materialize_ff_projection,
)

log = logging.getLogger(__name__)


def build_keyence_stitch_map(
    acquisition_inventory_df: pd.DataFrame,
    *,
    n_samples: int = 50,
    out_path: Path,
) -> None:
    """Build the experiment-grain Keyence stitch map and write it to ``out_path``.

    Samples up to ``n_samples`` ``(well_id, time_index)`` pairs from the inventory (fixed seed
    for determinism), focus-stacks each tile's Z planes, runs stitch2d alignment, and collects
    per-tile ``(dx_px, dy_px)`` transforms. The median across all good samples is written as
    ``{"coords": {tile_id: [median_x, median_y], ...}}`` JSON.

    Raises:
        RuntimeError: if no sample succeeds alignment (callers must know — silent skip is wrong).
    """
    if acquisition_inventory_df.empty:
        raise ValueError("acquisition_inventory_df is empty — nothing to sample.")

    orientation_raw = str(
        acquisition_inventory_df["orientation"].mode().iloc[0]
    ).lower()
    orientation = "horizontal" if orientation_raw == "horizontal" else "vertical"

    pairs = (
        acquisition_inventory_df[["well_id", "time_index"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(pairs), size=min(n_samples, len(pairs)), replace=False)
    sampled = pairs.iloc[sorted(idx)]

    log.info(
        "build_keyence_stitch_map: orientation=%s n_candidates=%d n_samples=%d",
        orientation, len(pairs), len(sampled),
    )

    # Collect per-tile-index [y, x] coords (raw stitch2d convention) across FULLY aligned
    # samples only — mirrors legacy build01A_compile_keyence_images.py lines ~495-514: a sample
    # that aligns fewer than n_images tiles is skipped, NOT treated as a failure of the whole
    # batch. Only a handful of fully-aligned frames are needed to get a good median.
    align_rows: list[np.ndarray] = []  # each entry: (n_images, 2) array of [y, x]
    n_good = 0
    n_tried = 0

    for _, row in sampled.iterrows():
        well_id = row["well_id"]
        time_index = row["time_index"]
        sample_rows = acquisition_inventory_df[
            (acquisition_inventory_df["well_id"] == well_id)
            & (acquisition_inventory_df["time_index"] == time_index)
        ]
        try:
            tile_specs = _build_tile_specs(sample_rows)
        except Exception as exc:
            log.debug("Sample well=%s time=%s skipped (tile build): %s", well_id, time_index, exc)
            continue

        n_tried += 1

        try:
            coords_raw = raw_stitch2d_align(tile_specs, orientation=orientation)
        except Exception as exc:
            log.debug("Sample well=%s time=%s skipped (align): %s", well_id, time_index, exc)
            continue

        # Mirror legacy: only keep samples where EVERY tile placed. Partial alignments are
        # silently skipped here (not an error) — that is the whole point of sampling many.
        if len(coords_raw) != len(tile_specs):
            log.debug(
                "Sample well=%s time=%s partial alignment (%d/%d tiles) — skipped",
                well_id, time_index, len(coords_raw), len(tile_specs),
            )
            continue

        arr = np.full((len(tile_specs), 2), np.nan, dtype=float)
        for tid_idx, yx in coords_raw.items():
            arr[int(tid_idx), 0] = float(yx[0])
            arr[int(tid_idx), 1] = float(yx[1])
        align_rows.append(arr)
        n_good += 1

    if n_good == 0:
        raise RuntimeError(
            f"build_keyence_stitch_map: no sample fully aligned all tiles "
            f"(tried {n_tried} candidates). Cannot write stitch map."
        )

    stacked = np.stack(align_rows, axis=0)  # (n_good, n_images, 2)
    med_coords = np.nanmedian(stacked, axis=0)  # (n_images, 2) in [y, x]

    # Written coords are keyed by tile INDEX (0-based, raster order) — matching stitch2d's own
    # convention and what `load_params`/`_stitch_with_stitch2d` expect. [y, x] order preserved.
    coords: dict[str, list[float]] = {
        str(idx): [float(med_coords[idx, 0]), float(med_coords[idx, 1])]
        for idx in range(med_coords.shape[0])
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"coords": coords}))
    log.info(
        "build_keyence_stitch_map: wrote %d tile coords to %s (from %d/%d good samples)",
        len(coords), out_path, n_good, len(sampled),
    )


def _build_tile_specs(sample_rows: pd.DataFrame) -> list[TileSpec]:
    """Focus-stack each tile's Z planes and return a sorted list of TileSpecs."""
    tile_specs: list[TileSpec] = []
    for tile_id, tile_rows in sample_rows.groupby("tile_id"):
        z_paths = (
            tile_rows.sort_values("z_index")["source_tiff_path"]
            .map(lambda p: Path(str(p)))
            .tolist()
        )
        if not z_paths:
            raise ValueError(f"No z-plane TIFFs for tile_id={tile_id!r}.")
        planes = [skio.imread(str(p)) for p in z_paths]
        stack_zyx = np.stack(planes, axis=0)
        norm, _, _ = im_rescale(stack_zyx)
        tile_ff, _ = materialize_ff_projection(norm.astype(np.float32), device="cpu")
        tile_specs.append(TileSpec(tile_id=str(tile_id), image=np.asarray(tile_ff)))
    return tile_specs
