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
    n_samples: int = 3,
    out_path: Path,
) -> None:
    """Build the experiment-grain Keyence stitch map from STAGE COORDINATES, not image alignment.

    Every Keyence plane's XML carries the absolute stage position (``stage_x_nm``/``stage_y_nm``)
    at which it was acquired. The per-tile DELTAS of those positions ARE the mosaic geometry, exact
    and free — so the tile offsets need no feature detection at all. Feature-based alignment
    (``raw_stitch2d_align``) fails outright on these frames: brightfield wells are low-texture and
    carry a strong illumination gradient across the strip, and every one of 50 sampled frames
    failed to align in both orientations.

    Samples ``n_samples`` wells (fixed seed), converts each well's stage deltas to pixel offsets via
    ``micrometers_per_pixel``, and writes the median across wells. Coords are keyed by 0-based tile
    INDEX in ``[y, x]`` order, matching stitch2d's convention and what ``load_params`` expects.

    Raises:
        ValueError: if the inventory lacks stage columns, or tiles do not form a single strip.
    """
    if acquisition_inventory_df.empty:
        raise ValueError("acquisition_inventory_df is empty — nothing to sample.")

    missing = {"stage_x_nm", "stage_y_nm"} - set(acquisition_inventory_df.columns)
    if missing:
        raise ValueError(
            f"acquisition_inventory is missing {sorted(missing)}; re-run scope ingest so the "
            f"Keyence stage positions are scraped into the inventory."
        )

    df = acquisition_inventory_df
    um_per_px = float(df["micrometers_per_pixel"].iloc[0])
    if not um_per_px > 0:
        raise ValueError(f"micrometers_per_pixel must be positive, got {um_per_px!r}.")

    wells = sorted(df["well_id"].unique())
    rng = np.random.default_rng(seed=42)
    picked = rng.choice(len(wells), size=min(n_samples, len(wells)), replace=False)
    sampled_wells = [wells[i] for i in sorted(picked)]

    # Per well: one (stage_x, stage_y) per tile, ordered by tile_id. Stage is constant across Z
    # and channel within a tile, so take the first row of each tile group.
    per_well: list[np.ndarray] = []
    for well_id in sampled_wells:
        w = df[df["well_id"] == well_id].sort_values("tile_id")
        t = w.groupby("tile_id", sort=True)[["stage_x_nm", "stage_y_nm"]].first()
        per_well.append(t.to_numpy(dtype=float))

    n_tiles = per_well[0].shape[0]
    if any(a.shape[0] != n_tiles for a in per_well):
        raise ValueError(
            f"Sampled wells disagree on tile count: {[a.shape[0] for a in per_well]}. "
            f"Cannot build one experiment-grain stitch map."
        )

    # Offsets relative to tile 0, in nm -> um -> px. Stage +x is image +x; stage +y is image +y.
    stage = np.stack(per_well, axis=0)                       # (n_wells, n_tiles, 2) [x, y] nm
    offsets_nm = stage - stage[:, :1, :]                     # relative to tile 0 within each well
    offsets_px = np.nanmedian(offsets_nm, axis=0) / 1000.0 / um_per_px   # (n_tiles, 2) [x, y]

    # Anchor at the minimum so all coords are >= 0 (stage x DECREASES with tile_id on this scope).
    offsets_px -= offsets_px.min(axis=0)

    spread = offsets_px.max(axis=0) - offsets_px.min(axis=0)  # [x_spread, y_spread]
    orientation = "horizontal" if spread[0] >= spread[1] else "vertical"
    minor = float(spread.min())
    if n_tiles > 1 and minor > 1.0:
        raise ValueError(
            f"Keyence tiles do not form a single row/column: stage spread is "
            f"{spread[0]:.1f}px x {spread[1]:.1f}px. A 2-D mosaic is not supported here."
        )

    # coords keyed by 0-based tile index, value [y, x] — note the axis swap from offsets_px [x, y].
    coords = {
        str(i): [float(offsets_px[i, 1]), float(offsets_px[i, 0])]
        for i in range(n_tiles)
    }
    shape = [n_tiles, 1] if orientation == "vertical" else [1, n_tiles]
    tile_shape = [int(df["image_height_px"].iloc[0]), int(df["image_width_px"].iloc[0])]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "metadata": {"shape": shape, "size": n_tiles, "tile_shape": tile_shape},
                "coords": coords,
            }
        )
    )
    step = float(np.abs(np.diff(offsets_px[:, 0])).max()) if n_tiles > 1 else 0.0
    log.info(
        "build_keyence_stitch_map: %s, %d tiles, step=%.1fpx, overlap=%.1fpx "
        "(from stage coords of %d wells) -> %s",
        orientation, n_tiles, step, tile_shape[1] - step, len(sampled_wells), out_path,
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
