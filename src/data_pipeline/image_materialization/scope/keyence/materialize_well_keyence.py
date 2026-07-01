"""Per-well Keyence image materializer — mosaic image products from per-Z-plane TIFFs.

This is the Keyence twin of ``scope/yx1/materialize_well_yx1.py``.  It consumes ONE well's
acquisition inventory rows (one row per raw TIFF plane: well × tile × z × channel × time) and
produces materialized mosaic frames + frame-inventory rows.

Key differences from YX1:
  - **Many source TIFFs per frame** (not one ND2).  Per ``(channel_id, time_index)`` frame:
      1. Group rows by tile (``tile_id`` / ``position_index_within_well``).
      2. Per tile: read its z-plane TIFFs → focus-stack via ``materialize_ff_projection``
         (same primitive as YX1 — shared via import, NOT copied).
      3. Assemble ``TileSpec(tile_id, image)`` in raster order.
      4. ``stitch_frame_tiles(tile_specs, FrameTilingConfig(...), PreComputeStitchParams(master_params_path))``
         → one mosaic.
  - **xy_composition must be ``"mosaic"``** — this backend asserts it (mirror of YX1's identity guard).
  - **``master_params_path`` is optional for now** (Stage C wires the experiment-grain stitch map;
    until then tiles are aligned from scratch via ``stitch2d`` on every frame).

Projection provenance (``focus_index_map``):
  The per-tile focus_index_map from ``materialize_ff_projection`` is painted onto a canvas-sized
  ``focus_index_map`` using the tile transforms from ``stitch_frame_tiles``.  The canvas map is
  written as a ``.npz`` (``focus_index_map`` + ``z_indices``) at the path returned by
  ``materialized_image_paths.focus_index_map_path``; the ``focus_index_map_path`` column in the
  frame-inventory row points at this file (same contract as YX1).

Import rules: imports image primitives from ``image_building/``, path helpers from
``materialized_image_paths``, ID helpers from ``shared/identifiers/``, and the shared
``frame_inventory_contract`` for column names.  MUST NOT import orchestration, tasks, Snakemake,
or YX1-specific logic (e.g. ``_determine_bf_channel``, ``_get_stack``, nd2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.image_building.shared.log_focus import im_rescale
from data_pipeline.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    PreComputeStitchParams,
    TileSpec,
    stitch_frame_tiles,
)
from data_pipeline.image_materialization import materialized_image_paths
from data_pipeline.image_materialization.frame_inventory_contract import (
    REQUIRED_FRAME_INVENTORY_COLUMNS,
    derive_image_id,
    derive_well_id,
)
from data_pipeline.image_materialization.materialization_plan import ResolvedImageProduct
from data_pipeline.image_materialization.materialized_image_write_policy import (
    MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
)
from data_pipeline.image_materialization.scope.yx1.materialize_well_yx1 import (
    materialize_ff_projection,
)
from data_pipeline.metadata_ingest.scope.keyence.acquisition_inventory import (
    validate_keyence_acquisition_inventory,
)

log = logging.getLogger(__name__)

# Same emitted schema as YX1 — the shared frame_inventory_contract governs both.
_EMITTED_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_index",
    "well_id",
    "channel_id",
    "time_index",
    "image_id",
    "elapsed_time_s",
    "acquisition_time_s",
    "z_index",
    "image_product_type",
    "projection_method",
    "source_image_path",
    "focus_index_map_path",
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    *MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
)


def materialize_keyence_product_for_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    well_acquisition_inventory_df: pd.DataFrame,
    built_image_data_dir: Path,
    resolved_product: ResolvedImageProduct,
    device: str = "cpu",
    candidate: bool = False,
    smoke_max_time_indices: int | None = None,
    master_params_path: Path | None = None,
) -> pd.DataFrame:
    """Materialize ONE Keyence image product for ONE well; return frame-inventory rows.

    Asserts ``xy_composition == 'mosaic'`` — this is the Keyence executor guard (mirror of
    YX1's identity guard).  Only ``projection / focus_stack`` is wired for now; ``z_stack``
    raises ``NotImplementedError`` (Keyence z_stack fanout deferred — requires per-tile plane
    grouping that projection already does, but no stitch step; reserved for a future commit).

    Args:
        master_params_path: optional path to the experiment-grain ``master_params.json`` produced
            by Stage C (``build_keyence_stitch_map``).  ``None`` = align each frame from scratch
            via ``stitch2d`` (slower; no fallback coords).  Becomes required once Stage C lands.
    """
    # --- Executor guard: mosaic-only, projection/focus_stack only for now ---
    if resolved_product.xy_composition != "mosaic":
        raise ValueError(
            f"materialize_keyence_product_for_well only executes xy_composition='mosaic'; got "
            f"{resolved_product.xy_composition!r}. Keyence is multi-tile — this indicates a resolver bug."
        )
    if resolved_product.image_product_type == "z_stack":
        raise NotImplementedError(
            "Keyence z_stack materialization is not yet wired (Stage B covers projection only). "
            "Request xy_composition='mosaic' + image_product_type='projection' for now."
        )
    if resolved_product.image_product_type != "projection":
        raise ValueError(
            f"Unsupported Keyence image_product_type {resolved_product.image_product_type!r}."
        )
    if resolved_product.projection_method != "focus_stack":
        raise ValueError(
            f"Keyence projection materialization requires projection_method='focus_stack'; "
            f"got {resolved_product.projection_method!r}."
        )

    # --- Entry guard: well/inventory consistency ---
    expected_well_id = derive_well_id(experiment_id, well_index)
    if expected_well_id != well_id:
        raise ValueError(
            f"well_id mismatch: derive_well_id({experiment_id!r}, {well_index!r}) "
            f"= {expected_well_id!r} but caller passed well_id={well_id!r}."
        )
    if well_acquisition_inventory_df.empty:
        raise ValueError(f"well_acquisition_inventory_df is empty for well {well_id!r}.")

    # Consume-boundary: validate inventory + source readability before any TIFF reads.
    validate_keyence_acquisition_inventory(well_acquisition_inventory_df, check_sources=True)

    channel_id = resolved_product.channel_id
    inv = well_acquisition_inventory_df[
        well_acquisition_inventory_df["channel_id"] == channel_id
    ].copy()
    if inv.empty:
        raise ValueError(
            f"No inventory rows for channel_id={channel_id!r} in well {well_id!r}."
        )

    # Per-frame calibration (constant within a well — use first row).
    um_per_px = float(inv["micrometers_per_pixel"].iloc[0])
    img_w = int(inv["image_width_px"].iloc[0])
    img_h = int(inv["image_height_px"].iloc[0])

    # Determine orientation for the stitcher (from inventory; default vertical if unknown).
    orientation_raw = str(inv["orientation"].iloc[0]).lower()
    orientation = "vertical" if orientation_raw not in ("horizontal",) else "horizontal"

    time_indices = sorted(inv["time_index"].unique())
    if smoke_max_time_indices is not None and smoke_max_time_indices > 0:
        n_full = len(time_indices)
        time_indices = time_indices[:smoke_max_time_indices]
        msg = (
            f"SMOKE_FRAME_CAP_ACTIVE: limiting materialization to first "
            f"{len(time_indices)} of {n_full} time_index values for well {well_id} "
            f"(development scaffolding — NOT a production frame-selection policy)."
        )
        log.warning(msg)
        print(msg, flush=True)

    # Time-column lookup (constant per time_index across tiles/z).
    time_cols = ["elapsed_time_s", "acquisition_time_s"]
    time_lookup = (
        inv.drop_duplicates("time_index")
        .set_index("time_index")[time_cols]
        .to_dict("index")
    )

    tiling_config = FrameTilingConfig(orientation=orientation)
    fallback = PreComputeStitchParams(master_params_path=master_params_path)

    log.info(
        "materialize_keyence_product_for_well: experiment=%s well=%s channel=%s "
        "product=%s n_time=%d orientation=%s master_params=%s candidate=%s",
        experiment_id, well_id, channel_id,
        resolved_product.image_product_type, len(time_indices),
        orientation, master_params_path, candidate,
    )

    rows: list[dict] = []

    for t in time_indices:
        t_rows = inv[inv["time_index"] == t]
        t_times = time_lookup[t]

        # Build one TileSpec per tile: per-tile focus-stack (raw Z planes → 2D).
        tile_specs: list[TileSpec] = []
        tile_fims: dict[str, np.ndarray] = {}  # tile_id → per-tile focus_index_map
        tile_z_indices: np.ndarray | None = None
        for tile_id, tile_rows in t_rows.groupby("tile_id"):
            sorted_rows = tile_rows.sort_values("z_index")
            z_paths = sorted_rows["source_tiff_path"].map(lambda p: Path(str(p))).tolist()
            if not z_paths:
                raise ValueError(
                    f"No z-plane TIFFs for well={well_id} time_index={t} tile_id={tile_id}."
                )
            # Read raw Z planes → (Z, Y, X) stack.
            planes = [skio.imread(str(p)) for p in z_paths]
            stack_zyx = np.stack(planes, axis=0)
            z_indices_tile = np.asarray(sorted_rows["z_index"].tolist(), dtype=np.int32)
            if tile_z_indices is None:
                tile_z_indices = z_indices_tile

            # Focus-stack: same primitive as YX1 (DRY — imported, not copied).
            norm, _, _ = im_rescale(stack_zyx)
            tile_ff, tile_fim = materialize_ff_projection(
                norm.astype(np.float32), device=device
            )
            tile_img = np.asarray(tile_ff)
            tile_specs.append(TileSpec(tile_id=str(tile_id), image=tile_img))
            tile_fims[str(tile_id)] = tile_fim.astype(np.int32)

        # Stitch the per-tile 2D images into one mosaic.
        result = stitch_frame_tiles(tile_specs, tiling_config, fallback)
        mosaic = result.stitched

        # Build canvas focus_index_map by painting each tile's fim at its stitched origin.
        # Then trim to the mosaic's actual shape (which may be smaller after legacy-canvas trim).
        canvas_h, canvas_w = result.canvas_shape
        canvas_fim = np.zeros((canvas_h, canvas_w), dtype=np.int32)
        for tile_spec in tile_specs:
            tr = result.tile_transforms[tile_spec.tile_id]
            dy = int(round(tr.dy_px))
            dx = int(round(tr.dx_px))
            fim = tile_fims[tile_spec.tile_id]
            th, tw = fim.shape
            ey = min(dy + th, canvas_h)
            ex = min(dx + tw, canvas_w)
            if ey > dy and ex > dx:
                canvas_fim[dy:ey, dx:ex] = fim[: ey - dy, : ex - dx]
        # Trim to match the final mosaic dimensions (post-finalize crop).
        canvas_fim = canvas_fim[: mosaic.shape[0], : mosaic.shape[1]]

        out_path = materialized_image_paths.projection_frame_path(
            built_image_data_dir,
            experiment_id=experiment_id,
            well_id=well_id,
            channel_id=channel_id,
            time_index=int(t),
            projection_method="focus_stack",
            candidate=candidate,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        skio.imsave(str(out_path), mosaic, check_contrast=False)

        fim_path = materialized_image_paths.focus_index_map_path(
            built_image_data_dir,
            experiment_id=experiment_id,
            well_id=well_id,
            channel_id=channel_id,
            time_index=int(t),
            candidate=candidate,
        )
        fim_path.parent.mkdir(parents=True, exist_ok=True)
        z_indices_out = tile_z_indices if tile_z_indices is not None else np.array([], dtype=np.int32)
        np.savez(fim_path, focus_index_map=canvas_fim, z_indices=z_indices_out)

        image_id = derive_image_id(well_id, channel_id, int(t))
        rows.append({
            "experiment_id": experiment_id,
            "well_index": well_index,
            "well_id": well_id,
            "channel_id": channel_id,
            "time_index": int(t),
            "image_id": image_id,
            "elapsed_time_s": t_times["elapsed_time_s"],
            "acquisition_time_s": t_times["acquisition_time_s"],
            "z_index": pd.NA,
            "image_product_type": "projection",
            "projection_method": "focus_stack",
            "source_image_path": str(out_path),
            "focus_index_map_path": str(fim_path),
            "source_micrometers_per_pixel": um_per_px,
            "source_image_width_px": mosaic.shape[1],
            "source_image_height_px": mosaic.shape[0],
            "image_width_px": mosaic.shape[1],
            "image_height_px": mosaic.shape[0],
            "image_file_format": "png",
            "pixel_dtype": "uint8",
            "downsample_factor": 1,
            "downsample_method": "none",
            "jpeg_quality": pd.NA,
        })

        if (len(rows) % 10) == 0:
            log.info("  %d/%d frames written", len(rows), len(time_indices))

    inv_df = pd.DataFrame(rows, columns=list(_EMITTED_COLUMNS))

    missing = [c for c in REQUIRED_FRAME_INVENTORY_COLUMNS if c not in inv_df.columns]
    if missing:
        raise RuntimeError(
            f"materialize_keyence_product_for_well produced a frame-inventory DataFrame missing "
            f"required columns: {missing}. Present: {list(inv_df.columns)}."
        )

    log.info(
        "materialize_keyence_product_for_well complete: well=%s product=%s frames=%d",
        well_id, resolved_product.image_product_type, len(inv_df),
    )
    return inv_df
