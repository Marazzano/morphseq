"""Per-well Keyence image materializer — mosaic image products from per-Z-plane TIFFs.

This is the Keyence twin of ``scope/yx1/materialize_well_yx1.py``.  It consumes ONE well's
acquisition inventory rows (one row per raw TIFF plane: well × tile × z × channel × time) and
produces materialized mosaic frames + frame-inventory rows.

Key differences from YX1:
  - **Many source TIFFs per frame** (not one ND2).  Per ``(channel_id, time_index)`` frame:
      1. Group rows by tile (``tile_id`` / ``position_index_within_well``).
      2. Read every tile's raw z-plane TIFFs → focus-stack the WHOLE frame in ONE
         ``focus_stack_group`` call (shared 0.1–99.9% bounds across all tiles/Z planes; the
         shared primitive owns all focus math — the adapter does NOT normalize).
      3. Assemble ``TileSpec(tile_id, image)`` in raster order.
      4. ``stitch_frame_tiles(tile_specs, FrameTilingConfig(...), PreComputeStitchParams(master_params_path))``
         → one mosaic.
  - **xy_composition must be ``"mosaic"``** — this backend asserts it (mirror of YX1's identity guard).
  - **``master_params_path`` is optional for now** (Stage C wires the experiment-grain stitch map;
    until then tiles are aligned from scratch via ``stitch2d`` on every frame).

Projection provenance (``focus_index_map``):
  The per-tile focus_index_map from ``focus_stack_group`` is painted onto a canvas-sized
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

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio
from PIL import Image

from data_pipeline.acquisition.image_building.shared.display_polarity import apply_display_polarity
from data_pipeline.acquisition.image_building.shared.focus_stack_group import (
    FocusStackConfig,
    focus_stack_group,
)
from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    PreComputeStitchParams,
    TileSpec,
    UnstitchableFrameError,
    stitch_frame_tiles,
)
from data_pipeline.acquisition.image_materialization import materialized_image_paths
from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    derive_image_id,
    derive_well_id,
)
from data_pipeline.acquisition.image_materialization.materialization_plan import ResolvedImageProduct
from data_pipeline.acquisition.image_materialization.resolved_product_plans import (
    image_product_key_for_resolved_product,
)
from data_pipeline.acquisition.image_materialization.materialized_image_write_policy import (
    MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
    orient_image_for_write,
    prepare_image_for_write,
    resolve_image_write_policy,
    suffix_for_policy,
    write_image,
)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import (
    validate_keyence_acquisition_inventory,
)
from data_pipeline.shared.path_roots import resolve_under_input_root

log = logging.getLogger(__name__)

# Label for resolve_under_input_root errors when reading inventory-stored TIFF paths.
_SCOPE_LABEL = "Keyence acquisition inventory"


def _resolve_tiff(stored, *, input_root) -> "Path":
    """Resolve a stored TIFF path, re-anchoring onto input_root if it has moved."""
    return resolve_under_input_root(
        stored, input_root=input_root, scope_label=_SCOPE_LABEL, full_root_fallback=True
    )

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
    "image_path",
    "image_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    "focus_index_map_path",
    "raw_tile_path",
    "raw_tile_manifest_path",
    "raw_tile_width_px",
    "raw_tile_height_px",
    "raw_tile_count",
    "raw_micrometers_per_pixel",
    *MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
)

_REQUIRED_IMAGE_CORE_COLUMNS: tuple[str, ...] = (
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
    "image_path",
    "image_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    "image_file_format",
    "pixel_dtype",
    "downsample_factor",
    "downsample_method",
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
    config: dict | None = None,
    input_root: Path | None = None,
) -> pd.DataFrame:
    """Materialize ONE Keyence image product for ONE well; return frame-inventory rows.

    Asserts ``xy_composition == 'mosaic'`` — this is the Keyence executor guard (mirror of
    YX1's identity guard). Projection products focus-stack each tile before stitching; z-stack
    products stitch the acquisition-inventory TIFFs independently for each declared z plane.

    Args:
        master_params_path: optional path to the experiment-grain ``master_params.json`` produced
            by Stage C (``build_keyence_stitch_map``).  ``None`` = align each frame from scratch
            via ``stitch2d`` (slower; no fallback coords).  Becomes required once Stage C lands.
    """
    # --- Executor guard: mosaic-only ---
    if resolved_product.xy_composition != "mosaic":
        raise ValueError(
            f"materialize_keyence_product_for_well only executes xy_composition='mosaic'; got "
            f"{resolved_product.xy_composition!r}. Keyence is multi-tile — this indicates a resolver bug."
        )
    if resolved_product.image_product_type == "projection":
        if resolved_product.projection_method != "focus_stack":
            raise ValueError(
                f"Keyence projection materialization requires projection_method='focus_stack'; "
                f"got {resolved_product.projection_method!r}."
            )
    elif resolved_product.image_product_type == "z_stack":
        if resolved_product.projection_method is not None:
            raise ValueError("Keyence z_stack materialization requires projection_method=None.")
    else:
        raise ValueError(
            f"Unsupported Keyence image_product_type {resolved_product.image_product_type!r}."
        )
    product_key = image_product_key_for_resolved_product(resolved_product)
    write_policy = resolve_image_write_policy(config, product_key)
    ext = suffix_for_policy(write_policy)

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
    validate_keyence_acquisition_inventory(
        well_acquisition_inventory_df, check_sources=True, input_root=input_root
    )

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

    # Determine orientation for the stitcher. Modern Keyence exports often leave this unknown;
    # legacy behavior treats those as horizontal strips.
    orientation_raw = str(inv["orientation"].iloc[0]).lower()
    orientation = "vertical" if orientation_raw == "vertical" else "horizontal"

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

    if resolved_product.image_product_type == "z_stack":
        expected_tiles = set(inv["tile_id"].tolist())
        declared_tile_counts = set(int(n) for n in inv["n_tiles_in_well"].unique())
        if len(declared_tile_counts) != 1 or declared_tile_counts != {len(expected_tiles)}:
            raise ValueError(
                f"Inconsistent Keyence tile inventory for well={well_id}: "
                f"observed tile_ids={sorted(expected_tiles)}, "
                f"declared n_tiles_in_well={sorted(declared_tile_counts)}."
            )
        selected = inv[inv["time_index"].isin(time_indices)]
        duplicate_cells = selected.duplicated(["time_index", "tile_id", "z_index"], keep=False)
        if duplicate_cells.any():
            cells = selected.loc[
                duplicate_cells, ["time_index", "tile_id", "z_index"]
            ].to_dict("records")
            raise ValueError(
                f"Duplicate Keyence time/tile/z inventory cells for well={well_id}: {cells}."
            )
        expected_z_indices = sorted(int(z) for z in selected["z_index"].unique())
        for t in time_indices:
            t_rows = selected[selected["time_index"] == t]
            actual_z_indices = sorted(int(z) for z in t_rows["z_index"].unique())
            if actual_z_indices != expected_z_indices:
                raise ValueError(
                    f"Incomplete Keyence z-stack for well={well_id} time_index={t}: "
                    f"expected z_indices={expected_z_indices}, got {actual_z_indices}."
                )
            for z_index in expected_z_indices:
                actual_tiles = set(
                    t_rows.loc[t_rows["z_index"] == z_index, "tile_id"].tolist()
                )
                if actual_tiles != expected_tiles:
                    raise ValueError(
                        f"Incomplete Keyence z plane for well={well_id} time_index={t} "
                        f"z_index={z_index}: expected tile_ids={sorted(expected_tiles)}, "
                        f"got {sorted(actual_tiles)}."
                    )

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

        if resolved_product.image_product_type == "z_stack":
            z_indices = sorted(int(z) for z in t_rows["z_index"].unique())
            for z_index in z_indices:
                z_rows = t_rows[t_rows["z_index"] == z_index]
                tile_specs = [
                    TileSpec(
                        tile_id=str(row.tile_id),
                        image=np.asarray(skio.imread(str(
                            _resolve_tiff(row.source_tiff_path, input_root=input_root)
                        ))),
                    )
                    for row in z_rows.sort_values("tile_id").itertuples(index=False)
                ]
                result = stitch_frame_tiles(tile_specs, tiling_config, fallback)
                if not result.qc.passed:
                    raise UnstitchableFrameError(
                        f"stitch_frame_tiles returned qc.passed=False for well={well_id} "
                        f"time_index={t} z_index={z_index}: reasons={result.qc.reasons} "
                        f"fallback_used={result.fallback_used}."
                    )
                # One shared, explicit polarity flip post-stitch (NOT hidden in the stitcher):
                # canonical bright-embryo/dark-background, same op every microscope uses.
                mosaic = apply_display_polarity(result.stitched)
                out_path = materialized_image_paths.z_stack_frame_path(
                    built_image_data_dir,
                    experiment_id=experiment_id,
                    well_id=well_id,
                    channel_id=channel_id,
                    time_index=int(t),
                    z_index=z_index,
                    ext=ext,
                    candidate=candidate,
                )
                prepared_mosaic = prepare_image_for_write(mosaic, write_policy)
                oriented_mosaic = orient_image_for_write(mosaic, write_policy)
                write_image(mosaic, out_path, write_policy)
                written_meta = _read_written_image_metadata(out_path)
                image_height_px = int(written_meta["image_height_px"])
                image_width_px = int(written_meta["image_width_px"])
                if prepared_mosaic.shape != (image_height_px, image_width_px):
                    raise RuntimeError(
                        f"Keyence materializer write dimensions disagree for {out_path}: "
                        f"prepared={prepared_mosaic.shape}, header={(image_height_px, image_width_px)}."
                    )
                if str(prepared_mosaic.dtype) != str(written_meta["pixel_dtype"]):
                    raise RuntimeError(
                        f"Keyence materializer write dtype disagrees for {out_path}: "
                        f"prepared={prepared_mosaic.dtype}, header={written_meta['pixel_dtype']}."
                    )
                raw_tile_path, raw_tile_manifest_path = _raw_tile_provenance_paths(
                    out_path=out_path,
                    experiment_id=experiment_id,
                    well_id=well_id,
                    channel_id=channel_id,
                    time_index=int(t),
                    t_rows=z_rows,
                    input_root=input_root,
                )
                rows.append({
                    "experiment_id": experiment_id,
                    "well_index": well_index,
                    "well_id": well_id,
                    "channel_id": channel_id,
                    "time_index": int(t),
                    "image_id": derive_image_id(well_id, channel_id, int(t), z_index=z_index),
                    "elapsed_time_s": t_times["elapsed_time_s"],
                    "acquisition_time_s": t_times["acquisition_time_s"],
                    "z_index": z_index,
                    "image_product_type": "z_stack",
                    "projection_method": pd.NA,
                    "image_path": str(out_path),
                    "image_micrometers_per_pixel": _materialized_micrometers_per_pixel(
                        raw_micrometers_per_pixel=um_per_px,
                        oriented_native_shape=oriented_mosaic.shape,
                        written_shape=(image_height_px, image_width_px),
                    ),
                    "image_width_px": image_width_px,
                    "image_height_px": image_height_px,
                    "focus_index_map_path": pd.NA,
                    "raw_tile_path": raw_tile_path,
                    "raw_tile_manifest_path": raw_tile_manifest_path,
                    "raw_tile_width_px": img_w,
                    "raw_tile_height_px": img_h,
                    "raw_tile_count": int(z_rows["tile_id"].nunique()),
                    "raw_micrometers_per_pixel": um_per_px,
                    "orientation": write_policy.orientation,
                    "image_file_format": str(written_meta["image_file_format"]),
                    "pixel_dtype": str(written_meta["pixel_dtype"]),
                    "downsample_factor": write_policy.downsample_factor,
                    "downsample_method": write_policy.downsample_method,
                    "jpeg_quality": (
                        write_policy.jpeg_quality if write_policy.file_format == "jpg" else pd.NA
                    ),
                })
            continue

        # Read all raw uint16 tile stacks for this frame, then focus-stack the WHOLE frame in
        # ONE shared-bounds group op (image_building/shared/README.md): a single 0.1–99.9%
        # intensity range across every tile/Z plane, so no tile renders on its own tone curve
        # and the per-tile normalization+double-normalization regression is impossible. The
        # adapter owns tile grouping/ordering only — the shared primitive owns all focus math.
        ordered_tile_ids: list[str] = []
        raw_stacks: list[np.ndarray] = []
        tile_z_indices: np.ndarray | None = None
        for tile_id, tile_rows in t_rows.groupby("tile_id"):
            sorted_rows = tile_rows.sort_values("z_index")
            z_paths = sorted_rows["source_tiff_path"].map(
                lambda p: _resolve_tiff(p, input_root=input_root)
            ).tolist()
            if not z_paths:
                raise ValueError(
                    f"No z-plane TIFFs for well={well_id} time_index={t} tile_id={tile_id}."
                )
            # Read raw Z planes → (Z, Y, X) uint16 stack (NOT normalized — that is the shared
            # primitive's job).
            planes = [skio.imread(str(p)) for p in z_paths]
            stack_zyx = np.stack(planes, axis=0)
            z_indices_tile = np.asarray(sorted_rows["z_index"].tolist(), dtype=np.int32)
            if tile_z_indices is None:
                tile_z_indices = z_indices_tile
            ordered_tile_ids.append(str(tile_id))
            raw_stacks.append(stack_zyx)

        group = focus_stack_group(raw_stacks, config=FocusStackConfig(), device=device)
        tile_specs: list[TileSpec] = [
            TileSpec(tile_id=tid, image=group.tiles[i].projection_u8)
            for i, tid in enumerate(ordered_tile_ids)
        ]
        tile_fims: dict[str, np.ndarray] = {
            tid: group.tiles[i].focus_index_map.astype(np.int32)
            for i, tid in enumerate(ordered_tile_ids)
        }

        # Stitch the per-tile 2D images into one mosaic. ``stitch_frame_tiles`` itself raises
        # ``UnstitchableFrameError`` when it cannot produce a trustworthy mosaic (incomplete/
        # implausible alignment + no usable master fallback) — this guard is defense-in-depth
        # for any path that returns a result with qc.passed=False instead of raising. Either
        # way: refuse to materialize a wrong-but-passing image.
        result = stitch_frame_tiles(tile_specs, tiling_config, fallback)
        if not result.qc.passed:
            raise UnstitchableFrameError(
                f"stitch_frame_tiles returned qc.passed=False for well={well_id} "
                f"time_index={t}: reasons={result.qc.reasons} fallback_used={result.fallback_used}. "
                f"Refusing to materialize a wrong-but-passing image."
            )
        # One shared, explicit polarity flip post-stitch (NOT hidden in the stitcher):
        # canonical bright-embryo/dark-background, same op every microscope uses.
        mosaic = apply_display_polarity(result.stitched)

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
            ext=ext,
            candidate=candidate,
        )
        prepared_mosaic = prepare_image_for_write(mosaic, write_policy)
        oriented_fim = orient_image_for_write(canvas_fim, write_policy)
        write_image(mosaic, out_path, write_policy)
        written_meta = _read_written_image_metadata(out_path)
        image_height_px = int(written_meta["image_height_px"])
        image_width_px = int(written_meta["image_width_px"])
        if prepared_mosaic.shape != (image_height_px, image_width_px):
            raise RuntimeError(
                "Keyence materializer wrote an image whose header dimensions disagree with the "
                f"shared writer output for {out_path}: prepared={prepared_mosaic.shape}, "
                f"header={(image_height_px, image_width_px)}."
            )
        if str(prepared_mosaic.dtype) != str(written_meta["pixel_dtype"]):
            raise RuntimeError(
                "Keyence materializer wrote an image whose read-back dtype disagrees with the "
                f"shared writer output for {out_path}: prepared={prepared_mosaic.dtype}, "
                f"header={written_meta['pixel_dtype']}."
            )
        image_micrometers_per_pixel = _materialized_micrometers_per_pixel(
            raw_micrometers_per_pixel=um_per_px,
            oriented_native_shape=oriented_fim.shape,
            written_shape=(image_height_px, image_width_px),
        )
        aligned_fim = _resize_index_map(
            oriented_fim,
            target_shape=(image_height_px, image_width_px),
        )

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
        np.savez(fim_path, focus_index_map=aligned_fim, z_indices=z_indices_out)

        raw_tile_path, raw_tile_manifest_path = _raw_tile_provenance_paths(
            out_path=out_path,
            experiment_id=experiment_id,
            well_id=well_id,
            channel_id=channel_id,
            time_index=int(t),
            t_rows=t_rows,
            input_root=input_root,
        )

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
            "image_path": str(out_path),
            "image_micrometers_per_pixel": image_micrometers_per_pixel,
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
            "focus_index_map_path": str(fim_path),
            "raw_tile_path": raw_tile_path,
            "raw_tile_manifest_path": raw_tile_manifest_path,
            "raw_tile_width_px": img_w,
            "raw_tile_height_px": img_h,
            "raw_tile_count": int(t_rows["tile_id"].nunique()),
            "raw_micrometers_per_pixel": um_per_px,
            "orientation": write_policy.orientation,
            "image_file_format": str(written_meta["image_file_format"]),
            "pixel_dtype": str(written_meta["pixel_dtype"]),
            "downsample_factor": write_policy.downsample_factor,
            "downsample_method": write_policy.downsample_method,
            "jpeg_quality": write_policy.jpeg_quality if write_policy.file_format == "jpg" else pd.NA,
        })

        if (len(rows) % 10) == 0:
            log.info("  %d/%d frames written", len(rows), len(time_indices))

    inv_df = pd.DataFrame(rows, columns=list(_EMITTED_COLUMNS))

    missing = [c for c in _REQUIRED_IMAGE_CORE_COLUMNS if c not in inv_df.columns]
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


def _read_written_image_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as im:
        arr = np.asarray(im)
        file_format = str(im.format or path.suffix.lstrip(".")).lower()
    if file_format == "jpeg":
        file_format = "jpg"
    if file_format == "tiff":
        file_format = "tif"
    return {
        "image_width_px": int(arr.shape[1]),
        "image_height_px": int(arr.shape[0]),
        "image_file_format": file_format,
        "pixel_dtype": str(arr.dtype),
    }


def _materialized_micrometers_per_pixel(
    *,
    raw_micrometers_per_pixel: float,
    oriented_native_shape: tuple[int, int],
    written_shape: tuple[int, int],
) -> float:
    native_height_px, native_width_px = oriented_native_shape
    written_height_px, written_width_px = written_shape
    if written_height_px <= 0 or written_width_px <= 0:
        raise ValueError(
            f"Written image shape must be positive; got {written_shape}."
        )
    scale_y = float(native_height_px) / float(written_height_px)
    scale_x = float(native_width_px) / float(written_width_px)
    return float(raw_micrometers_per_pixel) * ((scale_y + scale_x) / 2.0)


def _resize_index_map(index_map: np.ndarray, *, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(index_map)
    if arr.ndim != 2:
        raise ValueError(f"focus_index_map must be 2D; got shape {arr.shape}.")
    target_height_px, target_width_px = target_shape
    if (target_height_px, target_width_px) == arr.shape:
        return arr.astype(np.int32, copy=False)
    if target_height_px <= 0 or target_width_px <= 0:
        raise ValueError(f"target_shape must be positive; got {target_shape}.")
    y_idx = np.clip(
        np.floor((np.arange(target_height_px) + 0.5) * arr.shape[0] / target_height_px).astype(int),
        0,
        arr.shape[0] - 1,
    )
    x_idx = np.clip(
        np.floor((np.arange(target_width_px) + 0.5) * arr.shape[1] / target_width_px).astype(int),
        0,
        arr.shape[1] - 1,
    )
    return arr[np.ix_(y_idx, x_idx)].astype(np.int32, copy=False)


def _raw_tile_provenance_paths(
    *,
    out_path: Path,
    experiment_id: str,
    well_id: str,
    channel_id: str,
    time_index: int,
    t_rows: pd.DataFrame,
    input_root: Path | None = None,
) -> tuple[object, object]:
    # Provenance records ABSOLUTE paths (a self-contained point-in-time snapshot of the files used),
    # even though the inventory stores them relative — so resolve each stored path here.
    def _abs(p: object) -> str:
        return str(_resolve_tiff(p, input_root=input_root))

    raw_paths = sorted(_abs(p) for p in t_rows["source_tiff_path"].dropna().unique())
    if len(raw_paths) == 1:
        return raw_paths[0], pd.NA

    manifest_path = out_path.parent / "raw_tile_manifest" / f"{out_path.stem}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "well_id": well_id,
        "channel_id": channel_id,
        "time_index": int(time_index),
        "tiles": [
            {
                "tile_id": str(tile_id),
                "z_indices": [int(z) for z in tile_df.sort_values("z_index")["z_index"].tolist()],
                "source_tiff_paths": [
                    _abs(path)
                    for path in tile_df.sort_values("z_index")["source_tiff_path"].tolist()
                ],
            }
            for tile_id, tile_df in t_rows.groupby("tile_id")
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pd.NA, str(manifest_path)
