"""Per-well Keyence image materializer — mosaic image products from per-Z-plane TIFFs.

This is the Keyence twin of ``scope/yx1/materialize_well_yx1.py``.  It consumes ONE well's
acquisition inventory rows (one row per raw TIFF plane: well × tile × z × channel × time) and
produces materialized mosaic frames + frame-inventory rows.

Shape of the job: ONE well, EVERY requested product. ``materialize_keyence_products_for_well``
prepares each ``(channel_id, time_index)`` frame ONCE via
``frame_materials.prepare_keyence_frame`` — one raw read, at most one focus reduction, one tile
transform — and then fans out to ``_emit_projection_product`` / ``_emit_z_stack_product``, each of
which applies its OWN write policy and returns its OWN frame-inventory rows. The product branch
happens at the write boundary, not at the front of the job.

Each product still yields a SEPARATE inventory shard; only the process and the acquisition-derived
facts are shared. ``materialize_keyence_product_for_well`` is a thin single-product wrapper.

Key differences from YX1:
  - **Many source TIFFs per frame** (not one ND2), grouped by tile and read once per frame by
    ``prepare_keyence_frame``; the shared focus primitive owns all focus math (shared 0.1-99.9%
    bounds across all tiles/Z planes) — this adapter does NOT normalize.
  - **xy_composition must be ``"mosaic"``** — this backend asserts it (mirror of YX1's identity guard).
  - **``master_params_path``** is the experiment-grain stitch map from ``build_keyence_stitch_map``
    (Stage C, LANDED — the DAG passes it on every invocation). How it is USED depends on
    ``image_materialization.keyence_tiling_mode``: under the default ``"auto"`` the map is the QC
    reference and fallback while each frame is still aligned from scratch; under ``"prior_only"``
    the map supplies the coords directly and per-frame alignment is skipped.

Projection provenance (``focus_index_map``):
  The per-tile focus_index_map from ``focus_stack_group`` is painted onto a canvas-sized
  ``focus_index_map`` using the tile transforms from ``stitch_frame_tiles``.  The canvas map is
  written as a ``.npz`` (``focus_index_map`` + ``z_indices``) at the path returned by
  ``materialized_image_paths.index_map_path``; the ``index_map_path`` column in the
  frame-inventory row points at this file (same contract as YX1).

Import rules: imports image primitives from ``image_building/``, path helpers from
``materialized_image_paths``, ID helpers from ``shared/identifiers/``, and the shared
``frame_inventory_contract`` for column names.  MUST NOT import orchestration, tasks, Snakemake,
or YX1-specific logic (e.g. ``_determine_bf_channel``, ``_get_stack``, nd2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.acquisition.image_materialization.scope.keyence.frame_materials import (
    FrameMaterials,
    prepare_keyence_frame,
)
from data_pipeline.acquisition.image_building.shared.display_polarity import apply_display_polarity
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
    "index_map_path",
    "write_index_map",
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


@dataclass(frozen=True)
class _WellContext:
    """The product-independent setup for ONE well, derived once and shared by every product.

    Everything here is a fact about the ACQUISITION, not about any product: which timepoints exist,
    how the frames are calibrated, how the tiles are laid out. Deriving it per product is what the
    old one-job-per-product shape forced; deriving it once is the point of the fanout below.
    """

    channel_id: str
    inv: pd.DataFrame
    um_per_px: float
    img_w: int
    img_h: int
    time_indices: list
    time_lookup: dict
    tiling_config: FrameTilingConfig
    fallback: PreComputeStitchParams


def materialize_keyence_products_for_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    well_acquisition_inventory_df: pd.DataFrame,
    built_image_data_dir: Path,
    resolved_products: tuple[ResolvedImageProduct, ...],
    device: str = "cpu",
    candidate: bool = False,
    smoke_max_time_indices: int | None = None,
    master_params_path: Path | None = None,
    config: dict | None = None,
    input_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Materialize EVERY requested Keyence image product for ONE well, sharing the frame work.

    The product fanout happens at the WRITE boundary, not at the front of the job: each
    ``(channel_id, time_index)`` frame is prepared once (one raw read, at most one focus reduction,
    one tile transform) and every product emits from those same materials with its own write policy,
    its own files, and its own frame-inventory shard.

    Each product still yields a SEPARATE shard, exactly as the per-product jobs did — a shard must
    stay single-product, because ``PER_WELL_PRODUCT_CHECKS`` deliberately cannot answer
    cross-channel questions. Only the process is shared.

    Returns:
        ``{product_key: frame-inventory DataFrame}``, one entry per requested product.
    """
    if not resolved_products:
        raise ValueError(f"No resolved products requested for well {well_id!r}.")

    for product in resolved_products:
        _assert_supported_product(product)

    product_keys = [image_product_key_for_resolved_product(p) for p in resolved_products]
    if len(set(product_keys)) != len(product_keys):
        raise ValueError(
            f"Duplicate image product keys for well {well_id!r}: {product_keys}. Each requested "
            "product must resolve to a distinct key."
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

    # Consume-boundary: validate inventory + source readability before any TIFF reads. Hoisted to
    # the WELL level: this stats and opens every source_tiff_path, and it is a fact about the
    # acquisition, not about a product — running it once per product re-walked the whole tree.
    validate_keyence_acquisition_inventory(
        well_acquisition_inventory_df, check_sources=True, input_root=input_root
    )

    # Group products by channel. Materials are never shared across channels: pixel reductions and
    # write policies are per-channel concerns, so each channel gets its own prepared frames.
    products_by_channel: dict[str, list[ResolvedImageProduct]] = {}
    for product in resolved_products:
        products_by_channel.setdefault(product.channel_id, []).append(product)

    rows_by_product: dict[str, list[dict]] = {key: [] for key in product_keys}

    for channel_id, channel_products in products_by_channel.items():
        context = _build_well_context(
            well_acquisition_inventory_df,
            channel_id=channel_id,
            well_id=well_id,
            smoke_max_time_indices=smoke_max_time_indices,
            master_params_path=master_params_path,
            config=config,
        )

        # z_stack products need a complete tile x z grid; check once per channel, before reading.
        if any(p.image_product_type == "z_stack" for p in channel_products):
            _assert_complete_z_stack_inventory(context, well_id=well_id)

        write_policies = {
            image_product_key_for_resolved_product(p): resolve_image_write_policy(
                config,
                image_product_key_for_resolved_product(p),
                native_micrometers_per_pixel=context.um_per_px,
            )
            for p in channel_products
        }

        # Focus is required by any projection product, and by content-aligning geometry (auto /
        # align_only), which must match features on sharp composites rather than raw planes.
        compute_focus = any(
            p.image_product_type == "projection" for p in channel_products
        ) or context.tiling_config.mode != "prior_only"

        log.info(
            "materialize_keyence_products_for_well: experiment=%s well=%s channel=%s "
            "products=%s n_time=%d mode=%s compute_focus=%s candidate=%s",
            experiment_id, well_id, channel_id,
            [image_product_key_for_resolved_product(p) for p in channel_products],
            len(context.time_indices), context.tiling_config.mode, compute_focus, candidate,
        )

        for t in context.time_indices:
            frame_rows = context.inv[context.inv["time_index"] == t]
            materials = prepare_keyence_frame(
                frame_rows=frame_rows,
                channel_id=channel_id,
                time_index=int(t),
                well_id=well_id,
                tiling_config=context.tiling_config,
                fallback=context.fallback,
                compute_focus=compute_focus,
                device=device,
                resolve_tiff=lambda p: _resolve_tiff(p, input_root=input_root),
            )

            for product in channel_products:
                product_key = image_product_key_for_resolved_product(product)
                emit = (
                    _emit_projection_product
                    if product.image_product_type == "projection"
                    else _emit_z_stack_product
                )
                rows_by_product[product_key].extend(
                    emit(
                        materials=materials,
                        context=context,
                        experiment_id=experiment_id,
                        well_id=well_id,
                        well_index=well_index,
                        built_image_data_dir=built_image_data_dir,
                        write_policy=write_policies[product_key],
                        resolved_product=product,
                        candidate=candidate,
                        input_root=input_root,
                    )
                )

    result: dict[str, pd.DataFrame] = {}
    for product_key in product_keys:
        inv_df = pd.DataFrame(rows_by_product[product_key], columns=list(_EMITTED_COLUMNS))
        missing = [c for c in _REQUIRED_IMAGE_CORE_COLUMNS if c not in inv_df.columns]
        if missing:
            raise RuntimeError(
                f"materialize_keyence_products_for_well produced a frame-inventory DataFrame "
                f"missing required columns for {product_key!r}: {missing}. "
                f"Present: {list(inv_df.columns)}."
            )
        log.info(
            "materialize_keyence_products_for_well complete: well=%s product=%s frames=%d",
            well_id, product_key, len(inv_df),
        )
        result[product_key] = inv_df
    return result


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

    Single-product wrapper over :func:`materialize_keyence_products_for_well`. Kept because the
    per-product Snakemake rule (still the live path for non-Keyence scopes, and the fallback for
    Keyence) hands in exactly one product.
    """
    products = materialize_keyence_products_for_well(
        experiment_id=experiment_id,
        well_id=well_id,
        well_index=well_index,
        well_acquisition_inventory_df=well_acquisition_inventory_df,
        built_image_data_dir=built_image_data_dir,
        resolved_products=(resolved_product,),
        device=device,
        candidate=candidate,
        smoke_max_time_indices=smoke_max_time_indices,
        master_params_path=master_params_path,
        config=config,
        input_root=input_root,
    )
    return products[image_product_key_for_resolved_product(resolved_product)]


def _assert_supported_product(resolved_product: ResolvedImageProduct) -> None:
    """Executor guards: mosaic-only, and only the product shapes this backend has wired."""
    if resolved_product.xy_composition != "mosaic":
        raise ValueError(
            f"materialize_keyence_product_for_well only executes xy_composition='mosaic'; got "
            f"{resolved_product.xy_composition!r}. Keyence is multi-tile — this indicates a resolver bug."
        )
    if resolved_product.image_product_type == "projection":
        # EXECUTOR limit, not scope policy: only the focus-stack projection is wired for Keyence.
        # Method GRAMMAR and general capability are checked in the resolver; this guards what THIS
        # backend has actually implemented. Wiring max here means feeding the per-tile stacks through
        # a max reduce and then stitch_frame_tiles, the same shape as the focus path.
        if resolved_product.projection_method != "focus_stack":
            raise ValueError(
                f"Keyence projection materialization has only focus_stack wired; got "
                f"{resolved_product.projection_method!r}. (The YX1 backend implements 'max'.)"
            )
    elif resolved_product.image_product_type == "z_stack":
        if resolved_product.projection_method is not None:
            raise ValueError("Keyence z_stack materialization requires projection_method=None.")
    else:
        raise ValueError(
            f"Unsupported Keyence image_product_type {resolved_product.image_product_type!r}."
        )


def _build_well_context(
    well_acquisition_inventory_df: pd.DataFrame,
    *,
    channel_id: str,
    well_id: str,
    smoke_max_time_indices: int | None,
    master_params_path: Path | None,
    config: dict | None,
) -> _WellContext:
    """Derive the per-channel acquisition facts every product of that channel shares."""
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

    # Tile alignment strategy. The tiles of a frame do not move as the stage steps through focus, so
    # geometry is a property of the FRAME: prepare_keyence_frame resolves it once and every plane and
    # every product reuses it. "prior_only" reads the experiment-grain map instead of solving.
    # Default stays "auto": the per-frame align is also the per-frame QC signal, so trading it away
    # is an opt-in decision, not a silent one.
    tiling_mode = str(
        (config or {}).get("image_materialization", {}).get("keyence_tiling_mode", "auto")
    )
    return _WellContext(
        channel_id=str(channel_id),
        inv=inv,
        um_per_px=um_per_px,
        img_w=img_w,
        img_h=img_h,
        time_indices=time_indices,
        time_lookup=time_lookup,
        tiling_config=FrameTilingConfig(orientation=orientation, mode=tiling_mode),
        fallback=PreComputeStitchParams(master_params_path=master_params_path),
    )


def _assert_complete_z_stack_inventory(context: _WellContext, *, well_id: str) -> None:
    """A z_stack writes planes verbatim, so a missing tile/plane would silently change geometry."""
    inv = context.inv
    expected_tiles = set(inv["tile_id"].tolist())
    declared_tile_counts = set(int(n) for n in inv["n_tiles_in_well"].unique())
    if len(declared_tile_counts) != 1 or declared_tile_counts != {len(expected_tiles)}:
        raise ValueError(
            f"Inconsistent Keyence tile inventory for well={well_id}: "
            f"observed tile_ids={sorted(expected_tiles)}, "
            f"declared n_tiles_in_well={sorted(declared_tile_counts)}."
        )
    selected = inv[inv["time_index"].isin(context.time_indices)]
    duplicate_cells = selected.duplicated(["time_index", "tile_id", "z_index"], keep=False)
    if duplicate_cells.any():
        cells = selected.loc[
            duplicate_cells, ["time_index", "tile_id", "z_index"]
        ].to_dict("records")
        raise ValueError(
            f"Duplicate Keyence time/tile/z inventory cells for well={well_id}: {cells}."
        )
    expected_z_indices = sorted(int(z) for z in selected["z_index"].unique())
    for t in context.time_indices:
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


def _emit_z_stack_product(
    *,
    materials: FrameMaterials,
    context: _WellContext,
    experiment_id: str,
    well_id: str,
    well_index: str,
    built_image_data_dir: Path,
    write_policy,
    resolved_product: ResolvedImageProduct,
    candidate: bool,
    input_root: Path | None,
) -> list[dict]:
    """Stitch and write every declared z plane of ONE frame; return its inventory rows.

    Reuses ``materials.tile_transforms`` verbatim for every plane. No plane is feature-matched:
    out-of-focus planes carry too few descriptors to align reliably (planes of one stack could land
    in different coordinate frames from each other and from their own projection), and a plane
    yielding <2 descriptors aborts the whole well inside OpenCV's FLANN matcher.
    """
    t = materials.time_index
    t_rows = context.inv[context.inv["time_index"] == t]
    t_times = context.time_lookup[t]
    channel_id = materials.channel_id
    ext = suffix_for_policy(write_policy)
    use_transforms = materials.tile_transforms if len(materials.tile_ids) > 1 else None

    rows: list[dict] = []
    for z_index in [int(z) for z in materials.z_indices]:
        z_rows = t_rows[t_rows["z_index"] == z_index]
        # Planes come from the shared materials — already read once for this frame.
        tile_specs = [
            TileSpec(tile_id=tile_id, image=materials.plane_for(tile_id, z_index))
            for tile_id in materials.tile_ids
        ]
        result = stitch_frame_tiles(
            tile_specs, context.tiling_config, context.fallback, use_transforms=use_transforms
        )
        if not result.qc.passed:
            raise UnstitchableFrameError(
                f"stitch_frame_tiles returned qc.passed=False for well={well_id} "
                f"time_index={t} z_index={z_index}: reasons={result.qc.reasons} "
                f"fallback_used={result.fallback_used}."
            )
        # One shared, explicit polarity flip post-stitch (NOT hidden in the stitcher):
        # canonical bright-embryo/dark-background, governed by the per-product write policy.
        mosaic = apply_display_polarity(result.stitched, invert=write_policy.flip_polarity)
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
                raw_micrometers_per_pixel=context.um_per_px,
                oriented_native_shape=oriented_mosaic.shape,
                written_shape=(image_height_px, image_width_px),
            ),
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
            "index_map_path": pd.NA,
            "write_index_map": False,
            "raw_tile_path": raw_tile_path,
            "raw_tile_manifest_path": raw_tile_manifest_path,
            "raw_tile_width_px": context.img_w,
            "raw_tile_height_px": context.img_h,
            "raw_tile_count": int(z_rows["tile_id"].nunique()),
            "raw_micrometers_per_pixel": context.um_per_px,
            "orientation": write_policy.orientation,
            "image_file_format": str(written_meta["image_file_format"]),
            "pixel_dtype": str(written_meta["pixel_dtype"]),
            "downsample_factor": write_policy.downsample_factor,
            "downsample_method": write_policy.downsample_method,
            "jpeg_quality": (
                write_policy.jpeg_quality if write_policy.file_format == "jpg" else pd.NA
            ),
            "flip_polarity": bool(write_policy.flip_polarity),
        })
    return rows


def _emit_projection_product(
    *,
    materials: FrameMaterials,
    context: _WellContext,
    experiment_id: str,
    well_id: str,
    well_index: str,
    built_image_data_dir: Path,
    write_policy,
    resolved_product: ResolvedImageProduct,
    candidate: bool,
    input_root: Path | None,
) -> list[dict]:
    """Stitch and write ONE frame's focus projection + its canvas index map; return its row."""
    t = materials.time_index
    t_rows = context.inv[context.inv["time_index"] == t]
    t_times = context.time_lookup[t]
    channel_id = materials.channel_id
    ext = suffix_for_policy(write_policy)

    if materials.focus is None:
        raise RuntimeError(
            f"Projection product requested for well={well_id} time_index={t} but the frame was "
            "prepared without a focus reduction. This is a fanout bug: compute_focus must be True "
            "whenever any product is a projection."
        )

    tile_specs = [
        TileSpec(tile_id=tid, image=materials.focus.tiles[i].projection_u8)
        for i, tid in enumerate(materials.tile_ids)
    ]
    tile_fims = {
        tid: materials.focus.tiles[i].focus_index_map.astype(np.int32)
        for i, tid in enumerate(materials.tile_ids)
    }

    # Reuse the frame's already-resolved geometry rather than aligning again. ``stitch_frame_tiles``
    # itself raises ``UnstitchableFrameError`` when it cannot produce a trustworthy mosaic — this
    # guard is defense-in-depth for any path that returns qc.passed=False instead of raising.
    # Either way: refuse to materialize a wrong-but-passing image.
    use_transforms = materials.tile_transforms if len(materials.tile_ids) > 1 else None
    result = stitch_frame_tiles(
        tile_specs, context.tiling_config, context.fallback, use_transforms=use_transforms
    )
    if not result.qc.passed:
        raise UnstitchableFrameError(
            f"stitch_frame_tiles returned qc.passed=False for well={well_id} "
            f"time_index={t}: reasons={result.qc.reasons} fallback_used={result.fallback_used}. "
            f"Refusing to materialize a wrong-but-passing image."
        )
    # One shared, explicit polarity flip post-stitch (NOT hidden in the stitcher):
    # canonical bright-embryo/dark-background, governed by the per-product write policy.
    mosaic = apply_display_polarity(result.stitched, invert=write_policy.flip_polarity)

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
        raw_micrometers_per_pixel=context.um_per_px,
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
    np.savez(fim_path, focus_index_map=aligned_fim, z_indices=materials.z_indices)

    raw_tile_path, raw_tile_manifest_path = _raw_tile_provenance_paths(
        out_path=out_path,
        experiment_id=experiment_id,
        well_id=well_id,
        channel_id=channel_id,
        time_index=int(t),
        t_rows=t_rows,
        input_root=input_root,
    )

    return [{
        "experiment_id": experiment_id,
        "well_index": well_index,
        "well_id": well_id,
        "channel_id": channel_id,
        "time_index": int(t),
        "image_id": derive_image_id(well_id, channel_id, int(t)),
        "elapsed_time_s": t_times["elapsed_time_s"],
        "acquisition_time_s": t_times["acquisition_time_s"],
        "z_index": pd.NA,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "image_path": str(out_path),
        "image_micrometers_per_pixel": image_micrometers_per_pixel,
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
        "index_map_path": str(fim_path),
        # Keyence focus_stack always writes its canvas index map; when the plan gains real
        # control here, thread resolved_product.write_index_map through instead.
        "write_index_map": True,
        "raw_tile_path": raw_tile_path,
        "raw_tile_manifest_path": raw_tile_manifest_path,
        "raw_tile_width_px": context.img_w,
        "raw_tile_height_px": context.img_h,
        "raw_tile_count": int(t_rows["tile_id"].nunique()),
        "raw_micrometers_per_pixel": context.um_per_px,
        "orientation": write_policy.orientation,
        "image_file_format": str(written_meta["image_file_format"]),
        "pixel_dtype": str(written_meta["pixel_dtype"]),
        "downsample_factor": write_policy.downsample_factor,
        "downsample_method": write_policy.downsample_method,
        "jpeg_quality": write_policy.jpeg_quality if write_policy.file_format == "jpg" else pd.NA,
        "flip_polarity": bool(write_policy.flip_polarity),
    }]



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
