"""Per-well YX1 image materializer — image products from ND2 tensor slices.

This module owns the YX1-specific path from an acquisition-inventory row set to materialized
pixel files + frame-inventory rows.  It has two layers:

  Primitives (pure image math, no paths, no IDs):
    ``materialize_ff_projection``  — focus-stack a Z-stack → one uint8 2D frame (BF method)
    ``materialize_max_projection`` — max-project a Z-stack → one 2D frame (fluorescence method)

  Per-well orchestrator:
    ``materialize_yx1_product_for_well`` — reads ONE well's ND2 slices for ONE resolved product,
                                           writes files through ``materialized_image_paths.py``,
                                           returns inventory rows.
    ``materialize_yx1_well``             — compatibility wrapper requiring exactly one product.

Step 3 scope: BF only (``projection_method='focus_stack'``).  Fluorescence channels use
``materialize_max_projection``; that wiring is deferred until GFP acquisition inventory lands.

Import rules: this module imports image primitives from ``image_building/``, path resolution
from ``materialized_image_paths.py``, ID helpers from ``shared/identifiers/``, and the
frame-inventory contract for column names.  It MUST NOT import orchestration, tasks, or
Snakemake rules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
import skimage.util
import torch

from data_pipeline.acquisition.image_building.scope.yx1.stitched_ff_builder import (
    _determine_bf_channel,
    _get_stack,
)
from data_pipeline.acquisition.image_building.shared.log_focus import LoG_focus_stacker, im_rescale
from data_pipeline.acquisition.image_materialization import materialized_image_paths
from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    REQUIRED_FRAME_INVENTORY_COLUMNS,
    derive_image_id,
    derive_well_id,
)
from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
    ResolvedMaterializationPlan,
)
from data_pipeline.acquisition.image_materialization.resolved_product_plans import (
    image_product_key_for_resolved_product,
)
from data_pipeline.acquisition.image_materialization.materialized_image_write_policy import (
    ImageWritePolicy,
    MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
    expected_downsampled_dims,
    resolve_image_write_policy,
    suffix_for_policy,
    write_image,
)
from data_pipeline.acquisition.metadata_ingest.scope.yx1.acquisition_inventory import (
    validate_yx1_acquisition_inventory,
)

log = logging.getLogger(__name__)

# Frame-inventory columns emitted by this module (flat schema, decided 2026-06-17). Derived columns
# are written for consumers but never trusted: the validator recomputes them from atoms and fails
# loud on disagreement.
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


# ---------------------------------------------------------------------------
# Image-math primitives (pure — no paths, no IDs, no side effects)
# ---------------------------------------------------------------------------


def materialize_ff_projection(
    stack_zyx: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Focus-stack a Z-stack into one 2D frame (BF / brightfield projection method).

    This is the ``projection_method='focus_stack'`` primitive. Pure image math — does not know
    about well_id, time_index, or paths.

    Args:
        stack_zyx: a single ``(Z, Y, X)`` brightfield z-stack for one (well, channel, time).
        device: torch device for the LoG convolutions (``"cpu"`` or ``"cuda"``).

    Returns:
        ``(projection_u8, focus_index_map)``:
          - ``projection_u8``: the focus-stacked 2D frame, ``uint8 (Y, X)``.
          - ``focus_index_map``: per-pixel STACK-AXIS OFFSET ``int32 (Y, X)`` — for pixel ``(y, x)``
            the value ``k`` is the Z plane (0-based axis offset into ``stack_zyx``) with the sharpest
            LoG response. It is NOT an acquisition ``z_index`` label; the caller pairs it with an
            ordered ``z_indices`` array to recover the labels.
    """
    norm, _, _ = im_rescale(stack_zyx)
    # LoG_focus_stacker returns (ff, abs_log); abs_log is the per-plane LoG response magnitude with
    # shape (Z, Y, X). The focus index is argmax over the Z AXIS (axis 0) — the exact same selection
    # the stacker makes internally to gather ff.
    ff, abs_log = LoG_focus_stacker(norm.astype(np.float32), filter_size=3, device=device)
    arr = ff.cpu().numpy() if torch.is_tensor(ff) else np.asarray(ff)
    projection_u8 = skimage.util.img_as_ubyte(np.clip(arr, 0, 65535).astype(np.uint16))

    abs_log_np = abs_log.cpu().numpy() if torch.is_tensor(abs_log) else np.asarray(abs_log)
    focus_index_map = np.argmax(abs_log_np, axis=0).astype(np.int32)
    return projection_u8, focus_index_map


def materialize_max_projection(stack_zyx: np.ndarray) -> np.ndarray:
    """Max-project a Z-stack into one 2D frame (fluorescence projection method).

    Returns same dtype as input.  This is the ``projection_method='max_projection'``
    primitive.  Defined here now to lock the naming convention; wired when GFP lands.
    """
    return stack_zyx.max(axis=0)


# ---------------------------------------------------------------------------
# Per-well orchestrator
# ---------------------------------------------------------------------------


def materialize_yx1_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    well_acquisition_inventory_df: pd.DataFrame,
    built_image_data_dir: Path,
    resolved_plan: ResolvedMaterializationPlan,
    device: str = "cuda",
    candidate: bool = True,
    smoke_max_time_indices: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Materialize exactly one resolved product for ONE YX1 well.

    Back-compat wrapper around ``materialize_yx1_product_for_well``. Snakemake product fanout is the
    owner of multi-product execution, so this wrapper rejects plans containing more than one product.
    """
    if len(resolved_plan.products) != 1:
        raise ValueError(
            "materialize_yx1_well expects exactly one resolved product. "
            "Fan out requested products before calling the YX1 product executor."
        )
    return materialize_yx1_product_for_well(
        experiment_id=experiment_id,
        well_id=well_id,
        well_index=well_index,
        well_acquisition_inventory_df=well_acquisition_inventory_df,
        built_image_data_dir=built_image_data_dir,
        resolved_product=resolved_plan.products[0],
        device=device,
        candidate=candidate,
        smoke_max_time_indices=smoke_max_time_indices,
        config=config,
    )


def materialize_yx1_product_for_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    well_acquisition_inventory_df: pd.DataFrame,
    built_image_data_dir: Path,
    resolved_product: ResolvedImageProduct,
    device: str = "cuda",
    candidate: bool = True,
    smoke_max_time_indices: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Materialize ONE YX1 image product for ONE well and return frame-inventory rows.

    Reads Z-stacks from the ND2 tensor (one slice per time_index). For ``projection`` it focus-stacks
    each stack into one BF frame. For ``z_stack`` it writes the inventory-declared planes directly,
    one row per ``(time_index, z_index)``.

    The ND2 source path is NOT a parameter: it is read from the inventory's ``source_nd2_path``
    column (the acquisition inventory is the record of what was acquired and from which file). The
    entry guard asserts that column is unambiguous for the well, so there is one source of truth.

    This backend is an EXECUTOR: it accepts a ``ResolvedImageProduct`` (a commitment, never a request)
    and asserts it resolves to ``xy_composition='identity'`` — YX1 has one tile per
    well/channel/time, so identity is its only XY composition. The assert fails loud if a non-identity
    product ever reaches here (guards against a future resolver/router bug).

    Args:
        experiment_id: global experiment identifier.
        well_id: global well identifier (``{experiment_id}_{well_index}``).
        well_index: local well label (e.g. ``"B01"``).
        well_acquisition_inventory_df: acquisition inventory rows for THIS well only.
            Must carry ``position_index``, ``time_index``, ``source_nd2_path``,
            ``micrometers_per_pixel``, ``image_width_px``, ``image_height_px``.
        built_image_data_dir: stage root (``DATA_ROOT / "built_image_data"``), resolved
            by the caller.
        resolved_product: one scope-resolved product (from ``resolve_materialization_plan`` product
            fanout). Must be BF / identity; product type chooses projection vs z_stack behavior.
        device: PyTorch device for focus stacking (``"cuda"`` or ``"cpu"``).
        candidate: if ``True``, writes under ``materialized_images/candidate/`` so paths
            can never collide with the live tree.
        smoke_max_time_indices: TEMPORARY smoke cap — if set, only the first N time_indices are
            materialized (no-GPU / fast smoke). ``None`` = full well (production).
        config: pipeline config dict; only ``image_materialization.write_policies`` is consumed
            here, keyed by canonical product_key.

    Returns:
        DataFrame with columns ``_EMITTED_COLUMNS`` — one row per materialized frame.
        ``well_id`` and ``image_id`` are derived from atoms and written for downstream consumers;
        the contract validator recomputes them and fails loud on disagreement.

    Raises:
        ValueError: on well_id / well_index / experiment_id inconsistency, empty
            inventory, ambiguous position_index / source_nd2_path, or unsupported resolved product.
    """
    # --- Entry guard: executor (identity-only) + well/inventory consistency + source readability -
    # Executor guard — this backend only does identity XY composition. Fail loud otherwise.
    if resolved_product.xy_composition != "identity":
        raise ValueError(
            f"materialize_yx1_product_for_well only executes xy_composition='identity'; got "
            f"{resolved_product.xy_composition!r}. YX1 is single-tile — this indicates a resolver bug."
        )
    if resolved_product.channel_id != "BF":
        raise ValueError(
            f"materialize_yx1_product_for_well only supports channel_id='BF'; "
            f"got {resolved_product.channel_id!r}."
        )
    if resolved_product.image_product_type == "projection":
        if resolved_product.projection_method != "focus_stack":
            raise ValueError(
                "YX1 projection materialization requires projection_method='focus_stack'."
            )
    elif resolved_product.image_product_type == "z_stack":
        if resolved_product.projection_method is not None:
            raise ValueError("YX1 z_stack materialization requires projection_method=None.")
    else:
        raise ValueError(
            f"Unsupported YX1 image_product_type {resolved_product.image_product_type!r}."
        )
    product_key = image_product_key_for_resolved_product(resolved_product)
    write_policy = resolve_image_write_policy(config, product_key)
    ext = suffix_for_policy(write_policy)

    # Entry guard — fail loud before any disk work.
    expected_well_id = derive_well_id(experiment_id, well_index)
    if expected_well_id != well_id:
        raise ValueError(
            f"well_id mismatch: derive_well_id({experiment_id!r}, {well_index!r}) "
            f"= {expected_well_id!r} but caller passed well_id={well_id!r}."
        )
    if well_acquisition_inventory_df.empty:
        raise ValueError(
            f"well_acquisition_inventory_df is empty for well {well_id!r}."
        )
    if well_acquisition_inventory_df["position_index"].nunique() != 1:
        raise ValueError(
            f"Ambiguous position_index for well {well_id!r}: "
            f"{sorted(well_acquisition_inventory_df['position_index'].unique())}. "
            "Inventory must map exactly one position_index to one well."
        )
    if well_acquisition_inventory_df["source_nd2_path"].nunique() != 1:
        raise ValueError(
            f"Ambiguous source_nd2_path for well {well_id!r}: "
            f"{sorted(well_acquisition_inventory_df['source_nd2_path'].unique())}."
        )

    # Consume-boundary contract check: re-validate the acquisition inventory in source-checking mode
    # so a moved/deleted/corrupt ND2 fails loud HERE (named, with the fix) before any tensor read —
    # not as a raw nd2.ND2File traceback below. The readability logic is OWNED by the acquisition
    # contract; this backend only calls it. (The nunique()==1 guard above stays as a local tripwire.)
    validate_yx1_acquisition_inventory(well_acquisition_inventory_df, check_sources=True)

    position_index = int(well_acquisition_inventory_df["position_index"].iloc[0])
    um_per_px = float(well_acquisition_inventory_df["micrometers_per_pixel"].iloc[0])
    img_w = int(well_acquisition_inventory_df["image_width_px"].iloc[0])
    img_h = int(well_acquisition_inventory_df["image_height_px"].iloc[0])
    # The ND2 source comes from the inventory (the record of what was acquired), not a CLI arg.
    nd2_path = Path(well_acquisition_inventory_df["source_nd2_path"].iloc[0])

    log.info(
        "materialize_yx1_product_for_well: experiment=%s well=%s product=%s position_index=%d nd2=%s device=%s candidate=%s",
        experiment_id, well_id, resolved_product.image_product_type, position_index, nd2_path, device, candidate,
    )

    # --- ND2 source + tensor setup: open the one ND2, pick the BF channel axis ----------------
    nd = nd2.ND2File(nd2_path)
    try:
        dask_arr = nd.to_dask()
        channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
        bf_idx = _determine_bf_channel(channel_names)

        # Select BF channel axis if present (shape T,W,Z,C,Y,X → T,W,Z,Y,X).
        if dask_arr.ndim == 6:
            dask_arr = dask_arr[:, :, :, bf_idx, :, :]

        time_indices = sorted(well_acquisition_inventory_df["time_index"].unique())
        if smoke_max_time_indices is not None and smoke_max_time_indices > 0:
            # ┌─────────────────────────────────────────────────────────────────────────────────┐
            # │ DEVELOPMENT-ONLY SCAFFOLDING — NOT a frame-selection policy.                       │
            # │ smoke_max_time_indices limits materialization to the first N time_index values so  │
            # │ the live front-half spine can be smoke-tested cheaply during pipeline development.  │
            # │ It takes frames in raw time_index order — it is NOT a scientific sampling           │
            # │ mechanism and must not be used as one.                                             │
            # │ TODO(frame_selection): replace with an explicit frame_selection contract that      │
            # │ derives the time-index subset systematically from the acquisition inventory /      │
            # │ experiment design (or remove this) once full materialization is stable.            │
            # └─────────────────────────────────────────────────────────────────────────────────┘
            n_full = len(time_indices)
            time_indices = time_indices[:smoke_max_time_indices]
            # Unmistakable signal that this run is capped (so a capped shard is never mistaken
            # for a full one). Loud on BOTH the log and stdout.
            msg = (
                f"SMOKE_FRAME_CAP_ACTIVE: limiting materialization to first "
                f"{len(time_indices)} of {n_full} time_index values for well {well_id} "
                f"(development scaffolding — NOT a production frame-selection policy)."
            )
            log.warning(msg)
            print(msg, flush=True)
        rows: list[dict] = []

        # Per-time_index time-column lookup, carried through from the acquisition inventory (the
        # OWNER/deriver of the time block — see specs/acquisition_inventory_schema_policy.md). The
        # values are constant across z/channel within a time_index, so one row per time_index suffices.
        time_cols = ["elapsed_time_s", "acquisition_time_s"]
        time_lookup = (
            well_acquisition_inventory_df.drop_duplicates("time_index")
            .set_index("time_index")[time_cols]
            .to_dict("index")
        )
        z_lookup = (
            well_acquisition_inventory_df.groupby("time_index")["z_index"]
            .apply(lambda s: sorted(int(z) for z in s.dropna().unique()))
            .to_dict()
        )

        # --- Materialization loop: per time_index → product frame(s) → write PNG → record rows ---
        for t in time_indices:
            stack = _get_stack(dask_arr, t=t, w=position_index)
            t_times = time_lookup[t]
            if resolved_product.image_product_type == "projection":
                ff, focus_index_map = materialize_ff_projection(stack, device=device)

                out_path = materialized_image_paths.projection_frame_path(
                    built_image_data_dir,
                    experiment_id=experiment_id,
                    well_id=well_id,
                    channel_id="BF",
                    time_index=t,
                    projection_method="focus_stack",
                    ext=ext,
                    candidate=candidate,
                )
                write_image(ff, out_path, write_policy)
                out_w, out_h = expected_downsampled_dims(
                    img_w, img_h, write_policy.downsample_factor, write_policy.downsample_method
                )

                # --- Construction provenance: the focus_index_map .npz (focus_stack only) ----------
                # focus_index_map values are STACK-AXIS OFFSETS into `stack` (axis 0). z_indices is
                # the ordered list of acquisition z_index labels for those offsets — built from the
                # SAME inventory rows / order used to load `stack`. For YX1 the stack is loaded in
                # z_index order (range(n_z)), so sorted(z_lookup[t]) is exactly the axis order.
                z_indices = np.asarray(z_lookup.get(t, []), dtype=np.int32)
                if focus_index_map.shape[0] != stack.shape[1] or focus_index_map.shape[1] != stack.shape[2]:
                    raise ValueError(
                        f"focus_index_map shape {focus_index_map.shape} does not match the "
                        f"(Y, X) of stack {stack.shape} for well {well_id} time_index={t}."
                    )
                fim_path = materialized_image_paths.focus_index_map_path(
                    built_image_data_dir,
                    experiment_id=experiment_id,
                    well_id=well_id,
                    channel_id="BF",
                    time_index=t,
                    candidate=candidate,
                )
                fim_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    fim_path,
                    focus_index_map=focus_index_map.astype(np.int32),
                    z_indices=z_indices,
                )

                image_id = derive_image_id(well_id, "BF", int(t))
                rows.append(_frame_inventory_row(
                    experiment_id=experiment_id,
                    well_index=well_index,
                    well_id=well_id,
                    time_index=int(t),
                    image_id=image_id,
                    elapsed_time_s=t_times["elapsed_time_s"],
                    acquisition_time_s=t_times["acquisition_time_s"],
                    z_index=pd.NA,
                    image_product_type="projection",
                    projection_method="focus_stack",
                    source_image_path=out_path,
                    focus_index_map_path=fim_path,
                    source_micrometers_per_pixel=um_per_px,
                    source_image_width_px=img_w,
                    source_image_height_px=img_h,
                    image_width_px=out_w,
                    image_height_px=out_h,
                    write_policy=write_policy,
                ))
            else:
                for z_index in z_lookup.get(t, []):
                    if z_index >= stack.shape[0]:
                        raise ValueError(
                            f"Inventory declares z_index={z_index} for well {well_id} time_index={t}, "
                            f"but the ND2 stack has only {stack.shape[0]} plane(s). Re-run scope ingest "
                            "or restore the matching raw ND2."
                        )
                    out_path = materialized_image_paths.z_stack_frame_path(
                        built_image_data_dir,
                        experiment_id=experiment_id,
                        well_id=well_id,
                        channel_id="BF",
                        time_index=t,
                        z_index=z_index,
                        ext=ext,
                        candidate=candidate,
                    )
                    write_image(stack[z_index], out_path, write_policy)
                    out_w, out_h = expected_downsampled_dims(
                        img_w, img_h, write_policy.downsample_factor, write_policy.downsample_method
                    )

                    image_id = derive_image_id(well_id, "BF", int(t), z_index=z_index)
                    rows.append(_frame_inventory_row(
                        experiment_id=experiment_id,
                        well_index=well_index,
                        well_id=well_id,
                        time_index=int(t),
                        image_id=image_id,
                        elapsed_time_s=t_times["elapsed_time_s"],
                        acquisition_time_s=t_times["acquisition_time_s"],
                        z_index=int(z_index),
                        image_product_type="z_stack",
                        projection_method=pd.NA,
                        source_image_path=out_path,
                        focus_index_map_path=None,  # z_stack rows carry no focus provenance
                        source_micrometers_per_pixel=um_per_px,
                        source_image_width_px=img_w,
                        source_image_height_px=img_h,
                        image_width_px=out_w,
                        image_height_px=out_h,
                        write_policy=write_policy,
                    ))

            if (len(rows) % 10) == 0:
                log.info("  %d/%d frames written", len(rows), len(time_indices))

    finally:
        nd.close()

    # --- Inventory assembly: build the frame-inventory shard + final required-columns check ---
    inv_df = pd.DataFrame(rows, columns=list(_EMITTED_COLUMNS))

    # Sanity check — all required atom columns present.
    missing = [c for c in REQUIRED_FRAME_INVENTORY_COLUMNS if c not in inv_df.columns]
    if missing:
        raise RuntimeError(
            f"materialize_yx1_well produced a frame-inventory DataFrame missing required "
            f"columns: {missing}. Present: {list(inv_df.columns)}."
        )

    log.info(
        "materialize_yx1_product_for_well complete: well=%s product=%s frames=%d",
        well_id, resolved_product.image_product_type, len(inv_df),
    )
    return inv_df


def _frame_inventory_row(
    *,
    experiment_id: str,
    well_index: str,
    well_id: str,
    time_index: int,
    image_id: str,
    elapsed_time_s: float,
    acquisition_time_s: float,
    z_index: object,
    image_product_type: str,
    projection_method: object,
    source_image_path: Path,
    focus_index_map_path: object,
    source_micrometers_per_pixel: float,
    source_image_width_px: int,
    source_image_height_px: int,
    image_width_px: int,
    image_height_px: int,
    write_policy: ImageWritePolicy,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "well_index": well_index,
        "well_id": well_id,
        "channel_id": "BF",
        "time_index": time_index,
        "image_id": image_id,
        "elapsed_time_s": elapsed_time_s,
        "acquisition_time_s": acquisition_time_s,
        "z_index": z_index,
        "image_product_type": image_product_type,
        "projection_method": projection_method,
        "source_image_path": str(source_image_path),
        # Construction-provenance path (NOT a primary image): the focus_stack focus_index_map .npz,
        # populated for projection/focus_stack rows, NA otherwise.
        "focus_index_map_path": (
            pd.NA if focus_index_map_path is None or focus_index_map_path is pd.NA
            else str(focus_index_map_path)
        ),
        "source_micrometers_per_pixel": source_micrometers_per_pixel,
        "source_image_width_px": source_image_width_px,
        "source_image_height_px": source_image_height_px,
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
        "image_file_format": write_policy.file_format,
        "pixel_dtype": write_policy.pixel_dtype,
        "downsample_factor": write_policy.downsample_factor,
        "downsample_method": write_policy.downsample_method,
        "jpeg_quality": write_policy.jpeg_quality,
    }
