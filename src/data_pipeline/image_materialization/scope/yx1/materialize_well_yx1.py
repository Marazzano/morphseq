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
import skimage.io as skio
import skimage.util
import torch

from data_pipeline.image_building.scope.yx1.stitched_ff_builder import (
    _determine_bf_channel,
    _get_stack,
)
from data_pipeline.image_building.shared.log_focus import LoG_focus_stacker, im_rescale
from data_pipeline.image_materialization import materialized_image_paths
from data_pipeline.image_materialization.frame_inventory_contract import (
    REQUIRED_FRAME_INVENTORY_COLUMNS,
    derive_image_id,
    derive_well_id,
)
from data_pipeline.image_materialization.materialization_plan import (
    ResolvedImageProduct,
    ResolvedMaterializationPlan,
)
from data_pipeline.metadata_ingest.scope.yx1.acquisition_inventory import (
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
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
)


# ---------------------------------------------------------------------------
# Image-math primitives (pure — no paths, no IDs, no side effects)
# ---------------------------------------------------------------------------


def materialize_ff_projection(stack_zyx: np.ndarray, *, device: str) -> np.ndarray:
    """Focus-stack a Z-stack into one 2D frame (BF / brightfield projection method).

    Returns a uint8 2D array.  This is the ``projection_method='focus_stack'`` primitive.
    Does not know about well_id, time_index, or paths — pure image math.
    """
    norm, _, _ = im_rescale(stack_zyx)
    ff, _ = LoG_focus_stacker(norm.astype(np.float32), filter_size=3, device=device)
    arr = ff.cpu().numpy() if torch.is_tensor(ff) else np.asarray(ff)
    return skimage.util.img_as_ubyte(np.clip(arr, 0, 65535).astype(np.uint16))


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
                ff = materialize_ff_projection(stack, device=device)

                out_path = materialized_image_paths.projection_frame_path(
                    built_image_data_dir,
                    experiment_id=experiment_id,
                    well_id=well_id,
                    channel_id="BF",
                    time_index=t,
                    candidate=candidate,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                skio.imsave(str(out_path), ff, check_contrast=False)

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
                    source_micrometers_per_pixel=um_per_px,
                    image_width_px=img_w,
                    image_height_px=img_h,
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
                        candidate=candidate,
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    skio.imsave(str(out_path), stack[z_index], check_contrast=False)

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
                        source_micrometers_per_pixel=um_per_px,
                        image_width_px=img_w,
                        image_height_px=img_h,
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
    source_micrometers_per_pixel: float,
    image_width_px: int,
    image_height_px: int,
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
        "source_micrometers_per_pixel": source_micrometers_per_pixel,
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
    }
