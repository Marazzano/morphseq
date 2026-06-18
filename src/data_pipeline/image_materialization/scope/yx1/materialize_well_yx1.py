"""Per-well YX1 image materializer — projection frames from ND2 tensor slices.

This module owns the YX1-specific path from an acquisition-inventory row set to materialized
pixel files + frame-inventory rows.  It has two layers:

  Primitives (pure image math, no paths, no IDs):
    ``materialize_ff_projection``  — focus-stack a Z-stack → one uint8 2D frame (BF method)
    ``materialize_max_projection`` — max-project a Z-stack → one 2D frame (fluorescence method)

  Per-well orchestrator:
    ``materialize_yx1_well``       — reads ONE well's ND2 slices, calls the BF primitive,
                                     writes files through ``materialized_image_paths.py``,
                                     returns inventory rows.

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
    derive_well_id,
)
from data_pipeline.image_materialization.materialization_plan import (
    ResolvedMaterializationPlan,
)
from data_pipeline.metadata_ingest.scope.yx1.acquisition_inventory import (
    validate_yx1_acquisition_inventory,
)

log = logging.getLogger(__name__)

# Frame-inventory columns emitted by this module (flat schema, decided 2026-06-17).
# Derived columns (well_id, image_id) are intentionally absent — the contract validator
# recomputes them from atoms and fails loud on disagreement.
_EMITTED_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_index",
    "channel_id",
    "time_index",
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
    """Materialize projection frames for ONE YX1 well and return frame-inventory rows.

    Reads Z-stacks from the ND2 tensor (one slice per time_index), focus-stacks each into a
    BF projection frame, writes images through ``materialized_image_paths.py``, and records one frame-inventory
    row per (time_index, channel_id) pair as it goes.  BF only for Step 3.

    The ND2 source path is NOT a parameter: it is read from the inventory's ``source_nd2_path``
    column (the acquisition inventory is the record of what was acquired and from which file). The
    entry guard asserts that column is unambiguous for the well, so there is one source of truth.

    This backend is an EXECUTOR: it accepts a ``ResolvedMaterializationPlan`` (a commitment, never
    a request) and asserts every product resolves to ``xy_composition='identity'`` — YX1 has one
    tile per well/channel/time, so identity is its only XY composition. The assert fails loud if a
    non-identity product ever reaches here (guards against a future resolver/router bug).

    Args:
        experiment_id: global experiment identifier.
        well_id: global well identifier (``{experiment_id}_{well_index}``).
        well_index: local well label (e.g. ``"B01"``).
        well_acquisition_inventory_df: acquisition inventory rows for THIS well only.
            Must carry ``position_index``, ``time_index``, ``source_nd2_path``,
            ``micrometers_per_pixel``, ``image_width_px``, ``image_height_px``.
        built_image_data_dir: stage root (``DATA_ROOT / "built_image_data"``), resolved
            by the caller.
        resolved_plan: the scope-resolved product set (from ``resolve_materialization_plan``).
            Every product must be BF / projection / focus_stack / identity for Step 6.
        device: PyTorch device for focus stacking (``"cuda"`` or ``"cpu"``).
        candidate: if ``True``, writes under ``materialized_images/candidate/`` so paths
            can never collide with the live tree.
        smoke_max_time_indices: TEMPORARY smoke cap — if set, only the first N time_indices are
            materialized (no-GPU / fast smoke). ``None`` = full well (production).

    Returns:
        DataFrame with columns ``_EMITTED_COLUMNS`` — one row per materialized frame.
        ``well_id`` and ``image_id`` are absent; the contract validator derives them.

    Raises:
        ValueError: on well_id / well_index / experiment_id inconsistency, empty
            inventory, ambiguous position_index / source_nd2_path, or a non-identity resolved product.
    """
    # --- Entry guard: executor (identity-only) + well/inventory consistency + source readability -
    # Executor guard — this backend only does identity XY composition. Fail loud otherwise.
    if not resolved_plan.products:
        raise ValueError("resolved_plan has no products to materialize.")
    for product in resolved_plan.products:
        if product.xy_composition != "identity":
            raise ValueError(
                f"materialize_yx1_well only executes xy_composition='identity'; got "
                f"{product.xy_composition!r}. YX1 is single-tile — this indicates a resolver bug."
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
        "materialize_yx1_well: experiment=%s well=%s position_index=%d nd2=%s device=%s candidate=%s",
        experiment_id, well_id, position_index, nd2_path, device, candidate,
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

        # --- Materialization loop: per time_index → focus-stack → write PNG → record one row ---
        for t in time_indices:
            stack = _get_stack(dask_arr, t=t, w=position_index)
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

            rows.append({
                "experiment_id": experiment_id,
                "well_index": well_index,
                "channel_id": "BF",
                "time_index": t,
                "z_index": pd.NA,
                "image_product_type": "projection",
                "projection_method": "focus_stack",
                "source_image_path": str(out_path),
                "source_micrometers_per_pixel": um_per_px,
                "image_width_px": img_w,
                "image_height_px": img_h,
            })

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
        "materialize_yx1_well complete: well=%s frames=%d",
        well_id, len(inv_df),
    )
    return inv_df
