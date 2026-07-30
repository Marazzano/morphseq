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
from PIL import Image

from data_pipeline.acquisition.metadata_ingest.scope.yx1.nd2_axes import (
    array_axis_order_of,
    axes_of,
)
from data_pipeline.acquisition.image_building.scope.yx1.stitched_ff_builder import (
    _get_stack,
)
from data_pipeline.acquisition.image_building.shared.display_polarity import apply_display_polarity
from data_pipeline.acquisition.image_building.shared.focus_stack_group import (
    FocusStackConfig,
    focus_stack_group,
)
from data_pipeline.acquisition.image_materialization import materialized_image_paths
from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
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
    resolve_image_write_policy,
    suffix_for_policy,
    write_image,
)
from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_channels import (
    resolve_channel_index,
)
from data_pipeline.acquisition.metadata_ingest.scope.yx1.acquisition_inventory import (
    validate_yx1_acquisition_inventory,
)
from data_pipeline.shared.path_roots import resolve_under_input_root

log = logging.getLogger(__name__)

# Label for resolve_under_input_root errors when reading the inventory's stored ND2 path.
_SCOPE_LABEL = "YX1 acquisition inventory"

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
    "image_path",
    "focus_index_map_path",
    "image_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    "orientation",
    "image_file_format",
    "pixel_dtype",
    "downsample_factor",
    "downsample_method",
    "jpeg_quality",
    "flip_polarity",
    "raw_image_source_path",
    "raw_image_width_px",
    "raw_image_height_px",
    "raw_micrometers_per_pixel",
)


# ---------------------------------------------------------------------------
# Image-math primitives (pure — no paths, no IDs, no side effects)
# ---------------------------------------------------------------------------


def materialize_ff_projection(
    stack_zyx: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Focus-stack a Z-stack into one 2D frame (BF / brightfield projection method).

    This is the ``projection_method='focus_stack'`` primitive. It is a thin YX1-identity
    wrapper over the shared ``focus_stack_group`` (one-stack group) — the shared layer owns
    intensity bounds, LoG scoring, raw-pixel gather, and the uint8 display transform (see
    ``image_building/shared/README.md``). Pure image math — no well_id, time_index, or paths.

    Args:
        stack_zyx: a single ``(Z, Y, X)`` uint16 brightfield z-stack for one (well, channel,
            time).
        device: torch device for the LoG convolutions (``"cpu"`` or ``"cuda"``).

    Returns:
        ``(projection_u8, focus_index_map)``:
          - ``projection_u8``: the focus-stacked 2D frame, ``uint8 (Y, X)``.
          - ``focus_index_map``: per-pixel STACK-AXIS OFFSET ``int32 (Y, X)`` — for pixel ``(y, x)``
            the value ``k`` is the Z plane (0-based axis offset into ``stack_zyx``) with the sharpest
            LoG response. It is NOT an acquisition ``z_index`` label; the caller pairs it with an
            ordered ``z_indices`` array to recover the labels.
    """
    # Delegate the focus-projection image math to the shared group primitive on a ONE-stack
    # group (YX1 identity composition). This is the sole owner of intensity bounds, LoG
    # scoring, raw-pixel gather, and the uint8 display transform — the adapter must not
    # normalize itself (see image_building/shared/README.md). One stack ⇒ one shared bound
    # pair over exactly this frame, matching legacy per-frame behavior.
    # Pure focus math only — display polarity is a WRITE-POLICY concern applied by the orchestrator
    # (per-product flip_polarity), not baked into this primitive. This keeps polarity uniform and
    # config-driven across both microscopes and both image products.
    result = focus_stack_group([stack_zyx], config=FocusStackConfig(), device=device)
    tile = result.tiles[0]
    return tile.projection_u8, tile.focus_index_map


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
    input_root: Path | None = None,
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
        input_root=input_root,
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
    input_root: Path | None = None,
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
    validate_yx1_acquisition_inventory(
        well_acquisition_inventory_df, check_sources=True, input_root=input_root
    )

    position_index = int(well_acquisition_inventory_df["position_index"].iloc[0])
    um_per_px = float(well_acquisition_inventory_df["micrometers_per_pixel"].iloc[0])
    img_w = int(well_acquisition_inventory_df["image_width_px"].iloc[0])
    img_h = int(well_acquisition_inventory_df["image_height_px"].iloc[0])

    # Resolved here (not at function entry) because a product declaring a fixed µm/px target needs
    # this well's native calibration to compute its downsample factor.
    write_policy = _build_yx1_write_policy(
        config, product_key, native_micrometers_per_pixel=um_per_px
    )
    ext = suffix_for_policy(write_policy)
    # The ND2 source comes from the inventory (the record of what was acquired), not a CLI arg.
    # Stored as a full path; re-anchored onto input_root if it has moved.
    nd2_path = resolve_under_input_root(
        well_acquisition_inventory_df["source_nd2_path"].iloc[0],
        input_root=input_root,
        scope_label=_SCOPE_LABEL,
        full_root_fallback=True,
    )

    log.info(
        "materialize_yx1_product_for_well: experiment=%s well=%s product=%s position_index=%d nd2=%s device=%s candidate=%s",
        experiment_id, well_id, resolved_product.image_product_type, position_index, nd2_path, device, candidate,
    )

    # --- ND2 source + tensor setup: open the one ND2, pick the BF channel axis ----------------
    nd = nd2.ND2File(nd2_path)
    try:
        dask_arr = nd.to_dask()

        # CHANNEL INDEX IS READ, NEVER RE-DERIVED. The scope adapter minted the
        # channel_index/raw_channel_name/channel_id triple into the acquisition inventory (a 1:1:1
        # mapping guarded by assert_channel_mapping_consistent); this looks the requested product's
        # channel_id up in it. Matching channel NAMES here would invent a second channel vocabulary
        # that drifts from the scope's channel_map.py.
        channel_index = resolve_channel_index(
            well_acquisition_inventory_df, resolved_product.channel_id
        )

        # Axis selection is by NAME (see metadata_ingest/scope/yx1/nd2_axes.py). The channel is
        # picked per-slice inside _get_stack instead of by pre-slicing axis 3, because that
        # pre-slice was gated on `ndim == 6` and so was SKIPPED for a 5-D (P, Z, C, Y, X) snapshot —
        # leaving the channel axis in place to masquerade as the Z stack, which would make
        # focus-stacking run over [BF, fluorescence] as if they were focal planes.
        axes = axes_of(nd)
        array_axis_order = array_axis_order_of(nd)
        # The recorded index must fit THIS file — catches a stale/mismatched inventory here rather
        # than as a cryptic dask index error deep in the read.
        if not 0 <= channel_index < axes.n_c:
            raise ValueError(
                f"Acquisition inventory maps channel {resolved_product.channel_id!r} to "
                f"channel_index {channel_index}, but {nd2_path} has {axes.n_c} channel(s). The "
                "inventory does not match the file it points at; re-run scope ingest."
            )
        log.info(
            "ND2 axes: T=%d P=%d Z=%d C=%d (array order %s, channel_id %s -> channel_index %d)",
            axes.n_t, axes.n_p, axes.n_z, axes.n_c, array_axis_order,
            resolved_product.channel_id, channel_index,
        )

        time_indices = sorted(well_acquisition_inventory_df["time_index"].unique())

        # MERGED time_index -> SOURCE-NATIVE frame index. For a single experiment these are the same
        # value. For a COLLECTION they are not: the union block-offsets each source's frames onto one
        # plate-wide axis, so a well's merged time_index 96 may be its source's native frame 0. The
        # output identity (image_id, paths) uses the MERGED index; the ND2 lookup must use the NATIVE
        # one, or a collection would read pixels from the wrong frame — or off the end of the file.
        # `raw_time_index` is the union's record of the source-native value (see
        # metadata_ingest/collection_merge_primitives.py).
        if "raw_time_index" in well_acquisition_inventory_df.columns:
            native_by_merged = {
                int(merged): int(native)
                for merged, native in well_acquisition_inventory_df.groupby("time_index")[
                    "raw_time_index"
                ].first().items()
            }
        else:
            native_by_merged = {int(t): int(t) for t in time_indices}

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

        # --- Materialization loop: per time_index → product frame(s) → write image → record rows --
        for t in time_indices:
            # The ND2 lookup uses the SOURCE-NATIVE frame index; `t` (merged) keys the output identity.
            stack = _get_stack(
                dask_arr,
                t=native_by_merged[int(t)],
                w=position_index,
                axes=axes,
                array_axis_order=array_axis_order,
                channel=channel_index,
            )
            t_times = time_lookup[t]
            if resolved_product.image_product_type == "projection":
                ff, focus_index_map = materialize_ff_projection(stack, device=device)
                # Per-product display polarity (shared owner), same op/flag Keyence uses.
                ff = apply_display_polarity(ff, invert=write_policy.flip_polarity)

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
                out_w, out_h = _write_image_and_read_dims(ff, out_path, write_policy)

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
                    image_path=out_path,
                    focus_index_map_path=fim_path,
                    image_micrometers_per_pixel=_materialized_um_per_px(um_per_px, write_policy),
                    image_width_px=out_w,
                    image_height_px=out_h,
                    write_policy=write_policy,
                    raw_image_source_path=nd2_path,
                    raw_image_width_px=img_w,
                    raw_image_height_px=img_h,
                    raw_micrometers_per_pixel=um_per_px,
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
                    # Per-product display polarity (shared owner) — same flag as Keyence z_stack,
                    # so z_stacks are consistent across microscopes (previously YX1 z_stack was raw).
                    z_plane = apply_display_polarity(stack[z_index], invert=write_policy.flip_polarity)
                    out_w, out_h = _write_image_and_read_dims(z_plane, out_path, write_policy)

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
                        image_path=out_path,
                        focus_index_map_path=None,  # z_stack rows carry no focus provenance
                        image_micrometers_per_pixel=_materialized_um_per_px(um_per_px, write_policy),
                        image_width_px=out_w,
                        image_height_px=out_h,
                        write_policy=write_policy,
                        raw_image_source_path=nd2_path,
                        raw_image_width_px=img_w,
                        raw_image_height_px=img_h,
                        raw_micrometers_per_pixel=um_per_px,
                    ))

            if (len(rows) % 10) == 0:
                log.info("  %d/%d frames written", len(rows), len(time_indices))

    finally:
        nd.close()

    # --- Inventory assembly: build the frame-inventory shard + final required-columns check ---
    inv_df = pd.DataFrame(rows, columns=list(_EMITTED_COLUMNS))


    # Migration-local sanity check: Stage 3 emits the materialized-image-first contract shape even
    # if the shared frame_inventory contract module has not been updated yet on this branch.
    missing = [c for c in _EMITTED_COLUMNS if c not in inv_df.columns]
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
    image_path: Path,
    focus_index_map_path: object,
    image_micrometers_per_pixel: float,
    image_width_px: int,
    image_height_px: int,
    write_policy: ImageWritePolicy,
    raw_image_source_path: Path,
    raw_image_width_px: int,
    raw_image_height_px: int,
    raw_micrometers_per_pixel: float,
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
        "image_path": str(image_path),
        # Construction-provenance path (NOT a primary image): the focus_stack focus_index_map .npz,
        # populated for projection/focus_stack rows, NA otherwise.
        "focus_index_map_path": (
            pd.NA if focus_index_map_path is None or focus_index_map_path is pd.NA
            else str(focus_index_map_path)
        ),
        "image_micrometers_per_pixel": image_micrometers_per_pixel,
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
        "orientation": write_policy.orientation,
        "image_file_format": write_policy.file_format,
        "pixel_dtype": write_policy.pixel_dtype,
        "downsample_factor": write_policy.downsample_factor,
        "downsample_method": write_policy.downsample_method,
        "jpeg_quality": (
            pd.NA if write_policy.jpeg_quality is None else int(write_policy.jpeg_quality)
        ),
        "flip_polarity": bool(write_policy.flip_polarity),
        "raw_image_source_path": str(raw_image_source_path),
        "raw_image_width_px": int(raw_image_width_px),
        "raw_image_height_px": int(raw_image_height_px),
        "raw_micrometers_per_pixel": float(raw_micrometers_per_pixel),
    }


def _build_yx1_write_policy(
    config: dict | None,
    product_key: str,
    native_micrometers_per_pixel: float | None = None,
) -> ImageWritePolicy:
    """Construct the YX1 writer policy explicitly at the writer boundary.

    This stays integration-ready with Stage 1's optional orientation field without forcing the
    current branch to have landed that dataclass change yet.
    """
    resolved = resolve_image_write_policy(
        config, product_key, native_micrometers_per_pixel=native_micrometers_per_pixel
    )
    policy_kwargs = {
        "file_format": resolved.file_format,
        # float, not int: a fixed µm/px target yields a fractional factor.
        "downsample_factor": float(resolved.downsample_factor),
        "downsample_method": resolved.downsample_method,
        "pixel_dtype": resolved.pixel_dtype,
        "jpeg_quality": resolved.jpeg_quality,
        "flip_polarity": bool(resolved.flip_polarity),
    }
    if "orientation" in getattr(ImageWritePolicy, "__dataclass_fields__", {}):
        policy_kwargs["orientation"] = "none"
    return ImageWritePolicy(**policy_kwargs)


def _write_image_and_read_dims(
    image: np.ndarray,
    out_path: Path,
    write_policy: ImageWritePolicy,
) -> tuple[int, int]:
    """Write one image through the shared policy boundary and read back its header dimensions."""
    write_image(image, out_path, write_policy)
    with Image.open(out_path) as written:
        width_px, height_px = written.size
    return int(width_px), int(height_px)


def _materialized_um_per_px(raw_um_per_px: float, write_policy: ImageWritePolicy) -> float:
    """Return the materialized-image calibration after the writer downsample policy.

    ``downsample_factor`` may be fractional when the product targets a fixed µm/px, so this must NOT
    coerce to int — doing so truncated e.g. 2.012 to 2 and recorded the wrong calibration.
    """
    return float(raw_um_per_px) * float(write_policy.downsample_factor)

