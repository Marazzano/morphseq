"""run_materialize_well — the pipeline-facing SEQUENCER for one well.

Separation of concerns (the three roles):
  - ``run_materialize_well`` (THIS file) = sequencer: load plan, call resolver, call backend.
    It RECEIVES already-resolved raw inputs (nd2_path, inventory shard, config) from tasks.py /
    Snakemake plumbing — it does not gather or resolve file arguments itself.
  - ``scope_resolver_for_materialization_plan`` = semantic translator: request plan → resolved
    plan. It does NOT touch a backend.
  - ``scope/yx1/materialize_well_yx1`` = executor: resolved plan + raw inputs → images + inventory.

Call chain:

    tasks.py / Snakemake                       = parses CLI args, passes file paths + loaded tables
      └─► run_materialize_well()               = THIS file — the sequencer
            ├─ load_image_materialization_plan(config)        (the request)
            ├─ resolve_materialization_plan(scope, requested) (→ resolved plan; no backend)
            └─ call scope backend with resolved plan + the raw inputs it was handed

The resolver owns plan-narrowing scope quirks; the sequencer owns backend selection. Both are
scope-aware, for different reasons. Step 6 backend selection is YX1-only. The raw-source pointer
travels INSIDE the acquisition inventory (its ``source_nd2_path`` column) rather than as a bespoke
argument — so the sequencer's signature is already scope-neutral; only backend selection is
YX1-only. Making backend selection itself scope-neutral is the NEXT migration, done together with
the Keyence resolver route + backend. Until then this sequencer calls only the ``yx1`` backend.

Import rules: imports ``materialization_plan``, the resolver, and (lazily) the YX1 backend. It MUST
NOT import orchestration paths or Snakemake rules — paths are resolved by callers / the backend via
``materialized_image_paths.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    UnsupportedScopeError,
    load_image_materialization_plan,
)
from data_pipeline.acquisition.image_materialization.resolved_product_plans import (
    load_resolved_product_plan_for_well,
)
from data_pipeline.acquisition.image_materialization.scope.scope_resolver_for_materialization_plan import (
    resolve_materialization_plan,
)
from data_pipeline.utils.cuda_diagnostics import resolve_device

log = logging.getLogger(__name__)


def run_materialize_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    scope_name: str,
    well_acquisition_inventory_df: pd.DataFrame,  # carries source_nd2_path (the raw-source pointer)
    built_image_data_dir: Path,
    config: dict | None = None,
    device: str = "auto",
    candidate: bool = False,
    smoke_max_time_indices: int | None = None,
    master_params_path: Path | None = None,
    input_root: Path | None = None,
) -> pd.DataFrame:
    """Materialize the configured image-product set for ONE well; return frame-inventory rows.

    The workflow:
      1. Load the requested plan from ``config`` (defaults to the one accepted YX1 product).
      2. Resolve it for ``scope_name`` (request → commitment; scope quirks applied here).
      3. Route to the scope backend, handing it the RESOLVED plan and nothing to interpret.

    Live backends: ``"yx1"`` and ``"keyence"``; any other ``scope_name`` raises
    ``UnsupportedScopeError``.

    The ``device`` preference (``"auto"`` / ``"cuda"`` / ``"cpu"``) is auto-resolved HERE via the shared
    ``resolve_device`` (the same resolver the rest of the pipeline uses); the backend receives a concrete
    device. ``"auto"`` (the default) → CUDA if available, else CPU. The chosen device is announced on
    stdout as ``AUTO_MODE_CHOSEN: requested=... -> CUDA|CPU`` so a run's device decision is observable.

    Args:
        scope_name: microscope key. Supported: ``"yx1"``, ``"keyence"``.
        config: pipeline config dict (``image_materialization.products``); ``None`` → default plan.
        device: device preference — ``"auto"`` (default), ``"cuda"``, or ``"cpu"``. Resolved to a
            concrete device by ``resolve_device`` before the backend runs.
        candidate: ``True`` writes under ``materialized_images/candidate/`` (isolated from live).
        smoke_max_time_indices: TEMPORARY smoke cap — if set, the backend materializes only the
            first N time_indices (no-GPU / fast smoke). ``None`` = full well (production).

    Returns:
        Per-well frame-inventory DataFrame (flat schema; derived ids recomputed by the validator).

    Raises:
        UnsupportedScopeError on a non-yx1 scope; UnsupportedMaterializationRequest on a bad plan.
    """
    scope_name = str(scope_name).strip().lower()

    requested_plan = load_image_materialization_plan(config)
    resolved_plan = resolve_materialization_plan(
        scope_name=scope_name,
        requested_plan=requested_plan,
    )

    # Auto-resolve the device preference (auto/cuda/cpu) to a concrete device the backend can use,
    # and announce the choice so it is observable in the run output.
    resolved_device = resolve_device(device)
    log.info("AUTO_MODE_CHOSEN: requested=%s -> %s", device, resolved_device.upper())
    print(f"AUTO_MODE_CHOSEN: requested={device!r} -> {resolved_device.upper()}", flush=True)

    log.info(
        "run_materialize_well: experiment=%s well=%s scope=%s products=%d device=%s candidate=%s",
        experiment_id, well_id, scope_name, len(resolved_plan.products), resolved_device, candidate,
    )

    if scope_name == "yx1":
        from data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1 import (
            materialize_yx1_well,
        )
        return materialize_yx1_well(
            experiment_id=experiment_id,
            well_id=well_id,
            well_index=well_index,
            well_acquisition_inventory_df=well_acquisition_inventory_df,
            built_image_data_dir=built_image_data_dir,
            resolved_plan=resolved_plan,
            device=resolved_device,
            candidate=candidate,
            smoke_max_time_indices=smoke_max_time_indices,
            config=config,
            input_root=input_root,
        )
    if scope_name == "keyence":
        from data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence import (
            materialize_keyence_product_for_well,
        )
        if len(resolved_plan.products) != 1:
            raise ValueError(
                "run_materialize_well Keyence path expects exactly one resolved product; "
                f"got {len(resolved_plan.products)}. Fan out before calling."
            )
        return materialize_keyence_product_for_well(
            experiment_id=experiment_id,
            well_id=well_id,
            well_index=well_index,
            well_acquisition_inventory_df=well_acquisition_inventory_df,
            built_image_data_dir=built_image_data_dir,
            resolved_product=resolved_plan.products[0],
            device=resolved_device,
            candidate=candidate,
            smoke_max_time_indices=smoke_max_time_indices,
            master_params_path=master_params_path,
            input_root=input_root,
        )
    raise UnsupportedScopeError(
        f"No live materialization backend for scope {scope_name!r}. "
        "Supported scopes: 'yx1', 'keyence'."
    )


def run_materialize_image_product_for_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    scope_name: str,
    well_acquisition_inventory_df: pd.DataFrame,
    built_image_data_dir: Path,
    resolved_product_plan_json: Path,
    product_key: str,
    config: dict | None = None,
    device: str = "auto",
    candidate: bool = False,
    smoke_max_time_indices: int | None = None,
    master_params_path: Path | None = None,
    input_root: Path | None = None,
) -> pd.DataFrame:
    """Materialize one resolved image product for one well; return its product frame-inventory."""
    scope_name = str(scope_name).strip().lower()
    plan = load_resolved_product_plan_for_well(
        resolved_product_plan_json,
        expected_experiment_id=experiment_id,
        expected_well_id=well_id,
        expected_product_key=product_key,
    )
    if plan.scope_name != scope_name:
        raise ValueError(
            f"resolved_product_plan scope_name={plan.scope_name!r} disagrees with requested "
            f"scope_name={scope_name!r}."
        )

    resolved_device = resolve_device(device)
    log.info("AUTO_MODE_CHOSEN: requested=%s -> %s", device, resolved_device.upper())
    print(f"AUTO_MODE_CHOSEN: requested={device!r} -> {resolved_device.upper()}", flush=True)

    if scope_name == "yx1":
        from data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1 import (
            materialize_yx1_product_for_well,
        )
        return materialize_yx1_product_for_well(
            experiment_id=experiment_id,
            well_id=well_id,
            well_index=well_index,
            well_acquisition_inventory_df=well_acquisition_inventory_df,
            built_image_data_dir=built_image_data_dir,
            resolved_product=plan.product,
            device=resolved_device,
            candidate=candidate,
            smoke_max_time_indices=smoke_max_time_indices,
            config=config,
            input_root=input_root,
        )
    if scope_name == "keyence":
        from data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence import (
            materialize_keyence_product_for_well,
        )
        return materialize_keyence_product_for_well(
            experiment_id=experiment_id,
            well_id=well_id,
            well_index=well_index,
            well_acquisition_inventory_df=well_acquisition_inventory_df,
            built_image_data_dir=built_image_data_dir,
            resolved_product=plan.product,
            device=resolved_device,
            candidate=candidate,
            smoke_max_time_indices=smoke_max_time_indices,
            master_params_path=master_params_path,
            input_root=input_root,
        )
    raise UnsupportedScopeError(
        f"No live materialization backend for scope {scope_name!r}. "
        "Supported scopes: 'yx1', 'keyence'."
    )


def run_materialize_image_products_for_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    scope_name: str,
    well_acquisition_inventory_df: pd.DataFrame,
    built_image_data_dir: Path,
    resolved_product_plan_jsons: dict[str, Path],
    config: dict | None = None,
    device: str = "auto",
    candidate: bool = False,
    smoke_max_time_indices: int | None = None,
    master_params_path: Path | None = None,
    input_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Materialize EVERY requested image product for one well in ONE process.

    The multi-product twin of :func:`run_materialize_image_product_for_well`. Same sequencer role —
    load the commitments, resolve the device, hand a backend nothing to interpret — but it passes the
    well's WHOLE product plan so the backend can share the acquisition-derived work (one raw read,
    one focus reduction, one tile transform per frame) and branch only at the write boundary.

    Keyence-only: it exists to remove the multi-TIFF duplication that a per-product process shape
    forces. YX1 reads one ND2 per well and has no such duplication, so it keeps the per-product
    entry point.

    Args:
        resolved_product_plan_jsons: ``{product_key: path to that product's resolved plan JSON}``.

    Returns:
        ``{product_key: frame-inventory DataFrame}``, one entry per requested product.
    """
    scope_name = str(scope_name).strip().lower()
    if not resolved_product_plan_jsons:
        raise ValueError(
            f"run_materialize_image_products_for_well: no resolved product plans given for well "
            f"{well_id!r}."
        )

    plans = {}
    for product_key, plan_path in resolved_product_plan_jsons.items():
        plan = load_resolved_product_plan_for_well(
            plan_path,
            expected_experiment_id=experiment_id,
            expected_well_id=well_id,
            expected_product_key=product_key,
        )
        if plan.scope_name != scope_name:
            raise ValueError(
                f"resolved_product_plan scope_name={plan.scope_name!r} disagrees with requested "
                f"scope_name={scope_name!r} (product {product_key!r})."
            )
        plans[product_key] = plan

    resolved_device = resolve_device(device)
    log.info("AUTO_MODE_CHOSEN: requested=%s -> %s", device, resolved_device.upper())
    print(f"AUTO_MODE_CHOSEN: requested={device!r} -> {resolved_device.upper()}", flush=True)

    if scope_name == "keyence":
        from data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence import (
            materialize_keyence_products_for_well,
        )
        return materialize_keyence_products_for_well(
            experiment_id=experiment_id,
            well_id=well_id,
            well_index=well_index,
            well_acquisition_inventory_df=well_acquisition_inventory_df,
            built_image_data_dir=built_image_data_dir,
            resolved_products=tuple(plans[key].product for key in resolved_product_plan_jsons),
            device=resolved_device,
            candidate=candidate,
            smoke_max_time_indices=smoke_max_time_indices,
            master_params_path=master_params_path,
            config=config,
            input_root=input_root,
        )
    raise UnsupportedScopeError(
        f"Multi-product materialization is wired for scope 'keyence' only; got {scope_name!r}. "
        "Other scopes use run_materialize_image_product_for_well (one product per process)."
    )
