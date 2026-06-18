"""run_materialize_well — the pipeline-facing SEQUENCER for one well.

Separation of concerns (the three roles):
  - ``run_materialize_well`` (THIS file) = sequencer: load plan, call resolver, call backend.
    It RECEIVES already-resolved raw inputs (nd2_path, inventory shard, config) from tasks.py /
    Snakemake plumbing — it does not gather or resolve file arguments itself.
  - ``scope_resolver_for_materialization_plan`` = semantic translator: request plan → resolved
    plan. It does NOT touch a backend.
  - ``scope/yx1/materialize_well_yx1`` = executor: resolved plan + raw inputs → images + inventory.

Call chain:

    tasks.py / Snakemake                       = resolves CLI/file args (nd2_path, inventory, config)
      └─► run_materialize_well()               = THIS file — the sequencer
            ├─ load_image_materialization_plan(config)        (the request)
            ├─ resolve_materialization_plan(scope, requested) (→ resolved plan; no backend)
            └─ call scope backend with resolved plan + the raw inputs it was handed

The resolver owns plan-narrowing scope quirks; the sequencer owns backend selection. Both are
scope-aware, for different reasons. Step 6 backend selection is YX1-only: the raw-source argument
(``nd2_path``) is still YX1-shaped (Keyence has no ND2). Making raw-source inputs scope-neutral is
the NEXT migration, done together with the Keyence resolver route + backend. Until then this
sequencer calls only the ``yx1`` backend; any other scope fails loud.

Import rules: imports ``materialization_plan``, the resolver, and (lazily) the YX1 backend. It MUST
NOT import orchestration paths or Snakemake rules — paths are resolved by callers / the backend via
``materialized_image_paths.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.image_materialization.materialization_plan import (
    UnsupportedScopeError,
    load_image_materialization_plan,
)
from data_pipeline.image_materialization.scope.scope_resolver_for_materialization_plan import (
    resolve_materialization_plan,
)

log = logging.getLogger(__name__)


def run_materialize_well(
    *,
    experiment_id: str,
    well_id: str,
    well_index: str,
    scope_name: str,
    well_acquisition_inventory_df: pd.DataFrame,
    nd2_path: Path,  # YX1-only Step-6 live input; not scope-neutral yet (see module docstring)
    built_image_data_dir: Path,
    config: dict | None = None,
    device: str = "cuda",
    candidate: bool = False,
    smoke_max_time_indices: int | None = None,
) -> pd.DataFrame:
    """Materialize the configured image-product set for ONE well; return frame-inventory rows.

    The workflow:
      1. Load the requested plan from ``config`` (defaults to the one accepted YX1 product).
      2. Resolve it for ``scope_name`` (request → commitment; scope quirks applied here).
      3. Route to the scope backend, handing it the RESOLVED plan and nothing to interpret.

    Step 6 is YX1-only live: any ``scope_name`` other than ``"yx1"`` raises ``UnsupportedScopeError``.

    Args:
        scope_name: microscope key. Step 6 supports ``"yx1"`` only.
        config: pipeline config dict (``image_materialization.products``); ``None`` → default plan.
        candidate: ``True`` writes under ``materialized_images/candidate/`` (isolated from live).
        smoke_max_time_indices: TEMPORARY smoke cap — if set, the backend materializes only the
            first N time_indices (no-GPU / fast smoke). ``None`` = full well (production).

    Returns:
        Per-well frame-inventory DataFrame (flat schema; derived ids recomputed by the validator).

    Raises:
        UnsupportedScopeError on a non-yx1 scope; UnsupportedMaterializationRequest on a bad plan.
    """
    if scope_name != "yx1":
        raise UnsupportedScopeError(
            f"No live materialization backend for scope {scope_name!r}. "
            "Step 6 supports only 'yx1'. Keyence is planned but not wired."
        )

    requested_plan = load_image_materialization_plan(config)
    resolved_plan = resolve_materialization_plan(
        scope_name=scope_name,
        requested_plan=requested_plan,
    )
    log.info(
        "materialize_well: experiment=%s well=%s scope=%s products=%d candidate=%s",
        experiment_id, well_id, scope_name, len(resolved_plan.products), candidate,
    )

    from data_pipeline.image_materialization.scope.yx1.materialize_well_yx1 import (
        materialize_yx1_well,
    )

    return materialize_yx1_well(
        experiment_id=experiment_id,
        well_id=well_id,
        well_index=well_index,
        well_acquisition_inventory_df=well_acquisition_inventory_df,
        nd2_path=nd2_path,
        built_image_data_dir=built_image_data_dir,
        resolved_plan=resolved_plan,
        device=device,
        candidate=candidate,
        smoke_max_time_indices=smoke_max_time_indices,
    )
