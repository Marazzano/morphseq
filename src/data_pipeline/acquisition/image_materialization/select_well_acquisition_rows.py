"""select_well_acquisition_rows — narrow the acquisition inventory to ONE well's rows.

This is the pure domain step that used to live inline in ``tasks.py::cmd_materialize_well``: join
the scope acquisition inventory to the canonical ``position_well_mapping`` (so each acquisition row
gains its ``well_id``), then slice out the rows for the requested well. DataFrames in → a row-slice
out. No disk read, no ``argparse``, no CLI conventions — that plumbing stays in ``tasks.py``; this
function is directly unit-testable.

Why it lives in ``image_materialization/`` and NOT in orchestration: this is INPUT prep for the
materialize step (it produces the acquisition rows ``run_materialize_well`` consumes), not
well SCHEDULING. The well scheduler — "which wells run, and where are their shard FILES?" — is
``orchestration/well_runner.py``, which works in well_ids + paths and never opens a row. The two are
on opposite sides of the materialize step and must not be confused:

    well_runner:  which well_ids run? → per-well frame_inventory SHARD paths (the OUTPUT)
    THIS file:    acquisition_inventory rows → the ROWS for one well (the INPUT)

So this is "select the acquisition rows" (a slice of the input inventory), deliberately NOT named a
"shard" — a shard is well_runner's frame_inventory OUTPUT artifact. This module imports identity
(via the join columns the mapping carries) but NOT orchestration: a domain kingdom never reaches up
into the scheduler.
"""

from __future__ import annotations

import pandas as pd


def select_well_acquisition_rows(
    acquisition_inventory_df: pd.DataFrame,
    position_well_mapping_df: pd.DataFrame,
    *,
    experiment_id: str,
    well_id: str,
) -> pd.DataFrame:
    """Return the acquisition-inventory rows for ONE well (joined to its ``well_id``).

    Joins ``acquisition_inventory_df`` to ``position_well_mapping_df`` on
    ``(experiment_id, position_index)`` so each acquisition row gains ``well_index``/``well_id``,
    then filters to ``(experiment_id, well_id)``. The mapping is the canonical microscope-AGNOSTIC
    bridge from acquisition position to well identity, so this step learns no scope quirks.

    The caller is responsible for validating ``position_well_mapping_df`` before handing it in
    (``tasks.py`` calls ``validate_position_well_mapping`` at the file boundary) — this function
    assumes a valid mapping and does the domain join/filter.

    Args:
        acquisition_inventory_df: the scope acquisition inventory (``experiment_id`` already string).
        position_well_mapping_df: the canonical ``position_well_mapping`` (already validated), at
            least columns ``experiment_id, position_index, well_index, well_id``.
        experiment_id: the experiment to scope the join + filter to.
        well_id: the global well identifier whose rows to return.

    Returns:
        The acquisition rows for ``well_id`` (carrying ``source_nd2_path`` + the joined
        ``well_index``/``well_id``), ready to hand to ``run_materialize_well``.

    Raises:
        ValueError: if no acquisition rows resolve to ``well_id`` after the join (names the fix).
    """
    experiment_id = str(experiment_id)
    well_id = str(well_id)

    acquisition_inventory_df = acquisition_inventory_df.copy()
    acquisition_inventory_df["experiment_id"] = acquisition_inventory_df["experiment_id"].astype(str)
    # Drop identity columns that will be authoritatively provided by the mapping join.
    # Some scope inventories (e.g. Keyence) already carry well_index / well_id minted at ingest;
    # keeping them causes pandas to emit _x/_y suffixes, breaking the downstream filter.
    for _col in ("well_index", "well_id"):
        if _col in acquisition_inventory_df.columns:
            acquisition_inventory_df = acquisition_inventory_df.drop(columns=[_col])

    mapping = position_well_mapping_df.copy()
    mapping["experiment_id"] = mapping["experiment_id"].astype(str)
    mapping = mapping[mapping["experiment_id"] == experiment_id]

    joined = acquisition_inventory_df.merge(
        mapping[["experiment_id", "position_index", "well_index", "well_id"]],
        on=["experiment_id", "position_index"],
        how="left",
        validate="many_to_one",
    )

    well_rows = joined[
        (joined["experiment_id"].astype(str) == experiment_id)
        & (joined["well_id"].astype(str) == well_id)
    ]
    if well_rows.empty:
        raise ValueError(
            f"No acquisition inventory rows resolve to experiment={experiment_id!r}, "
            f"well_id={well_id!r} after joining position_well_mapping. Check that the mapping "
            f"covers this well's position_index and that the well_id is well-formed."
        )
    return well_rows
