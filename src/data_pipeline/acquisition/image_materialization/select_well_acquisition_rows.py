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

    # An inventory that already carries well identity is authoritative and must NOT be re-derived.
    # Multi-tile scopes (Keyence) number position_index per TILE, whereas position_well_mapping
    # numbers it per WELL; joining across those two spaces silently assembles wells out of tiles
    # that belong to other wells. Such inventories mint well_index/well_id at ingest, where the
    # tile→well relationship is still known, so trust them and skip the join entirely.
    if {"well_index", "well_id"}.issubset(acquisition_inventory_df.columns):
        joined = acquisition_inventory_df
    else:
        joined = _join_well_identity(
            acquisition_inventory_df, position_well_mapping_df, experiment_id=experiment_id
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


def _join_well_identity(
    acquisition_inventory_df: pd.DataFrame,
    position_well_mapping_df: pd.DataFrame,
    *,
    experiment_id: str,
) -> pd.DataFrame:
    """Attach ``well_index``/``well_id`` to an inventory that lacks them, via the mapping.

    SOURCE_ORDINAL IS PART OF THE KEY FOR COLLECTIONS. A single-source experiment has one row per
    position, so ``(experiment_id, position_index)`` is unique and the join works. A COLLECTION
    merges several acquisitions into one experiment_id, and each source contributes its own
    position 0..N -- so that pair repeats once per source and the many_to_one validation fails with
    "Merge keys are not unique in right dataset".

    ``collection_position_mapping`` states it outright: source_ordinal is "the JOIN KEY ... (see the
    docstring above for why it is not time_index)". Both frames carry it. Measured on the pbx
    collection: 288 mapping rows = 96 positions x 3 sources, all 288 duplicated on the two-column
    key and ZERO duplicated once source_ordinal joins it.

    Included only when BOTH frames have it, so single-source experiments -- which predate the
    column -- keep their existing two-column join and their existing behavior.
    """
    mapping = position_well_mapping_df.copy()
    mapping["experiment_id"] = mapping["experiment_id"].astype(str)
    mapping = mapping[mapping["experiment_id"] == experiment_id]

    join_keys = ["experiment_id", "position_index"]
    if "source_ordinal" in mapping.columns:
        if "source_ordinal" not in acquisition_inventory_df.columns:
            raise ValueError(
                "position_well_mapping.csv carries 'source_ordinal' (a collection mapping, one "
                "block per raw source) but the acquisition inventory does not. Both sides must "
                "speak the same source key — the collection acquisition union stamps "
                "source_ordinal; a mapping and inventory from different pipelines cannot be joined."
            )
        # Coerce BOTH sides. A CSV round-trip can leave one int64 and the other object/float64,
        # which yields ZERO matches rather than an error -- and the failure then surfaces as a
        # misleading "no rows resolve to this well". Mirrors apply_position_to_well_mapping, which
        # hit exactly this.
        acquisition_inventory_df = acquisition_inventory_df.copy()
        for frame in (acquisition_inventory_df, mapping):
            frame["source_ordinal"] = pd.to_numeric(
                frame["source_ordinal"], errors="raise"
            ).astype(int)
        join_keys.append("source_ordinal")

    return acquisition_inventory_df.merge(
        mapping[join_keys + ["well_index", "well_id"]],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
