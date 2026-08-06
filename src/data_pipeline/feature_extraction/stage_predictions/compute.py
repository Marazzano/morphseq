"""stage_predictions compute — Kimmel1995 stage (hpf) per snip from plate metadata + frame timing.

No mask reading: stage is predicted from start_age_hpf + temperature and elapsed_time_s
(frame_inventory, by image_id). Reuses the legacy pure predict_stage_hpf.

``start_age_hpf`` source depends on the DECLARED collection-provenance fact (CLASSIFY ONCE, CONSUME
EVERYWHERE — see docs/EXPERIMENT_GROUP_PLATE_MODEL.md). This is the ONE consumer that branches on it:

Both branches key on ``source_ordinal``; only the age's HOME differs:

  - COLLECTION (``is_collection`` true) → ``start_age_by_source_ordinal[str(source_ordinal)]``. A
    collection's age varies per raw acquisition, which cannot live in per-well plate_metadata.
  - SINGLE (non-collection) → ``plate_by_well[well_id]["start_age_hpf"]``, BYTE-IDENTICAL to the
    pre-collection behavior. Age there is per-WELL biology; its one source is ordinal 0.

Every experiment declares a provenance artifact (a single experiment is a collection of ONE source),
so ``source_ordinal`` is a real fact everywhere — no backfill, no presence branch.

``temperature`` always comes from plate_metadata by well_id (a per-well fact, unchanged either way).
This consumer never re-derives ``is_collection`` — it reads the declared bool.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import SNIP_FEATURE_TABLE_SPINE_COLUMNS
from data_pipeline.feature_extraction.stage_inference import predict_stage_hpf

from .contract import STAGE_PREDICTION_TABLE_COLUMNS

MODEL_VERSION = "kimmel1995_temp_rate_v1"

_TIME_COLUMNS: tuple[str, ...] = ("elapsed_time_s", "experiment_time_s", "time_s")


def _elapsed_time_s(frame_inventory_by_image: pd.DataFrame, image_id: str, snip_id: str) -> float:
    row = frame_inventory_by_image.loc[image_id]
    for col in _TIME_COLUMNS:
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    raise ValueError(
        f"stage_predictions: frame_inventory row for image_id {image_id!r} (snip {snip_id!r}) has "
        f"no frame timing. Expected one of {_TIME_COLUMNS}."
    )


def _source_ordinal_by_frame(acquisition_inventory_df: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Map ``(well_id, time_index) -> source_ordinal`` from the acquisition inventory.

    WHY A JOIN, not a carried column. frame_inventory deliberately carries NO per-frame source
    LABEL — only the per-well ``n_sources`` COUNT survives the merge seam (see
    collection_acquisition_union: "no downstream stage needs to know WHICH source a frame came
    from"). A frame has exactly ONE source, so labelling every frame would add nothing that the
    acquisition inventory does not already state. This reads the fact from its OWNER at the consume
    boundary instead of propagating it downstream.

    Returns an empty map when the column is absent (a single experiment's inventory need not carry
    it — its only source is ordinal 0, and the caller does not consult this map for singles).
    """
    if "source_ordinal" not in acquisition_inventory_df.columns:
        return {}
    frame = acquisition_inventory_df[["well_id", "time_index", "source_ordinal"]].drop_duplicates()
    by_frame: dict[tuple[str, int], int] = {}
    for well_id, time_index, source_ordinal in frame.itertuples(index=False):
        key = (str(well_id), int(time_index))
        previous = by_frame.setdefault(key, int(source_ordinal))
        if previous != int(source_ordinal):
            raise ValueError(
                f"stage_predictions: acquisition inventory maps frame {key} to multiple "
                f"source_ordinals ({previous}, {int(source_ordinal)}). A frame belongs to exactly "
                "ONE source; the merge assigns each source a disjoint time_index block."
            )
    return by_frame


def _start_age_hpf_for_snip(
    *,
    collection_provenance: dict | None,
    plate: pd.Series,
    source_ordinal: object,
    well_id: str,
    snip_id: str,
) -> float:
    """Resolve ``start_age_hpf`` for one snip, keyed on its SOURCE ORDINAL.

    The staging formula is ``start_age_hpf + elapsed_within_that_source * rate``, so the age is a
    per-SOURCE fact. Both branches key on ``source_ordinal``; only the age's HOME differs:

      * COLLECTION — the age varies per source (each raw acquisition declares its own ``t<NN>hpf``),
        which cannot live in per-well plate_metadata. It comes from the provenance artifact's
        ``start_age_by_source_ordinal``.
      * SINGLE — the age is per-WELL biology (a plate can hold wells of different ages), so it comes
        from plate_metadata exactly as it always has. Its one source is ordinal 0.

    Keying on ``source_ordinal`` rather than the merged ``time_index`` is what makes a TIMELAPSE
    source correct. One source spans MANY merged time_index values; they coincide only when every
    source contributes a single frame. Using time_index staged source A's second frame with source
    B's declared age — a 48-hour error on a real two-timelapse plate.
    """
    is_collection = bool(collection_provenance.get("is_collection")) if collection_provenance else False

    if not is_collection:
        # SINGLE path — unchanged. plate_metadata's start_age_hpf, validated by the caller.
        return float(plate["start_age_hpf"])

    # Prefer the canonical field; fall back to the legacy alias for an artifact written before the
    # rename. See TODO(collection-legacy-age-map) in collection_provenance.
    age_map = collection_provenance.get("start_age_by_source_ordinal")
    if age_map is None:
        age_map = collection_provenance["start_age_by_time_index"]

    if source_ordinal is None or pd.isna(source_ordinal):
        raise ValueError(
            f"stage_predictions: snip {snip_id!r} (well {well_id!r}) has no 'source_ordinal'. It is "
            "frame provenance stamped by the collection union and carried through frame_inventory; "
            "a collection's age is keyed by it."
        )

    key = str(int(source_ordinal))
    if key not in age_map or age_map[key] is None:
        experiment_id = collection_provenance.get("experiment_id", well_id)
        raise ValueError(
            f"stage_predictions: collection {experiment_id!r} declares no start_age_hpf for "
            f"source_ordinal {key} (snip {snip_id!r}). Every source must declare a t<NN>hpf age in "
            "the provenance artifact."
        )
    return float(age_map[key])


def compute_stage_prediction_features(
    snip_inventory_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    plate_metadata_df: pd.DataFrame,
    *,
    acquisition_inventory_df: pd.DataFrame | None = None,
    collection_provenance: dict | None = None,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Return one stage_prediction_features row per snip.

    ``collection_provenance`` is the DECLARED collection-provenance payload (or ``None`` for a
    single experiment / the pre-collection call path). When it declares ``is_collection`` true, each
    snip's ``start_age_hpf`` is read from ``start_age_by_source_ordinal``;
    otherwise it is the per-well ``plate_metadata`` value (byte-identical to before).
    """
    frame_inventory_by_image = frame_inventory_df.set_index("image_id")
    plate_by_well = plate_metadata_df.set_index("well_id")
    source_ordinal_by_frame = _source_ordinal_by_frame(
        acquisition_inventory_df if acquisition_inventory_df is not None else pd.DataFrame()
    )

    rows: list[dict] = []
    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        well_id = str(snip["well_id"])
        image_id = str(snip["image_id"])

        if well_id not in plate_by_well.index:
            raise ValueError(
                f"stage_predictions: well_id {well_id!r} (snip {snip_id!r}) not in plate_metadata. "
                "Every snip's well must have a plate_metadata row with start_age_hpf + temperature."
            )
        plate = plate_by_well.loc[well_id]
        # MERGE NOTE. Two designs met here and only one of them is right per field.
        #
        # An unresolvable stage is a DATA condition, not a bug: the run must produce a row saying so
        # rather than aborting the job, which is why `stage_prediction_status` (and its validator in
        # contract.py) wins over the raise-on-missing branch this branch had.
        #
        # But `start_age_hpf` is no longer a plate_metadata-only fact. For a COLLECTION it lives in
        # the provenance artifact keyed by source_ordinal, so testing `plate["start_age_hpf"]`
        # would flag every collection snip as `missing_start_age_hpf` while the age sat available
        # one lookup away. Resolution goes through _start_age_hpf_for_snip; only a genuinely
        # unresolvable age produces the status.
        #
        # temperature stays a per-well plate_metadata fact in both worlds.
        elapsed = _elapsed_time_s(frame_inventory_by_image, image_id, snip_id)
        is_collection = bool(collection_provenance and collection_provenance.get("is_collection"))

        if is_collection:
            # A COLLECTION RESOLVES ITS AGE OR FAILS — it does NOT degrade to a status flag.
            # The nullable-status path exists for a plate that simply never declared an age. A
            # collection always declares one; failing to find it means the provenance artifact or
            # the source_ordinal mapping is broken, i.e. a structural fault that would otherwise
            # mis-stage every frame of that source by hours. Those raises are asserted by
            # test_collection_missing_age_for_source_fails_loud and
            # test_collection_without_the_acquisition_inventory_fails_loud.
            #
            # A frame belongs to ONE source; the acquisition inventory owns that mapping.
            source_ordinal = source_ordinal_by_frame.get((well_id, int(snip["time_index"])))
            start_age_hpf = _start_age_hpf_for_snip(
                collection_provenance=collection_provenance,
                plate=plate,
                source_ordinal=source_ordinal,
                well_id=well_id,
                snip_id=snip_id,
            )
        else:
            # SINGLE experiment: an absent or null start_age_hpf is a data condition, reported as
            # `missing_start_age_hpf` rather than aborting the run.
            raw_age = plate["start_age_hpf"] if "start_age_hpf" in plate.index else None
            start_age_hpf = None if raw_age is None or pd.isna(raw_age) else float(raw_age)

        if start_age_hpf is None or pd.isna(start_age_hpf):
            predicted = None
            status = "missing_start_age_hpf"
        elif "temperature" not in plate.index or pd.isna(plate["temperature"]):
            predicted = None
            status = "missing_temperature"
        else:
            predicted = predict_stage_hpf(start_age_hpf, elapsed, float(plate["temperature"]))
            status = "predicted"

        row = {col: snip[col] for col in SNIP_FEATURE_TABLE_SPINE_COLUMNS}
        row["predicted_stage_hpf"] = predicted
        row["model_version"] = model_version
        row["stage_prediction_status"] = status
        rows.append(row)

    return pd.DataFrame(rows, columns=STAGE_PREDICTION_TABLE_COLUMNS)
