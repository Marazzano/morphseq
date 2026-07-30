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
        # temperature is always a per-well plate_metadata fact. start_age_hpf is per-well ONLY for a
        # single experiment; for a collection it comes from the provenance artifact, so
        # a collection well legitimately need not carry start_age_hpf here.
        if "temperature" not in plate.index or pd.isna(plate["temperature"]):
            raise ValueError(
                f"stage_predictions: plate_metadata for well {well_id!r} is missing 'temperature'."
            )
        if not (collection_provenance and collection_provenance.get("is_collection")):
            if "start_age_hpf" not in plate.index or pd.isna(plate["start_age_hpf"]):
                raise ValueError(
                    f"stage_predictions: plate_metadata for well {well_id!r} is missing 'start_age_hpf'."
                )

        start_age_hpf = _start_age_hpf_for_snip(
            collection_provenance=collection_provenance,
            plate=plate,
            source_ordinal=snip.get("source_ordinal"),
            well_id=well_id,
            snip_id=snip_id,
        )

        elapsed = _elapsed_time_s(frame_inventory_by_image, image_id, snip_id)
        predicted = predict_stage_hpf(start_age_hpf, elapsed, float(plate["temperature"]))

        row = {col: snip[col] for col in SNIP_FEATURE_TABLE_SPINE_COLUMNS}
        row["predicted_stage_hpf"] = predicted
        row["model_version"] = model_version
        rows.append(row)

    return pd.DataFrame(rows, columns=STAGE_PREDICTION_TABLE_COLUMNS)
