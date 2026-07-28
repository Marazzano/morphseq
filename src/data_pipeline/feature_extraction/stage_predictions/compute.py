"""stage_predictions compute — Kimmel1995 stage (hpf) per snip from plate metadata + frame timing.

No mask reading: stage is predicted from start_age_hpf + temperature and elapsed_time_s
(frame_inventory, by image_id). Reuses the legacy pure predict_stage_hpf.

``start_age_hpf`` source depends on the DECLARED collection-classify fact (CLASSIFY ONCE, CONSUME
EVERYWHERE — see docs/EXPERIMENT_GROUP_PLATE_MODEL.md). This is the ONE consumer that branches on it:

  - COLLECTION (``is_collection`` true) → ``start_age_by_time_index[str(time_index)]`` — a snapshot
    collection is a coarse timelapse whose per-timepoint age can't live in per-well plate_metadata,
    so the age RIDES IN the classify artifact keyed by the union ``time_index``.
  - SINGLE (non-collection, or no classify artifact) → ``plate_by_well[well_id]["start_age_hpf"]``,
    BYTE-IDENTICAL to the pre-collection behavior.

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
    collection_classification: dict | None,
    plate: pd.Series,
    time_index: object,
    well_id: str,
    snip_id: str,
) -> float:
    """Resolve ``start_age_hpf`` for one snip, branching on the DECLARED collection fact.

    Collection → the per-timepoint age from ``start_age_by_time_index[str(time_index)]`` (the union
    time_index keys the map). Single (no classify artifact, or ``is_collection`` false) → the
    per-well ``plate_metadata`` value, byte-identical to the pre-collection path.
    """
    is_collection = bool(collection_classification.get("is_collection")) if collection_classification else False

    if not is_collection:
        # SINGLE path — unchanged. plate_metadata's start_age_hpf, validated by the caller.
        return float(plate["start_age_hpf"])

    age_map = collection_classification["start_age_by_time_index"]
    key = str(int(time_index))
    if key not in age_map or age_map[key] is None:
        experiment_id = collection_classification.get("experiment_id", well_id)
        raise ValueError(
            f"stage_predictions: collection {experiment_id!r} declares no start_age_hpf "
            f"for time_index {key} (snip {snip_id!r}). The collection-classify artifact's "
            "start_age_by_time_index must cover every timepoint with a declared t<NN>hpf age."
        )
    return float(age_map[key])


def compute_stage_prediction_features(
    snip_inventory_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    plate_metadata_df: pd.DataFrame,
    *,
    collection_classification: dict | None = None,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Return one stage_prediction_features row per snip.

    ``collection_classification`` is the DECLARED collection-classify payload (or ``None`` for a
    single experiment / the pre-collection call path). When it declares ``is_collection`` true, each
    snip's ``start_age_hpf`` is read from ``start_age_by_time_index`` by the snip's ``time_index``;
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
        # single experiment; for a collection it comes from the classify artifact by time_index, so
        # a collection well legitimately need not carry start_age_hpf here.
        if "temperature" not in plate.index or pd.isna(plate["temperature"]):
            raise ValueError(
                f"stage_predictions: plate_metadata for well {well_id!r} is missing 'temperature'."
            )
        if not (collection_classification and collection_classification.get("is_collection")):
            if "start_age_hpf" not in plate.index or pd.isna(plate["start_age_hpf"]):
                raise ValueError(
                    f"stage_predictions: plate_metadata for well {well_id!r} is missing 'start_age_hpf'."
                )

        start_age_hpf = _start_age_hpf_for_snip(
            collection_classification=collection_classification,
            plate=plate,
            time_index=snip["time_index"],
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
