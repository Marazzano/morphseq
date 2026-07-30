"""Build the well_id <-> seq_sample_id crosswalk.

One row per imaging well of every keyed experiment. The crosswalk is deliberately *only* the
identity bridge: no latents, no stages, no QC. Those join onto it later, keyed on ``well_id`` for
the morphology side and ``seq_sample_id`` for the sequencing side.

``well_id`` — not ``snip_id`` — is the join key. A hash well identifies a physical well of embryos,
and the pipeline resolves multiple embryos per well; collapsing that to a single snip (as the
predecessor notebook did, by hardcoding ``_e00_t0000``) silently drops the ambiguity instead of
recording it. Resolving well -> embryo is a separate, explicit decision made downstream.

Every row carries a ``pairing_status`` so an unpaired or ambiguous well is visible in the output
rather than dropped:

    paired        one imaging well <-> one sequencing sample
    blank_well    well exists on the plate but has no authored hash assignment
    no_hash_map   the experiment is keyed, but neither plate source carries a hash map
"""

from __future__ import annotations

import pandas as pd

from .experiment_key import ExperimentKey, load_experiment_key
from .hash_map import load_experiment_hash_map
from .identifiers import build_seq_sample_id, is_blank, normalize_hash_well, format_hash_plate
from .paths import PipelinePaths, default_paths

STATUS_PAIRED = "paired"
STATUS_BLANK_WELL = "blank_well"
STATUS_NO_HASH_MAP = "no_hash_map"

CROSSWALK_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "well_index",
    "sci_expt",
    "hash_plate",
    "hash_well",
    "seq_sample_id",
    "pairing_status",
    "hash_map_source",
)


def build_crosswalk(
    experiment_ids: "list[str] | None" = None,
    *,
    key: ExperimentKey | None = None,
    paths: PipelinePaths | None = None,
    key_path: "str | None" = None,
) -> pd.DataFrame:
    """Assemble the crosswalk for ``experiment_ids`` (default: every keyed experiment).

    Args:
        experiment_ids: Restrict to these imaging experiments. Each must have a key entry.
        key: Pre-loaded experiment key; loaded from disk when omitted.
        paths: Pre-resolved pipeline paths; resolved from env/default when omitted.
        key_path: Override the key file location (ignored when ``key`` is given).

    Returns:
        A DataFrame with ``CROSSWALK_COLUMNS``, sorted by ``experiment_id`` then ``well_index``.

    Raises:
        ValueError: if a requested experiment has no key entry, or if the assembled table is not
            1-to-1 (see ``validate_crosswalk``).
    """
    resolved_key = key if key is not None else load_experiment_key(key_path)
    resolved_paths = paths if paths is not None else default_paths()

    targets = (
        [str(experiment_id) for experiment_id in experiment_ids]
        if experiment_ids is not None
        else resolved_key.experiment_ids
    )

    unmapped = [
        experiment_id
        for experiment_id in targets
        if resolved_key.sci_expt_for(experiment_id) is None
    ]
    if unmapped:
        raise ValueError(
            f"[morphseq_integration] no experiment key entry for {unmapped}. "
            "Add a row to experiment_sequencing_key.csv mapping each experiment_id to its sci_expt."
        )

    frames = [
        _crosswalk_one(experiment_id, resolved_key.sci_expt_for(experiment_id), resolved_paths)
        for experiment_id in targets
    ]
    crosswalk = pd.concat(frames, ignore_index=True) if frames else _empty_crosswalk()
    crosswalk = crosswalk.sort_values(["experiment_id", "well_index"]).reset_index(drop=True)

    validate_crosswalk(crosswalk)
    return crosswalk


def _crosswalk_one(
    experiment_id: str, sci_expt: str, paths: PipelinePaths
) -> pd.DataFrame:
    """Crosswalk rows for one experiment; a single ``no_hash_map`` row when no map is available."""
    hash_map = load_experiment_hash_map(experiment_id, paths=paths)
    if hash_map is None:
        return pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "well_id": "",
                    "well_index": "",
                    "sci_expt": sci_expt,
                    "hash_plate": "",
                    "hash_well": "",
                    "seq_sample_id": "",
                    "pairing_status": STATUS_NO_HASH_MAP,
                    "hash_map_source": "",
                }
            ]
        )

    records = []
    for row in hash_map.table.itertuples(index=False):
        blank = is_blank(row.hash_well_raw) or is_blank(row.hash_plate_raw)
        if blank:
            hash_plate = hash_well = seq_sample_id = ""
            status = STATUS_BLANK_WELL
        else:
            hash_plate = format_hash_plate(row.hash_plate_raw)
            hash_well = normalize_hash_well(row.hash_well_raw)
            seq_sample_id = build_seq_sample_id(sci_expt, hash_plate, hash_well)
            status = STATUS_PAIRED
        records.append(
            {
                "experiment_id": experiment_id,
                "well_id": row.well_id,
                "well_index": row.well_index,
                "sci_expt": sci_expt,
                "hash_plate": hash_plate,
                "hash_well": hash_well,
                "seq_sample_id": seq_sample_id,
                "pairing_status": status,
                "hash_map_source": hash_map.source,
            }
        )
    return pd.DataFrame(records, columns=list(CROSSWALK_COLUMNS))


def _empty_crosswalk() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in CROSSWALK_COLUMNS})


def validate_crosswalk(crosswalk: pd.DataFrame) -> None:
    """Fail unless the paired rows form a 1-to-1 mapping.

    Both directions matter and fail for different reasons: a duplicate ``well_id`` means the plate
    table was read twice or a well was authored twice; a duplicate ``seq_sample_id`` means two
    imaging wells claim the same sequenced embryo, which is a curation error in the hash map or a
    wrong ``sci_expt``/``hash_plate_num``.

    Raises:
        ValueError: on a missing column or a duplicate on either side.
    """
    missing = [column for column in CROSSWALK_COLUMNS if column not in crosswalk.columns]
    if missing:
        raise ValueError(f"[morphseq_integration] crosswalk is missing column(s) {missing}.")

    paired = crosswalk.loc[crosswalk["pairing_status"] == STATUS_PAIRED]

    duplicated_wells = paired.loc[paired.duplicated(subset=["well_id"], keep=False), "well_id"]
    if not duplicated_wells.empty:
        raise ValueError(
            "[morphseq_integration] duplicate well_id in the crosswalk: "
            f"{sorted(set(duplicated_wells))[:10]}. Each imaging well maps to at most one sample."
        )

    duplicated_samples = paired.loc[
        paired.duplicated(subset=["seq_sample_id"], keep=False),
        ["experiment_id", "well_id", "seq_sample_id"],
    ]
    if not duplicated_samples.empty:
        preview = duplicated_samples.head(10).to_dict(orient="records")
        raise ValueError(
            "[morphseq_integration] two imaging wells claim the same seq_sample_id: "
            f"{preview}. Check hash_plate_num and image_to_hash_map for the listed experiments, "
            "and that each is keyed to the right sci_expt."
        )


ALL_STATUSES: tuple[str, ...] = (STATUS_PAIRED, STATUS_BLANK_WELL, STATUS_NO_HASH_MAP)


def summarize(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Per-experiment row counts by ``pairing_status`` — the quick "did this work" view.

    Every status in ``ALL_STATUSES`` is always a column, even at zero. A pivot that omits absent
    statuses makes the headline number (``paired``) disappear exactly when the answer is "nothing
    paired", which is the one case you most need to see.
    """
    status = pd.Categorical(crosswalk["pairing_status"], categories=ALL_STATUSES)
    counts = (
        crosswalk.assign(pairing_status=status)
        .groupby(["experiment_id", "sci_expt", "pairing_status"], observed=False)
        .size()
        .unstack("pairing_status", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return counts.loc[:, ["experiment_id", "sci_expt", *ALL_STATUSES]]
