"""Write the explicit MorphSeq <-> sequencing embryo mapping table.

Everything needed to go from an imaging well to a sequencing embryo_ID now lives in each
plate's Excel (hash_plate_num / image_to_hash_map / rt_block, all keyed by imaging well). This
script resolves that once and writes it down, so downstream work joins a table instead of
re-deriving the mapping.

The mapping is an explicit composition:

    representative MorphSeq embryo ID
      -> (experiment, imaging_well)                    from the MorphSeq pipeline table
      -> (hash_plate, hash_well, rt_block)             from the plate Excel
      -> seq_embryo_ID                                 exact sequencing-coordinate lookup

Grain: one row per `physical_embryo_id`. `embryo_registry.csv` has already selected the
representative MorphSeq acquisition (timeseries before snapshot), so this script does not collapse
frames or choose between acquisitions.

Important: the registry contains sequenced embryos that survived the MorphSeq pipeline. It is the
right input for attaching morphology to sequencing, but it is not a census of every sequenced well
and must not be used to calculate QC loss rates.

Columns:
    morphseq_embryo_id MorphSeq pipeline identity
    physical_embryo_id MorphSeq physical identity shared by redundant acquisitions
    experiment          imaging experiment / plate metadata file this came from
    imaging_well        well on the imaging plate
    hash_plate          \\
    hash_well            > the sequencing coordinate, read from the plate Excel
    rt_block            /
    seq_embryo_ID       GENE14_{hash_plate}_{hash_well}_{rt_block}, if that embryo was sequenced
    provenance          how seq_embryo_ID was obtained, or why it is blank

Usage:  python 3_build_embryo_id_map.py [--out PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
QC_DIR = HERE.parent.parent / "20260607_sci_cilia_gene14_imaging_qc"
MORPHSEQ_REGISTRY = QC_DIR / "tables/embryo_registry.csv"
DEFAULT_OUT = HERE / "embryo_id_map.csv"

_spec = importlib.util.spec_from_file_location("cw", HERE / "2_seq_imaging_crosswalk.py")
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)

# how seq_embryo_ID was obtained -- one value per row, so a blank is always explained
RESOLVED = "excel_coordinate"          # plate Excel gave the full coordinate and it exists in seq
NOT_SEQUENCED = "not_in_seq_metadata"  # coordinate is complete, but no such sequencing embryo
INCOMPLETE = "incomplete_coordinate"   # the plate Excel is missing plate, well or block


def load_morphseq_embryos(path: Path = MORPHSEQ_REGISTRY) -> pd.DataFrame:
    """Load the registry, which must already contain one row per physical embryo."""
    required = {
        "physical_embryo_id",
        "representative_embryo_id",
        "experiment",
        "well",
        "collection_time_hpf",
    }
    registry = pd.read_csv(path, low_memory=False)
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    registry = registry.rename(
        columns={
            "representative_embryo_id": "morphseq_embryo_id",
            "well": "imaging_well",
        }
    )
    registry = registry[
        [
            "morphseq_embryo_id",
            "physical_embryo_id",
            "experiment",
            "imaging_well",
            "collection_time_hpf",
        ]
    ].copy()

    if registry.isna().any().any():
        null_counts = registry.isna().sum()
        null_counts = null_counts[null_counts > 0]
        raise ValueError("MorphSeq registry has missing identity values:\n" + null_counts.to_string())

    registry["morphseq_embryo_id"] = registry["morphseq_embryo_id"].astype(str)
    registry["imaging_well"] = registry["imaging_well"].map(cw.norm_well)

    duplicate_physical = registry.duplicated("physical_embryo_id", keep=False)
    if duplicate_physical.any():
        raise ValueError(
            "physical_embryo_id is not unique in the MorphSeq registry:\n"
            + registry[duplicate_physical].to_string(index=False)
        )

    duplicate_morphseq = registry.duplicated("morphseq_embryo_id", keep=False)
    if duplicate_morphseq.any():
        raise ValueError(
            "representative MorphSeq embryo ID is not unique in the registry:\n"
            + registry[duplicate_morphseq].to_string(index=False)
        )

    duplicate_well = registry.duplicated(["experiment", "imaging_well"], keep=False)
    if duplicate_well.any():
        raise ValueError(
            "More than one physical embryo occupies an experiment/well:\n"
            + registry[duplicate_well].to_string(index=False)
        )

    return registry


def attach_plate_coordinates(morphseq: pd.DataFrame) -> pd.DataFrame:
    """Add hash plate, hash well, and RT block from the plate Excels."""
    plate_coordinates = cw.build_crosswalk()
    return morphseq.merge(
        plate_coordinates,
        on=["experiment", "imaging_well"],
        how="left",
        validate="many_to_one",
        indicator="plate_join",
    )


def attach_sequencing_ids(embryo_map: pd.DataFrame) -> pd.DataFrame:
    """Resolve each complete Excel coordinate to an exact sequencing embryo ID."""
    # This index is built by PARSING every seq embryo_ID:
    # GENE14_P18_A4_Bl1 -> (P18, A4, Bl1). We deliberately do not trust the separate
    # hash_plate/hash_well/rt_block columns in the sequencing metadata.
    seq_id_by_coordinate = cw.build_seq_index()
    seq_ids = []
    provenance = []

    for _, embryo in embryo_map.iterrows():
        coordinate = [embryo["hash_plate"], embryo["hash_well"], embryo["rt_block"]]
        coordinate_is_complete = embryo["plate_join"] == "both" and not any(
            pd.isna(value) or value == "" for value in coordinate
        )

        if not coordinate_is_complete:
            seq_id = None
            source = INCOMPLETE
        else:
            seq_id = cw.resolve_seq_embryo_id(*coordinate, seq_id_by_coordinate)
            source = RESOLVED if seq_id else NOT_SEQUENCED

        seq_ids.append(seq_id)
        provenance.append(source)

    embryo_map["seq_embryo_ID"] = seq_ids
    embryo_map["provenance"] = provenance
    return embryo_map.drop(columns="plate_join")


def build_map() -> pd.DataFrame:
    """Run the three identity joins in biological order."""
    morphseq = load_morphseq_embryos()
    embryo_map = attach_plate_coordinates(morphseq)
    return attach_sequencing_ids(embryo_map)


def validate_map(embryo_map: pd.DataFrame) -> None:
    """Check the biological one-to-one rules before writing anything."""
    resolved = embryo_map[embryo_map.seq_embryo_ID.notna()]
    problems = []

    # A missing Excel coordinate is a metadata error. A complete coordinate that is absent from
    # sequencing metadata is allowed: some embryos were marked/attempted for sequencing but did
    # not produce a sequencing embryo record. Those rows retain `not_in_seq_metadata` provenance.
    incomplete = embryo_map[embryo_map.provenance == INCOMPLETE]
    if not incomplete.empty:
        problems.append(
            "MorphSeq embryos are missing an Excel sequencing coordinate:\n"
            + incomplete[
                [
                    "experiment", "imaging_well", "morphseq_embryo_id",
                    "hash_plate", "hash_well", "rt_block", "provenance",
                ]
            ].to_string(index=False)
        )

    duplicated_ids = embryo_map[embryo_map.duplicated("morphseq_embryo_id", keep=False)]
    if not duplicated_ids.empty:
        problems.append(
            "MorphSeq embryo IDs appear more than once:\n"
            + duplicated_ids.to_string(index=False)
        )

    # More than one MorphSeq object ID is okay when both name the same physical embryo.
    positions = embryo_map[
        ["experiment", "imaging_well", "physical_embryo_id"]
    ].drop_duplicates()
    bad_positions = positions.duplicated(["experiment", "imaging_well"], keep=False)
    if bad_positions.any():
        problems.append(
            "Multiple physical embryos occupy one experiment/well:\n"
            + positions[bad_positions].to_string(index=False)
        )

    # Two distinct physical embryos in one experiment cannot become one sequencing embryo.
    physical_links = resolved[
        [
            "experiment", "imaging_well", "morphseq_embryo_id", "physical_embryo_id",
            "hash_plate", "hash_well", "rt_block", "seq_embryo_ID",
        ]
    ].drop_duplicates(["experiment", "physical_embryo_id", "seq_embryo_ID"])
    bad_links = physical_links.duplicated(["experiment", "seq_embryo_ID"], keep=False)
    if bad_links.any():
        detail = physical_links[bad_links][
            ["experiment", "imaging_well", "morphseq_embryo_id",
             "physical_embryo_id", "hash_plate", "hash_well", "rt_block", "seq_embryo_ID"]
        ].sort_values(["experiment", "seq_embryo_ID", "imaging_well", "morphseq_embryo_id"])
        problems.append(
            "Distinct physical embryos converge on one sequencing embryo inside an experiment:\n"
            + detail.to_string(index=False)
        )

    rebuilt = ("GENE14_" + resolved.hash_plate + "_" + resolved.hash_well + "_" + resolved.rt_block)
    mismatched = (rebuilt != resolved.seq_embryo_ID).sum()
    if mismatched:
        problems.append(f"{mismatched} rows where the coordinate does not rebuild seq_embryo_ID")

    physical_to_seq = (
        resolved.dropna(subset=["physical_embryo_id"])
        .groupby("physical_embryo_id")
        .seq_embryo_ID.nunique()
    )
    ambiguous_physical = physical_to_seq[physical_to_seq > 1]
    if not ambiguous_physical.empty:
        problems.append(
            "Physical embryos map to multiple sequencing embryos:\n"
            + ambiguous_physical.to_string()
        )

    # This is a validation only, never part of ID resolution. It catches a valid-looking
    # coordinate that points to an embryo from the wrong collection time.
    seq_metadata = pd.read_csv(
        cw.SEQ_METADATA_TSV,
        sep="\t",
        usecols=["embryo_ID", "timepoint"],
    ).rename(columns={"embryo_ID": "seq_embryo_ID", "timepoint": "seq_timepoint_hpf"})
    resolved_with_time = resolved.merge(
        seq_metadata,
        on="seq_embryo_ID",
        how="left",
        validate="many_to_one",
    )
    wrong_time = resolved_with_time[
        resolved_with_time["collection_time_hpf"] != resolved_with_time["seq_timepoint_hpf"]
    ]
    if not wrong_time.empty:
        problems.append(
            "Resolved embryos disagree on MorphSeq versus sequencing collection time:\n"
            + wrong_time[
                [
                    "experiment", "imaging_well", "morphseq_embryo_id", "seq_embryo_ID",
                    "collection_time_hpf", "seq_timepoint_hpf",
                ]
            ].to_string(index=False)
        )

    if problems:
        raise ValueError("\n\n".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out_path = parser.parse_args().out

    embryo_map = build_map()
    validate_map(embryo_map)
    embryo_map.sort_values(
        ["experiment", "imaging_well", "morphseq_embryo_id"]
    ).to_csv(out_path, index=False)

    counts = embryo_map.provenance.value_counts()
    print(f"wrote {out_path}  ({len(embryo_map)} rows)")
    print(f"  MorphSeq embryo IDs    {embryo_map.morphseq_embryo_id.nunique()}")
    print(f"  experiments            {embryo_map.experiment.nunique()}")
    print(f"  resolved to a seq ID   {counts.get(RESOLVED, 0)}")
    print(f"    distinct seq embryos {embryo_map.seq_embryo_ID.nunique()}")
    print(f"  not in seq metadata    {counts.get(NOT_SEQUENCED, 0)}")
    print(f"  incomplete coordinate  {counts.get(INCOMPLETE, 0)}")
    print(f"  with physical_embryo_id {embryo_map.physical_embryo_id.notna().sum()}")


if __name__ == "__main__":
    main()
