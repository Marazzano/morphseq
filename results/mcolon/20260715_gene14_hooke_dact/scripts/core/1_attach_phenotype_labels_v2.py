"""
1 - Attach MorphSeq imaging labels + measurements onto the GENE14 SEQUENCING ordering.

This script does two things, in order:

  PHASE A  build the mapping   imaging embryo  -->  sequencing embryo_ID
           (+ carry the morphseq phenotype label and per-embryo measurements along)
           and WRITE it to disk as a reusable crosswalk artifact.

  PHASE B  use the mapping     apply those labels/measurements onto the reconciled
           520-embryo SEQUENCING table (one row per sequenced embryo).

All the imaging<->sequencing id logic (the tricky reformatted/collision cases) lives in the
sibling module `seq_imaging_crosswalk.py` — read its docstring first.

Nature of the merge: the sequencing table is the SPINE (every sequenced embryo is kept).
The morphseq phenotype + measurements are EXTRA columns, NaN for embryos with no imaging
prediction. Phenotype models exist only for b9d2 (CE/HTA) and cep290 (High_to_Low/Low_to_High)
homozygotes, so most embryos are legitimately unlabeled.

Run:
  conda run -n segmentation_grounded_sam --no-capture-output python \
      results/mcolon/20260715_gene14_hooke_dact/scripts/core/1_attach_phenotype_labels_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seq_imaging_crosswalk import (  # noqa: E402
    build_seq_index, collection_time_hpf, load_plate_maps,
    norm_well, resolve_seq_embryo_id,
)

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
QC_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
PHENO_CSV = QC_DIR / "predictions/sequenced_homozygous_phenotype_cross_bin.csv"
# Per-imaging-embryo features we borrow: breeding `pair` (cross/spawn) + length/curvature.
# (On disk this is the opaquely-named `query_all_rows_clean.csv`, written by the QC dir's
#  0_load_and_clean_datasets.py — keep the filename, name the variable for what we use it for.)
IMAGING_EMBRYO_FEATURES_CSV = QC_DIR / "tables/query_all_rows_clean.csv"

RUN_DIR = PROJECT_ROOT / "results/mcolon/20260715_gene14_hooke_dact"
# The reconciled SEQUENCING ordering — one row per sequenced embryo. This is the spine we
# attach onto; it is NOT a DACT product (DACTs are computed later, by script 5).
SEQ_ORDERING_TSV = RUN_DIR / "output/gene14_embryos_reconciled.tsv"
OUT_DIR = RUN_DIR / "output"

# Phase A artifact: the reusable imaging->sequencing crosswalk (one row per resolved embryo).
CROSSWALK_TSV = OUT_DIR / "imaging_to_seq_crosswalk.tsv"

PROB_COLS = ["prob_CE", "prob_HTA", "prob_High_to_Low", "prob_Low_to_High"]

# Columns the crosswalk carries from imaging onto each sequencing embryo.
# Identity + provenance first, then the winning acquisition's call:
#   physical_embryo_id  the fish (biology-scoped; 1:1 with seq_embryo_ID)
#   collection_time     ONE per fish — the collection/destruction event that MADE the seq
#                       embryo (30to48 -> 48). A per-FISH property, not per-acquisition.
#   imaging_embryo_id   which acquisition won the priority collapse
#   acquisition_type    what kind that winning acquisition was (timeseries/t02/t01/snapshot)
CARRY_COLS = ["physical_embryo_id", "collection_time", "imaging_embryo_id", "acquisition_type",
              "morphseq_phenotype", *PROB_COLS, "pair",
              "total_length_um", "baseline_deviation_normalized"]

# The phenotype only EXISTS once morphological divergence has emerged. Before that the imaging
# classifier is guessing a phenotype that isn't real yet, so labels earlier than this are noise.
#   cep290 (HtL/LtH): meaningful only at tp >= 30 (18hpf was all-LtH, 24hpf all-HtL == defaults)
#   b9d2   (CE/HTA) : meaningful only at tp >= 18 (drops the 14hpf calls)
PHENO_MIN_TP = {"cep290": 30, "b9d2": 18}


# ===========================================================================
# PHASE A — build the imaging -> sequencing mapping
# ===========================================================================
def load_phenotype_predictions() -> pd.DataFrame:
    """Read the imaging phenotype classifier output; keep only confident, labeled rows.

    IMPORTANT: one row per IMAGING embryo (there can be several per physical embryo — snapshot
    + backups + timeseries). `well` is normalized to the bare 'A9' form so it can key the
    plate maps later. Rows with no predicted_label are unusable and dropped here.
    """
    pheno = pd.read_csv(PHENO_CSV, low_memory=False)
    pheno = pheno.dropna(subset=["predicted_label"]).copy()
    pheno["imaging_well"] = pheno["well"].map(norm_well)
    return pheno


def attach_pair_and_features(pheno: pd.DataFrame) -> pd.DataFrame:
    """Attach breeding pair + two physical measurements from the imaging query table.

    IMPORTANT: length/curvature are PER-FRAME for timeseries embryos, so we collapse them to
    the per-embryo MEDIAN (order-independent; a no-op for single-frame snapshots). Joins are
    keyed on imaging `embryo_id`.
        pair                          = breeding pair / spawn (cep290_P1, cep290_spawn, ab, ...)
        total_length_um               = embryo length
        baseline_deviation_normalized = curvature
    """
    query = pd.read_csv(IMAGING_EMBRYO_FEATURES_CSV, low_memory=False)

    pair_by_embryo = (
        query.dropna(subset=["pair"])
        .drop_duplicates("embryo_id")
        .set_index("embryo_id")["pair"]
    )
    feature_cols = ["total_length_um", "baseline_deviation_normalized"]
    features_by_embryo = query.groupby("embryo_id")[feature_cols].median()

    pheno["pair"] = pheno["embryo_id"].map(pair_by_embryo)
    for col in feature_cols:
        pheno[col] = pheno["embryo_id"].map(features_by_embryo[col])
    return pheno


def resolve_to_sequencing_ids(pheno: pd.DataFrame) -> pd.DataFrame:
    """Resolve each imaging embryo to its sequencing embryo_ID via the crosswalk.

    IMPORTANT: this is the whole imaging->sequencing bridge (all the tricky cases live in
    seq_imaging_crosswalk). Three lookups feed the resolver:
        (experiment, imaging_well) --plate maps-->  hash_well, hash_plate   (both regimes)
        experiment + collection_time_hpf --30to48 rule-->  collection_time  (seq `timepoint`)
    Unresolved embryos get seq_embryo_ID = NaN and are reported, not dropped.
    """
    plate_maps = load_plate_maps()
    seq_index_full, seq_index_partial = build_seq_index()

    def map_imaging_well_to_hash(row: pd.Series) -> pd.Series:
        # plate map: imaging_well -> (hash_well, hash_plate). Missing plate/well -> identity well.
        experiment_map = plate_maps.get(row["experiment"], {})
        hash_well, hash_plate = experiment_map.get(row["imaging_well"], (row["imaging_well"], None))
        return pd.Series({"hash_well": hash_well, "hash_plate_img": hash_plate})

    pheno[["hash_well", "hash_plate_img"]] = pheno.apply(map_imaging_well_to_hash, axis=1)

    # collection time (30to48 -> 48) is what the sequencing `timepoint` encodes
    pheno["collection_time"] = pheno.apply(
        lambda row: collection_time_hpf(
            row["experiment"],
            pd.to_numeric(row["collection_time_hpf"], errors="coerce"),
        ),
        axis=1,
    )

    pheno["seq_embryo_ID"] = pheno.apply(
        lambda row: resolve_seq_embryo_id(
            row["gene"], row["collection_time"],
            row["hash_well"], row["hash_plate_img"],
            seq_index_full, seq_index_partial,
        ),
        axis=1,
    )
    return pheno


# Acquisition kinds a single fish can be imaged as, in priority order (index = rank, lower wins).
# One fish is often photographed several times (30hpf snapshot, 48hpf snapshot, timeseries) before
# the single collection/destruction event. We keep the most informative acquisition per fish.
_ACQ_PRIORITY = ["timeseries", "snapshot_t02", "snapshot", "snapshot_t01"]


def classify_acquisition(imaging_id: str) -> str:
    """Name the acquisition KIND of one imaging embryo_id (an imaging event of a fish).

    IMPORTANT: distinguishes how the fish was photographed, matching the confidence-plot
    convention timeseries > t02(48hpf) > plain snapshot > t01(30hpf). This is IMAGING time /
    acquisition type — NOT the fish's collection time (see collection_time, which is per-fish).
        timeseries    = _sci timeseries (spans 30-48hpf)   — most informative
        snapshot_t02  = 48hpf backup snapshot
        snapshot      = plain snapshot (no t-token)
        snapshot_t01  = 30hpf snapshot                     — least (only if it's all there is)
    """
    s = str(imaging_id)
    if "_sci_" in s or s.rstrip("_e01").endswith("sci"):
        return "timeseries"
    if "_t02_" in s:
        return "snapshot_t02"
    if "_t01_" in s:
        return "snapshot_t01"
    return "snapshot"


def collapse_to_one_row_per_fish(pheno: pd.DataFrame) -> pd.DataFrame:
    """Collapse to ONE row per fish (== per sequencing embryo_ID), keeping the best acquisition.

    GRAIN: one row per fish. physical_embryo_id <-> seq_embryo_ID is 1:1 ("a fish is destroyed
    once to make its sequencing library"), so the fish is the natural row. Each fish carries ONE
    collection_time (its collection/destruction event) — a per-fish property.

    Why 1:1 already holds here: a fish's several imaging acquisitions (t01/t02/timeseries) were
    already narrowed upstream, and we make the choice EXPLICIT — sort by acquisition priority and
    keep the winner per fish. imaging_embryo_id + acquisition_type record WHICH acquisition won,
    so the collapse is auditable rather than silent.
    """
    resolved = pheno.dropna(subset=["seq_embryo_ID"]).copy()
    resolved["acquisition_type"] = resolved["embryo_id"].map(classify_acquisition)
    resolved["_acq_rank"] = resolved["acquisition_type"].map(_ACQ_PRIORITY.index)
    resolved = resolved.sort_values(["seq_embryo_ID", "_acq_rank"])
    return (
        resolved.drop_duplicates("seq_embryo_ID", keep="first")
        .set_index("seq_embryo_ID")
        .rename(columns={"predicted_label": "morphseq_phenotype",
                         "embryo_id": "imaging_embryo_id"})
    )


def report_resolution(pheno: pd.DataFrame) -> None:
    """Print how many phenotyped imaging embryos resolved to a sequencing embryo_ID."""
    n_resolved = pheno["seq_embryo_ID"].notna().sum()
    print("=== PHASE A: phenotype -> sequencing bridge ===")
    print(f"  phenotyped imaging embryos         : {len(pheno)}")
    print(f"  resolved to a sequencing embryo_ID : {n_resolved}")
    unresolved = pheno[pheno["seq_embryo_ID"].isna()]
    if len(unresolved):
        print(f"  UNRESOLVED: {len(unresolved)}")
        print(unresolved.groupby("experiment").size().to_string())


def write_crosswalk(crosswalk: pd.DataFrame) -> None:
    """Write the crosswalk as a reusable artifact (one row per fish).

    seq_embryo_ID (index) -> physical_embryo_id, collection_time, imaging_embryo_id,
    acquisition_type, label + probs + measurements.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crosswalk[CARRY_COLS].to_csv(CROSSWALK_TSV, sep="\t", index=True)
    print(f"  wrote crosswalk: {CROSSWALK_TSV}  ({len(crosswalk)} embryos)")


# ===========================================================================
# PHASE B — apply the mapping onto the sequencing ordering
# ===========================================================================
def merge_crosswalk_onto_sequencing(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Left-join the crosswalk onto ALL reconciled SEQUENCING embryos (the 520 spine).

    IMPORTANT: every sequencing embryo is KEPT; the carried columns are NaN for embryos with no
    imaging prediction. Join key is embryo_ID (sequencing spine) == seq_embryo_ID (crosswalk index).
    """
    seq = pd.read_csv(SEQ_ORDERING_TSV, sep="\t")
    return seq.merge(
        crosswalk[CARRY_COLS],
        left_on="embryo_ID", right_index=True, how="left",
    )


def null_pre_emergence_labels(seq: pd.DataFrame) -> pd.DataFrame:
    """Null phenotype labels from BEFORE the phenotype biologically exists (per PHENO_MIN_TP).

    IMPORTANT: pre-divergence, the classifier's call is not a real phenotype yet, so we NULL it
    (the embryo keeps its perturbation label). This is why 18/24hpf mutants show up unlabeled.
    """
    for gene, min_tp in PHENO_MIN_TP.items():
        pre_emergence = (
            (seq["target"] == gene)
            & seq["morphseq_phenotype"].notna()
            & (seq["timepoint"] < min_tp)
        )
        n = int(pre_emergence.sum())
        seq.loc[pre_emergence, "morphseq_phenotype"] = pd.NA
        print(f"  [cutoff] nulled {n} {gene} phenotype labels at tp < {min_tp} (pre-emergence)")
    return seq


def report_and_write(seq: pd.DataFrame, pheno_full: pd.DataFrame) -> None:
    """Print the final attach summary and write both output TSVs.

    pheno_full = the full per-imaging-embryo detail (all 169 rows) -> bridge_detail dump.
    """
    print("\n=== PHASE B: attached to reconciled sequencing embryos ===")
    print(f"  sequencing embryos total  : {len(seq)}")
    print(f"  with a MorphSeq phenotype : {seq['morphseq_phenotype'].notna().sum()}")
    print("\n  phenotype by perturbation:")
    print(seq.dropna(subset=["morphseq_phenotype"])
          .groupby(["perturbation", "morphseq_phenotype"]).size().to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seq.to_csv(OUT_DIR / "gene14_embryos_with_phenotype.tsv", sep="\t", index=False)
    pheno_full.to_csv(OUT_DIR / "phenotype_bridge_detail.tsv", sep="\t", index=False)
    print("\nWrote:")
    print(f"  {OUT_DIR / 'gene14_embryos_with_phenotype.tsv'}")
    print(f"  {OUT_DIR / 'phenotype_bridge_detail.tsv'}")


def main() -> None:
    # ===== PHASE A — build the imaging -> sequencing mapping =====
    pheno = load_phenotype_predictions()             # 1. read + clean imaging classifier calls
    pheno = attach_pair_and_features(pheno)           # 2. add breeding pair + length/curvature
    pheno = resolve_to_sequencing_ids(pheno)          # 3. imaging (expt,well,time) -> seq embryo_ID
    report_resolution(pheno)                          #    print resolved / unresolved
    crosswalk = collapse_to_one_row_per_fish(pheno)   # 4. pick winning acquisition -> one row/fish
    write_crosswalk(crosswalk)                        #    -> imaging_to_seq_crosswalk.tsv

    # ===== PHASE B — apply the mapping onto the sequencing ordering =====
    seq = merge_crosswalk_onto_sequencing(crosswalk)  # 5. left-join crosswalk onto the 520 seq embryos
    seq = null_pre_emergence_labels(seq)              # 6. drop labels from before phenotype exists
    report_and_write(seq, pheno)                      #    summary -> gene14_embryos_with_phenotype.tsv


if __name__ == "__main__":
    main()
