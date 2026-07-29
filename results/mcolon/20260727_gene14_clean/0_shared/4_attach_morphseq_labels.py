#!/usr/bin/env python
"""
4_attach_morphseq_labels.py  --  the ONLY script that writes labels onto the spine
=============================================================================
Left-joins imaging-derived labels onto `embryo_table.tsv` (from step 0) and
rewrites it as `embryo_table_labeled.tsv`.

    step 0 output  ->  + morphseq phenotype  ->  embryo_table_labeled.tsv
                       + morphology metrics
                       + imaging QC flags

THE JOIN: same fish, two naming systems
    imaging  20260415_cep290_18hpf_plate03_C01_e01   date_gene_stage_plate_well_e##
    seq      GENE14_P18_F10_Bl2                      GENE14_P{hashplate}_{hashwell}_Bl{rtblock}

The ID strings share nothing, so they are joined on PLATE COORDINATES:

    imaging (experiment, well) --image_to_hash_map|identity--> hash_well
                              --hash_plate_num-------------->  hash_plate
                              --collection_time (30to48->48)-> timepoint
    lookup (gene, timepoint, hash_plate, hash_well) -> one seq embryo_ID

`physical_embryo_id` <-> seq `embryo_ID` is 1:1 -- a fish is destroyed once to
make its library. Imaging `embryo_id` is many-per-fish (snapshot + _t01/_t02 +
_sci timeseries), so acquisitions are collapsed by explicit priority and the
winner is recorded, never silently chosen.

All of that lives in 2_seq_imaging_crosswalk.py, which is imported, not rewritten.

PHENOTYPE ONLY, BY DESIGN
    Predicted GENOTYPE also exists (sequenced_genotype_qc_cross_bin.csv, 496
    embryos including 205 crispants) and is resolvable by the same crosswalk.
    Out of scope for now -- `attach_labels(...)` takes the label file as an
    argument, so adding it later is one more call, not a rewrite.

    conda run -n segmentation_grounded_sam --no-capture-output python \
        0_shared/4_attach_morphseq_labels.py
=============================================================================
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
QC_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"

SPINE_IN = HERE / "embryo_table.tsv"                    # from 0_load_mcclintock.R
SPINE_OUT = HERE / "embryo_table_labeled.tsv"
CROSSWALK_OUT = HERE / "imaging_to_seq_crosswalk.tsv"

PHENO_CSV = QC_DIR / "predictions/sequenced_homozygous_phenotype_cross_bin.csv"
MORPH_CSV = QC_DIR / "tables/query_all_rows_clean.csv"
REGISTRY_CSV = QC_DIR / "tables/embryo_registry.csv"

PROB_COLS = ["prob_CE", "prob_HTA", "prob_High_to_Low", "prob_Low_to_High"]
MORPH_COLS = ["total_length_um", "baseline_deviation_normalized"]

# A phenotype only EXISTS once morphological divergence has emerged; before that
# the classifier is guessing something that is not there yet.
#   cep290 (HtL/LtH): tp >= 30  (18hpf was all-LtH, 24hpf all-HtL == the defaults)
#   b9d2   (CE/HTA) : tp >= 18  (drops the 14hpf calls)
PHENO_MIN_TP = {"cep290": 30, "b9d2": 18}

# One fish is often photographed several times before the single collection event.
# Keep the most informative acquisition, and record which one won.
ACQ_PRIORITY = ["timeseries", "snapshot_t02", "snapshot", "snapshot_t01"]

# import the resolver by path -- its filename starts with a digit
_spec = importlib.util.spec_from_file_location("cw", HERE / "2_seq_imaging_crosswalk.py")
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)


def classify_acquisition(imaging_id: str) -> str:
    """Which KIND of imaging event this is (not the fish's collection time)."""
    s = str(imaging_id)
    if "_sci_" in s or s.rstrip("_e01").endswith("sci"):
        return "timeseries"
    if "_t02_" in s:
        return "snapshot_t02"
    if "_t01_" in s:
        return "snapshot_t01"
    return "snapshot"


def resolve_labels(label_csv, label_col="predicted_label", out_col="morphseq_phenotype",
                   prob_cols=PROB_COLS, min_tp=None):
    """Map one imaging label file onto sequencing embryo_IDs. One row per fish.

    label_csv / label_col / out_col / min_tp are arguments precisely so this can
    be pointed at the genotype file later without editing anything.
    """
    lab = pd.read_csv(label_csv, low_memory=False)
    print(f"  {label_csv.name}: {len(lab)} rows")

    # Three pieces of the resolver, used exactly as it intends:
    #   build_crosswalk()  (experiment, imaging_well) -> hash_well, hash_plate
    #   build_seq_index()  indexes OUR spine for lookup + the P18->P02 tie-break
    #   resolve_seq_embryo_id()  one imaging embryo -> one sequencing embryo_ID
    plate_map = cw.build_crosswalk().set_index(["experiment", "imaging_well"])
    spine = pd.read_csv(SPINE_IN, sep="\t", low_memory=False)
    seq_index = cw.build_seq_index(meta=spine)
    print(f"  plate map: {len(plate_map)} (experiment, well) entries; "
          f"seq index built from our own {len(spine)}-embryo spine")

    seq_ids, phys_ids, coll_times = [], [], []
    for r in lab.itertuples():
        well = cw.norm_well(r.well)
        # populated image_to_hash_map -> use it; absent -> the imaging well IS
        # the hash well (identity plates). hash_plate comes from the map either way.
        hit = plate_map.loc[(r.experiment, well)] if (r.experiment, well) in plate_map.index else None
        hash_well = hit["hash_well"] if hit is not None else well
        hash_plate = hit["hash_plate"] if hit is not None else None
        rt_block = hit["rt_block"] if hit is not None else None
        tp = cw.collection_time_hpf(r.experiment, getattr(r, "collection_time_hpf", None))
        seq_ids.append(cw.resolve_seq_embryo_id(hash_plate, hash_well, rt_block, seq_index))
        phys_ids.append(getattr(r, "physical_embryo_id", None))
        coll_times.append(tp)

    lab["seq_embryo_ID"] = seq_ids
    lab["physical_embryo_id"] = phys_ids
    lab["collection_time"] = coll_times

    resolved = lab.dropna(subset=["seq_embryo_ID"]).copy()
    orphans = len(lab) - len(resolved)
    print(f"  resolved {len(resolved)}/{len(lab)}  (unresolved: {orphans})")
    if orphans:
        print("    unresolved:",
              lab[lab.seq_embryo_ID.isna()].embryo_id.head(5).tolist())

    # collapse to one row per fish, keeping the best acquisition -- explicitly
    resolved["acquisition_type"] = resolved.embryo_id.map(classify_acquisition)
    resolved["_rank"] = resolved.acquisition_type.map(ACQ_PRIORITY.index)
    resolved = (resolved.sort_values(["seq_embryo_ID", "_rank"])
                        .drop_duplicates("seq_embryo_ID", keep="first"))
    print(f"  collapsed to {len(resolved)} fish "
          f"({resolved.acquisition_type.value_counts().to_dict()})")

    resolved = resolved.rename(columns={label_col: out_col,
                                        "embryo_id": "imaging_embryo_id"})

    # null out pre-emergence calls: the label is not wrong, it does not exist yet
    if min_tp:
        for gene, floor in min_tp.items():
            mask = (resolved.gene.astype(str).str.contains(gene, case=False, na=False)
                    & (resolved.collection_time < floor))
            if mask.any():
                print(f"  {gene}: nulling {mask.sum()} calls before tp{floor} "
                      f"(pre-divergence)")
                resolved.loc[mask, [out_col, *prob_cols]] = pd.NA

    keep = ["seq_embryo_ID", "physical_embryo_id", "collection_time",
            "imaging_embryo_id", "acquisition_type", out_col,
            *[c for c in prob_cols if c in resolved.columns]]
    return resolved[keep]


def main():
    print("=" * 74)
    print("4_attach_morphseq_labels.py")
    print("=" * 74)

    for p in (SPINE_IN, PHENO_CSV, MORPH_CSV, REGISTRY_CSV):
        if not p.exists():
            sys.exit(f"MISSING: {p}")

    spine = pd.read_csv(SPINE_IN, sep="\t", low_memory=False)
    n0 = len(spine)
    print(f"\nspine: {n0} embryos (from 0_load_mcclintock.R)")

    print("\n--- resolving phenotype labels ---")
    pheno = resolve_labels(PHENO_CSV, min_tp=PHENO_MIN_TP)
    pheno.to_csv(CROSSWALK_OUT, sep="\t", index=False)
    print(f"  wrote {CROSSWALK_OUT.name}")

    # ---- morphology, median-collapsed --------------------------------------
    # query_all_rows_clean.csv is PER FRAME; a timeseries fish has many rows, so
    # take the median per fish rather than an arbitrary one.
    print("\n--- morphology metrics ---")
    morph = pd.read_csv(MORPH_CSV, low_memory=False,
                        usecols=["physical_embryo_id", *MORPH_COLS])
    morph = (morph.dropna(subset=["physical_embryo_id"])
                  .groupby("physical_embryo_id")[MORPH_COLS].median().reset_index())
    print(f"  {len(morph)} fish with morphology (median over frames)")

    print("\n--- imaging QC flags ---")
    reg = pd.read_csv(REGISTRY_CSV, low_memory=False)
    reg_cols = [c for c in ("physical_embryo_id", "imaging_qc_disposition",
                            "final_usable", "has_latents", "representative_embryo_id")
                if c in reg.columns]
    reg = reg[reg_cols].drop_duplicates("physical_embryo_id")
    print(f"  {len(reg)} fish in embryo_registry.csv")

    # ---- join, spine-preserving -------------------------------------------
    # Suffix collisions matter: the spine already has `timepoint` (sequencing) and
    # `pheno` (sequencing-side label). collection_time and morphseq_phenotype are
    # DIFFERENT things and must not overwrite them.
    out = (spine.merge(pheno, left_on="embryo_ID", right_on="seq_embryo_ID", how="left")
                .merge(morph, on="physical_embryo_id", how="left")
                .merge(reg, on="physical_embryo_id", how="left"))

    assert len(out) == n0, f"join changed row count: {n0} -> {len(out)}"
    out.to_csv(SPINE_OUT, sep="\t", index=False)

    print(f"\nwrote {SPINE_OUT.name}: {len(out)} rows x {out.shape[1]} cols")
    print(f"  morphseq_phenotype non-null : {out.morphseq_phenotype.notna().sum()}")
    print(f"  physical_embryo_id resolved : {out.physical_embryo_id.notna().sum()}")
    print(f"  total_length_um present     : {out.total_length_um.notna().sum()}")
    print("\n  phenotype by gene target:")
    print(out[out.morphseq_phenotype.notna()]
          .groupby(["target", "morphseq_phenotype"]).size().to_string())
    print("\ndone. next: 5_celltype_gate.py")


if __name__ == "__main__":
    main()
