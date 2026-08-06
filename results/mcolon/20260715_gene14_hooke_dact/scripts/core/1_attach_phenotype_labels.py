"""
1 - Attach MorphSeq phenotype labels to the GENE14 sequencing / DACT embryos.

All the mapping logic + the full record of what is clean vs tricky lives in the sibling
module `seq_imaging_crosswalk.py` (read its docstring first). This script is a thin consumer:

  phenotype prediction (imaging embryo)
    --seq_imaging_crosswalk-->  sequencing embryo_ID
    --merge-->                  the reconciled 520-embryo DACT table

Phenotype is added as an EXTRA column: every DACT embryo is kept; embryos without a MorphSeq
phenotype prediction simply keep their perturbation label (morphseq_phenotype = NaN).
Phenotype models exist only for b9d2 (CE/HTA) and cep290 (High_to_Low/Low_to_High) homozygotes.

Run:
  conda run -n segmentation_grounded_sam --no-capture-output python \
      results/mcolon/20260715_gene14_hooke_dact/scripts/core/1_attach_phenotype_labels.py
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
# query_all_rows_clean carries the breeding-`pair` (cross/spawn) per imaging embryo_id.
QUERY_ROWS_CSV = QC_DIR / "tables/query_all_rows_clean.csv"

RUN_DIR = PROJECT_ROOT / "results/mcolon/20260715_gene14_hooke_dact"
RECONCILED_TSV = RUN_DIR / "output/gene14_embryos_reconciled.tsv"
OUT_DIR = RUN_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROB_COLS = ["prob_CE", "prob_HTA", "prob_High_to_Low", "prob_Low_to_High"]


def main() -> None:
    plate_maps = load_plate_maps()
    by_full, by_part = build_seq_index()

    pheno = pd.read_csv(PHENO_CSV, low_memory=False)
    pheno = pheno.dropna(subset=["predicted_label"]).copy()
    pheno["imaging_well"] = pheno["well"].map(norm_well)

    # From the query table (keyed on imaging embryo_id), attach where available:
    #   pair                            = breeding pair / spawn (cep290_P1, cep290_spawn, ab, ...)
    #   total_length_um                 = embryo length (physical feature)
    #   baseline_deviation_normalized   = curvature (physical feature)
    # length/curvature are PER-FRAME for timeseries embryos, so collapse to the per-embryo
    # median (order-independent; snapshots have a single frame so this is a no-op for them).
    q = pd.read_csv(QUERY_ROWS_CSV, low_memory=False)
    q_pair = q.dropna(subset=["pair"]).drop_duplicates("embryo_id").set_index("embryo_id")["pair"]
    q_feat = q.groupby("embryo_id")[["total_length_um", "baseline_deviation_normalized"]].median()
    pheno["pair"] = pheno["embryo_id"].map(q_pair)
    pheno["total_length_um"] = pheno["embryo_id"].map(q_feat["total_length_um"])
    pheno["baseline_deviation_normalized"] = pheno["embryo_id"].map(q_feat["baseline_deviation_normalized"])

    # imaging (experiment, well) -> (hash_well, hash_plate) via the crosswalk (both regimes)
    def hw_hp(r):
        m = plate_maps.get(r["experiment"], {})
        return pd.Series(m.get(r["imaging_well"], (r["imaging_well"], None)))
    pheno[["hash_well", "hash_plate_img"]] = pheno.apply(hw_hp, axis=1)

    # collection time (30to48 -> 48) is what the sequencing `timepoint` encodes
    pheno["collection_time"] = pheno.apply(
        lambda r: collection_time_hpf(r["experiment"],
                                      pd.to_numeric(r["collection_time_hpf"], errors="coerce")),
        axis=1,
    )

    # resolve each phenotyped embryo -> its sequencing embryo_ID
    pheno["seq_embryo_ID"] = pheno.apply(
        lambda r: resolve_seq_embryo_id(r["gene"], r["collection_time"],
                                        r["hash_well"], r["hash_plate_img"],
                                        by_full, by_part),
        axis=1,
    )

    n_res = pheno["seq_embryo_ID"].notna().sum()
    print(f"=== phenotype -> sequencing bridge ===")
    print(f"  phenotyped imaging embryos         : {len(pheno)}")
    print(f"  resolved to a sequencing embryo_ID : {n_res}")
    unresolved = pheno[pheno["seq_embryo_ID"].isna()]
    if len(unresolved):
        print(f"  UNRESOLVED: {len(unresolved)}")
        print(unresolved.groupby("experiment").size().to_string())

    # attach onto the reconciled DACT embryos — keep ALL 520; phenotype is an EXTRA column
    recon = pd.read_csv(RECONCILED_TSV, sep="\t")

    # ONE physical embryo -> ONE sequencing embryo, in its collection x acquisition column.
    # When several imaging rows (t01 30hpf snapshot / t02 48hpf snapshot / _sci timeseries)
    # resolve to the SAME seq_embryo_ID, we must NOT take whichever sorts first — we prioritize
    # by acquisition, matching the confidence-plot convention:
    #   timeseries (_sci)  >  48hpf snapshot (_t02)  >  30hpf snapshot (_t01)  >  other
    # (timeseries is prioritized for label transfer; snapshot-only plates like cep290/b9d2
    # plate02 use their 48hpf t02 snapshot as the answer, NOT the 30hpf t01). Without this,
    # the plate02 t01 (30hpf) row wrongly won over t02 (48hpf) for E01/F01, mislabeling E01
    # (t01=High_to_Low@0.56 vs the correct t02=Low_to_High@0.99) and bucketing it at 48.
    def _acq_priority(imaging_id: str) -> int:
        s = str(imaging_id)
        if "_sci_" in s or s.rstrip("_e01").endswith("sci"):
            return 0  # timeseries — highest priority
        if "_t02_" in s:
            return 1  # 48hpf backup snapshot
        if "_t01_" in s:
            return 3  # 30hpf snapshot — lowest (only if it's the only acquisition)
        return 2      # plain snapshot (no t-token)
    ph = pheno.dropna(subset=["seq_embryo_ID"]).copy()
    ph["_acq_rank"] = ph["embryo_id"].map(_acq_priority)
    ph = ph.sort_values(["seq_embryo_ID", "_acq_rank"])
    pheno_by_seq = (
        ph.drop_duplicates("seq_embryo_ID", keep="first")
        .set_index("seq_embryo_ID")
        .rename(columns={"predicted_label": "morphseq_phenotype", "embryo_id": "imaging_embryo_id"})
    )
    recon = recon.merge(
        pheno_by_seq[["morphseq_phenotype", *PROB_COLS, "pair",
                      "total_length_um", "baseline_deviation_normalized", "imaging_embryo_id"]],
        left_on="embryo_ID", right_index=True, how="left",
    )

    # BIOLOGICAL CUTOFF: the phenotype only EXISTS once morphological divergence has
    # emerged. Before that, the imaging classifier is guessing a phenotype that isn't
    # real yet, so we NULL the label there (the embryo keeps its perturbation label).
    #   cep290 (HtL/LtH): meaningful only at tp >= 30 (pre-30 was the classifier default;
    #                     e.g. 18hpf was all-LtH, 24hpf all-HtL -- pre-phenotype noise)
    #   b9d2   (CE/HTA) : meaningful only at tp >= 18 (drops the 14hpf calls)
    PHENO_MIN_TP = {"cep290": 30, "b9d2": 18}
    for gene, mintp in PHENO_MIN_TP.items():
        pre = (recon["target"] == gene) & recon["morphseq_phenotype"].notna() & (recon["timepoint"] < mintp)
        n = int(pre.sum())
        recon.loc[pre, "morphseq_phenotype"] = pd.NA
        print(f"  [cutoff] nulled {n} {gene} phenotype labels at tp < {mintp} (pre-emergence)")

    print(f"\n=== attached to reconciled DACT embryos ===")
    print(f"  DACT embryos total        : {len(recon)}")
    print(f"  with a MorphSeq phenotype : {recon['morphseq_phenotype'].notna().sum()}")
    print("\n  phenotype by perturbation:")
    print(recon.dropna(subset=["morphseq_phenotype"])
          .groupby(["perturbation", "morphseq_phenotype"]).size().to_string())

    recon.to_csv(OUT_DIR / "gene14_embryos_with_phenotype.tsv", sep="\t", index=False)
    pheno.to_csv(OUT_DIR / "phenotype_bridge_detail.tsv", sep="\t", index=False)
    print(f"\nWrote:")
    print(f"  {OUT_DIR / 'gene14_embryos_with_phenotype.tsv'}")
    print(f"  {OUT_DIR / 'phenotype_bridge_detail.tsv'}")


if __name__ == "__main__":
    main()
