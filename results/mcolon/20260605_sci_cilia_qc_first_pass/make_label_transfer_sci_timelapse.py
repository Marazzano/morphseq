"""
Label transfer for the sci_ timelapse plates (30–48 hpf Yokogawa acquisitions).

Two models per gene:
  • homozygous_focus  — reference filtered to homozygous only (no Not_Penetrant / wildtype leakage)
  • all_phenotypes    — full reference including het, wt, Not_Penetrant

Both models are run on ALL sci_ embryos; the trajectory scripts filter to sequenced > 0.

Outputs (relative to this script's directory):
  time_series/sequenced_focus/homozygous_focus/{gene}_homo_predictions.csv
  time_series/sequenced_focus/all_phenotypes/{gene}_all_predictions.csv

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/make_label_transfer_sci_timelapse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

import build_reference_and_transfer as T  # noqa: E402

PLATE_META = PROJECT_ROOT / "metadata/plate_metadata"
WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]
OUT_HOMO = RUN_DIR / "time_series" / "sequenced_focus" / "homozygous_focus"
OUT_ALL = RUN_DIR / "time_series" / "sequenced_focus" / "all_phenotypes"

SCI_PLATES = {
    "20260414_sci_b9d2_48hpf_plate01": "b9d2",
    "20260415_sci_cep290_48hpf_plate01": "cep290",
}


def sequenced_grid(exp: str) -> dict[str, int]:
    """well -> sequenced code {0,1,2}. Returns {} if not found."""
    for cand in (f"{exp}_well_metadata.xlsx", f"{exp}.xlsx"):
        p = PLATE_META / cand
        if not p.exists():
            continue
        with pd.ExcelFile(p) as xlf:
            if "sequenced" not in xlf.sheet_names:
                return {}
            df = xlf.parse("sequenced", header=0)
            block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
            arr = block.to_numpy(dtype=str).ravel()
        out: dict[str, int] = {}
        for w, v in zip(WELLS, arr):
            s = v.strip()
            try:
                out[w] = int(float(s)) if s not in ("", "nan") else 0
            except ValueError:
                out[w] = 0
        return out
    return {}


def build_well_seq_map(exps: list[str]) -> pd.DataFrame:
    """embryo_id -> (well, sequenced_code) from build06 CSVs + plate Excel grids."""
    rows = []
    for exp in exps:
        p = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
        wdf = pd.read_csv(p, usecols=[T.GROUP_COL, "well"], low_memory=False)
        wdf = wdf.drop_duplicates(T.GROUP_COL)
        grid = sequenced_grid(exp)
        wdf["sequenced"] = wdf["well"].map(grid).fillna(0).astype(int)
        rows.append(wdf)
    return pd.concat(rows, ignore_index=True).set_index(T.GROUP_COL)


def pool_b9d2_phenotypes(df: pd.DataFrame, col: str = T.PHENO_COL) -> pd.DataFrame:
    """Merge BA_rescue into HTA for b9d2 (pooled phenotype convention)."""
    df = df.copy()
    df[col] = df[col].replace("BA_rescue", "HTA")
    return df


def run_transfer(gene: str, ref: pd.DataFrame, qry: pd.DataFrame, feat: list[str],
                 tag: str, well_seq: pd.DataFrame) -> pd.DataFrame:
    """Run phenotype transfer, attach predicted_stage_hpf and sequenced code. Returns embryo df."""
    emb_df = T.run_phenotype_transfer(gene, ref, qry, feat)
    if emb_df.empty:
        return emb_df

    # join predicted_stage_hpf from query (group by embryo_id, take min/max/median across frames)
    hpf_map = (
        qry.dropna(subset=[T.TIME_COL])
        .groupby(T.GROUP_COL)[T.TIME_COL]
        .agg(["min", "max", "median"])
        .rename(columns={"min": "hpf_min", "max": "hpf_max", "median": "hpf_median"})
    )
    emb_df = emb_df.join(hpf_map, on="query_embryo_id", how="left")

    # join well + sequenced code from the pre-built map (from raw build06 CSV + Excel)
    emb_df["well"] = emb_df["query_embryo_id"].map(well_seq["well"])
    emb_df["sequenced"] = emb_df["query_embryo_id"].map(well_seq["sequenced"]).fillna(0).astype(int)
    emb_df["model"] = tag

    return emb_df


def main() -> None:
    OUT_HOMO.mkdir(parents=True, exist_ok=True)
    OUT_ALL.mkdir(parents=True, exist_ok=True)

    # group sci_ plates by gene
    by_gene: dict[str, list[str]] = {"b9d2": [], "cep290": []}
    for exp, gene in SCI_PLATES.items():
        p = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
        if not p.exists():
            print(f"  MISSING build06: {exp} — skipping")
            continue
        by_gene[gene].append(exp)

    ref_paths = {"b9d2": T.B9D2_REF, "cep290": T.CEP290_REF}

    for gene, exps in by_gene.items():
        if not exps:
            continue
        ref_path = ref_paths[gene]
        qpaths = [T.B6 / f"df03_final_output_with_latents_{e}.csv" for e in exps]
        feat = T.resolve_feature_cols([ref_path, *qpaths])
        print(f"\n{'='*60}")
        print(f"{gene.upper()}  |  feature dims={len(feat)}  |  exps={exps}")

        ref_full = T._load(ref_path, feat, gene_hint=gene)
        if gene == "b9d2":
            ref_full = pool_b9d2_phenotypes(ref_full)
        qparts = []
        for exp, p in zip(exps, qpaths):
            q = T._load(p, feat, gene_hint=gene)
            q["query_experiment"] = exp
            qparts.append(q)
        qry = pd.concat(qparts, ignore_index=True)

        # build well + sequenced lookup BEFORE transfer (needs raw build06 CSV, not T._load filtered)
        well_seq = build_well_seq_map(exps)

        # predicted_stage_hpf range sanity check
        hpf = qry[T.TIME_COL].dropna()
        print(f"  query predicted_stage_hpf: [{hpf.min():.2f}, {hpf.max():.2f}]  n={len(hpf)}")

        # ── Model A: homozygous only reference ───────────────────────────────
        # Filter to homozygous rows AND only the penetrant phenotype classes.
        # For cep290: keep only High_to_Low + Low_to_High (drop Not Penetrant entirely).
        # For b9d2:   keep only CE + HTA (BA_rescue already pooled into HTA above).
        # This ensures zero Not_Penetrant / wildtype leakage into the homo model.
        HOMO_KEEP = {
            "b9d2":   {"CE", "HTA"},
            "cep290": {"High_to_Low", "Low_to_High"},
        }
        homo_mask = ref_full[T.GENO_COL].str.endswith("_homozygous", na=False)
        keep_labels = HOMO_KEEP.get(gene, set())
        ref_homo = ref_full[homo_mask & ref_full[T.PHENO_COL].isin(keep_labels)].copy()
        print(f"  ref_homo rows: {len(ref_homo)}  |  cluster_categories: "
              f"{ref_homo[T.PHENO_COL].value_counts().to_dict()}")

        homo_preds = run_transfer(gene, ref_homo, qry, feat, tag="homozygous_focus", well_seq=well_seq)
        if not homo_preds.empty:
            out_path = OUT_HOMO / f"{gene}_homo_predictions.csv"
            homo_preds.to_csv(out_path, index=False)
            print(f"  [homo_focus] predicted_label dist: "
                  f"{homo_preds['predicted_label'].value_counts().to_dict()}")
            seq_preds = homo_preds[homo_preds["sequenced"] > 0]
            print(f"  [homo_focus] sequenced embryos: {len(seq_preds)}  "
                  f"label dist: {seq_preds['predicted_label'].value_counts().to_dict()}")
            print(f"  → wrote {out_path.name}")

        # ── Model B: full reference ──────────────────────────────────────────
        all_preds = run_transfer(gene, ref_full, qry, feat, tag="all_phenotypes", well_seq=well_seq)
        if not all_preds.empty:
            out_path = OUT_ALL / f"{gene}_all_predictions.csv"
            all_preds.to_csv(out_path, index=False)
            print(f"  [all_pheno] predicted_label dist: "
                  f"{all_preds['predicted_label'].value_counts().to_dict()}")
            seq_preds = all_preds[all_preds["sequenced"] > 0]
            print(f"  [all_pheno] sequenced embryos: {len(seq_preds)}  "
                  f"label dist: {seq_preds['predicted_label'].value_counts().to_dict()}")
            print(f"  → wrote {out_path.name}")


if __name__ == "__main__":
    main()
