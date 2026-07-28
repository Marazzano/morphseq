"""
3_ablation_auroc.py
-------------------
Ablation test: are dims 71 and 33 really as important as their coefficients
claim? If the classifier collapsed the phenotype onto them, removing them should
tank discriminability. If discriminability SURVIVES, the phenotype info is
redundantly spread across the other ~40 dims (as the concentration analysis in
script 2 suggested) and 71/33 are convenient handles, not the only carriers.

Pooled case only: (CE + HTA) vs WT. Four feature sets, refit the classifier on
each and read k-fold CV AUROC over developmental time:
  full    all 80 z_mu_b dims (baseline, no ablation)
  drop71  79 dims
  drop33  79 dims
  drop_both 78 dims

Uses the existing classification machinery end-to-end: run_classification does
the k-fold CV AUROC per time bin + permutation p-values, and
plot_aurocs_over_time renders the four curves. Quick pass — permutations kept
modest, no bootstrap.

Data: reference_b9d2_clean.csv (phenotype_clean in {CE, HTA, wildtype},
zygosity, 80 z_mu_b dims, predicted_stage_hpf).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.classification import run_classification
from analyze.classification.viz import plot_aurocs_over_time

FIGURES = _HERE / "figures"
TABLES = _HERE / "tables"
B9D2_CSV = (
    _REPO / "results" / "mcolon" / "20260607_sci_cilia_gene14_imaging_qc"
    / "tables" / "reference_b9d2_clean.csv"
)

TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
N_PERM = 20           # quick first pass; light permutation count, no bootstrap
N_SPLITS = 5          # k-fold CV

ABLATIONS = {
    "full": [],
    "drop71": ["z_mu_b_71"],
    "drop33": ["z_mu_b_33"],
    "drop_both": ["z_mu_b_71", "z_mu_b_33"],
}
ABLATION_COLORS = {
    "full": "#000000",
    "drop71": "#1b7837",
    "drop33": "#762a83",
    "drop_both": "#d95f02",
}


def load_pooled():
    df = pd.read_csv(B9D2_CSV, low_memory=False)
    # pooled binary problem: GRP = CE+HTA phenotype embryos, WT = wildtype
    wt = df[df["zygosity"] == "wildtype"].copy()
    grp = df[df["phenotype_clean"].isin(["CE", "HTA"])].copy()
    wt["label"], grp["label"] = "WT", "GRP"
    pooled = pd.concat([wt, grp], ignore_index=True)
    z_cols = sorted(
        [c for c in pooled.columns if c.startswith("z_mu_b")],
        key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")),
    )
    return pooled, z_cols


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_pooled()
    print(f"Loaded {len(df)} frames; GRP embryos={df[df.label=='GRP'].embryo_id.nunique()}, "
          f"WT embryos={df[df.label=='WT'].embryo_id.nunique()}; {len(z_cols)} dims")

    all_scores = []
    for name, drop in ABLATIONS.items():
        feats = [c for c in z_cols if c not in drop]
        print(f"\n=== ablation '{name}': {len(feats)} dims (dropped {drop or 'none'}) ===")
        result = run_classification(
            df, class_col="label", id_col="embryo_id", time_col=TIME_COL,
            positive="GRP", negative="WT",
            features={"emb": feats},
            bin_width=BIN_WIDTH, n_permutations=N_PERM, n_splits=N_SPLITS,
            random_state=42, class_weight="balanced", verbose=False,
        )
        s = result.scores.copy()
        s["ablation"] = name
        all_scores.append(s)

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(TABLES / "ablation_scores.csv", index=False)

    # four curves, one per ablation, on the pooled GRP-vs-WT AUROC over time
    plot_aurocs_over_time(
        scores,
        curve_col="ablation",
        color_lookup=ABLATION_COLORS,
        show_significance=True, sig_threshold=0.01,
        show_chance_line=True,
        title="b9d2 pooled (CE+HTA) vs WT — AUROC over time under dim ablation\n"
              "does removing 71 / 33 / both actually cost discriminability?",
        backend="matplotlib",
        output_path=FIGURES / "ablation_auroc_over_time.png",
    )
    print(f"\nSaved figure -> {FIGURES / 'ablation_auroc_over_time.png'}")

    # compact readout: mean AUROC per ablation (over bins) and the cost vs full
    summary = scores.groupby("ablation")["auroc_obs"].mean()
    base = summary.get("full", float("nan"))
    print("\n=== mean AUROC over time bins (cost vs full) ===")
    for name in ABLATIONS:
        v = summary.get(name, float("nan"))
        print(f"  {name:10} mean AUROC={v:.3f}   Δ vs full={v-base:+.3f}")


if __name__ == "__main__":
    main()
