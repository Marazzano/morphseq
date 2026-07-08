"""
3f - Experiment-separability batch controls for the early CEP290 AUROC.

The cep290_phenotype_auroc.png (3e) shows surprisingly high separability at 14–20 hpf.
At those early bins only two experiments contribute homozygous embryos:
    20251212  ->  mostly Low_to_High
    20250512  ->  mostly High_to_Low
Any batch/imaging difference between those experiments could masquerade as phenotype signal.

This script runs six diagnostic tests to characterise that risk:

  A  WT-only:   can experiment_id (20251212 vs 20250512) be predicted from morphology?
  B  HET-only:  same question in heterozygotes
  C  Homo from 20251212 only:  can Low_to_High vs High_to_Low be separated within that experiment?
  D  Homo from 20250512 only:  same within the other experiment
  E/F Homo vs WT within each early experiment (transitivity check)

Interpretation:
  A/B elevated at early hpf  -> batch recoverable even in non-phenotypic embryos
  C/D elevated               -> real biology present within a single experiment
  E/F flat at early hpf      -> confirms no within-experiment phenotype signal early on

All outputs are written to plots/separability/3f_batch_controls/ to keep the top-level
separability dir clean (3e outputs remain at plots/separability/).

Features: emb (z_mu_b latents) + curvature (baseline_deviation_normalized) only.
All plots note this in the title.

Also produces a class-balance summary figure (class_balance.png) showing, for each of the
two early experiments, the per-zygosity and per-phenotype embryo counts. This documents the
confound structure that motivates the controls.

Run:
    PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output \\
        python results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3f_experiment_batch_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.classification import run_classification
from analyze.classification.viz import plot_aurocs_over_time

TABLE_DIR  = RUN_DIR / "tables"
CLASS_DIR  = RUN_DIR / "classification"
# All 3f outputs go into a dedicated subdir; 3e outputs stay at plots/separability/
PLOT_DIR   = RUN_DIR / "plots" / "separability" / "3f_batch_controls"
CLASS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL   = "embryo_id"
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 2.0
N_SPLITS  = 5
N_PERM    = 500
MAX_HPF   = 48.0

EARLY_EXPERIMENTS = ["20251212", "20250512"]

# emb + curvature only (sufficient to diagnose batch vs biology)
FEATURES = {
    "emb":       "z_mu_b",
    "curvature": ["baseline_deviation_normalized"],
}

KEEP_COLS = {ID_COL, TIME_COL, "zygosity", "experiment_id",
             "baseline_deviation_normalized", "cluster_categories"}

FEAT_NOTE = "features: emb + curvature only"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_ref() -> pd.DataFrame:
    df = pd.read_csv(
        TABLE_DIR / "reference_cep290_clean.csv",
        usecols=lambda c: c in KEEP_COLS or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    df = df[df["experiment_id"].isin(EARLY_EXPERIMENTS)].copy()
    df = df.dropna(subset=[TIME_COL, "baseline_deviation_normalized"])
    return df[df[TIME_COL] <= MAX_HPF].copy()


def load_exp_ctrl(zygosity: str) -> pd.DataFrame:
    """Tests A/B: label = experiment_id, within a single zygosity class."""
    return _load_ref().pipe(lambda d: d[d["zygosity"] == zygosity].copy())


def load_homo_single_exp(exp: str) -> pd.DataFrame:
    """Tests C/D: label = phenotype_label, homo embryos from one experiment only."""
    df = _load_ref()
    df = df[(df["zygosity"] == "homozygous") & (df["experiment_id"] == exp)].copy()
    df["phenotype_label"] = df["cluster_categories"].replace({"Intermediate": "Low_to_High"})
    return df[df["phenotype_label"].isin(["High_to_Low", "Low_to_High"])].copy()


def load_homo_vs_wt_single_exp(exp: str) -> pd.DataFrame:
    """Tests E/F: homo vs WT within one experiment (no cross-experiment batch)."""
    df = _load_ref()
    df = df[(df["experiment_id"] == exp) & (df["zygosity"].isin(["homozygous", "wildtype"]))].copy()
    return df.copy()


# ---------------------------------------------------------------------------
# Class-balance summary figure
# ---------------------------------------------------------------------------

def plot_class_balance() -> None:
    """
    Two-panel figure showing embryo counts per zygosity and per phenotype for each
    of the two early experiments. Documents the confound structure that motivates
    the batch controls.
    """
    df_all = pd.read_csv(
        TABLE_DIR / "reference_cep290_clean.csv",
        usecols=["embryo_id", "experiment_id", "zygosity", "cluster_categories"],
        low_memory=False,
    )
    df_all = df_all[df_all["experiment_id"].isin(EARLY_EXPERIMENTS)].copy()
    df_emb = df_all.drop_duplicates("embryo_id")

    # Panel 1: zygosity distribution per experiment
    zyg_counts = (
        df_emb.groupby(["experiment_id", "zygosity"])["embryo_id"]
        .count()
        .unstack(fill_value=0)
    )
    zyg_order = ["wildtype", "heterozygous", "homozygous", "unknown"]
    zyg_counts = zyg_counts.reindex(
        columns=[c for c in zyg_order if c in zyg_counts.columns]
    )

    # Panel 2: phenotype distribution among homos per experiment
    homo = df_emb[df_emb["zygosity"] == "homozygous"].copy()
    homo["phenotype_label"] = homo["cluster_categories"].replace({"Intermediate": "Low_to_High"})
    phen_counts = (
        homo.groupby(["experiment_id", "phenotype_label"])["embryo_id"]
        .count()
        .unstack(fill_value=0)
    )
    phen_order = ["Low_to_High", "High_to_Low", "Not Penetrant", "Intermediate"]
    phen_counts = phen_counts.reindex(
        columns=[c for c in phen_order if c in phen_counts.columns]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "CEP290 class balance in the two early-timepoint experiments\n"
        "(motivates the batch controls in 3f)",
        fontsize=11,
    )

    zyg_colors = ["#2166AC", "#F7B267", "#B2182B", "#808080"]
    zyg_counts.plot(
        kind="bar", ax=axes[0], color=zyg_colors[: len(zyg_counts.columns)],
        edgecolor="white", width=0.6,
    )
    axes[0].set_title("All embryos — zygosity", fontsize=10)
    axes[0].set_xlabel("experiment_id")
    axes[0].set_ylabel("# embryos")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="zygosity", fontsize=8, title_fontsize=8)

    phen_colors = ["#4DAF4A", "#E41A1C", "#984EA3", "#FF7F00"]
    phen_counts.plot(
        kind="bar", ax=axes[1], color=phen_colors[: len(phen_counts.columns)],
        edgecolor="white", width=0.6,
    )
    axes[1].set_title("Homozygous only — phenotype class", fontsize=10)
    axes[1].set_xlabel("experiment_id")
    axes[1].set_ylabel("# embryos")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="phenotype", fontsize=8, title_fontsize=8)

    # Annotate bars with counts
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", label_type="edge", fontsize=7, padding=2)

    fig.tight_layout()
    out = PLOT_DIR / "class_balance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(RUN_DIR)}")


# ---------------------------------------------------------------------------
# Helper: run one comparison and save a plot
# ---------------------------------------------------------------------------

def _run_and_plot(
    df: pd.DataFrame,
    *,
    class_col: str,
    positive: str,
    negative: str,
    save_name: str,
    title: str,
    min_samples_per_group: int = 3,
) -> None:
    n_emb = df[ID_COL].nunique()
    vc    = df.drop_duplicates(ID_COL)[class_col].value_counts().to_dict()
    print(f"  {n_emb} embryos | {class_col} counts: {vc}")

    result = run_classification(
        df,
        class_col=class_col,
        id_col=ID_COL,
        time_col=TIME_COL,
        positive=positive,
        negative=negative,
        features=FEATURES,
        bin_width=BIN_WIDTH,
        n_splits=N_SPLITS,
        n_permutations=N_PERM,
        n_jobs=-1,
        min_samples_per_group=min_samples_per_group,
        save_dir=str(CLASS_DIR / save_name),
        overwrite=True,
        verbose=False,
    )

    plot_aurocs_over_time(
        result.scores,
        curve_col="feature_set",
        show_null_band=True,
        show_significance=True,
        sig_threshold=0.05,
        show_chance_line=True,
        title=title,
        y_label="cross-validated AUROC",
        backend="matplotlib",
        output_path=str(PLOT_DIR / f"{save_name}_auroc.png"),
    )
    print(f"  saved plots/separability/{save_name}_auroc.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("3f - experiment batch controls for early CEP290 AUROC")
print(f"Early experiments: {EARLY_EXPERIMENTS}")
print(f"Features: {list(FEATURES.keys())} ({FEAT_NOTE})")
print(f"All outputs -> {PLOT_DIR.relative_to(RUN_DIR)}/")

# -- Class balance summary --------------------------------------------------
print("\n[0] Class balance summary")
plot_class_balance()

# -- A: WT experiment separability ------------------------------------------
print("\n[A] WT: experiment_id separability (20251212 vs 20250512)")
_run_and_plot(
    load_exp_ctrl("wildtype"),
    class_col="experiment_id",
    positive="20251212",
    negative="20250512",
    save_name="cep290_wt_expctrl",
    title=(
        "cep290 WT — experiment separability\n"
        "20251212 vs 20250512  |  2 hpf bins, leave-one-out CV, 500 perm\n"
        f"{FEAT_NOTE}"
    ),
)

# -- B: HET experiment separability -----------------------------------------
print("\n[B] HET: experiment_id separability (20251212 vs 20250512)")
_run_and_plot(
    load_exp_ctrl("heterozygous"),
    class_col="experiment_id",
    positive="20251212",
    negative="20250512",
    save_name="cep290_het_expctrl",
    title=(
        "cep290 HET — experiment separability\n"
        "20251212 vs 20250512  |  2 hpf bins, leave-one-out CV, 500 perm\n"
        f"{FEAT_NOTE}"
    ),
)

# -- C/D: within-experiment phenotype separability --------------------------
# 20251212 has only 2 High_to_Low homos -> relax min_samples_per_group to 2
for exp in EARLY_EXPERIMENTS:
    print(f"\n[C/D] Homo within {exp}: Low_to_High vs High_to_Low")
    _run_and_plot(
        load_homo_single_exp(exp),
        class_col="phenotype_label",
        positive="Low_to_High",
        negative="High_to_Low",
        save_name=f"cep290_homo_{exp}",
        title=(
            f"cep290 homo ({exp} only) — phenotype separability\n"
            "Low→High vs High→Low  |  2 hpf bins, leave-one-out CV, 500 perm\n"
            f"{FEAT_NOTE}"
        ),
        min_samples_per_group=2,
    )

# -- E/F: homo vs WT within each early experiment --------------------------
# Transitivity check: if homo is not distinguishable from WT within an experiment,
# there cannot be genuine phenotype signal driving cross-experiment separability.
for exp in EARLY_EXPERIMENTS:
    print(f"\n[E/F] Homo vs WT within {exp}")
    _run_and_plot(
        load_homo_vs_wt_single_exp(exp),
        class_col="zygosity",
        positive="homozygous",
        negative="wildtype",
        save_name=f"cep290_homo_vs_wt_{exp}",
        title=(
            f"cep290 homo vs WT ({exp} only) — within-experiment zygosity separability\n"
            "homozygous vs wildtype  |  2 hpf bins, leave-one-out CV, 500 perm\n"
            f"{FEAT_NOTE}"
        ),
    )

print(f"\nAll plots written to: plots/separability/")
print(f"All classification runs written to: classification/")
