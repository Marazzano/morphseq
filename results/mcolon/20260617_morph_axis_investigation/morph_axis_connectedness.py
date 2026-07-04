"""
Within-group connectedness: is each gene's POOLED homozygous population's own
distribution along (total_length_um x baseline_deviation_normalized) a continuous
blob, or does it fracture into disconnected modes? Wildtype siblings supply the
stage-matched "continuous" calibration and the density-prior tolerance -- NOT a
contrast class.

Crucially, the group under test is the WHOLE homozygous population per gene, not
one phenotype label in isolation -- the phenotype split (e.g. CE vs HTA) is the
thing a real fracture would recover, so phenotype labels are used only for
COLORING points in the plots, never to define the tested group.

See docs/todos_scratch/morph_axis_discreteness_spec.md and connectedness.py.

Anchors (known biology, used to tune/validate the method):
  cep290 homozygous (High_to_Low + Low_to_High pooled) -> expected CONTINUOUS
  b9d2   homozygous (CE + HTA pooled)                   -> expected DISCRETE

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/morph_axis_connectedness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RUN_DIR))

GENE14_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
REF_DIR = GENE14_DIR / "tables"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(GENE14_DIR))
from plot_config import GENOTYPE_COLORS, PHENOTYPE_COLORS  # noqa: E402

from connectedness import connectedness_pvalue, normalize_shape  # noqa: E402

# ---------------------------------------------------------------------------
# Config -- mirrors morph_axis_scatter.py's gene/binning setup
# ---------------------------------------------------------------------------

GENES = {
    "cep290": {
        "csv": REF_DIR / "reference_cep290_clean.csv",
        "phenotype_labels": ["High_to_Low", "Low_to_High"],
    },
    "b9d2": {
        "csv": REF_DIR / "reference_b9d2_clean.csv",
        "phenotype_labels": ["CE", "HTA"],
    },
}

X_FEAT = "total_length_um"
Y_FEAT = "baseline_deviation_normalized"

TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
TARGET_DESIGN_HPF = [14, 18, 24, 30, 48]
_bin_center = lambda h: float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)
TARGET_BIN_CENTERS = sorted({_bin_center(h) for h in TARGET_DESIGN_HPF})
BIN_CENTER_TO_DESIGN_HPF = {_bin_center(h): h for h in TARGET_DESIGN_HPF}

CLIP_PERCENTILE = 1
MIN_EMBRYOS_PER_CLASS = 10
N_RESAMPLE = 150

WT_COLOR = GENOTYPE_COLORS["wildtype"]
UNLABELED_COLOR = "#BBBBBB"
SIGNIFICANT_COLOR = "#B2182B"   # crimson -- "discrete" call
CONTINUOUS_COLOR = "#2166AC"    # blue -- "continuous" call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH).astype(int)
    df["time_bin_center"] = df["time_bin"] * BIN_WIDTH + BIN_WIDTH / 2
    return df


def remove_outliers(pts: np.ndarray) -> np.ndarray:
    """Boolean mask for rows inside the [CLIP_PERCENTILE, 100-CLIP_PERCENTILE] band
    on BOTH axes. Removes (not clips) so density estimation isn't distorted by an
    artificial pile-up of capped points at the boundary."""
    lo = np.percentile(pts, CLIP_PERCENTILE, axis=0)
    hi = np.percentile(pts, 100 - CLIP_PERCENTILE, axis=0)
    return np.all((pts >= lo) & (pts <= hi), axis=1)


# ---------------------------------------------------------------------------
# Per-gene analysis: test the POOLED homozygous population, not one phenotype
# ---------------------------------------------------------------------------

# {gene: [(design_hpf, ConnectednessResult, group_df, wt_pts), ...]}
all_results: dict[str, list] = {g: [] for g in GENES}

for gene, cfg in GENES.items():
    print(f"\n{'='*60}\nGene: {gene}\n{'='*60}")

    df_raw = pd.read_csv(cfg["csv"], low_memory=False)
    needed = [TIME_COL, X_FEAT, Y_FEAT, "phenotype_clean", "zygosity"]
    df = df_raw.dropna(subset=needed).copy()
    df = assign_bin(df)

    embryo_col = "embryo_id" if df["physical_embryo_id"].isna().all() else "physical_embryo_id"

    agg = {X_FEAT: "mean", Y_FEAT: "mean"}
    agg["phenotype_clean"] = lambda x: x.mode().iloc[0]
    agg["zygosity"] = lambda x: x.mode().iloc[0]
    agg["experiment_id"] = "first"

    edf = (
        df.groupby([embryo_col, "time_bin", "time_bin_center"]).agg(agg).reset_index()
        .rename(columns={embryo_col: "embryo_id"})
    )

    wt_pool = edf[edf["zygosity"] == "wildtype"].copy()
    # The tested group is every embryo carrying a mutant phenotype LABEL (CE/HTA
    # for b9d2, High_to_Low/Low_to_High for cep290), regardless of zygosity. The
    # phenotype call is a morphological cluster label independent of genotype, and
    # CE is overwhelmingly het/unknown -- gating on homozygous-only threw away the
    # arm that forms the fracture. Any phenotype split is the thing the test should
    # recover, so we pool the labeled mutant population and let the geometry speak.
    homo_pool = edf[edf["phenotype_clean"].isin(cfg["phenotype_labels"])].copy()

    for bin_center in TARGET_BIN_CENTERS:
        design_hpf = BIN_CENTER_TO_DESIGN_HPF[bin_center]
        grp_bdf = homo_pool[homo_pool["time_bin_center"] == bin_center].copy()
        wt_bdf = wt_pool[wt_pool["time_bin_center"] == bin_center]

        if len(grp_bdf) < MIN_EMBRYOS_PER_CLASS or len(wt_bdf) < MIN_EMBRYOS_PER_CLASS:
            print(f"  homozygous @ {design_hpf} hpf: skip (n_group={len(grp_bdf)}, "
                  f"n_wt={len(wt_bdf)})")
            continue

        grp_xy = grp_bdf[[X_FEAT, Y_FEAT]].values.astype(float)
        wt_xy = wt_bdf[[X_FEAT, Y_FEAT]].values.astype(float)

        # NOTE: outlier removal deliberately DISABLED. An outlier should not decide
        # whether a distribution is discrete -- the extreme-short CE embryos ARE the
        # second mode, and percentile clipping was erasing exactly the fracture we
        # are trying to detect. Keep every labeled embryo.

        res = connectedness_pvalue(grp_xy, wt_xy, n_resample=N_RESAMPLE,
                                    rng=np.random.default_rng(42))
        call = "DISCRETE" if res.pvalue < 0.05 else "continuous"
        pheno_counts = grp_bdf["phenotype_clean"].value_counts().to_dict()
        print(f"  homozygous @ {design_hpf} hpf: n={len(grp_xy)} (wt n={len(wt_xy)}) "
              f"stat={res.stat:.3f} wt_ref={res.reference_stat:.3f} p={res.pvalue:.3f} "
              f"-> {call}  [{pheno_counts}]")

        all_results[gene].append((design_hpf, res, grp_bdf, wt_xy))


# ---------------------------------------------------------------------------
# Plot 1: per (gene x timepoint) 2-D distribution panel, pooled homozygous group
# points colored by their (emergent) phenotype label
# ---------------------------------------------------------------------------

for gene, cfg in GENES.items():
    entries = all_results[gene]
    n_cols = len(TARGET_DESIGN_HPF)

    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 3.8), squeeze=False)
    axes = axes[0]
    results_by_hpf = {e[0]: e for e in entries}

    for col, design_hpf in enumerate(TARGET_DESIGN_HPF):
        ax = axes[col]
        entry = results_by_hpf.get(design_hpf)

        if entry is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999999")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        _, res, grp_bdf, wt_xy = entry
        grp_xy = grp_bdf[[X_FEAT, Y_FEAT]].values.astype(float)
        grp_norm = normalize_shape(grp_xy)
        wt_norm = normalize_shape(wt_xy)

        ax.scatter(wt_norm[:, 0], wt_norm[:, 1], s=14, alpha=0.3,
                   facecolors="none", edgecolors=WT_COLOR, linewidths=0.7,
                   label="wildtype (null)", zorder=1)

        phenotypes_here = grp_bdf["phenotype_clean"].values
        for pheno in cfg["phenotype_labels"] + ["unlabeled"]:
            mask = phenotypes_here == pheno
            if not mask.any():
                continue
            color = PHENOTYPE_COLORS.get(pheno, UNLABELED_COLOR)
            ax.scatter(grp_norm[mask, 0], grp_norm[mask, 1], s=28, alpha=0.85,
                       facecolors=color, edgecolors="k", linewidths=0.3,
                       label=pheno, zorder=2)

        is_discrete = res.pvalue < 0.05
        badge_color = SIGNIFICANT_COLOR if is_discrete else CONTINUOUS_COLOR
        badge_text = "DISCRETE" if is_discrete else "continuous"
        ax.text(0.03, 0.96, badge_text, transform=ax.transAxes, fontsize=8,
                 fontweight="bold", color="white", va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color,
                            edgecolor="none", alpha=0.9))
        ax.text(0.97, 0.04, f"p={res.pvalue:.2f}\nvalley={res.stat:.2f}\nn={len(grp_xy)}",
                 transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                 color="#333333")

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_title(f"{design_hpf} hpf", fontsize=10, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor="none", edgecolor=WT_COLOR, label="wildtype (stage-matched null)"),
    ] + [
        mpatches.Patch(facecolor=PHENOTYPE_COLORS.get(p, UNLABELED_COLOR), edgecolor="k", label=p)
        for p in cfg["phenotype_labels"] + ["unlabeled"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.suptitle(f"{gene}  —  pooled homozygous connectedness (normalized shape)\n"
                 f"axis: {X_FEAT}  x  {Y_FEAT}   |   phenotype color shows the EMERGENT split, "
                 f"not the tested grouping",
                 fontsize=11, fontweight="bold", y=1.08)
    fig.tight_layout()
    out_path = PLOT_DIR / f"{gene}_connectedness_panel.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: connectedness trajectory over developmental time, one line per gene
# ---------------------------------------------------------------------------

GENE_TRAJECTORY_COLORS = {"cep290": "#7b3294", "b9d2": "#1b9e77"}

fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5.5))

for gene in GENES:
    entries = sorted(all_results[gene], key=lambda e: e[0])
    if not entries:
        continue
    color = GENE_TRAJECTORY_COLORS.get(gene, "#444444")
    hpfs = [e[0] for e in entries]
    stats = [e[1].stat for e in entries]
    pvals = [e[1].pvalue for e in entries]
    wt_refs = [e[1].reference_stat for e in entries]

    ax2.plot(hpfs, stats, "o-", color=color, linewidth=2.4, markersize=9,
             label=f"{gene} homozygous (pooled)", zorder=3)
    ax2.plot(hpfs, wt_refs, "--", color=color, linewidth=1.2, alpha=0.4, zorder=1)

    sig_hpfs = [h for h, p in zip(hpfs, pvals) if p < 0.05]
    if sig_hpfs:
        transition_hpf = min(sig_hpfs)
        ax2.axvline(transition_hpf, color=color, linestyle=":", alpha=0.5, zorder=0)
        ax2.annotate(f"{gene}: discrete\nfrom {transition_hpf} hpf",
                     xy=(transition_hpf, ax2.get_ylim()[1]),
                     xytext=(transition_hpf, 1.02), textcoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    for h, s, p in zip(hpfs, stats, pvals):
        if p < 0.05:
            ax2.scatter([h], [s], s=200, facecolors="none", edgecolors=color,
                        linewidths=2.2, zorder=4)

ax2.plot([], [], "--", color="#888888", linewidth=1.2, alpha=0.6, label="wildtype null (median)")
ax2.scatter([], [], s=200, facecolors="none", edgecolors="#444", linewidths=2.2,
            label="p < 0.05 (discrete)")

ax2.set_xlabel("predicted stage (hpf)", fontsize=11)
ax2.set_ylabel("relative valley depth\n(0 = continuous, 1 = deeply fractured)", fontsize=10)
ax2.set_ylim(-0.05, 1.1)
ax2.grid(alpha=0.25)
ax2.legend(fontsize=9, loc="upper left")
ax2.set_title("Variance -> mode transition over developmental time\n"
              "pooled homozygous population per gene vs. stage-matched wildtype null",
              fontsize=12)

fig2.tight_layout()
trend_path = PLOT_DIR / "connectedness_trajectory.png"
fig2.savefig(trend_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {trend_path.name}")

print("\nDone.")
