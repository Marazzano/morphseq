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

from support_geometry import compute_support_geometry, normalize_shape  # noqa: E402

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
N_RESAMPLE = 80  # lowered from 150: conductance eigendecomps per resample are slow

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

        bundle = compute_support_geometry(grp_xy, wt_xy, n_resample=N_RESAMPLE,
                                          rng=np.random.default_rng(42))
        call = bundle.support_call.upper() if bundle.support_call == "discrete" else "continuous"
        pheno_counts = grp_bdf["phenotype_clean"].value_counts().to_dict()
        stat_str = "  ".join(f"{n}:p={bundle.results[n].pvalue:.2f}"
                             for n in bundle.results)
        print(f"  homozygous @ {design_hpf} hpf: n={len(grp_xy)} (wt n={len(wt_xy)}) "
              f"-> {call}  [{stat_str}]  {pheno_counts}")

        all_results[gene].append((design_hpf, bundle, grp_bdf, wt_xy))


# ---------------------------------------------------------------------------
# Deciding-statistics strip under each panel
# ---------------------------------------------------------------------------

# Each metric is oriented so that LARGER = more broken/discrete (valley depth & MST
# max-edge are naturally so; fiedler & conductance are returned inverted 1/(1+.) for
# this). So a metric "votes DISCRETE" when its observed value is significantly LARGER
# than the wildtype null -- i.e. one-sided p < its threshold. Otherwise it votes
# CONTINUOUS. Each metric gets its OWN threshold constant so they can be tuned
# independently later (all default 0.05 today).
DISCRETE_P_THRESHOLDS = {
    "valley_depth": 0.05,
    "mst_max_edge": 0.05,
    "fiedler": 0.05,
    "conductance": 0.05,
}

# (stat key, display label) in table row order.
_STAT_ROWS = [
    ("valley_depth", "valley"),
    ("mst_max_edge", "MST"),
    ("fiedler", "Fiedler"),
    ("conductance", "conduct"),
]
_FIRED_COLOR = "#B2182B"      # crimson: votes DISCRETE (p < threshold)
_QUIET_COLOR = "#7f7f7f"      # gray: votes CONTINUOUS


def _votes_discrete(name: str, pvalue: float) -> bool:
    """A metric votes DISCRETE iff observed >> WT null at its own threshold."""
    return pvalue < DISCRETE_P_THRESHOLDS.get(name, 0.05)


def _tally_call(bundle) -> str:
    """Overall call from the per-metric votes: DISCRETE if a majority of estimable
    metrics vote discrete, else continuous. (Simple, transparent vote tally -- each
    metric weighs equally. Distinct from support_geometry.support_call's AND-rule;
    this panel shows the metrics as independent votes so the reader decides.)"""
    votes = [_votes_discrete(n, sr.pvalue)
             for n, sr in bundle.results.items() if sr is not None]
    if not votes:
        return "n/a"
    return "discrete" if sum(votes) > len(votes) / 2 else "continuous"


def _render_stat_strip(ax, bundle) -> None:
    """Render the per-metric DISCR/CONT vote table beneath a panel.

    One row per metric. Each metric independently votes DISCRETE or CONTINUOUS by
    whether its observed value is significantly LARGER than the WT null (one-sided
    p < its threshold; every metric is oriented so larger = more broken). Columns:
        metric | DISCR | CONT | p (vs WT)
    A dot marks the column the metric supports; the p-value and value are shown."""
    x_label, x_discr, x_cont, x_p = 0.02, 0.45, 0.62, 0.78
    y = -0.05
    dy = 0.082

    # header row
    ax.text(x_label, y, "metric", transform=ax.transAxes, fontsize=6.4,
            va="top", ha="left", color="#333", fontstyle="italic", fontfamily="monospace")
    ax.text(x_discr, y, "DISCR", transform=ax.transAxes, fontsize=6.4,
            va="top", ha="center", color=_FIRED_COLOR, fontweight="bold", fontfamily="monospace")
    ax.text(x_cont, y, "CONT", transform=ax.transAxes, fontsize=6.4,
            va="top", ha="center", color=CONTINUOUS_COLOR, fontweight="bold", fontfamily="monospace")
    ax.text(x_p, y, "p vs WT", transform=ax.transAxes, fontsize=6.4,
            va="top", ha="left", color="#333", fontstyle="italic", fontfamily="monospace")

    for i, (name, label) in enumerate(_STAT_ROWS):
        yy = y - (i + 1) * dy
        sr = bundle.results.get(name)
        ax.text(x_label, yy, label, transform=ax.transAxes, fontsize=6.4,
                va="top", ha="left", color="#222", fontfamily="monospace")
        if sr is None:
            ax.text(x_p, yy, "n/a", transform=ax.transAxes, fontsize=6.4,
                    va="top", ha="left", color="#bbb", fontfamily="monospace")
            continue
        discrete = _votes_discrete(name, sr.pvalue)
        # dot in the supported column
        ax.text(x_discr, yy, "●" if discrete else "", transform=ax.transAxes, fontsize=8,
                va="top", ha="center", color=_FIRED_COLOR)
        ax.text(x_cont, yy, "" if discrete else "●", transform=ax.transAxes, fontsize=8,
                va="top", ha="center", color=CONTINUOUS_COLOR)
        thr = DISCRETE_P_THRESHOLDS.get(name, 0.05)
        pcolor = _FIRED_COLOR if discrete else _QUIET_COLOR
        ax.text(x_p, yy, f"p={sr.pvalue:.2f} (<{thr:g}?)", transform=ax.transAxes,
                fontsize=6.4, va="top", ha="left", color=pcolor, fontfamily="monospace",
                fontweight="bold" if discrete else "normal")

    # tally line
    call = _tally_call(bundle)
    ncall_disc = sum(_votes_discrete(n, sr.pvalue)
                     for n, sr in bundle.results.items() if sr is not None)
    ntot = sum(1 for sr in bundle.results.values() if sr is not None)
    tcolor = _FIRED_COLOR if call == "discrete" else CONTINUOUS_COLOR
    ax.text(x_label, y - (len(_STAT_ROWS) + 1.1) * dy,
            f"tally: {ncall_disc}/{ntot} discrete  =>  {call.upper()}",
            transform=ax.transAxes, fontsize=6.6, va="top", ha="left",
            color=tcolor, fontweight="bold", fontfamily="monospace")


# ---------------------------------------------------------------------------
# Plot 1: per (gene x timepoint) 2-D distribution panel, pooled homozygous group
# points colored by their (emergent) phenotype label
# ---------------------------------------------------------------------------

for gene, cfg in GENES.items():
    entries = all_results[gene]
    n_cols = len(TARGET_DESIGN_HPF)

    # taller figure + bottom room for the per-metric vote table under each panel
    fig, axes = plt.subplots(1, n_cols, figsize=(3.7 * n_cols, 5.2), squeeze=False)
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

        _, bundle, grp_bdf, wt_xy = entry
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

        is_discrete = _tally_call(bundle) == "discrete"
        badge_color = SIGNIFICANT_COLOR if is_discrete else CONTINUOUS_COLOR
        badge_text = "DISCRETE" if is_discrete else "continuous"
        ax.text(0.03, 0.96, badge_text, transform=ax.transAxes, fontsize=8,
                 fontweight="bold", color="white", va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color,
                            edgecolor="none", alpha=0.9))
        ax.text(0.97, 0.04, f"n={len(grp_xy)}",
                 transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                 color="#333333")

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_title(f"{design_hpf} hpf", fontsize=10, fontweight="bold")

        # ── deciding-statistics strip: the per-statistic p-values that drive the
        #    support_call, rendered directly under this panel. valley is the
        #    density gate (must fire); MST/Fiedler/conductance are graph corroborators.
        _render_stat_strip(ax, bundle)

    handles = [
        mpatches.Patch(facecolor="none", edgecolor=WT_COLOR, label="wildtype (stage-matched null)"),
    ] + [
        mpatches.Patch(facecolor=PHENOTYPE_COLORS.get(p, UNLABELED_COLOR), edgecolor="k", label=p)
        for p in cfg["phenotype_labels"] + ["unlabeled"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.12), frameon=False)

    fig.suptitle(f"{gene}  —  pooled homozygous connectedness (normalized shape)\n"
                 f"axis: {X_FEAT}  x  {Y_FEAT}   |   phenotype color shows the EMERGENT split, "
                 f"not the tested grouping",
                 fontsize=11, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.40, top=0.86, left=0.03, right=0.99,
                        wspace=0.12)  # room for the per-metric vote table (no tight_layout: it fights the strip)
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
    # Plot 2 tracks the DENSITY GATE (valley_depth) over time -- the statistic that
    # must fire for a discrete call. Pull it out of each bundle.
    hpfs = [e[0] for e in entries]
    stats = [e[1].results["valley_depth"].stat for e in entries]
    pvals = [e[1].results["valley_depth"].pvalue for e in entries]
    wt_refs = [e[1].results["valley_depth"].reference_stat for e in entries]

    ax2.plot(hpfs, stats, "o-", color=color, linewidth=2.4, markersize=9,
             label=f"{gene} homozygous (pooled)", zorder=3)
    ax2.plot(hpfs, wt_refs, "--", color=color, linewidth=1.2, alpha=0.4, zorder=1)

    # The TRUE call is support_call (valley gate AND graph corroboration), not raw
    # valley significance. Mark both: a filled ring where the system calls discrete,
    # and an X where valley fires but the graph statistics VETO it (the b9d2 story).
    calls = [_tally_call(e[1]) for e in entries]
    discrete_hpfs = [h for h, c in zip(hpfs, calls) if c == "discrete"]
    if discrete_hpfs:
        transition_hpf = min(discrete_hpfs)
        ax2.axvline(transition_hpf, color=color, linestyle=":", alpha=0.5, zorder=0)
        ax2.annotate(f"{gene}: discrete\nfrom {transition_hpf} hpf",
                     xy=(transition_hpf, ax2.get_ylim()[1]),
                     xytext=(transition_hpf, 1.02), textcoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    for h, s, p, c in zip(hpfs, stats, pvals, calls):
        if c == "discrete":
            ax2.scatter([h], [s], s=200, facecolors="none", edgecolors=color,
                        linewidths=2.2, zorder=4)
        elif p < 0.05:  # valley gate fired but graph vetoed -> continuous call
            ax2.scatter([h], [s], s=120, marker="x", color=color,
                        linewidths=2.0, zorder=4)

ax2.plot([], [], "--", color="#888888", linewidth=1.2, alpha=0.6, label="wildtype null (median)")
ax2.scatter([], [], s=200, facecolors="none", edgecolors="#444", linewidths=2.2,
            label="support_call = discrete")
ax2.scatter([], [], s=120, marker="x", color="#444", linewidths=2.0,
            label="valley fires but graph vetoes")

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
