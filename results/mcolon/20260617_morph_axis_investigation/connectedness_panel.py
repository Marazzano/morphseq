"""
connectedness_panel.py  -- reusable connectedness-panel utility
================================================================
One place to build a "connectedness panel" for a gene: per hpf bin, a scatter of
the phenotype-labeled mutant population vs. a stage-matched wildtype null, with a
per-metric DISCR/CONT vote table beneath each panel (valley / MST / Fiedler /
conductance, each voting by whether it exceeds the WT null at its own threshold).

The 2-D AXIS is pluggable via an `axis_provider`: a callable
    axis_provider(group_df, wt_df) -> (group_xy: (n,2), wt_xy: (m,2), axis_label: str)
so the SAME panel machinery serves the hand-picked morphometric axis and a PCA of
the raw z_mu_b embeddings (or anything else) without re-implementing the panel.

Callers: `morph_axis_connectedness.py` (hand axis) and
`embedding_pca_connectedness.py` (PC1/PC2 of z_mu_b).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

GENE14_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
REF_DIR = GENE14_DIR / "tables"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(GENE14_DIR))
from plot_config import GENOTYPE_COLORS, PHENOTYPE_COLORS  # noqa: E402

from support_geometry import compute_support_geometry, normalize_shape  # noqa: E402

# ── shared config ────────────────────────────────────────────────────────────
GENES = {
    "cep290": {"csv": REF_DIR / "reference_cep290_clean.csv",
               "phenotype_labels": ["High_to_Low", "Low_to_High"]},
    "b9d2":   {"csv": REF_DIR / "reference_b9d2_clean.csv",
               "phenotype_labels": ["CE", "HTA"]},
}
TIME_COL, BIN_WIDTH = "predicted_stage_hpf", 4.0
TARGET_DESIGN_HPF = [14, 18, 24, 30, 48]
_bin_center = lambda h: float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)
BIN_CENTER_TO_DESIGN_HPF = {_bin_center(h): h for h in TARGET_DESIGN_HPF}
MIN_EMBRYOS = 10
N_RESAMPLE = 80

WT_COLOR = GENOTYPE_COLORS["wildtype"]
UNLABELED_COLOR = "#BBBBBB"
SIGNIFICANT_COLOR = "#B2182B"
CONTINUOUS_COLOR = "#2166AC"

# ── per-metric DISCRETE vote thresholds (each metric oriented larger=more broken;
#    votes DISCRETE iff observed >> WT null at one-sided p < threshold). Tunable. ──
DISCRETE_P_THRESHOLDS = {
    "valley_depth": 0.05, "mst_max_edge": 0.05, "fiedler": 0.05, "conductance": 0.05,
}
_STAT_ROWS = [("valley_depth", "valley"), ("mst_max_edge", "MST"),
              ("fiedler", "Fiedler"), ("conductance", "conduct")]
_FIRED_COLOR = "#B2182B"
_QUIET_COLOR = "#7f7f7f"


# Metrics DEMOTED from the vote tally (still shown in the table for transparency, but
# they do NOT count toward the discrete/continuous decision). MST is excluded because
# it is fragile at low n -- a single outlier forces one long spanning-tree edge, so it
# fires spuriously on stragglers rather than real modes (verified: b9d2 14/18 hpf).
TALLY_EXCLUDED = {"mst_max_edge"}


def _votes_discrete(name: str, pvalue: float) -> bool:
    return pvalue < DISCRETE_P_THRESHOLDS.get(name, 0.05)


def _tally_call(bundle) -> str:
    """Majority vote over the TALLY metrics (MST excluded -- see TALLY_EXCLUDED)."""
    votes = [_votes_discrete(n, sr.pvalue)
             for n, sr in bundle.results.items()
             if sr is not None and n not in TALLY_EXCLUDED]
    if not votes:
        return "n/a"
    return "discrete" if sum(votes) > len(votes) / 2 else "continuous"


# ── data loading (phenotype-labeled pool + WT, per hpf bin) ──────────────────

def load_gene_bins(cfg: dict, extra_cols: list[str] | None = None) -> dict[int, dict]:
    """Return {design_hpf: {"group": df, "wt": df}} for the phenotype-labeled mutant
    pool (all zygosities) and the stage-matched wildtype pool. Embryo-collapsed.

    `extra_cols` are additional per-embryo columns to carry through (e.g. z_mu_b_*
    for a PCA axis); they are mean-aggregated.
    """
    df = pd.read_csv(cfg["csv"], low_memory=False)
    feat_hand = ["total_length_um", "baseline_deviation_normalized"]
    extra_cols = extra_cols or []
    # always carry any z_mu_b embedding columns through, so a PCA axis provider works
    zmub = [c for c in df.columns if c.startswith("z_mu_b")]
    needed = [TIME_COL, "phenotype_clean", "zygosity"] + feat_hand + extra_cols + zmub
    needed = [c for c in dict.fromkeys(needed) if c in df.columns]
    df = df.dropna(subset=needed).copy()
    df["time_bin_center"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2

    embryo_col = "embryo_id" if df["physical_embryo_id"].isna().all() else "physical_embryo_id"
    num_cols = [c for c in needed if c not in (TIME_COL, "phenotype_clean", "zygosity")]
    agg = {c: "mean" for c in num_cols}
    agg["phenotype_clean"] = lambda x: x.mode().iloc[0]
    agg["zygosity"] = lambda x: x.mode().iloc[0]
    edf = df.groupby([embryo_col, "time_bin_center"]).agg(agg).reset_index()

    out = {}
    for bc, hpf in BIN_CENTER_TO_DESIGN_HPF.items():
        sub = edf[edf["time_bin_center"] == bc]
        wt = sub[sub["zygosity"] == "wildtype"]
        grp = sub[sub["phenotype_clean"].isin(cfg["phenotype_labels"])]
        if len(wt) < MIN_EMBRYOS or len(grp) < MIN_EMBRYOS:
            continue
        out[hpf] = {"group": grp, "wt": wt}
    return out


# ── axis providers ───────────────────────────────────────────────────────────

def hand_axis_provider(group_df, wt_df):
    """The hand-picked morphometric axis: total_length_um x baseline_deviation_normalized."""
    cols = ["total_length_um", "baseline_deviation_normalized"]
    return (group_df[cols].values.astype(float),
            wt_df[cols].values.astype(float),
            "total_length_um x baseline_deviation_normalized")


def make_embedding_pca_provider(n_z: int = 80):
    """Axis = PC1/PC2 of the raw z_mu_b embeddings, FIT ON THE MUTANT GROUP, with WT
    projected into the same mutant-defined frame (so the null lives in the coordinates
    where a mutant CE/HTA split would appear)."""
    def provider(group_df, wt_df):
        zcols = [c for c in group_df.columns if c.startswith("z_mu_b")]
        if len(zcols) < 2:
            raise ValueError("z_mu_b columns not found for embedding PCA axis")
        G = group_df[zcols].values.astype(float)
        W = wt_df[zcols].values.astype(float)
        mean = G.mean(axis=0)
        # PCA on the mutant group via SVD
        _, _, vt = np.linalg.svd(G - mean, full_matrices=False)
        comps = vt[:2]  # PC1, PC2
        return ((G - mean) @ comps.T, (W - mean) @ comps.T,
                f"z_mu_b PC1 x PC2 (fit on mutant, {len(zcols)}-dim)")
    return provider


# ── the panel builder ────────────────────────────────────────────────────────

def build_panel(
    gene: str,
    cfg: dict,
    axis_provider: Callable,
    out_name: str,
    title_extra: str = "",
    extra_cols: list[str] | None = None,
    rng_seed: int = 42,
) -> list[tuple]:
    """Compute + render one connectedness panel for `gene` using `axis_provider`.

    Returns the per-bin entries [(hpf, bundle, group_df, wt_xy, group_xy), ...] so a
    caller can build trajectory plots without recomputing.
    """
    bins = load_gene_bins(cfg, extra_cols=extra_cols)
    entries = []
    axis_label = ""
    print(f"\n=== {gene} :: {out_name} ===")
    for hpf in TARGET_DESIGN_HPF:
        if hpf not in bins:
            continue
        gdf, wdf = bins[hpf]["group"], bins[hpf]["wt"]
        group_xy, wt_xy, axis_label = axis_provider(gdf, wdf)
        bundle = compute_support_geometry(group_xy, wt_xy, n_resample=N_RESAMPLE,
                                          rng=np.random.default_rng(rng_seed))
        call = _tally_call(bundle)
        stat_str = "  ".join(f"{n.split('_')[0]}:p={bundle.results[n].pvalue:.2f}"
                             for n in bundle.results)
        print(f"  {hpf:>2} hpf n={len(group_xy):>3} -> {call:<10s} [{stat_str}]")
        entries.append((hpf, bundle, gdf, wt_xy, group_xy))

    _render_panel(gene, cfg, entries, axis_label, out_name, title_extra)
    return entries


def _render_panel(gene, cfg, entries, axis_label, out_name, title_extra):
    n_cols = len(TARGET_DESIGN_HPF)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.7 * n_cols, 5.2), squeeze=False)
    axes = axes[0]
    by_hpf = {e[0]: e for e in entries}

    for col, hpf in enumerate(TARGET_DESIGN_HPF):
        ax = axes[col]
        entry = by_hpf.get(hpf)
        if entry is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#999")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            continue
        _, bundle, gdf, wt_xy, group_xy = entry
        g_norm = normalize_shape(group_xy)
        w_norm = normalize_shape(wt_xy)

        ax.scatter(w_norm[:, 0], w_norm[:, 1], s=14, alpha=0.3, facecolors="none",
                   edgecolors=WT_COLOR, linewidths=0.7, zorder=1)
        phenos = gdf["phenotype_clean"].values
        for pheno in cfg["phenotype_labels"] + ["unlabeled"]:
            m = phenos == pheno
            if not m.any():
                continue
            ax.scatter(g_norm[m, 0], g_norm[m, 1], s=28, alpha=0.85,
                       facecolors=PHENOTYPE_COLORS.get(pheno, UNLABELED_COLOR),
                       edgecolors="k", linewidths=0.3, label=pheno, zorder=2)

        is_disc = _tally_call(bundle) == "discrete"
        bc = SIGNIFICANT_COLOR if is_disc else CONTINUOUS_COLOR
        ax.text(0.03, 0.96, "DISCRETE" if is_disc else "continuous", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=bc, edgecolor="none", alpha=0.9))
        ax.text(0.97, 0.04, f"n={len(group_xy)}", transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right", color="#333")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ccc")
        ax.set_title(f"{hpf} hpf", fontsize=10, fontweight="bold")
        _render_vote_table(ax, bundle)

    handles = [mpatches.Patch(facecolor="none", edgecolor=WT_COLOR, label="wildtype (null)")] + [
        mpatches.Patch(facecolor=PHENOTYPE_COLORS.get(p, UNLABELED_COLOR), edgecolor="k", label=p)
        for p in cfg["phenotype_labels"] + ["unlabeled"]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.suptitle(f"{gene} — connectedness  |  axis: {axis_label}{title_extra}\n"
                 f"per-metric DISCR/CONT vote (metric >> WT null?) beneath each panel; "
                 f"phenotype color = emergent split, not the tested grouping",
                 fontsize=11, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.40, top=0.86, left=0.03, right=0.99, wspace=0.12)
    out = PLOT_DIR / out_name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.name}")


def _render_vote_table(ax, bundle) -> None:
    x_label, x_discr, x_cont, x_p = 0.02, 0.45, 0.62, 0.78
    y, dy = -0.05, 0.082
    ax.text(x_label, y, "metric", transform=ax.transAxes, fontsize=6.4, va="top",
            ha="left", color="#333", fontstyle="italic", fontfamily="monospace")
    ax.text(x_discr, y, "DISCR", transform=ax.transAxes, fontsize=6.4, va="top",
            ha="center", color=_FIRED_COLOR, fontweight="bold", fontfamily="monospace")
    ax.text(x_cont, y, "CONT", transform=ax.transAxes, fontsize=6.4, va="top",
            ha="center", color=CONTINUOUS_COLOR, fontweight="bold", fontfamily="monospace")
    ax.text(x_p, y, "p vs WT", transform=ax.transAxes, fontsize=6.4, va="top",
            ha="left", color="#333", fontstyle="italic", fontfamily="monospace")
    for i, (name, label) in enumerate(_STAT_ROWS):
        yy = y - (i + 1) * dy
        sr = bundle.results.get(name)
        ax.text(x_label, yy, label, transform=ax.transAxes, fontsize=6.4, va="top",
                ha="left", color="#222", fontfamily="monospace")
        if sr is None:
            ax.text(x_p, yy, "n/a", transform=ax.transAxes, fontsize=6.4, va="top",
                    ha="left", color="#bbb", fontfamily="monospace")
            continue
        excluded = name in TALLY_EXCLUDED
        disc = _votes_discrete(name, sr.pvalue)
        if excluded:
            # shown for transparency but greyed; does not vote
            ax.text(x_label, yy, label, transform=ax.transAxes, fontsize=6.4, va="top",
                    ha="left", color="#aaa", fontfamily="monospace")  # re-draw label greyed
            ax.text((x_discr + x_cont) / 2, yy, "(excluded)", transform=ax.transAxes,
                    fontsize=6.0, va="top", ha="center", color="#aaa",
                    fontfamily="monospace", fontstyle="italic")
            ax.text(x_p, yy, f"p={sr.pvalue:.2f}", transform=ax.transAxes, fontsize=6.4,
                    va="top", ha="left", color="#aaa", fontfamily="monospace")
            continue
        ax.text(x_discr, yy, "●" if disc else "", transform=ax.transAxes, fontsize=8,
                va="top", ha="center", color=_FIRED_COLOR)
        ax.text(x_cont, yy, "" if disc else "●", transform=ax.transAxes, fontsize=8,
                va="top", ha="center", color=CONTINUOUS_COLOR)
        thr = DISCRETE_P_THRESHOLDS.get(name, 0.05)
        ax.text(x_p, yy, f"p={sr.pvalue:.2f} (<{thr:g}?)", transform=ax.transAxes,
                fontsize=6.4, va="top", ha="left",
                color=_FIRED_COLOR if disc else _QUIET_COLOR, fontfamily="monospace",
                fontweight="bold" if disc else "normal")
    call = _tally_call(bundle)
    nd = sum(_votes_discrete(n, sr.pvalue) for n, sr in bundle.results.items()
             if sr is not None and n not in TALLY_EXCLUDED)
    nt = sum(1 for n, sr in bundle.results.items()
             if sr is not None and n not in TALLY_EXCLUDED)
    ax.text(x_label, y - (len(_STAT_ROWS) + 1.1) * dy,
            f"tally: {nd}/{nt} discrete  =>  {call.upper()}", transform=ax.transAxes,
            fontsize=6.6, va="top", ha="left",
            color=_FIRED_COLOR if call == "discrete" else CONTINUOUS_COLOR,
            fontweight="bold", fontfamily="monospace")
