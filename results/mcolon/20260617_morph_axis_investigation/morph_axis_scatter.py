"""
Morphological axis investigation: how do model predictions stratify in the
(total_length_um × mean_curvature_per_um) scatter space?

Layout: 1 row per figure, 5 columns left→right:
  Col 0: true labels (ground truth reference)
  Col 1: z_mu_b embeddings (80-dim)  | LOEO P(class1) probability
  Col 2: length + curvature combined | LOEO P(class1) probability
  Col 3: length only                 | LOEO P(class1) probability
  Col 4: curvature only              | LOEO P(class1) probability

All probability columns share the same colormap (RdBu_r, 0→1).
True-label column uses PHENOTYPE_COLORS from plot_config.py.
One figure per (gene × time-bin).

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/morph_axis_scatter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GENE14_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
REF_DIR = GENE14_DIR / "tables"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(GENE14_DIR))
from plot_config import PHENOTYPE_COLORS  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GENES = {
    "cep290": {
        "csv": REF_DIR / "reference_cep290_clean.csv",
        "keep_labels": ["High_to_Low", "Low_to_High"],
        "class0": "High_to_Low",
        "class1": "Low_to_High",
    },
    "b9d2": {
        "csv": REF_DIR / "reference_b9d2_clean.csv",
        "keep_labels": ["CE", "HTA"],
        "class0": "CE",
        "class1": "HTA",
    },
}

# 5 columns: col 0 = true labels; cols 1-4 = classifiers
CLASSIFIERS = [
    {"name": "z_mu_b\n(80-dim)",        "feature_type": "embedding"},
    {"name": "length +\ncurvature",     "feature_type": "both"},
    {"name": "length\nonly",            "feature_type": "length"},
    {"name": "curvature\nonly",         "feature_type": "curvature"},
]

MORPH_FEATURES = {
    "both":      ["total_length_um", "mean_curvature_per_um"],
    "length":    ["total_length_um"],
    "curvature": ["mean_curvature_per_um"],
}

X_FEAT = "total_length_um"
Y_FEAT = "mean_curvature_per_um"

TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
# Design stages from plot_config; snap each to nearest 4-hpf bin center
# 14→14, 18→18, 24→26, 30→30, 48→46
TARGET_DESIGN_HPF = [14, 18, 24, 30, 48]
# Map each design stage to the bin center it falls in (floor(hpf/4)*4 + 2),
# and keep a reverse lookup so files/titles use the design stage label (e.g. 48 not 50)
_bin_center = lambda h: float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)
TARGET_BIN_CENTERS = sorted({_bin_center(h) for h in TARGET_DESIGN_HPF})
BIN_CENTER_TO_DESIGN_HPF = {_bin_center(h): h for h in TARGET_DESIGN_HPF}
# All bins are used for the trend plot (not just design stages)
ALL_BIN_CENTERS: list[float] = []  # populated during data loading

CLIP_PERCENTILE = 1  # clip both axes to [1st, 99th] percentile to remove outliers
MIN_EMBRYOS_PER_CLASS = 3

PROB_CMAP = None  # set per gene from phenotype colors (class0 → class1 gradient)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH).astype(int)
    df["time_bin_center"] = df["time_bin"] * BIN_WIDTH + BIN_WIDTH / 2
    return df


def make_pipe() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, solver="liblinear",
            class_weight="balanced", random_state=42,
        )),
    ])


def loeo_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """P(class_1) via LOEO (>=2 experiments) or 5-fold CV (single experiment)."""
    pipe = make_pipe()
    if len(np.unique(groups)) >= 2:
        proba = cross_val_predict(pipe, X, y, groups=groups,
                                  cv=LeaveOneGroupOut(), method="predict_proba")
    else:
        n_splits = min(5, int(np.bincount(y.astype(int)).min()))
        proba = cross_val_predict(pipe, X, y,
                                  cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                                  method="predict_proba")
    return proba[:, 1]


def emb_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("z_mu_b_")],
        key=lambda c: int(c.split("_")[-1]),
    )


# ---------------------------------------------------------------------------
# Per-gene main loop
# ---------------------------------------------------------------------------

# Collect bacc and scatter data for trend + big-panel plots
all_bacc: dict[str, dict[str, list]] = {g: {clf["feature_type"]: [] for clf in CLASSIFIERS}
                                         for g in GENES}
# {gene: [(design_hpf, x_vals, y_vals, true_labels, clf_results, c0, c1, gene_cmap, n_exp)]}
all_scatter: dict[str, list] = {g: [] for g in GENES}

for gene, cfg in GENES.items():
    print(f"\n{'='*60}\nGene: {gene}\n{'='*60}")

    df_raw = pd.read_csv(cfg["csv"], low_memory=False)

    df = df_raw[
        (df_raw["zygosity"] == "homozygous") &
        (df_raw["phenotype_clean"].isin(cfg["keep_labels"]))
    ].copy()

    needed = [TIME_COL, X_FEAT, Y_FEAT, "phenotype_clean"]
    df = df.dropna(subset=needed)

    ecols = emb_cols(df)
    df = df.dropna(subset=ecols)
    df = assign_bin(df)

    embryo_col = "embryo_id" if df["physical_embryo_id"].isna().all() else "physical_embryo_id"

    agg = {c: "mean" for c in ecols + [X_FEAT, Y_FEAT]}
    agg["phenotype_clean"] = lambda x: x.mode().iloc[0]
    agg["experiment_id"] = "first"
    agg["time_bin_center"] = "first"

    edf = (
        df.groupby([embryo_col, "time_bin"]).agg(agg).reset_index()
        .rename(columns={embryo_col: "embryo_id"})
    )

    class0, class1 = cfg["class0"], cfg["class1"]
    edf = edf[edf["phenotype_clean"].isin([class0, class1])].copy()
    edf["label_int"] = (edf["phenotype_clean"] == class1).astype(int)

    for bin_center in sorted(edf["time_bin_center"].unique()):
        bdf = edf[edf["time_bin_center"] == bin_center].copy()
        counts = bdf["phenotype_clean"].value_counts()
        if counts.min() < MIN_EMBRYOS_PER_CLASS:
            continue

        make_scatter = bin_center in TARGET_BIN_CENTERS
        design_hpf = BIN_CENTER_TO_DESIGN_HPF.get(bin_center, int(bin_center))
        n_emb = len(bdf)
        if make_scatter:
            print(f"  Scatter {design_hpf} hpf (bin {bin_center}) | n={n_emb} | {counts.to_dict()}")
        print(f"  Bin {bin_center} hpf (design {design_hpf} hpf) | n={n_emb} | {counts.to_dict()}")

        # Clip outliers to [CLIP_PERCENTILE, 100-CLIP_PERCENTILE] for display only
        x_raw = bdf[X_FEAT].values
        y_raw = bdf[Y_FEAT].values
        x_lo, x_hi = np.percentile(x_raw, [CLIP_PERCENTILE, 100 - CLIP_PERCENTILE])
        y_lo, y_hi = np.percentile(y_raw, [CLIP_PERCENTILE, 100 - CLIP_PERCENTILE])
        x_vals = np.clip(x_raw, x_lo, x_hi)
        y_vals = np.clip(y_raw, y_lo, y_hi)
        true_labels = bdf["phenotype_clean"].values
        y_int = bdf["label_int"].values
        groups = bdf["experiment_id"].astype(str).values

        feat_mats = {
            "embedding": bdf[ecols].values,
            "both":      bdf[MORPH_FEATURES["both"]].values,
            "length":    bdf[MORPH_FEATURES["length"]].values,
            "curvature": bdf[MORPH_FEATURES["curvature"]].values,
        }

        clf_results = {}
        for clf in CLASSIFIERS:
            ft = clf["feature_type"]
            try:
                p1 = loeo_proba(feat_mats[ft], y_int, groups)
                bacc = balanced_accuracy_score(y_int, (p1 >= 0.5).astype(int))
                clf_results[ft] = {"prob1": p1, "bacc": bacc}
                all_bacc[gene][ft].append((bin_center, bacc))  # use actual bin center for x-axis
                if make_scatter:
                    print(f"    {clf['name'].replace(chr(10), ' ')}: bacc={bacc:.3f}")
            except Exception as e:
                if make_scatter:
                    print(f"    {clf['name'].replace(chr(10), ' ')}: FAILED ({e})")
                clf_results[ft] = None

        if not make_scatter:
            continue

        # -------------------------------------------------------------------
        # Figure: 1 row × 5 columns
        # -------------------------------------------------------------------
        n_cols = 1 + len(CLASSIFIERS)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.5),
                                 constrained_layout=False)
        n_exp = bdf["experiment_id"].nunique()
        cv_label = f"leave-one-experiment-out CV  (n={n_exp} experiments)" if n_exp >= 2 else "5-fold CV  (n=1 experiment)"
        fig.suptitle(f"{gene}  |  {design_hpf} hpf  |  n={n_emb} embryos\n"
                     f"{cv_label}",
                     fontsize=10, fontweight="bold", y=1.02)

        marker_map = {class0: "s", class1: "o"}
        c0_color = PHENOTYPE_COLORS.get(class0, "#888888")
        c1_color = PHENOTYPE_COLORS.get(class1, "#444444")

        # Build a custom colormap: class0 color → white (midpoint) → class1 color
        from matplotlib.colors import LinearSegmentedColormap
        gene_cmap = LinearSegmentedColormap.from_list(
            f"{gene}_pheno", [c0_color, "#f7f7f7", c1_color]
        )
        true_color_map = {class0: c0_color, class1: c1_color}

        def _scatter_by_class(ax, xs, ys, labels, color_fn, alpha=0.75, s=45):
            """Scatter with per-class markers; color_fn(cls) → color or array."""
            for cls in [class0, class1]:
                mask = labels == cls
                if not mask.any():
                    continue
                colors = color_fn(cls, mask)
                ax.scatter(xs[mask], ys[mask], c=colors,
                           marker=marker_map[cls], s=s, alpha=alpha,
                           edgecolors="k", linewidths=0.3)

        # Col 0: true labels
        ax0 = axes[0]
        _scatter_by_class(ax0, x_vals, y_vals, true_labels,
                          lambda cls, m: [true_color_map[cls]] * m.sum())
        ax0.set_title("true labels", fontsize=9, fontweight="bold")
        ax0.set_xlabel(X_FEAT, fontsize=8)
        ax0.set_ylabel(Y_FEAT, fontsize=8)
        ax0.tick_params(labelsize=7)

        # Cols 1-4: classifier probability (shared colormap)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        has_clf_col = False

        for col_i, clf in enumerate(CLASSIFIERS, start=1):
            ax = axes[col_i]
            ft = clf["feature_type"]
            r = clf_results.get(ft)

            if r is None:
                ax.text(0.5, 0.5, "CV failed", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(clf["name"], fontsize=9)
                continue

            prob1 = r["prob1"]
            for cls in [class0, class1]:
                mask = true_labels == cls
                if not mask.any():
                    continue
                colors = gene_cmap(norm(prob1[mask]))
                ax.scatter(x_vals[mask], y_vals[mask], c=colors,
                           marker=marker_map[cls], s=45, alpha=0.8,
                           edgecolors="k", linewidths=0.3)

            ax.set_title(f"{clf['name']}\nbacc={r['bacc']:.2f}", fontsize=9)
            ax.set_xlabel(X_FEAT, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yticklabels([])
            has_clf_col = True

        # Shared colorbar attached to rightmost axis
        if has_clf_col:
            sm = cm.ScalarMappable(cmap=gene_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[-1], fraction=0.08, pad=0.03)
            cbar.set_label(f"P({class1})", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([class0, "0.5", class1], fontsize=7)

        # Legend: shape = true class, color = phenotype
        legend_handles = [
            mpatches.Patch(facecolor=c0_color, edgecolor="k", label=class0),
            mpatches.Patch(facecolor=c1_color, edgecolor="k", label=class1),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                       markeredgecolor="k", markersize=7, label=f"shape: {class0}"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markeredgecolor="k", markersize=7, label=f"shape: {class1}"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=8, bbox_to_anchor=(0.5, -0.09))

        fig.tight_layout()
        out_path = PLOT_DIR / f"{gene}_{design_hpf}hpf.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path.name}")

        # Store for big panel
        all_scatter[gene].append({
            "design_hpf": design_hpf, "n_emb": n_emb, "n_exp": n_exp,
            "x_vals": x_vals, "y_vals": y_vals, "true_labels": true_labels,
            "clf_results": clf_results,
            "c0_color": c0_color, "c1_color": c1_color,
            "gene_cmap": gene_cmap, "norm": norm,
            "class0": class0, "class1": class1,
        })

# ---------------------------------------------------------------------------
# Trend plot: balanced accuracy over time, one panel per gene
# ---------------------------------------------------------------------------
CLF_COLORS = {
    "embedding": "#333333",
    "both":      "#7b3294",
    "length":    "#1f78b4",
    "curvature": "#e66101",
}
CLF_MARKERS = {"embedding": "D", "both": "o", "length": "s", "curvature": "^"}
CLF_LABELS = {
    "embedding": "z_mu_b (80-dim)",
    "both":      "length + curvature",
    "length":    "length only",
    "curvature": "curvature only",
}

fig_trend, axes_trend = plt.subplots(1, len(GENES), figsize=(7 * len(GENES), 5), sharey=True)
if len(GENES) == 1:
    axes_trend = [axes_trend]

for ax, (gene, _) in zip(axes_trend, GENES.items()):
    # Vertical lines at canonical sequenced timepoints
    for dhpf in TARGET_DESIGN_HPF:
        ax.axvline(_bin_center(dhpf), color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7,
                   zorder=0)
        ax.text(_bin_center(dhpf), 1.02, f"{dhpf}", ha="center", va="bottom",
                fontsize=8, color="#888888")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.5, label="chance (0.5)")

    for clf in CLASSIFIERS:
        ft = clf["feature_type"]
        pts = sorted(all_bacc[gene][ft])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=CLF_COLORS[ft], marker=CLF_MARKERS[ft],
                linewidth=2.0, markersize=7, label=CLF_LABELS[ft])

    ax.set_title(gene, fontsize=14, fontweight="bold")
    ax.set_xlabel("predicted stage (hpf)", fontsize=12)
    ax.set_ylim(0.3, 1.08)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)

axes_trend[0].set_ylabel("balanced accuracy\n(leave-one-experiment-out CV)", fontsize=12)
handles, labels = axes_trend[0].get_legend_handles_labels()
fig_trend.legend(handles, labels, loc="lower center", ncol=3, fontsize=11,
                 bbox_to_anchor=(0.5, -0.15))
fig_trend.suptitle("Classifier accuracy over developmental time\n"
                   "homozygous reference embryos  |  4-hpf bins",
                   fontsize=12, y=1.02)
fig_trend.subplots_adjust(bottom=0.22, wspace=0.08)
trend_path = PLOT_DIR / "accuracy_over_time.png"
fig_trend.savefig(trend_path, dpi=150, bbox_inches="tight")
plt.close(fig_trend)
print(f"Saved trend plot: {trend_path.name}")

# ---------------------------------------------------------------------------
# Big panel: all timepoints side by side, one figure per gene
# ---------------------------------------------------------------------------
for gene, cfg in GENES.items():
    slices = all_scatter[gene]
    if not slices:
        continue

    n_tp = len(slices)
    n_cols_big = 1 + len(CLASSIFIERS)
    fig_big, axes_big = plt.subplots(
        n_tp, n_cols_big,
        figsize=(3.2 * n_cols_big, 3.0 * n_tp),
        constrained_layout=False,
    )
    if n_tp == 1:
        axes_big = axes_big[np.newaxis, :]

    for row, s in enumerate(slices):
        dhpf = s["design_hpf"]
        x_v, y_v = s["x_vals"], s["y_vals"]
        tl = s["true_labels"]
        cr = s["clf_results"]
        c0c, c1c = s["c0_color"], s["c1_color"]
        gcmap, gnorm = s["gene_cmap"], s["norm"]
        cls0, cls1 = s["class0"], s["class1"]
        mm = {cls0: "s", cls1: "o"}
        tcm = {cls0: c0c, cls1: c1c}
        cv_lbl = (f"LOEO n={s['n_exp']} exp" if s["n_exp"] >= 2 else "5-fold CV")

        # Col 0: true labels
        ax = axes_big[row, 0]
        for cls in [cls0, cls1]:
            mask = tl == cls
            if mask.any():
                ax.scatter(x_v[mask], y_v[mask], c=tcm[cls], marker=mm[cls],
                           s=30, alpha=0.75, edgecolors="k", linewidths=0.25)
        ax.set_ylabel(f"{dhpf} hpf\n({cv_lbl})", fontsize=7)
        if row == 0:
            ax.set_title("true labels", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.set_xlabel(X_FEAT if row == n_tp - 1 else "", fontsize=7)

        for col_i, clf in enumerate(CLASSIFIERS, start=1):
            ax = axes_big[row, col_i]
            ft = clf["feature_type"]
            r = cr.get(ft)
            if r is None:
                ax.text(0.5, 0.5, "failed", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            else:
                for cls in [cls0, cls1]:
                    mask = tl == cls
                    if not mask.any():
                        continue
                    colors = gcmap(gnorm(r["prob1"][mask]))
                    ax.scatter(x_v[mask], y_v[mask], c=colors, marker=mm[cls],
                               s=30, alpha=0.8, edgecolors="k", linewidths=0.25)
                if row == 0:
                    ax.set_title(f"{clf['name']}", fontsize=8, fontweight="bold")
                ax.set_title((ax.get_title() + "\n" if ax.get_title() else "") +
                             f"bacc={r['bacc']:.2f}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_yticklabels([])
            ax.set_xlabel(X_FEAT if row == n_tp - 1 else "", fontsize=7)

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=slices[0]["gene_cmap"], norm=slices[0]["norm"])
    sm.set_array([])
    cbar = fig_big.colorbar(sm, ax=axes_big[:, -1], fraction=0.04, pad=0.02)
    cbar.set_label(f"P({slices[0]['class1']})", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([slices[0]["class0"], "0.5", slices[0]["class1"]], fontsize=7)

    # Legend
    c0c, c1c = slices[0]["c0_color"], slices[0]["c1_color"]
    cls0, cls1 = slices[0]["class0"], slices[0]["class1"]
    leg = [
        mpatches.Patch(facecolor=c0c, edgecolor="k", label=cls0),
        mpatches.Patch(facecolor=c1c, edgecolor="k", label=cls1),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markeredgecolor="k", markersize=7, label=f"shape: {cls0}"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="k", markersize=7, label=f"shape: {cls1}"),
    ]
    fig_big.legend(handles=leg, loc="lower center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

    fig_big.suptitle(f"{gene}  —  all timepoints  |  leave-one-experiment-out CV",
                     fontsize=11, fontweight="bold", y=1.01)
    fig_big.tight_layout()
    big_path = PLOT_DIR / f"{gene}_all_timepoints.png"
    fig_big.savefig(big_path, dpi=150, bbox_inches="tight")
    plt.close(fig_big)
    print(f"Saved big panel: {big_path.name}")

print("\nDone.")
