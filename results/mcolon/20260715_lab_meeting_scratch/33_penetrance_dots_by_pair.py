"""Per-pair penetrance dots, in the style of composite view C.

Replaces the stacked-bar `phenotype_distribution_by_pair` figure with the dot layout: one dot
per embryo, non-penetrant as the baseline band, and a rule marking the predicted-penetrant
fraction with its rate printed on top.

Grid is  rows = zygosity  x  cols = pair, per gene, using the DERIVED 3-class labels
(trained once on the whole gene, then read out per pair -- the pairs are too small to train
on individually).

The rate is the classifier's own call ("% pen."), not a measured biological penetrance.
`unknown` zygosity is excluded, as in script 32.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/33_penetrance_dots_by_pair.py
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
sys.path.insert(0, str(RUN_DIR))

# Reuse the loaders + scorer so the labels match the pooled figure exactly.
_both = __import__("32_penetrance_dots_both_genes")
load_b9d2 = _both.load_b9d2
load_cep290 = _both.load_cep290
GENES = _both.GENES
NP_LABEL = _both.NP_LABEL
ZYGOSITY_ORDER = _both.ZYGOSITY_ORDER
ZYG_LABEL = _both.ZYG_LABEL
ID_COL = _both.ID_COL
_pipe = _both._pipe

OUTPUT_DIR = RUN_DIR / "figures" / "penetrance_by_pair"

from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

N_FOLDS = _both.N_FOLDS
RANDOM_STATE = _both.RANDOM_STATE


def score_with_pair(df: pd.DataFrame, z: list[str], classes: list[str]) -> pd.DataFrame:
    """Same scoring as script 32, but carries `pair` through to the per-embryo table."""
    tr_mask = df["train_label"].notna()
    tr = df[tr_mask]
    y = tr["train_label"].to_numpy()
    X = tr[z].to_numpy()
    groups = tr[ID_COL].to_numpy()

    oof = np.full((len(tr), len(classes)), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for a, b in cv.split(X, y, groups=groups):
        m = _pipe().fit(X[a], y[a])
        pr = m.predict_proba(X[b])
        for j, c in enumerate(classes):
            oof[b, j] = pr[:, list(m.classes_).index(c)] if c in m.classes_ else 0.0

    full = _pipe().fit(X, y)
    df = df.copy()
    pcols = [f"p_{i}" for i in range(len(classes))]
    for c in pcols:
        df[c] = np.nan
    df.loc[tr.index, pcols] = oof
    rest = df.index[~tr_mask]
    if len(rest):
        pr = full.predict_proba(df.loc[rest, z].to_numpy())
        for j, c in enumerate(classes):
            df.loc[rest, pcols[j]] = (pr[:, list(full.classes_).index(c)]
                                      if c in full.classes_ else 0.0)

    per = df.groupby(ID_COL)[pcols].mean()
    pred = per.idxmax(axis=1).map({p: c for p, c in zip(pcols, classes)})
    keep = [ID_COL, "zygosity"] + (["pair"] if "pair" in df.columns else [])
    out = df.drop_duplicates(ID_COL)[keep].copy()
    out["phenotype"] = out[ID_COL].map(pred)
    out = out[out["zygosity"].isin(ZYGOSITY_ORDER)].dropna(subset=["phenotype"])
    return out


def cell(ax, sub: pd.DataFrame, gene: str, rng, show_y: bool) -> None:
    """One zygosity x pair cell: stacked dots + penetrance rule."""
    classes = GENES[gene]["classes"]
    colors = GENES[gene]["colors"]
    total = len(sub)

    if total == 0:
        ax.text(0.5, 0.5, "–", ha="center", va="center",
                fontsize=11, color="#CCCCCC", transform=ax.transAxes)
    else:
        n_pen = int((sub["phenotype"] != NP_LABEL).sum())
        pen = n_pen / total
        y0 = 0.0
        for pheno in reversed(classes):          # NP at the bottom
            n = int((sub["phenotype"] == pheno).sum())
            if n == 0:
                continue
            frac = n / total
            ys = y0 + rng.uniform(0.04, 0.96, size=n) * frac
            xs = rng.uniform(-0.20, 0.20, size=n)
            ax.scatter(xs, ys, s=26, color=colors[pheno], alpha=0.75,
                       edgecolors="white", linewidths=0.35, zorder=3)
            y0 += frac
        ax.plot([-0.30, 0.30], [1 - pen, 1 - pen], color="#222222",
                linewidth=1.8, zorder=5)
        ax.annotate(f"{pen*100:.0f}%", (0, 1 - pen), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#222222", zorder=6)
        ax.annotate(f"n={total}", (0.5, 0.005), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=7, color="#999999")

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.04, 1.14)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"] if show_y else [], fontsize=8)
    ax.grid(True, axis="y", color="#F0F0F0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure_for_gene(emb: pd.DataFrame, gene: str, rng) -> None:
    if "pair" not in emb.columns:
        print(f"[{gene}] no pair column — skipped")
        return
    pairs = sorted(p for p in emb["pair"].dropna().unique())
    if not pairs:
        print(f"[{gene}] no pairs — skipped")
        return

    print(f"\n[{gene}] {len(pairs)} pairs")
    print(pd.crosstab(emb["pair"], [emb["zygosity"], emb["phenotype"]]).to_string())

    nrow, ncol = len(ZYGOSITY_ORDER), len(pairs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.05 * ncol + 1.0, 2.05 * nrow),
                             squeeze=False)
    for r, zyg in enumerate(ZYGOSITY_ORDER):
        for c, pair in enumerate(pairs):
            sub = emb[(emb["zygosity"] == zyg) & (emb["pair"] == pair)]
            cell(axes[r][c], sub, gene, rng, show_y=(c == 0))
            if r == 0:
                axes[r][c].set_title(str(pair).replace(f"{gene}_", ""),
                                     fontsize=8.5, fontweight="bold")
        axes[r][0].set_ylabel(ZYG_LABEL[zyg], fontsize=10, fontweight="bold")

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                          markerfacecolor=GENES[gene]["colors"][c],
                          markeredgecolor="white", label=c)
               for c in GENES[gene]["classes"]]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(handles))
    fig.suptitle(f"{gene} — predicted penetrance by pair (rule = % penetrant)",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"penetrance_by_pair__{gene}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    rng = np.random.default_rng(0)
    b9, zb = load_b9d2()
    cep, zc = load_cep290()
    figure_for_gene(score_with_pair(b9, zb, GENES["b9d2"]["classes"]), "b9d2", rng)
    figure_for_gene(score_with_pair(cep, zc, GENES["cep290"]["classes"]), "cep290", rng)


if __name__ == "__main__":
    main()
