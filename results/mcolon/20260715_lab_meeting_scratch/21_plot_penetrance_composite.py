"""Penetrance composite: phenotype spread across wildtype / het / homozygous, everything pooled.

The claim this figure has to carry:
    cep290  -- only homozygotes are penetrant; hets look like wildtype.
    b9d2    -- homozygotes are penetrant AND a real fraction of hets carry the CE phenotype.

Fractions are computed WITHIN each zygosity (standard penetrance framing: "what % of b9d2 hets
are CE"). Raw embryo counts are annotated on every view so a percentage over a tiny n can never
be mistaken for a strong result.

Phenotype calls come from the same all-zygosity label transfer used by script 14 -- the models
are trained on homozygotes and applied to every zygosity -- so these panels are the composition
summary of exactly the embryos in figures/resolve_phenotypes_from_mess.

Three views of the identical table, to pick from:
    A  dot matrix       -- zygosity x phenotype grid, dot area = n, fill = within-zygosity %.
                           Empty cells are visually loud, which is the cep290 point.
    B  skinny bars      -- 100% stacked composition, one thin bar per zygosity.
    C  per-embryo dots  -- one jittered dot per embryo; n is countable, not inferred.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/21_plot_penetrance_composite.py
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

# Reuse the exact loader + phenotype transfer from the "resolve from mess" figure so the two
# figures can never disagree about which embryo got which label.
_mess = __import__("14_plot_resolve_phenotypes_from_mess")
load_gene = _mess.load_gene
GENE_ORDER = _mess.GENE_ORDER
ZYGOSITY_ORDER = _mess.ZYGOSITY_ORDER
GENE_CONFIG = _mess.GENE_CONFIG

OUTPUT_DIR = RUN_DIR / "figures" / "penetrance_composite"
ID_COL = _mess.ID_COL

NP_LABEL = "Not Penetrant"
NP_COLOR = "#BBBBBB"

# Phenotype row order per gene: penetrant classes first, non-penetrant baseline last.
GENE_PHENOTYPES = {
    "cep290": ["High_to_Low", "Low_to_High", NP_LABEL],
    "b9d2": ["CE", "HTA", NP_LABEL],
}
ZYG_LABEL = {"wildtype": "WT", "heterozygous": "het", "homozygous": "homo"}
GENE_TITLE = {"cep290": "cep290", "b9d2": "b9d2"}


def phenotype_colors(gene: str) -> dict:
    colors = dict(GENE_CONFIG[gene]["pheno_colors"])
    colors[NP_LABEL] = NP_COLOR
    return colors


def build_table() -> pd.DataFrame:
    """One row per embryo: gene, zygosity, transferred phenotype."""
    frames = []
    for gene in GENE_ORDER:
        df = load_gene(gene)
        df = df[df["phenotype_clean"].notna()]
        frames.append(
            df.drop_duplicates(ID_COL)[[ID_COL, "gene", "zygosity", "phenotype_clean"]]
        )
    table = pd.concat(frames, ignore_index=True)
    table = table[table["zygosity"].isin(ZYGOSITY_ORDER)].copy()
    return table


def composition(table: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Within-zygosity composition for one gene: counts, totals, and percentages."""
    sub = table[table["gene"] == gene]
    phenos = GENE_PHENOTYPES[gene]
    counts = (
        sub.groupby(["zygosity", "phenotype_clean"]).size()
        .unstack("phenotype_clean")
        .reindex(index=ZYGOSITY_ORDER, columns=phenos)
        .fillna(0)
        .astype(int)
    )
    totals = counts.sum(axis=1)
    # Guard against an empty zygosity so a 0/0 cell renders as 0% rather than NaN.
    pct = counts.div(totals.replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    return counts, totals, pct


def _penetrant_pct(counts: pd.DataFrame, totals: pd.Series, gene: str) -> pd.Series:
    """% of each zygosity carrying ANY penetrant (non-NP) phenotype."""
    penetrant = [p for p in GENE_PHENOTYPES[gene] if p != NP_LABEL]
    return counts[penetrant].sum(axis=1).div(totals.replace(0, np.nan)).fillna(0.0) * 100.0


# --------------------------------------------------------------------------------------- view A
def plot_dot_matrix(table: pd.DataFrame, out: Path) -> None:
    """Zygosity x phenotype grid. Dot area ~ n embryos, fill saturation ~ within-zygosity %."""
    fig, axes = plt.subplots(1, len(GENE_ORDER), figsize=(11.5, 4.4))

    for ax, gene in zip(np.atleast_1d(axes), GENE_ORDER):
        counts, totals, pct = composition(table, gene)
        # Scale dot area PER GENE. A global max would shrink the whole b9d2 panel just because
        # cep290 has ~3x the embryos, hiding the b9d2-het signal this figure exists to show.
        max_n = max(int(counts.to_numpy().max()), 1)
        phenos = GENE_PHENOTYPES[gene]
        colors = phenotype_colors(gene)

        for yi, pheno in enumerate(phenos):
            for xi, zyg in enumerate(ZYGOSITY_ORDER):
                n = int(counts.loc[zyg, pheno])
                p = float(pct.loc[zyg, pheno])
                if n == 0:
                    # Draw the absence explicitly -- an empty cell is the cep290-het message.
                    ax.scatter(xi, yi, s=26, facecolors="none", edgecolors="#D8D8D8",
                               linewidths=1.0, zorder=2)
                    continue
                # Area scales with n; alpha carries the within-zygosity percentage.
                size = 90 + 1500 * (n / max_n)
                ax.scatter(xi, yi, s=size, color=colors[pheno],
                           alpha=0.30 + 0.70 * (p / 100.0),
                           edgecolors="white", linewidths=1.4, zorder=3)
                ax.annotate(f"{p:.0f}%", (xi, yi), ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if p > 45 else "#333333", zorder=4)
                ax.annotate(f"n={n}", (xi, yi), xytext=(0, -17), textcoords="offset points",
                            ha="center", va="top", fontsize=7.5, color="#666666", zorder=4)

        ax.set_xticks(range(len(ZYGOSITY_ORDER)))
        ax.set_xticklabels([f"{ZYG_LABEL[z]}\n(n={int(totals[z])})" for z in ZYGOSITY_ORDER],
                           fontsize=10)
        ax.set_yticks(range(len(phenos)))
        ax.set_yticklabels(phenos, fontsize=10)
        ax.set_xlim(-0.6, len(ZYGOSITY_ORDER) - 0.4)
        ax.set_ylim(-0.6, len(phenos) - 0.4)
        ax.invert_yaxis()
        ax.set_title(GENE_TITLE[gene], fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Phenotype composition within each zygosity  (dot area = n, label = % of column)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------------------- view B
def plot_skinny_bars(table: pd.DataFrame, out: Path) -> None:
    """100% stacked composition, one skinny bar per zygosity."""
    fig, axes = plt.subplots(1, len(GENE_ORDER), figsize=(10.5, 4.8), sharey=True)

    for ax, gene in zip(np.atleast_1d(axes), GENE_ORDER):
        counts, totals, pct = composition(table, gene)
        phenos = GENE_PHENOTYPES[gene]
        colors = phenotype_colors(gene)

        bottoms = np.zeros(len(ZYGOSITY_ORDER))
        xs = np.arange(len(ZYGOSITY_ORDER))
        for pheno in phenos:
            vals = pct[pheno].to_numpy()
            ax.bar(xs, vals, bottom=bottoms, width=0.26, color=colors[pheno],
                   edgecolor="white", linewidth=1.0, label=pheno, zorder=3)
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                n = int(counts.iloc[xi][pheno])
                if v <= 0:
                    continue
                if v >= 12:
                    # Fits inside, but stack pct over n on two lines -- one line overflows the
                    # 0.26-wide bars.
                    ax.annotate(f"{v:.0f}%\nn={n}", (xi, b + v / 2), ha="center", va="center",
                                fontsize=8.5, fontweight="bold", linespacing=1.25,
                                color="white" if pheno != NP_LABEL else "#444444", zorder=4)
                else:
                    # Thin segment: label outside to the right with a leader, so the small-but-
                    # real classes (cep290 het) stay readable instead of overflowing the bar.
                    ax.annotate(f"{v:.0f}% ({n})", (xi + 0.16, b + v / 2),
                                xytext=(14, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=8, fontweight="bold",
                                color=colors[pheno] if pheno != NP_LABEL else "#666666",
                                arrowprops=dict(arrowstyle="-", linewidth=0.7,
                                                color="#AAAAAA", shrinkA=0, shrinkB=2),
                                zorder=4)
            bottoms += vals

        for xi, zyg in enumerate(ZYGOSITY_ORDER):
            ax.annotate(f"n={int(totals[zyg])}", (xi, 101), ha="center", va="bottom",
                        fontsize=9, color="#444444")

        ax.set_xticks(xs)
        ax.set_xticklabels([ZYG_LABEL[z] for z in ZYGOSITY_ORDER], fontsize=11)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(GENE_TITLE[gene], fontsize=13, fontweight="bold", pad=14)
        ax.legend(frameon=False, fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.10), ncol=len(phenos))
        ax.grid(True, axis="y", color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    np.atleast_1d(axes)[0].set_ylabel("% of embryos within zygosity", fontsize=11)
    fig.suptitle("Phenotype composition within each zygosity", fontsize=12, y=1.00)
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------------------- view C
def plot_embryo_dots(table: pd.DataFrame, out: Path, seed: int = 0) -> None:
    """One dot per embryo, jittered per zygosity column, colored by phenotype."""
    rng = np.random.default_rng(seed)
    # Skinny columns: narrower figure keeps the penetrance rule the dominant read.
    fig, axes = plt.subplots(1, len(GENE_ORDER), figsize=(7.0, 5.0), sharey=True)

    for ax, gene in zip(np.atleast_1d(axes), GENE_ORDER):
        counts, totals, pct = composition(table, gene)
        phenos = GENE_PHENOTYPES[gene]
        colors = phenotype_colors(gene)
        sub = table[table["gene"] == gene]
        pen_pct = _penetrant_pct(counts, totals, gene)

        for xi, zyg in enumerate(ZYGOSITY_ORDER):
            col = sub[sub["zygosity"] == zyg]
            # Stack phenotypes vertically within the column so classes stay readable, and
            # keep NP at the bottom as the baseline band.
            y0 = 0.0
            for pheno in reversed(phenos):
                members = col[col["phenotype_clean"] == pheno]
                n = len(members)
                if n == 0:
                    continue
                frac = n / max(int(totals[zyg]), 1)
                ys = y0 + rng.uniform(0.02, 0.98, size=n) * frac
                # Tighter jitter to match the narrower columns; bolder dots with alpha so
                # overlap reads as density rather than a solid blob.
                xs = xi + rng.uniform(-0.11, 0.11, size=n)
                ax.scatter(xs, ys, s=34, color=colors[pheno], alpha=0.72,
                           edgecolors="white", linewidths=0.4, zorder=3)
                y0 += frac

            # Penetrant-fraction marker: the top of the non-NP stack.
            pen = pen_pct[zyg] / 100.0
            ax.plot([xi - 0.19, xi + 0.19], [1 - pen, 1 - pen], color="#222222",
                    linewidth=2.0, zorder=5)
            # Rate sits directly ON TOP of the rule it describes.
            ax.annotate(f"{pen_pct[zyg]:.0f}% pen.", (xi, 1 - pen),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color="#222222", zorder=6)

        ax.set_xticks(range(len(ZYGOSITY_ORDER)))
        ax.set_xticklabels([f"{ZYG_LABEL[z]}\nn={int(totals[z])}" for z in ZYGOSITY_ORDER],
                           fontsize=10)
        ax.set_ylim(-0.03, 1.14)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        # Extra right margin so the penetrance label on the last column has room.
        ax.set_xlim(-0.55, len(ZYGOSITY_ORDER) - 0.05)
        ax.set_title(GENE_TITLE[gene], fontsize=13, fontweight="bold", pad=10)
        # Build the legend from the gene's full phenotype set, not from whichever classes
        # happened to be drawn -- b9d2 WT is 100% NP, which would otherwise drop CE/HTA.
        handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                              markerfacecolor=colors[p], markeredgecolor="white", label=p)
                   for p in phenos]
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=len(phenos))
        ax.grid(True, axis="y", color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    np.atleast_1d(axes)[0].set_ylabel("% of embryos within zygosity", fontsize=11)
    fig.suptitle("One dot per embryo, grouped within zygosity  (bar = penetrant fraction)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    table = build_table()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "penetrance_table.csv", index=False)

    for gene in GENE_ORDER:
        counts, totals, pct = composition(table, gene)
        print(f"\n[{gene}] within-zygosity composition (n embryos)")
        print(counts.assign(total=totals).to_string())
        print(f"  penetrant %: {_penetrant_pct(counts, totals, gene).round(1).to_dict()}")

    plot_dot_matrix(table, OUTPUT_DIR / "A_dot_matrix.png")
    plot_skinny_bars(table, OUTPUT_DIR / "B_skinny_bars.png")
    plot_embryo_dots(table, OUTPUT_DIR / "C_embryo_dots.png")


if __name__ == "__main__":
    main()
