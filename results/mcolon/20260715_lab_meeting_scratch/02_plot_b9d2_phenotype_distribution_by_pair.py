"""Recreate the B9D2 pair-by-zygosity phenotype distribution."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.styling import (  # noqa: E402
    B9D2_PHENOTYPE_COLORS,
    B9D2_PHENOTYPE_ORDER,
    canonicalize_b9d2_phenotype,
)


INPUT_CSV = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv"
)
OUTPUT_DIR = RUN_DIR / "figures" / "b9d2_phenotype_distribution"
OUTPUT_PNG = OUTPUT_DIR / "phenotype_distribution_by_pair.png"

PAIR_ORDER = [f"b9d2_pair_{pair}" for pair in (2, 4, 5, 6, 7, 8)]
ZYGOSITY_ORDER = ["wildtype", "heterozygous", "homozygous"]


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_embryos() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_CSV,
        usecols=["embryo_id", "pair", "zygosity", "phenotype_clean", "use_embryo_flag"],
        low_memory=False,
    )
    df = df[_qc_mask(df["use_embryo_flag"])].drop_duplicates("embryo_id").copy()
    df = df[df["pair"].isin(PAIR_ORDER) & df["zygosity"].isin(ZYGOSITY_ORDER)].copy()
    df["phenotype_binary"] = df["phenotype_clean"].map(canonicalize_b9d2_phenotype)
    return df


def main() -> None:
    df = load_embryos()
    grouped = (
        df.groupby(["zygosity", "pair", "phenotype_binary"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )

    fig, axes = plt.subplots(
        len(ZYGOSITY_ORDER),
        len(PAIR_ORDER),
        figsize=(21.5, 10.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_positions = dict(zip(B9D2_PHENOTYPE_ORDER, (-0.32, 0.0, 0.32)))
    for row_index, zygosity in enumerate(ZYGOSITY_ORDER):
        for col_index, pair in enumerate(PAIR_ORDER):
            ax = axes[row_index, col_index]
            panel = grouped[(grouped["zygosity"] == zygosity) & (grouped["pair"] == pair)]
            counts = panel.set_index("phenotype_binary")["count"].to_dict()
            total = sum(counts.values())
            for phenotype in B9D2_PHENOTYPE_ORDER:
                count = int(counts.get(phenotype, 0))
                if count == 0 or total == 0:
                    continue
                proportion = count / total
                ax.bar(
                    x_positions[phenotype],
                    proportion,
                    width=0.29,
                    color=B9D2_PHENOTYPE_COLORS[phenotype],
                )
                ax.text(
                    x_positions[phenotype],
                    proportion + 0.012,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                )
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(0, 1.15)
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.18, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=12)
            if row_index == 0:
                ax.set_title(pair, fontsize=11, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(zygosity, fontweight="bold")
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    fig.suptitle(
        "B9D2 Phenotype Distribution by Pair and Genotype (Exp 20251121 & 20251125)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    handles = [
        Patch(
            facecolor=B9D2_PHENOTYPE_COLORS[label],
            label="Not Penetrant" if label == "non_penetrant" else label,
        )
        for label in B9D2_PHENOTYPE_ORDER
    ]
    fig.legend(
        handles=handles,
        title="phenotype",
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        fontsize=12,
        title_fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.89, 0.94))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Embryos plotted: {df['embryo_id'].nunique()}")
    print(grouped.to_string(index=False))
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
