"""Replot the emb emergence figures at significance threshold p<0.01.

Reuses the CACHED classification scores from script 13 (500-perm p-values already computed) --
no CV, no permutations rerun. Only the significance threshold and the rendering change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
SOURCE_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SOURCE_DIR))

from analyze.classification.viz import plot_aurocs_over_time  # noqa: E402
from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import StyleSpec  # noqa: E402
from plot_config import PHENOTYPE_COLORS, GENOTYPE_COLORS  # noqa: E402

CLASS_DIR = RUN_DIR / "classification"
PLOT_DIR = RUN_DIR / "figures" / "phenotype_emergence"
GENE_ORDER = ["cep290", "b9d2"]
SIG = 0.01
N_SPLITS = 5
N_PERM = 500

PHENO_COLORS = {
    k: PHENOTYPE_COLORS[k]
    for k in ("High_to_Low", "Low_to_High", "CE", "HTA", "Not Penetrant")
    if k in PHENOTYPE_COLORS
}
PHENO_COLORS.setdefault("Not Penetrant", "#BBBBBB")
GENO_COLORS = {k: GENOTYPE_COLORS[k] for k in ("homozygous", "heterozygous")}


def _load(tag: str) -> pd.DataFrame:
    frames = []
    for gene in GENE_ORDER:
        s = pd.read_parquet(CLASS_DIR / f"{gene}_{tag}" / "scores.parquet")
        s["gene"] = gene
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _figure(scores: pd.DataFrame, colors: dict, title: str, out: Path) -> None:
    fig = plot_aurocs_over_time(
        scores,
        curve_col="positive_label",
        facet_row="feature_set",
        facet_col="gene",
        layout=FacetSpec(row_order=["emb"], col_order=GENE_ORDER, sharex=True, sharey=True),
        color_lookup=colors,
        show_null_band=True,
        show_significance=True,
        sig_threshold=SIG,
        show_chance_line=True,
        title=title,
        x_label="Hours Post Fertilization (hpf)",
        y_label="cross-validated AUROC",
        style=StyleSpec(legend_fontsize=8, legend_loc="per-panel"),
        backend="matplotlib",
        output_path=None,
    )
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(RUN_DIR)}")


def main() -> None:
    _figure(
        _load("genotype_emergence"), GENO_COLORS,
        "Genotype emergence vs wildtype (emb latents)\n"
        f"pooled homozygous / heterozygous vs wildtype  |  2 hpf bins, {N_SPLITS}-fold CV, "
        f"{N_PERM} perm, p<{SIG}",
        PLOT_DIR / "genotype_emergence_emb.png",
    )
    _figure(
        _load("phenotype_emergence"), PHENO_COLORS,
        "Homozygous phenotype emergence vs wildtype (emb latents)\n"
        f"per phenotype class vs wildtype  |  2 hpf bins, {N_SPLITS}-fold CV, "
        f"{N_PERM} perm, p<{SIG}",
        PLOT_DIR / "phenotype_emergence_emb.png",
    )


if __name__ == "__main__":
    main()
