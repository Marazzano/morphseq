"""Phenotype/genotype emergence (emb features) for the base cilia genes cep290 + b9d2.

Regenerates the emb-feature-set row of
``20260607_sci_cilia_gene14_imaging_qc/plots/separability/phenotype_emergence.png`` as TWO
separate figures, dropping crispant:

  figure 1 (per GENOTYPE): pooled homozygous-vs-wt and heterozygous-vs-wt. Answers "when does
      carrying the mutation become detectable from embryo latents", regardless of phenotype.
      Classes are whole zygosity groups (all homozygotes / all heterozygotes), negative=wildtype.

  figure 2 (per PHENOTYPE): each homozygous phenotype class vs wildtype, 3-class:
      cep290 -> High_to_Low / Low_to_High / Not Penetrant ; b9d2 -> CE / HTA / Not Penetrant.
      NP is expected to sit near chance (a non-penetrant homozygote looks like wildtype), so its
      curve doubles as a sanity check.

Each figure is a 1-row (emb) x 2-col (cep290, b9d2) AUROC-over-time grid with permutation null
bands + significance markers, reusing ``run_classification`` and ``plot_aurocs_over_time`` exactly
as script 3g does.

Run:
    PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output \\
        python results/mcolon/20260715_lab_meeting_scratch/13_plot_phenotype_emergence_emb.py

Set N_PERM=0 for a fast smoke test (no null bands / significance).
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SOURCE_DIR))  # for plot_config

from analyze.classification import run_classification  # noqa: E402
from analyze.classification.viz import plot_aurocs_over_time  # noqa: E402
from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import StyleSpec  # noqa: E402
from plot_config import PHENOTYPE_COLORS, GENOTYPE_COLORS  # noqa: E402

TABLE_DIR = SOURCE_DIR / "tables"
CLASS_DIR = RUN_DIR / "classification"
PLOT_DIR = RUN_DIR / "figures" / "phenotype_emergence"
CLASS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
FEAT_COLS = ["baseline_deviation_normalized", "total_length_um"]
BIN_WIDTH = 2.0
N_SPLITS = 5
N_PERM = 500
MAX_HPF = 48.0

# emb feature set only (the 80 z_mu_b_* biological latents).
FEATURES = {"emb": "z_mu_b"}
GENE_ORDER = ["cep290", "b9d2"]

# cep290 has a real 3-class scheme incl. Not Penetrant. b9d2 stays 2-class: its NP-equivalent is
# 'unlabeled' (n=5, and means 'not scored', not 'confirmed non-penetrant'), too few + ambiguous.
PHENO_CLASSES = {
    "cep290": ["High_to_Low", "Low_to_High", "Not Penetrant"],
    "b9d2": ["CE", "HTA"],
}
GENOTYPE_CLASSES = ["homozygous", "heterozygous"]

GENE_TABLE = {
    "cep290": "reference_cep290_clean.csv",
    "b9d2": "reference_b9d2_clean.csv",
}

# Colors: phenotype figure uses the phenotype palette; genotype figure uses zygosity palette.
PHENO_COLORS = {
    k: PHENOTYPE_COLORS[k]
    for k in ("High_to_Low", "Low_to_High", "CE", "HTA", "Not Penetrant")
    if k in PHENOTYPE_COLORS
}
PHENO_COLORS.setdefault("Not Penetrant", "#BBBBBB")
GENO_COLORS = {k: GENOTYPE_COLORS[k] for k in ("homozygous", "heterozygous")}


def _load(gene: str) -> pd.DataFrame:
    """Reference table with wildtype + het + homo, capped at MAX_HPF, feature-complete."""
    df = pd.read_csv(
        TABLE_DIR / GENE_TABLE[gene],
        usecols=lambda c: c in {ID_COL, TIME_COL, "zygosity", "phenotype_clean", *FEAT_COLS}
        or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    df = df.dropna(subset=[TIME_COL, *FEAT_COLS])
    df = df[df[TIME_COL] <= MAX_HPF]
    return df.copy()


def _run(df: pd.DataFrame, class_col: str, positive: list[str], gene: str, tag: str):
    result = run_classification(
        df,
        class_col=class_col,
        id_col=ID_COL,
        time_col=TIME_COL,
        positive=positive,
        negative="wildtype",
        features=FEATURES,
        bin_width=BIN_WIDTH,
        n_splits=N_SPLITS,
        n_permutations=N_PERM,
        n_jobs=-1,
        save_dir=str(CLASS_DIR / f"{gene}_{tag}"),
        overwrite=True,
        verbose=False,
    )
    scores = result.scores.copy()
    scores["gene"] = gene
    return scores


def _figure(all_scores: list[pd.DataFrame], colors: dict, title: str, out: Path) -> None:
    combined = pd.concat(all_scores, ignore_index=True)
    fig = plot_aurocs_over_time(
        combined,
        curve_col="positive_label",
        facet_row="feature_set",
        facet_col="gene",
        layout=FacetSpec(row_order=["emb"], col_order=GENE_ORDER, sharex=True, sharey=True),
        color_lookup=colors,
        show_null_band=(N_PERM > 0),
        show_significance=(N_PERM > 0),
        sig_threshold=0.05,
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
    print(f"13 - phenotype/genotype emergence (emb), N_PERM={N_PERM}")

    geno_scores, pheno_scores = [], []
    for gene in GENE_ORDER:
        df = _load(gene)

        # --- per-genotype: pooled homo-vs-wt, het-vs-wt ---
        geno = df[df["zygosity"].isin(["wildtype", *GENOTYPE_CLASSES])].copy()
        geno["plot_label"] = geno["zygosity"]
        vc = geno.drop_duplicates(ID_COL)["plot_label"].value_counts().to_dict()
        print(f"[{gene}] genotype embryos={geno[ID_COL].nunique()} labels={vc}")
        geno_scores.append(_run(geno, "plot_label", GENOTYPE_CLASSES, gene, "genotype_emergence"))

        # --- per-phenotype: homozygous phenotype classes vs wt (3-class) ---
        pheno = df[df["zygosity"].isin(["homozygous", "wildtype"])].copy()
        pheno["plot_label"] = pheno["phenotype_clean"].where(
            pheno["zygosity"] == "homozygous", "wildtype"
        )
        keep = set(PHENO_CLASSES[gene]) | {"wildtype"}
        pheno = pheno[pheno["plot_label"].isin(keep)]
        vc = pheno.drop_duplicates(ID_COL)["plot_label"].value_counts().to_dict()
        print(f"[{gene}] phenotype embryos={pheno[ID_COL].nunique()} labels={vc}")
        pheno_scores.append(
            _run(pheno, "plot_label", PHENO_CLASSES[gene], gene, "phenotype_emergence")
        )

    _figure(
        geno_scores, GENO_COLORS,
        "Genotype emergence vs wildtype (emb latents)\n"
        f"pooled homozygous / heterozygous vs wildtype  |  2 hpf bins, {N_SPLITS}-fold CV, "
        f"{N_PERM} perm",
        PLOT_DIR / "genotype_emergence_emb.png",
    )
    _figure(
        pheno_scores, PHENO_COLORS,
        "Homozygous phenotype emergence vs wildtype (emb latents)\n"
        f"per phenotype class vs wildtype  |  2 hpf bins, {N_SPLITS}-fold CV, {N_PERM} perm",
        PLOT_DIR / "phenotype_emergence_emb.png",
    )


if __name__ == "__main__":
    main()
