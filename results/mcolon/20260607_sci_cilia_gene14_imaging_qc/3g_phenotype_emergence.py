"""
3g - Phenotype emergence: when do homozygous phenotypes become separable from wildtype?

For each gene group (cep290, b9d2, crispant), runs cross-validated AUROC classification
of each phenotype/genotype class vs. wildtype reference, using run_classification with
positive=[phenotype_classes], negative=<wt_label>.

Produces a 4-row × 3-column figure:
    rows = feature sets: emb | curvature | length | CURV + length
    cols = genes:        cep290 | b9d2 | crispant

Label conventions follow 0_load_and_clean_datasets.py:
    cep290  : phenotype_clean in {High_to_Low, Low_to_High}  (homo only), WT = wildtype
    b9d2    : phenotype_clean in {CE, HTA}                   (homo only), WT = wildtype
    crispant: genotype_clean in {foxj1a_crispant, ift88_crispant, sspo_crispant},
              negative = injection_control

Output:
    plots/separability/phenotype_emergence.png

Run:
    PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output \\
        python results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3g_phenotype_emergence.py

Note: set N_PERM=500 for the final run; N_PERM=0 for a fast smoke-test (no null band).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))   # for the sibling plot_config module

from analyze.classification import run_classification
from analyze.classification.viz import plot_aurocs_over_time
from analyze.viz.plotting.faceting_engine import FacetSpec
from analyze.viz.plotting.faceting_engine.style.defaults import StyleSpec
from plot_config import PHENOTYPE_COLORS, GENOTYPE_COLORS  # single source of truth for colors

TABLE_DIR = RUN_DIR / "tables"
CLASS_DIR = RUN_DIR / "classification"
PLOT_DIR  = RUN_DIR / "plots" / "separability"
CLASS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL   = "embryo_id"
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 2.0
N_SPLITS  = 5
N_PERM    = 500      # final run: full-resolution significance + null bands
MAX_HPF   = 48.0

FEATURES = {
    "emb":          "z_mu_b",
    "curvature":    ["baseline_deviation_normalized"],
    "length":       ["total_length_um"],
    "CURV + length": ["baseline_deviation_normalized", "total_length_um"],
}

FEATURE_ORDER = ["emb", "curvature", "length", "CURV + length"]
GENE_ORDER    = ["cep290", "b9d2", "crispant"]

# Colors sourced from plot_config (the single source of truth for the 3x figures):
#   cep290 phenotype pink/teal, b9d2 phenotype green/orange, crispant genotype colors.
CEP290_COLORS = {k: PHENOTYPE_COLORS[k] for k in ("High_to_Low", "Low_to_High")}
B9D2_PHENOTYPE_COLORS = {k: PHENOTYPE_COLORS[k] for k in ("CE", "HTA")}
CRISPANT_COLORS = {
    k: GENOTYPE_COLORS[k]
    for k in ("foxj1a_crispant", "ift88_crispant", "sspo_crispant")
}

ALL_COLORS = {
    "cep290":   CEP290_COLORS,
    "b9d2":     B9D2_PHENOTYPE_COLORS,
    "crispant": CRISPANT_COLORS,
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_cep290() -> pd.DataFrame:
    df = pd.read_csv(
        TABLE_DIR / "reference_cep290_clean.csv",
        usecols=lambda c: c in {ID_COL, TIME_COL, "zygosity", "phenotype_clean",
                                 "baseline_deviation_normalized", "total_length_um"}
                          or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    df = df[df["zygosity"].isin(["homozygous", "wildtype"])].copy()
    df = df.dropna(subset=[TIME_COL, "baseline_deviation_normalized", "total_length_um"])
    df = df[df[TIME_COL] <= MAX_HPF].copy()
    # Build plot_label: homos use phenotype_clean; WT rows get "wildtype"
    df["plot_label"] = df["phenotype_clean"].where(df["zygosity"] == "homozygous", "wildtype")
    # Keep only the two main phenotype classes + wildtype
    keep = set(CEP290_COLORS) | {"wildtype"}
    return df[df["plot_label"].isin(keep)].copy()


def load_b9d2() -> pd.DataFrame:
    df = pd.read_csv(
        TABLE_DIR / "reference_b9d2_clean.csv",
        usecols=lambda c: c in {ID_COL, TIME_COL, "zygosity", "phenotype_clean",
                                 "baseline_deviation_normalized", "total_length_um"}
                          or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    df = df[df["zygosity"].isin(["homozygous", "wildtype"])].copy()
    df = df.dropna(subset=[TIME_COL, "baseline_deviation_normalized", "total_length_um"])
    df = df[df[TIME_COL] <= MAX_HPF].copy()
    df["plot_label"] = df["phenotype_clean"].where(df["zygosity"] == "homozygous", "wildtype")
    keep = set(B9D2_PHENOTYPE_COLORS) | {"wildtype"}
    return df[df["plot_label"].isin(keep)].copy()


def load_crispant() -> pd.DataFrame:
    # injection_control is stored as ab_wildtype in this reference table
    crispant_genotypes = set(CRISPANT_COLORS) | {"ab_wildtype"}
    df = pd.read_csv(
        TABLE_DIR / "reference_crispant_clean.csv",
        usecols=lambda c: c in {ID_COL, TIME_COL, "genotype_clean",
                                 "baseline_deviation_normalized", "total_length_um"}
                          or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    df = df[df["genotype_clean"].isin(crispant_genotypes)].copy()
    df = df.dropna(subset=[TIME_COL, "baseline_deviation_normalized", "total_length_um"])
    df = df[df[TIME_COL] <= MAX_HPF].copy()
    return df.copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("3g - phenotype emergence (4×3 AUROC grid)")
print(f"  N_PERM={N_PERM}  (set to 500 for final run with null bands)")

GENE_SPEC = {
    "cep290": {
        "load":      load_cep290,
        "class_col": "plot_label",
        "positive":  list(CEP290_COLORS),
        "negative":  "wildtype",
    },
    "b9d2": {
        "load":      load_b9d2,
        "class_col": "plot_label",
        "positive":  list(B9D2_PHENOTYPE_COLORS),
        "negative":  "wildtype",
    },
    "crispant": {
        "load":      load_crispant,
        "class_col": "genotype_clean",
        "positive":  list(CRISPANT_COLORS),
        "negative":  "ab_wildtype",
    },
}

# Run classification for each gene and collect scores
all_scores = []
for gene, spec in GENE_SPEC.items():
    print(f"\n[{gene}]")
    df = spec["load"]()
    vc = df.drop_duplicates(ID_COL)[spec["class_col"]].value_counts().to_dict()
    print(f"  embryos: {df[ID_COL].nunique()} | labels: {vc}")

    result = run_classification(
        df,
        class_col=spec["class_col"],
        id_col=ID_COL,
        time_col=TIME_COL,
        positive=spec["positive"],
        negative=spec["negative"],
        features=FEATURES,
        bin_width=BIN_WIDTH,
        n_splits=N_SPLITS,
        n_permutations=N_PERM,
        n_jobs=-1,
        save_dir=str(CLASS_DIR / f"{gene}_emergence"),
        overwrite=True,
        verbose=False,
    )

    scores = result.scores.copy()
    scores["gene"] = gene
    all_scores.append(scores)
    print(f"  {len(scores)} score rows, bins: {sorted(scores['time_bin_center'].unique())[:5]}...")

combined = pd.concat(all_scores, ignore_index=True)

# Build combined color lookup across all genes
combined_colors: dict[str, str] = {}
for gene_colors in ALL_COLORS.values():
    combined_colors.update(gene_colors)

# 4×3 grid: facet_row=feature_set, facet_col=gene
print("\nGenerating 4×3 emergence plot...")
out = PLOT_DIR / "phenotype_emergence.png"

fig = plot_aurocs_over_time(
    combined,
    curve_col="positive_label",
    facet_row="feature_set",
    facet_col="gene",
    layout=FacetSpec(
        row_order=FEATURE_ORDER,
        col_order=GENE_ORDER,
        sharex=True,
        sharey=True,
    ),
    color_lookup=combined_colors,
    show_null_band=(N_PERM > 0),
    show_significance=(N_PERM > 0),
    sig_threshold=0.05,
    show_chance_line=True,
    title="Homozygous phenotype emergence vs wildtype\n"
          f"positive=[phenotype classes], negative=wildtype/injection_control  |  "
          f"2 hpf bins, {N_SPLITS}-fold CV"
          + (f", {N_PERM} perm" if N_PERM > 0 else "  |  no permutations (smoke-test)"),
    x_label="Hours Post Fertilization (hpf)",
    y_label="cross-validated AUROC",
    # 'per-panel' makes the faceting engine draw a compact legend on every populated
    # axis (each panel lists only its own curves + the significance marker).
    style=StyleSpec(legend_fontsize=7, legend_loc="per-panel"),
    backend="matplotlib",
    output_path=None,   # save below so we control dpi + tight bbox
)

fig.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved plots/separability/phenotype_emergence.png")
