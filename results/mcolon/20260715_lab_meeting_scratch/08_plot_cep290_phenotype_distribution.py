"""CEP290 predicted-phenotype distribution: OG (2025) vs F1 offspring (2026), by pair.

Story panel for the lab meeting. Two claims sit side by side with the severity overlay
(``09_plot_cep290_pair_severity_overlay.py``):

  1. The phenotype *distribution* did NOT translate from the parents to the F1 offspring.
     OG pair_2 is HtL-dominant (46/10) and pair_3 is LtH-dominant (13/41) -- near-opposite.
     In the F1 offspring both pairs collapse to ~55/45, i.e. the split converged.
  2. The *severity* (curvature magnitude within a phenotype) DID translate -- see script 09.

Layout: rows = generation (OG 2025 / F1 2026), cols = pair (pair_2 / pair_3). Bars are the two
predicted phenotypes as a percentage within each pair-generation cell, with the raw embryo count
as a bold label -- matching the b9d2 reference figure style (black-outlined bars, bold counts,
percent axis). ``cep290_spawn`` is excluded: it is not a pair_2/pair_3 lineage.

OG labels are curated ground truth (``reference_cep290_clean.csv``); F1 labels are model-transferred
via ``07_...load_data``. OG has ~2x the labeled homozygotes of the F1s, so bars are normalized to
percent (counts differ, proportions are the comparison).
"""

from __future__ import annotations

import pickle
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

from analyze.viz.styling import CEP290_PHENOTYPE_COLORS  # noqa: E402
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402

REFERENCE_CSV = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_cep290_clean.csv"
)
BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
MODEL_PATH = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/"
    "cep290_homozygous_phenotype_with_np.pkl"
)
OG_EXPERIMENTS = ["20251106", "20251113"]
# Only the F1 experiment that reaches late stages; the ~47 hpf experiments classify as all-NP
# from stage cutoff, so including them would put spurious NP bars on the F1 row.
F1_EXPERIMENTS = ["20260210"]

PHENOTYPE_ORDER = ["High_to_Low", "Low_to_High", "Not Penetrant"]
# (generation label, pair value in that generation's data) for each column.
PAIR_COLUMNS = [("pair_2", "cep290_pair_2", "cep290_pair_2_F1s"),
                ("pair_3", "cep290_pair_3", "cep290_pair_3_F1s")]
ROW_ORDER = ["OG (2025)", "F1 offspring (2026, 20260210)"]

OUTPUT_DIR = RUN_DIR / "figures" / "cep290_phenotype_distribution"
OUTPUT_PNG = OUTPUT_DIR / "phenotype_distribution_og_vs_f1_by_pair.png"


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _normalize_genotype(genotype: object) -> str:
    value = str(genotype).strip().lower().replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.replace("cep290_unkown", "cep290_unknown").replace(
        "cep290_homozyous", "cep290_homozygous"
    )


def _og_counts() -> pd.DataFrame:
    """Per-embryo curated 3-class labels for OG pair_2/pair_3."""
    ref = pd.read_csv(
        REFERENCE_CSV,
        usecols=["embryo_id", "pair", "phenotype_clean", "experiment_id",
                 "use_embryo_flag", "zygosity"],
        low_memory=False,
    )
    ref = ref[_qc_mask(ref["use_embryo_flag"])].drop_duplicates("embryo_id")
    ref = ref[
        ref["experiment_id"].astype(str).isin(OG_EXPERIMENTS)
        & ref["pair"].isin([p[1] for p in PAIR_COLUMNS])
        & ref["zygosity"].eq("homozygous")
        & ref["phenotype_clean"].isin(PHENOTYPE_ORDER)
    ]
    return ref[["embryo_id", "pair", "phenotype_clean"]]


def _f1_counts() -> pd.DataFrame:
    """Per-embryo 3-class transferred labels for F1 pair_2/pair_3, 20260210 only."""
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    feature_cols = list(model["config"]["feature_cols"])
    base_cols = ["embryo_id", "pair", "genotype", "predicted_stage_hpf", "use_embryo_flag"]

    frames = []
    for experiment in F1_EXPERIMENTS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        frames.append(pd.read_csv(path, usecols=base_cols + feature_cols, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    df = df[_qc_mask(df["use_embryo_flag"])].copy()
    df["pair"] = df["pair"].astype(str)
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df = df[
        (df["genotype"] == "cep290_homozygous")
        & df["pair"].isin([p[2] for p in PAIR_COLUMNS])
    ].dropna(subset=["predicted_stage_hpf", *feature_cols])

    cross = transfer_labels_perbin(model, df, verbose=False)["embryo_support"][
        "embryo_cross_bin_prediction"
    ].set_index("query_embryo_id")
    df = df.drop_duplicates("embryo_id").copy()
    df["phenotype_clean"] = df["embryo_id"].map(cross["predicted_label"])
    df = df[df["phenotype_clean"].isin(PHENOTYPE_ORDER)]
    return df[["embryo_id", "pair", "phenotype_clean"]]


def main() -> None:
    og = _og_counts()
    f1 = _f1_counts()
    data = {ROW_ORDER[0]: og, ROW_ORDER[1]: f1}

    fig, axes = plt.subplots(
        len(ROW_ORDER), len(PAIR_COLUMNS),
        figsize=(9.5, 8.0), sharex=True, sharey=True, squeeze=False,
    )
    x_positions = dict(zip(PHENOTYPE_ORDER, (-0.28, 0.0, 0.28)))

    for row_index, generation in enumerate(ROW_ORDER):
        frame = data[generation]
        for col_index, (pair_label, og_pair, f1_pair) in enumerate(PAIR_COLUMNS):
            ax = axes[row_index, col_index]
            pair_value = og_pair if generation.startswith("OG") else f1_pair
            panel = frame[frame["pair"] == pair_value]
            counts = panel.groupby("phenotype_clean", observed=True)["embryo_id"].nunique()
            total = int(counts.sum())

            for phenotype in PHENOTYPE_ORDER:
                count = int(counts.get(phenotype, 0))
                if count == 0 or total == 0:
                    continue
                proportion = count / total
                ax.bar(
                    x_positions[phenotype], proportion, width=0.26,
                    color=CEP290_PHENOTYPE_COLORS[phenotype],
                    edgecolor="black", linewidth=1.2,
                )
                ax.text(
                    x_positions[phenotype], proportion + 0.015, str(count),
                    ha="center", va="bottom", fontsize=15, fontweight="bold",
                )

            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 1.15)
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.18, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=12)
            if row_index == 0:
                ax.set_title(pair_label, fontsize=13, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(generation, fontsize=12, fontweight="bold")
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    fig.suptitle(
        "CEP290 homozygotes: predicted phenotype distribution did not translate to F1 offspring",
        fontsize=14, fontweight="bold", y=0.98,
    )
    handles = [
        Patch(facecolor=CEP290_PHENOTYPE_COLORS[label], edgecolor="black", label=label)
        for label in PHENOTYPE_ORDER
    ]
    fig.legend(
        handles=handles, title="predicted phenotype", loc="center left",
        bbox_to_anchor=(0.92, 0.5), fontsize=12, title_fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.90, 0.95))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")

    summary = pd.concat([
        og.assign(generation="OG (2025)"), f1.assign(generation="F1 offspring (2026)")
    ])
    print(summary.groupby(["generation", "pair", "phenotype_clean"], observed=True)
          .size().rename("n_embryos").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
