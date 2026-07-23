"""CEP290 pair_2 vs pair_3 curvature severity, overlaid by lineage within phenotype.

Motivation: in the 2026 F1 data the HtL/LtH *proportions* look similar across pairs (~55/45 in
both pair_2 and pair_3), but the curvature *severity* may differ. These two figures check whether
severity tracks lineage, in both the original 2025 collections and the 2026 F1 offspring.

Two figures, same layout (columns = phenotype, color = pair / lineage), restricted to 40-90 hpf
so OG and F1 are directly comparable:

- OG (2025): ``cep290_pair_2`` / ``cep290_pair_3`` from experiments 20251106 + 20251113. These
  carry *curated* phenotype labels in ``reference_cep290_clean.csv`` -- ground truth, no transfer.
  Compositions are near-opposite (pair_2 -> 46 HtL/10 LtH; pair_3 -> 13 HtL/41 LtH).
- F1 offspring (2026): ``cep290_pair_2_F1s`` / ``cep290_pair_3_F1s``. No curated labels, so the
  HtL/LtH calls come from the label-transfer model via ``07_...load_data`` -- the same filtering
  and transfer path as every other F1 figure here. ``cep290_spawn`` is excluded (not a lineage).

The model has only two classes (High_to_Low, Low_to_High), so no Not-Penetrant column is possible
for the F1s; the OG curated Not-Penetrant embryos are also dropped to keep the two views parallel.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time  # noqa: E402
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402

REFERENCE_CSV = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_cep290_clean.csv"
)
BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"

OG_EXPERIMENTS = ["20251106", "20251113"]
OG_PAIR_ORDER = ["cep290_pair_2", "cep290_pair_3"]
F1_PAIR_ORDER = ["cep290_pair_2_F1s", "cep290_pair_3_F1s"]
PHENOTYPE_ORDER = ["High_to_Low", "Low_to_High", "Not Penetrant"]
FEATURE = "baseline_deviation_normalized"
STAGE_WINDOW = (40.0, 90.0)

# 3-class NP model (script 11). Used to give the F1 side an honest Not-Penetrant call so the
# HtL/LtH severity trends are no longer polluted by force-labeled NP embryos.
MODEL_PATH = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/"
    "cep290_homozygous_phenotype_with_np.pkl"
)
# Only 20260210 reaches late enough stages for the F1 3-way split to be real; 20260208/20260219
# stop at ~47 hpf and would classify as all-NP purely from stage cutoff.
F1_EXPERIMENTS = ["20260210"]

# Blue = pair_2, red = pair_3 (distinct from the phenotype pink/teal palette).
OG_PAIR_COLORS = {"cep290_pair_2": "#4C78A8", "cep290_pair_3": "#E45756"}
F1_PAIR_COLORS = {"cep290_pair_2_F1s": "#4C78A8", "cep290_pair_3_F1s": "#E45756"}

OUTPUT_DIR = RUN_DIR / "figures" / "cep290_pair_severity_overlay"


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_og_data() -> pd.DataFrame:
    """OG pair_2/pair_3 curvature trajectories tagged with curated phenotype labels."""
    ref = pd.read_csv(
        REFERENCE_CSV,
        usecols=["embryo_id", "pair", "phenotype_clean", "experiment_id", "use_embryo_flag"],
        low_memory=False,
    )
    ref = ref[_qc_mask(ref["use_embryo_flag"])].drop_duplicates("embryo_id")
    # Keep all three curated classes now (HtL / LtH / Not Penetrant).
    ref = ref[
        ref["experiment_id"].astype(str).isin(OG_EXPERIMENTS)
        & ref["pair"].isin(OG_PAIR_ORDER)
        & ref["phenotype_clean"].isin(PHENOTYPE_ORDER)
    ]
    labels = ref.set_index("embryo_id")[["pair", "phenotype_clean"]]

    frames = []
    for experiment in OG_EXPERIMENTS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        frames.append(pd.read_csv(
            path,
            usecols=["embryo_id", "predicted_stage_hpf", FEATURE, "use_embryo_flag"],
            low_memory=False,
        ))
    traj = pd.concat(frames, ignore_index=True)
    traj = traj[_qc_mask(traj["use_embryo_flag"])].copy()
    traj["predicted_stage_hpf"] = pd.to_numeric(traj["predicted_stage_hpf"], errors="coerce")
    traj[FEATURE] = pd.to_numeric(traj[FEATURE], errors="coerce")

    df = traj.join(labels, on="embryo_id", how="inner")
    return df.dropna(subset=["predicted_stage_hpf", FEATURE])


def _normalize_genotype(genotype: object) -> str:
    value = str(genotype).strip().lower().replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.replace("cep290_unkown", "cep290_unknown").replace(
        "cep290_homozyous", "cep290_homozygous"
    )


def load_f1_data() -> pd.DataFrame:
    """F1 pair_2/pair_3 homozygote curvature, tagged with 3-class transferred labels.

    Restricted to F1_EXPERIMENTS (20260210) so the Not-Penetrant column reflects a real 3-way
    split rather than the stage-cutoff artifact seen in the ~47 hpf experiments.
    """
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    feature_cols = list(model["config"]["feature_cols"])
    base_cols = list(dict.fromkeys(
        ["embryo_id", "pair", "genotype", "predicted_stage_hpf", "use_embryo_flag", FEATURE]
    ))

    frames = []
    for experiment in F1_EXPERIMENTS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        frames.append(pd.read_csv(path, usecols=base_cols + feature_cols, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    df = df[_qc_mask(df["use_embryo_flag"])].copy()
    df["pair"] = df["pair"].astype(str)
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")

    df = df[
        (df["genotype"] == "cep290_homozygous") & df["pair"].isin(F1_PAIR_ORDER)
    ].dropna(subset=["predicted_stage_hpf", *feature_cols])

    cross = transfer_labels_perbin(model, df, verbose=False)["embryo_support"][
        "embryo_cross_bin_prediction"
    ].set_index("query_embryo_id")
    df["phenotype_clean"] = df["embryo_id"].map(cross["predicted_label"])
    return df[df["phenotype_clean"].isin(PHENOTYPE_ORDER)].copy()


def _plot(df: pd.DataFrame, pair_order: list[str], pair_colors: dict[str, str],
          title: str, output_png: Path) -> pd.Series:
    if df.empty:
        raise RuntimeError(f"No embryos to plot for: {title}")

    counts = (
        df.drop_duplicates("embryo_id")
        .groupby(["phenotype_clean", "pair"], observed=True)
        .size()
    )

    # Darker lines than the first pass: higher individual alpha + thicker trend.
    style = update_style(
        presentation_style(),
        height_per_row=380,
        width_per_col=430,
        min_width=1200,
        individual_alpha=0.30,
        individual_width=1.0,
        trend_width=4.0,
        axis_label_fontsize=13,
        legend_fontsize=12,
    )

    fig = plot_feature_over_time(
        df,
        features=FEATURE,
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="pair",
        color_lookup=pair_colors,
        facet_col="phenotype_clean",
        layout=FacetSpec(col_order=PHENOTYPE_ORDER, sharex=True, sharey=True),
        xlim=STAGE_WINDOW,
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=title,
        style=style,
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=False,
    )
    for col_index, ax in enumerate(fig.axes[: len(PHENOTYPE_ORDER)]):
        phenotype = PHENOTYPE_ORDER[col_index]
        lines = [
            f"pair_{pair.split('_')[2] if '_F1s' in pair else pair.rsplit('_', 1)[-1]}:"
            f" n={int(counts.get((phenotype, pair), 0))}"
            for pair in pair_order
        ]
        ax.text(
            0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7,
                  "edgecolor": "none"},
        )
        ax.set_xlabel("Hours post fertilization")
        ax.set_ylabel("Curvature (normalized)" if col_index == 0 else "")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return counts


def main() -> None:
    og_counts = _plot(
        load_og_data(),
        OG_PAIR_ORDER,
        OG_PAIR_COLORS,
        "OG CEP290 (2025) curvature severity: pair_2 vs pair_3 within phenotype",
        OUTPUT_DIR / "og_curvature_by_phenotype_colored_by_pair.png",
    )
    f1_counts = _plot(
        load_f1_data(),
        F1_PAIR_ORDER,
        F1_PAIR_COLORS,
        "F1 offspring CEP290 curvature severity: pair_2 vs pair_3 within phenotype",
        OUTPUT_DIR / "f1_curvature_by_phenotype_colored_by_pair.png",
    )
    print(f"Saved figures under: {OUTPUT_DIR}")
    print("\nOG (2025):")
    print(og_counts.rename("n_embryos").reset_index().to_string(index=False))
    print("\nF1 offspring (2026):")
    print(f1_counts.rename("n_embryos").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
