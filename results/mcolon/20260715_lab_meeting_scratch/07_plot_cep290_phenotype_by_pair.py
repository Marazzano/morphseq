"""CEP290 F1-offspring homozygotes: length + curvature split by transferred phenotype.

Remakes the pair-faceted trend figure from
``results/mcolon/20260302_cep290_genotype_pair_trends_20260208_20260210`` with two changes:

1. Homozygotes only. Het / wildtype / unknown are dropped.
2. The single homozygous trend is split into ``High_to_Low`` vs ``Low_to_High`` using the
   per-bin label-transfer model at
   ``results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/cep290_homozygous_phenotype.pkl``.

Naming note: ``pair`` identifies the *parents*, so ``cep290_pair_2_F1s`` means "F1 offspring of
pair 2". The imaged embryos are F1s, not F2s. Titles say "F1 offspring" to remove the ambiguity.

Provenance note: the reference set backing the model is 2025 experiments only, so predictions on
these 2026 experiments are genuine out-of-sample transfer, not re-prediction of training data.

Small-n caveat: 20260210 carries the result (30 homozygotes). 20260208 (7) and 20260219 (5) yield
1-5 embryos per phenotype facet, and 20260219 has zero High_to_Low. Faceting by experiment keeps
that visible rather than diluting a pooled trend; stage spans also barely overlap
(20260210 is 42-88 hpf, the others start near 10 hpf), so pooling would stitch a batch seam.
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
from analyze.viz.styling import CEP290_PHENOTYPE_COLORS  # noqa: E402
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402


BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
MODEL_PATH = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/cep290_homozygous_phenotype.pkl"
)
OUTPUT_DIR = RUN_DIR / "figures" / "cep290_phenotype_by_pair"
RESULTS_DIR = RUN_DIR / "results"

EXPERIMENTS = ["20260208", "20260210", "20260219"]
PHENOTYPE_ORDER = ["High_to_Low", "Low_to_High"]
FEATURES = ["total_length_um", "baseline_deviation_normalized"]
FEATURE_LABELS = {
    "total_length_um": "Total length (um)",
    "baseline_deviation_normalized": "Curvature (normalized)",
}

# Homozygote counts verified during planning. Guards against the ``cep290_homozyous``
# spelling regression silently shrinking the figure.
EXPECTED_HOMOZYGOTES = {"20260208": 7, "20260210": 30, "20260219": 5}


def _normalize_genotype(genotype: object) -> str:
    """Repair known genotype spellings in Build06 output.

    Ported from ``20260302_cep290_genotype_pair_trends_20260208_20260210/
    01_plot_cep290_genotype_pair_trends.py``. 20260210 spells the genotype
    ``cep290_homozyous`` (missing the "g") while the other experiments spell it correctly, so a
    plain ``endswith("_homozygous")`` filter drops all 30 good homozygotes and keeps only the 12
    weak ones. The source .xlsx is being corrected separately, but Build06 CSVs are regenerated
    output and still carry the typo until the next rebuild, so this stays as a safety net.
    """
    value = str(genotype).strip().lower().replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    value = value.replace("cep290_unkown", "cep290_unknown")
    value = value.replace("cep290_homozyous", "cep290_homozygous")
    return value


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load homozygotes across experiments and attach transferred phenotype labels."""
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    feature_cols = list(model["config"]["feature_cols"])

    base_cols = list(dict.fromkeys(
        ["embryo_id", "experiment_id", "pair", "genotype", "predicted_stage_hpf",
         "use_embryo_flag", *FEATURES]
    ))

    frames = []
    for experiment in EXPERIMENTS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        header = pd.read_csv(path, nrows=0).columns
        missing = sorted(set(base_cols + feature_cols) - set(header))
        if missing:
            raise RuntimeError(f"{path.name} is missing required columns: {missing}")
        frame = pd.read_csv(path, usecols=base_cols + feature_cols, low_memory=False)
        frame["experiment_id"] = experiment
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df[_qc_mask(df["use_embryo_flag"])].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["pair"] = df["pair"].astype(str)
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df[df["genotype"] == "cep290_homozygous"].copy()
    df = df.dropna(subset=["predicted_stage_hpf", *feature_cols])

    observed = df.groupby("experiment_id")["embryo_id"].nunique().to_dict()
    for experiment, expected in EXPECTED_HOMOZYGOTES.items():
        actual = observed.get(experiment, 0)
        if actual != expected:
            raise RuntimeError(
                f"{experiment}: expected {expected} homozygotes, found {actual}. "
                "Genotype spelling or QC flags changed -- check _normalize_genotype()."
            )

    # Transfer per experiment so per-experiment support diagnostics stay separable.
    predictions = []
    for experiment, group in df.groupby("experiment_id", sort=True):
        cross_bin = transfer_labels_perbin(model, group, verbose=False)["embryo_support"][
            "embryo_cross_bin_prediction"
        ]
        cross_bin = cross_bin.copy()
        cross_bin["experiment_id"] = experiment
        predictions.append(cross_bin)
    prediction = pd.concat(predictions, ignore_index=True)

    indexed = prediction.set_index("query_embryo_id")
    df["phenotype_clean"] = df["embryo_id"].map(indexed["predicted_label"])
    # Confidence is only ~0.65-0.71 median, so carry it through rather than presenting hard
    # labels as certain.
    df["top_probability"] = df["embryo_id"].map(indexed["top_probability"])
    df["n_bins_contributed"] = df["embryo_id"].map(indexed["n_bins_contributed"])
    df = df[df["phenotype_clean"].isin(PHENOTYPE_ORDER)].copy()

    calls = (
        df.drop_duplicates("embryo_id")[
            ["embryo_id", "experiment_id", "pair", "phenotype_clean",
             "top_probability", "n_bins_contributed"]
        ]
        .sort_values(["experiment_id", "pair", "phenotype_clean", "embryo_id"])
        .reset_index(drop=True)
    )

    pair_order = sorted(df["pair"].dropna().unique().tolist())
    return df, calls, pair_order


def _annotate_counts(fig, df: pd.DataFrame, row_order: list[str], col_order: list[str]) -> None:
    """Write per-facet embryo n onto each axis so thin facets read as thin."""
    counts = (
        df.drop_duplicates("embryo_id")
        .groupby(["experiment_id", "pair", "phenotype_clean"], observed=True)
        .size()
    )
    n_cols = len(col_order)
    for axis_index, ax in enumerate(fig.axes):
        row_index, col_index = divmod(axis_index, n_cols)
        if row_index >= len(row_order) or col_index >= n_cols:
            continue
        experiment, pair = row_order[row_index], col_order[col_index]
        parts = [
            f"{label.replace('_to_', '->')}: n={int(counts.get((experiment, pair, label), 0))}"
            for label in PHENOTYPE_ORDER
        ]
        ax.text(
            0.02, 0.97, "\n".join(parts),
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7,
                  "edgecolor": "none"},
        )


def main() -> None:
    df, calls, pair_order = load_data()
    if df.empty:
        raise RuntimeError("No CEP290 homozygotes received a High_to_Low/Low_to_High prediction.")

    experiment_order = sorted(df["experiment_id"].unique())
    colors = {label: CEP290_PHENOTYPE_COLORS[label] for label in PHENOTYPE_ORDER}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    style = update_style(
        presentation_style(),
        height_per_row=330,
        width_per_col=320,
        min_width=1400,
        individual_alpha=0.18,
        individual_width=0.75,
        trend_width=3.0,
        axis_label_fontsize=13,
        legend_fontsize=12,
    )

    for feature in FEATURES:
        fig = plot_feature_over_time(
            df,
            features=feature,
            time_col="predicted_stage_hpf",
            id_col="embryo_id",
            color_by="phenotype_clean",
            color_lookup=colors,
            facet_row="experiment_id",
            facet_col="pair",
            layout=FacetSpec(
                row_order=experiment_order,
                col_order=pair_order,
                sharex=True,
                sharey=True,
            ),
            show_individual=True,
            show_trend=True,
            show_error_band=True,
            trend_statistic="median",
            trend_smooth_sigma=1.5,
            bin_width=3.0,
            smooth_method="gaussian",
            smooth_params={"sigma": 1.0},
            backend="matplotlib",
            title=(
                f"CEP290 F1 offspring, homozygotes only: {FEATURE_LABELS[feature]}"
                " by predicted phenotype (label transfer)"
            ),
            style=style,
            legend_loc="outside",
            repeat_xlabels=False,
            repeat_ylabels=False,
            repeat_xticklabels=True,
            repeat_yticklabels=True,
        )
        for axis_index, ax in enumerate(fig.axes):
            row_index, col_index = divmod(axis_index, len(pair_order))
            ax.set_xlabel(
                "Hours post fertilization" if row_index == len(experiment_order) - 1 else ""
            )
            ax.set_ylabel(FEATURE_LABELS[feature] if col_index == 0 else "")
        _annotate_counts(fig, df, experiment_order, pair_order)

        output_path = OUTPUT_DIR / f"{feature}_by_pair_and_phenotype.png"
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    calls_path = RESULTS_DIR / "cep290_phenotype_transfer_calls.csv"
    calls.to_csv(calls_path, index=False)
    print(f"Saved: {calls_path}")

    summary = (
        calls.groupby(["experiment_id", "pair", "phenotype_clean"], observed=True)
        .agg(n_embryos=("embryo_id", "size"), median_top_prob=("top_probability", "median"))
        .reset_index()
    )
    print("\nPer-facet phenotype calls:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
