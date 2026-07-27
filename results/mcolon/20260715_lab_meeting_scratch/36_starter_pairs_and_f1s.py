"""Starter pairs and their F1 offspring: phenotype distribution + trajectories.

Two independent figures, both colored by the DERIVED (model-predicted) phenotype:

  Figure 1  starter pairs  cep290_pair_1 / pair_2 / pair_3   (2025 experiments)
            1a  predicted phenotype distribution, counts labelled
            1b  curvature + length over time, with AND without error bands

  Figure 2  F1 offspring   cep290_pair_2_F1s / cep290_pair_3_F1s   (experiment 20260210)
            2a  predicted phenotype distribution
            2b  curvature + length over time, with AND without error bands

Generation structure (confirmed): the bare `cep290_pair_N` are the starter generation and live
in the reference table; only pairs 2 and 3 have offspring, which are separate `*_F1s` pairs in
the 2026 Build06 experiments. The F1s have NO curated labels, so derived labels are the only
option there -- which is also why both figures use derived labels, to keep them comparable.

F1 sample sizes are small (pair_3 F1 cells reach n=4), so counts are annotated everywhere and
the trend for a tiny cell should not be over-read.

Only 20260210 is used for the F1s: the other 2026 experiments stop near ~47 hpf, where a 3-way
split would be a stage-cutoff artifact rather than a real phenotype call (see script 09).

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/36_starter_pairs_and_f1s.py
"""

from __future__ import annotations

import pickle
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

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style, update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402
from analyze.viz.plotting import plot_proportions  # noqa: E402
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402

REFERENCE_CSV = (PROJECT_ROOT /
                 "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_cep290_clean.csv")
BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
MODEL_PATH = (PROJECT_ROOT /
              "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/"
              "cep290_homozygous_phenotype_with_np.pkl")

OUTPUT_DIR = RUN_DIR / "figures" / "starter_pairs_and_f1s"

STARTER_PAIRS = ["cep290_pair_1", "cep290_pair_2", "cep290_pair_3"]
F1_PAIRS = ["cep290_pair_2_F1s", "cep290_pair_3_F1s"]
F1_EXPERIMENTS = ["20260210"]

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]

PHENOTYPE_ORDER = ["High_to_Low", "Low_to_High", "Not Penetrant"]
PHENO_COLORS = {"High_to_Low": "#E76FA2", "Low_to_High": "#2FB7B0",
                "Not Penetrant": "#9AA0A6"}


def _qc(v: pd.Series) -> pd.Series:
    if v.dtype == bool:
        return v.fillna(False)
    return v.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _normalize_genotype(g: object) -> str:
    v = str(g).strip().lower().replace(" ", "_")
    while "__" in v:
        v = v.replace("__", "_")
    return v.replace("cep290_unkown", "cep290_unknown").replace(
        "cep290_homozyous", "cep290_homozygous")


def _model():
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


def _derive(df: pd.DataFrame, model) -> pd.DataFrame:
    """Attach the transferred 3-class label to every embryo in df."""
    cross = transfer_labels_perbin(model, df, verbose=False)["embryo_support"][
        "embryo_cross_bin_prediction"].set_index("query_embryo_id")
    df = df.copy()
    df["phenotype"] = df[ID_COL].map(cross["predicted_label"])
    return df[df["phenotype"].isin(PHENOTYPE_ORDER)]


def load_starters() -> pd.DataFrame:
    """Starter-pair homozygotes with CURATED labels.

    The starters are the generation that actually has hand-curated calls, so use them as
    ground truth here. (Derived labels differ by only one embryo in pair_1 and pair_2, and
    not at all in pair_3 -- but curated is the honest source when it exists.) The F1s have no
    curated labels, so that figure necessarily stays on derived.
    """
    df = pd.read_csv(
        REFERENCE_CSV,
        usecols=lambda c: c in {ID_COL, TIME_COL, "pair", "zygosity", "use_embryo_flag",
                                "phenotype_clean", *FEATURES},
        low_memory=False)
    df = df[_qc(df["use_embryo_flag"])] if "use_embryo_flag" in df else df
    df = df[df["pair"].isin(STARTER_PAIRS) & (df["zygosity"] == "homozygous")].copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *FEATURES])
    df["phenotype"] = df["phenotype_clean"]
    return df[df["phenotype"].isin(PHENOTYPE_ORDER)]


def load_f1s() -> pd.DataFrame:
    """F1 offspring homozygotes; no curated labels exist, so derived is the only option."""
    model = _model()
    feats = list(model["config"]["feature_cols"])
    base = list(dict.fromkeys([ID_COL, "pair", "genotype", TIME_COL,
                               "use_embryo_flag", *FEATURES]))
    frames = [pd.read_csv(BUILD06_DIR / f"df03_final_output_with_latents_{e}.csv",
                          usecols=base + feats, low_memory=False)
              for e in F1_EXPERIMENTS]
    df = pd.concat(frames, ignore_index=True)
    df = df[_qc(df["use_embryo_flag"])].copy()
    df["pair"] = df["pair"].astype(str)
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df[(df["genotype"] == "cep290_homozygous") & df["pair"].isin(F1_PAIRS)]
    df = df.dropna(subset=[TIME_COL, *FEATURES, *feats])
    return _derive(df, model)


def plot_distribution(df: pd.DataFrame, pairs: list[str], title: str, out: Path) -> None:
    """Predicted phenotype composition per pair, via the package's plot_proportions."""
    emb = df.drop_duplicates(ID_COL)
    counts = (emb.groupby(["pair", "phenotype"]).size().unstack("phenotype")
                 .reindex(index=pairs, columns=PHENOTYPE_ORDER).fillna(0).astype(int))
    print(f"\n{title}")
    print(counts.assign(total=counts.sum(axis=1)).to_string())

    out.parent.mkdir(parents=True, exist_ok=True)
    plot_proportions(
        emb,
        color_by_grouping="phenotype",
        col_by="pair",
        count_by=ID_COL,
        facet_order={"pair": pairs},
        color_order=PHENOTYPE_ORDER,
        color_palette=PHENO_COLORS,
        normalize=True,
        bar_mode="grouped",
        title=title,
        show_counts=True,
        output_path=out,
    )
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def plot_trajectories(df: pd.DataFrame, pairs: list[str], title: str, out: Path,
                      *, bands: bool) -> None:
    """Curvature + length over time, cols = pair, colored by derived phenotype."""
    present = [p for p in PHENOTYPE_ORDER if p in set(df["phenotype"])]
    style = update_style(
        presentation_style(),
        height_per_row=280, width_per_col=300, min_width=760,
        individual_alpha=0.22, individual_width=0.7, trend_width=3.2,
        axis_label_fontsize=11, legend_fontsize=9,
    )
    fig = plot_feature_over_time(
        df,
        features=FEATURES,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="phenotype",
        color_preset=ColorPreset(colors=PHENO_COLORS, order=present),
        facet_col="pair",
        # Every column is the same feature across pairs, so the panels MUST share y to be
        # comparable. (Rows are different features and are scaled independently by the engine.)
        layout=FacetSpec(col_order=pairs, sharex=True, sharey=True),
        show_individual=not bands,      # bands view drops the individual spaghetti
        show_trend=True,
        show_error_band=bands,
        trend_statistic="median",
        # Bands over small per-pair groups are jagged at 3 hpf bins -- widen the bin and smooth
        # harder so the band reads as an envelope rather than per-bin noise.
        trend_smooth_sigma=2.5 if bands else 1.5,
        bin_width=6.0 if bands else 3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 2.0 if bands else 1.0},
        backend="matplotlib",
        title=title,
        style=style,
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=False,           # sharey makes per-cell y-labels redundant clutter
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    emb = df.drop_duplicates(ID_COL)
    note = "   ".join(
        f"{p.replace('cep290_', '')}: "
        + "/".join(f"{c}={int(((emb['pair'] == p) & (emb['phenotype'] == c)).sum())}"
                   for c in present)
        for p in pairs)
    fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#555555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    starters = load_starters()
    f1s = load_f1s()

    # ---- Figure 1: starter pairs ----
    plot_distribution(starters, STARTER_PAIRS,
                      "Starter pairs — curated phenotype distribution (homozygotes)",
                      OUTPUT_DIR / "1a_starter_pairs_distribution.png")
    for bands in (False, True):
        suffix = "bands" if bands else "traces"
        plot_trajectories(
            starters, STARTER_PAIRS,
            f"Starter pairs — curvature & length by curated phenotype ({suffix})",
            OUTPUT_DIR / f"1b_starter_pairs_trajectories__{suffix}.png", bands=bands)

    # ---- Figure 2: F1 offspring ----
    plot_distribution(f1s, F1_PAIRS,
                      "F1 offspring — predicted phenotype distribution (homozygotes)",
                      OUTPUT_DIR / "2a_f1_distribution.png")
    for bands in (False, True):
        suffix = "bands" if bands else "traces"
        plot_trajectories(
            f1s, F1_PAIRS,
            f"F1 offspring — curvature & length by derived phenotype ({suffix})",
            OUTPUT_DIR / f"2b_f1_trajectories__{suffix}.png", bands=bands)


if __name__ == "__main__":
    main()
