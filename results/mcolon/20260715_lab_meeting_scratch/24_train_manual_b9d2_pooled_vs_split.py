"""Train b9d2 phenotype models on the MANUAL labels, two ways, and compare transferred calls.

The open question: is BA_rescue its own class, or should it be pooled into HTA?

    variant "split"   -- CE / HTA / BA_rescue / wildtype   (4 classes, BA_rescue n=7)
    variant "pooled"  -- CE / HTA+BA_rescue / wildtype     (3 classes, matches old convention)

Both are trained on the curated labels from
    results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv
which -- unlike the shipped model -- has CE called in HETEROZYGOTES by eye (n=15), and uses
genuinely curated `wildtype` embryos as the unaffected class instead of relabeling all WT.

Each variant is then transferred onto ALL b9d2 embryos (every zygosity, incl. the 83 'unlabeled')
so the two labelings can be compared on the same embryos. Output per variant:
    - reference CV confusion (does BA_rescue survive as its own class?)
    - multifeature figure: rows = curvature/length, col = PREDICTED phenotype, color = genotype

Read the two figures side by side: if the pooled HTA column looks like a coherent single class,
pooling is fine; if it looks like two superimposed shapes, BA_rescue should stay split.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/24_train_manual_b9d2_pooled_vs_split.py
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

from analyze.classification.label_transfer import (  # noqa: E402
    prepare_reference_perbin,
    transfer_labels_perbin,
)
from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402

SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "manual_b9d2_pooled_vs_split"
MODEL_DIR = RUN_DIR / "models"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
LABEL_COL = "cluster_categories"
CV_GROUP_COL = "experiment_id"
BIN_WIDTH = 4.0
FEATURES = ["baseline_deviation_normalized", "total_length_um"]

TRAIN_CLASSES = ["CE", "HTA", "BA_rescue", "wildtype"]

# The curated `wildtype` class is the unaffected baseline. Renamed on output for readability.
WT_CLASS = "wildtype"

GENO_ORDER = ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous", "b9d2_unknown"]
GENO_COLORS = {
    "b9d2_wildtype": "#7F7F7F",
    "b9d2_heterozygous": "#F7B267",
    "b9d2_homozygous": "#B2182B",
    "b9d2_unknown": "#4C9F70",
}
PHENO_COLORS = {
    "CE": "#1B9E77",
    "HTA": "#D95F02",
    "BA_rescue": "#7570B3",
    "wildtype": "#999999",
}

VARIANTS = {
    # name -> mapping applied to the training label column
    "split": {},                       # leave BA_rescue alone
    "pooled": {"BA_rescue": "HTA"},    # fold BA_rescue into HTA
}
VARIANT_COL_ORDER = {
    "split": ["CE", "HTA", "BA_rescue", "wildtype"],
    "pooled": ["CE", "HTA", "wildtype"],
}


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=300,
        width_per_col=340,
        min_width=1100,
        individual_alpha=0.25,
        individual_width=0.7,
        trend_width=3.4,
        axis_label_fontsize=12,
        legend_fontsize=10,
    )


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, low_memory=False)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    feature_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    df = df.dropna(subset=[TIME_COL, *FEATURES, *feature_cols])
    return df, feature_cols


def run_variant(name: str, df: pd.DataFrame,
                feature_cols: list[str]) -> tuple[pd.DataFrame, set]:
    """Train on curated labels (with this variant's pooling), transfer onto all embryos."""
    mapping = VARIANTS[name]

    ref = df[df[LABEL_COL].isin(TRAIN_CLASSES)].copy()
    # Drop curated phenotype calls that contradict the genotype call -- genotype Excel is truth,
    # so a `b9d2_wildtype` embryo labeled CE/HTA is a curation conflict, not a training example.
    conflict = (ref["genotype"] == "b9d2_wildtype") & (ref[LABEL_COL] != WT_CLASS)
    n_conflict = ref.loc[conflict, ID_COL].nunique()
    ref = ref[~conflict].copy()
    ref[LABEL_COL] = ref[LABEL_COL].replace(mapping)

    n_ref = ref[ID_COL].nunique()
    print(f"\n=== variant '{name}' ===")
    print(f"dropped {n_conflict} genotype/phenotype-conflict embryos")
    print(f"reference: {n_ref} embryos, classes: "
          f"{ref.drop_duplicates(ID_COL)[LABEL_COL].value_counts().to_dict()}")

    model = prepare_reference_perbin(
        ref, feature_cols,
        label_col=LABEL_COL, group_col=ID_COL, time_col=TIME_COL,
        bin_width=BIN_WIDTH, cv_mode="auto", cv_group_col=CV_GROUP_COL,
        verbose=False,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / f"b9d2_manual_{name}.pkl").open("wb") as fh:
        pickle.dump(model, fh)

    perf = model["reference_performance"]
    print("held-out embryo-support recall:",
          {k: round(v, 2) for k, v in perf["embryo_support_recall"].items()})
    print("held-out embryo-support precision:",
          {k: round(v, 2) for k, v in perf["embryo_support_precision"].items()})
    conf = perf["embryo_support_confusion"]
    print("confusion (true rows x pred cols), labels =", conf["labels"])
    print(pd.DataFrame(conf["matrix"], index=conf["labels"],
                       columns=conf["labels"]).round(2).to_string())

    # Transfer onto EVERY embryo, including the 83 'unlabeled'.
    cross = transfer_labels_perbin(model, df, verbose=False)["embryo_support"][
        "embryo_cross_bin_prediction"
    ].set_index("query_embryo_id")
    out = df.copy()
    out["predicted_phenotype"] = out[ID_COL].map(cross["predicted_label"])
    out = out[out["predicted_phenotype"].notna()].copy()

    emb = out.drop_duplicates(ID_COL)
    print(f"transferred onto {emb[ID_COL].nunique()} embryos")
    print(pd.crosstab(emb["genotype"], emb["predicted_phenotype"]).to_string())
    return out, set(ref[ID_COL].unique())


def plot_variant(out: pd.DataFrame, name: str, path: Path,
                 title_suffix: str = "") -> None:
    cols = [c for c in VARIANT_COL_ORDER[name] if c in set(out["predicted_phenotype"])]
    genos = [g for g in GENO_ORDER if g in set(out["genotype"])]
    fig = plot_feature_over_time(
        out,
        features=FEATURES,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="genotype",
        color_preset=ColorPreset(colors=GENO_COLORS, order=genos),
        facet_col="predicted_phenotype",
        layout=FacetSpec(col_order=cols, sharex=True, sharey=False),
        show_individual=True,
        show_trend=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"b9d2 manual-label model, '{name}' — predicted phenotype, "
               f"colored by genotype{title_suffix}"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    emb = out.drop_duplicates(ID_COL)
    note = "   ".join(
        f"{p}/{g.replace('b9d2_','')}={n}"
        for (p, g), n in emb.groupby(["predicted_phenotype", "genotype"]).size().items()
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#555555")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.relative_to(RUN_DIR)}")


def main() -> None:
    df, feature_cols = load()
    print(f"loaded {df[ID_COL].nunique()} embryos, {len(feature_cols)} latent features")

    for name in VARIANTS:
        out, train_ids = run_variant(name, df, feature_cols)

        # All embryos (training embryos are scored IN-sample here).
        plot_variant(out, name, OUTPUT_DIR / f"{name}__multifeature_by_genotype.png")

        # Out-of-sample only: embryos the model never saw a label for. This is the honest
        # view of what the transferred labels look like on new data.
        oos = out[~out[ID_COL].isin(train_ids)].copy()
        n_oos = oos[ID_COL].nunique()
        print(f"\n--- '{name}' OUT-OF-SAMPLE: {n_oos} embryos (never in training) ---")
        emb_oos = oos.drop_duplicates(ID_COL)
        print(pd.crosstab(emb_oos["genotype"], emb_oos["predicted_phenotype"]).to_string())
        plot_variant(oos, name,
                     OUTPUT_DIR / f"{name}__OUT_OF_SAMPLE__multifeature_by_genotype.png",
                     title_suffix=f" — OUT-OF-SAMPLE only (n={n_oos}, never trained on)")


if __name__ == "__main__":
    main()
