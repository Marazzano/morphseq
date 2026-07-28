"""3-class probability explorer: one slider per class, ANDed across the simplex.

The interactive page itself lives in ``analyze.classification.viz.simplex_explorer``;
this script only fits the model and hands that module probabilities.

Model (the script-25 class definition):
    CE              -- every manually curated CE embryo, any genotype      n=38
    HTA             -- homozygotes curated HTA / BA_rescue                 n=29
    Not Penetrant   -- the curated unaffected embryos                      n=35

"Not Penetrant" is a PHENOTYPE class, not a genotype: an embryo of any zygosity can be
predicted into it, which is exactly what makes the non-penetrant-mutant query interesting.
The zygosity facet keeps the genotype vocabulary (wildtype / heterozygous / ...) separately,
so a "wildtype" zygosity and a "Not Penetrant" class deliberately read differently.

Multinomial logistic, out-of-fold via random k-fold GROUPED BY EMBRYO; every other embryo
(unlabeled / het / unknown) is scored by the full model.

Because the three probabilities sum to 1, thresholding them independently and ANDing the
results carves out a REGION OF THE SIMPLEX rather than slicing one axis:
    P(CE) >= 0.7                          -> confidently CE
    P(CE) >= 0.3 AND P(CE) <= 0.6         -> boundary cases worth eyeballing
    P(Not Penetrant) >= 0.6, facet homo   -> candidate non-penetrant mutants

Impossible combinations (e.g. all three >= 0.5) are ALLOWED -- they just return nothing -- and
the control panel flags the constraint instead of silently showing an empty plot.

NOTE the classes are not equally learnable: CE-vs-unaffected was d=6.66 / AUC 1.00, while the
HTA vs Not-Penetrant boundary is the weak one (recall ~0.46). Per-class one-vs-rest AUC is
printed and shown in the page so a soft score is not read as if it were the strong one.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/30_three_class_simplex_explorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.classification.viz import (  # noqa: E402
    build_simplex_payload,
    render_simplex_explorer_html,
)

SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "three_class_simplex"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
LABEL_COL = "cluster_categories"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]
FEATURE_LABEL = {
    "baseline_deviation_normalized": "curvature (baseline deviation, normalized)",
    "total_length_um": "total length (µm)",
}

N_FOLDS = 5
RANDOM_STATE = 42
MAX_POINTS = 60

# Class names are PHENOTYPE classes, deliberately not genotypes: an embryo of any zygosity can
# be Not Penetrant. Calling this class "wildtype" conflated the two and caused real confusion.
CLASS_CE = "CE"
CLASS_HTA = "HTA"
CLASS_NP = "Not Penetrant"
CLASS_ORDER = [CLASS_CE, CLASS_HTA, CLASS_NP]
CLASS_KEY = {CLASS_CE: "ce", CLASS_HTA: "hta", CLASS_NP: "np"}
CLASS_COLOR = {CLASS_CE: "#1B9E77", CLASS_HTA: "#D95F02", CLASS_NP: "#9AA0A6"}
# Penetrant = predicted anything other than Not Penetrant.
NONPENETRANT_CLASS = CLASS_NP

DROP_IDS = {"20251125_B06_e01", "20251125_F12_e01"}

ZYG_LABEL = {
    "b9d2_wildtype": "wildtype",
    "b9d2_heterozygous": "heterozygous",
    "b9d2_homozygous": "homozygous",
    "b9d2_unknown": "unknown",
}
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous", "unknown"]
ZYG_COLOR = {
    "wildtype": "#7F7F7F",
    "heterozygous": "#F7B267",
    "homozygous": "#B2182B",
    "unknown": "#4C9F70",
}


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(SOURCE, low_memory=False)
    df = df[~df[ID_COL].isin(DROP_IDS)]
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    z_cols = sorted([c for c in df.columns if c.startswith("z_mu_b_")],
                    key=lambda c: int(c.split("_")[-1]))
    df = df.dropna(subset=[TIME_COL, *FEATURES, *z_cols])

    is_ce = df[LABEL_COL] == "CE"
    # HTA pools the curated HTA + BA_rescue homozygotes (BA_rescue alone was unlearnable at n=7).
    is_hta = (df["genotype"] == "b9d2_homozygous") & df[LABEL_COL].isin(["HTA", "BA_rescue"])
    # The unaffected class is seeded from curated wildtype embryos, but it is a PHENOTYPE class:
    # embryos of any zygosity can be predicted into it.
    is_np = df[LABEL_COL] == "wildtype"
    df["train_label"] = pd.NA
    df.loc[is_ce, "train_label"] = CLASS_CE
    df.loc[is_hta, "train_label"] = CLASS_HTA
    df.loc[is_np, "train_label"] = CLASS_NP
    return df, z_cols


def score(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    train_mask = df["train_label"].notna()
    train = df[train_mask]
    y = train["train_label"].to_numpy()
    X = train[feature_cols].to_numpy()
    groups = train[ID_COL].to_numpy()

    def _pipe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=RANDOM_STATE),
        )

    oof = np.full((len(train), len(CLASS_ORDER)), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(X, y, groups=groups):
        m = _pipe().fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])
        # Reindex to CLASS_ORDER -- a fold's class order follows its own label set.
        for j, cls in enumerate(CLASS_ORDER):
            if cls in m.classes_:
                oof[te, j] = proba[:, list(m.classes_).index(cls)]
            else:
                oof[te, j] = 0.0

    full = _pipe().fit(X, y)
    df = df.copy()
    for j, cls in enumerate(CLASS_ORDER):
        df[f"p_{CLASS_KEY[cls]}"] = np.nan
    df.loc[train.index, [f"p_{CLASS_KEY[c]}" for c in CLASS_ORDER]] = oof

    other = df.index[~train_mask]
    if len(other):
        proba = full.predict_proba(df.loc[other, feature_cols].to_numpy())
        for j, cls in enumerate(CLASS_ORDER):
            col = f"p_{CLASS_KEY[cls]}"
            df.loc[other, col] = (proba[:, list(full.classes_).index(cls)]
                                  if cls in full.classes_ else 0.0)

    df["scoring"] = np.where(train_mask, "out-of-fold", "applied (not in training)")
    df["in_train"] = train_mask  # the explorer's tooltip distinguishes OOF from applied

    # Embryo-level = mean over timepoints, renormalized so the simplex still sums to 1.
    pcols = [f"p_{CLASS_KEY[c]}" for c in CLASS_ORDER]
    per_emb = df.groupby(ID_COL)[pcols].mean()
    per_emb = per_emb.div(per_emb.sum(axis=1), axis=0)
    for c in pcols:
        df[c + "_embryo"] = df[ID_COL].map(per_emb[c])

    # One-vs-rest AUC on the training embryos, per class.
    emb = df.drop_duplicates(ID_COL)
    tr_emb = emb[emb["train_label"].notna()]
    aucs = {}
    for cls in CLASS_ORDER:
        yy = (tr_emb["train_label"] == cls).astype(int)
        if yy.nunique() < 2:
            continue
        aucs[cls] = float(roc_auc_score(yy, tr_emb[f"p_{CLASS_KEY[cls]}_embryo"]))
    return df, aucs


def build_payload(df: pd.DataFrame, aucs: dict) -> dict:
    """Hand the scored frame to the shared viz builder.

    The embryo-level (``_embryo``) probability columns are the ones the explorer
    thresholds on; the per-timepoint columns stay in the frame for the CSV roster.
    """
    return build_simplex_payload(
        df,
        id_col=ID_COL,
        time_col=TIME_COL,
        features=FEATURES,
        class_prob_cols={c: f"p_{CLASS_KEY[c]}_embryo" for c in CLASS_ORDER},
        group_col="genotype",
        curated_col=LABEL_COL,
        in_train_col="in_train",
        class_colors=CLASS_COLOR,
        class_aucs=aucs,
        feature_labels=FEATURE_LABEL,
        group_labels=ZYG_LABEL,
        group_order=ZYG_ORDER,
        group_colors=ZYG_COLOR,
        non_penetrant_class=NONPENETRANT_CLASS,
        max_points=MAX_POINTS,
    )


def main() -> None:
    df, feature_cols = load()
    emb0 = df.drop_duplicates(ID_COL)
    print(f"loaded {emb0[ID_COL].nunique()} embryos, {len(feature_cols)} features")
    print(emb0["train_label"].value_counts(dropna=False).to_string())

    df, aucs = score(df, feature_cols)
    emb = df.drop_duplicates(ID_COL)

    print("\n=== one-vs-rest AUC (out-of-fold, training embryos) ===")
    for c in CLASS_ORDER:
        print(f"  {c:20s} AUC = {aucs.get(c, float('nan')):.3f}")

    print("\nmean probability by curated label:")
    pcols = [f"p_{CLASS_KEY[c]}_embryo" for c in CLASS_ORDER]
    print(emb.groupby(LABEL_COL)[pcols].mean().round(3).to_string())

    print("\npredicted class x zygosity:")
    emb2 = emb.assign(
        pred=emb[pcols].idxmax(axis=1).map(
            {f"p_{CLASS_KEY[c]}_embryo": c for c in CLASS_ORDER}),
        z=emb["genotype"].map(ZYG_LABEL))
    print(pd.crosstab(emb2["pred"], emb2["z"]).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roster = emb2[[ID_COL, "experiment_id", "genotype", LABEL_COL, "pred", *pcols, "scoring"]]
    roster.to_csv(OUTPUT_DIR / "three_class_scores.csv", index=False)
    print(f"\nSaved: {(OUTPUT_DIR / 'three_class_scores.csv').relative_to(RUN_DIR)}")

    payload = build_payload(df, aucs)
    out = OUTPUT_DIR / "simplex_explorer.html"
    render_simplex_explorer_html(
        payload,
        title="b9d2 — 3-class probability explorer",
        subtitle=(
            "One slider per class, <strong>ANDed</strong>: an embryo is kept only if it "
            "satisfies every constraint at once. Because the three probabilities sum to 1, "
            "this carves out a region of the simplex rather than slicing a single axis. Try "
            "<em>P(CE) &ge; 0.7</em> for the confident CE set, or "
            "<em>P(Not Penetrant) &ge; 0.6</em> with the homozygous facet for candidate "
            "non-penetrant mutants. Note <em>Not Penetrant</em> is a phenotype class — the "
            "zygosity facet is the genotype, and the two are meant to disagree."
        ),
        output_path=out,
    )
    print(f"Saved: {out.relative_to(RUN_DIR)}  "
          f"({len(payload['embryos'])} embryos, {out.stat().st_size/1024:.0f} KB)")



if __name__ == "__main__":
    main()
