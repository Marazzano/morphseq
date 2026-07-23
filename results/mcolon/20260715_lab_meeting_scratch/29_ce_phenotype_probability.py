"""Threshold explorer driven by the CURATED-CE phenotype predictor (not genotype).

Scripts 26/28 scored P(homozygous) -- a GENOTYPE model. It worked for cep290 (d=2.60) and
failed for b9d2 (d~1.0), but in both cases it answers "does this embryo look like a mutant",
which is the wrong question when many mutants are non-penetrant.

This scores P(CE) from a model trained on the MANUALLY CURATED CE labels:

    train  : binary logistic, curated CE  vs  curated wildtype
             CE = every manually curated CE embryo, any genotype (38)
             wildtype = the 35 curated wildtype embryos
             -- hets/homos NOT curated as CE are excluded from training, exactly as before,
                because their penetrance status is what we are trying to measure.
    score  : out-of-fold P(CE) via random k-fold GROUPED BY EMBRYO
    apply  : every other embryo (unlabeled, HTA, BA_rescue, ...) scored by the full model
    view   : the same threshold explorer, faceted by zygosity, colored by zygosity

This is the separability test that matters for the phenotype claim: if curated CE is a real,
learnable morphology, P(CE) should cleanly split curated CE from curated wildtype, and the
HTA / BA_rescue embryos should land in between rather than on top of CE.

Facet columns stay ZYGOSITY so the question "do penetrant hets exist" is answerable: a het
scoring high P(CE) is a carrier with the CE morphology.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/29_ce_phenotype_probability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

_expl = __import__("27_build_threshold_explorer")
HTML = _expl.HTML
MAX_POINTS = _expl.MAX_POINTS

SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "ce_phenotype_probability"

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

POS_CLASS = "CE"
NEG_CLASS = "wildtype"

ZYG_ORDER = ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous", "b9d2_unknown"]
ZYG_LABEL = {
    "b9d2_wildtype": "wildtype",
    "b9d2_heterozygous": "heterozygous",
    "b9d2_homozygous": "homozygous",
    "b9d2_unknown": "unknown",
}
ZYG_COLOR = {
    "wildtype": "#7F7F7F",
    "heterozygous": "#F7B267",
    "homozygous": "#B2182B",
    "unknown": "#4C9F70",
}


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(SOURCE, low_memory=False)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    df = df.dropna(subset=[TIME_COL, *FEATURES, *z_cols])
    return df, z_cols


def score_embryos(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Out-of-fold P(CE) for the curated CE/wildtype embryos; full-model P(CE) for the rest."""
    train_mask = df[LABEL_COL].isin([POS_CLASS, NEG_CLASS])
    train = df[train_mask]
    y = (train[LABEL_COL] == POS_CLASS).astype(int).to_numpy()
    X = train[feature_cols].to_numpy()
    groups = train[ID_COL].to_numpy()

    def _pipe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE),
        )

    oof = np.full(len(train), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, te_idx in cv.split(X, y, groups=groups):
        model = _pipe().fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]

    full = _pipe().fit(X, y)

    df = df.copy()
    df["p_ce"] = np.nan
    df.loc[train.index, "p_ce"] = oof
    other = df.index[~train_mask]
    if len(other):
        df.loc[other, "p_ce"] = full.predict_proba(
            df.loc[other, feature_cols].to_numpy())[:, 1]

    df["scoring"] = np.where(train_mask, "out-of-fold", "applied (not in training)")
    df["p_homo_embryo"] = df[ID_COL].map(df.groupby(ID_COL)["p_ce"].mean())
    return df


def build_payload(df: pd.DataFrame) -> dict:
    embryos = []
    for eid, grp in df.groupby(ID_COL):
        grp = grp.sort_values(TIME_COL)
        if len(grp) > MAX_POINTS:
            grp = grp.iloc[np.linspace(0, len(grp) - 1, MAX_POINTS).astype(int)]
        embryos.append({
            "id": eid,
            "zyg": ZYG_LABEL.get(grp["genotype"].iloc[0], "unknown"),
            "p": round(float(grp["p_homo_embryo"].iloc[0]), 4),
            "pheno": str(grp[LABEL_COL].iloc[0]),
            "t": [round(float(v), 2) for v in grp[TIME_COL]],
            "y0": [round(float(v), 4) for v in grp[FEATURES[0]]],
            "y1": [round(float(v), 1) for v in grp[FEATURES[1]]],
        })
    return {
        "embryos": embryos,
        "zygOrder": [ZYG_LABEL[z] for z in ZYG_ORDER
                     if any(e["zyg"] == ZYG_LABEL[z] for e in embryos)],
        "zygColor": ZYG_COLOR,
        "featureLabels": [FEATURE_LABEL[f] for f in FEATURES],
    }


def main() -> None:
    df, feature_cols = load()
    emb0 = df.drop_duplicates(ID_COL)
    print(f"loaded {emb0[ID_COL].nunique()} embryos, {len(feature_cols)} latent features")
    print(f"training on curated {POS_CLASS} vs {NEG_CLASS}:")
    print(emb0[emb0[LABEL_COL].isin([POS_CLASS, NEG_CLASS])][LABEL_COL]
          .value_counts().to_string())

    df = score_embryos(df, feature_cols)
    emb = df.drop_duplicates(ID_COL)

    # ---- separability of the TRAINING classes (the actual question) ----
    ce = emb[emb[LABEL_COL] == POS_CLASS]["p_homo_embryo"]
    wt = emb[emb[LABEL_COL] == NEG_CLASS]["p_homo_embryo"]
    pooled_sd = np.sqrt((ce.var() + wt.var()) / 2)
    print(f"\n=== SEPARABILITY (out-of-fold, curated {POS_CLASS} vs {NEG_CLASS}) ===")
    print(f"  {POS_CLASS:9s} n={len(ce):3d}  mean={ce.mean():.3f}  median={ce.median():.3f}")
    print(f"  {NEG_CLASS:9s} n={len(wt):3d}  mean={wt.mean():.3f}  median={wt.median():.3f}")
    print(f"  gap={ce.mean()-wt.mean():.3f}   Cohen's d = {(ce.mean()-wt.mean())/pooled_sd:.2f}")
    # Threshold-free separability.
    from sklearn.metrics import roc_auc_score
    yy = np.r_[np.ones(len(ce)), np.zeros(len(wt))]
    print(f"  AUC = {roc_auc_score(yy, np.r_[ce.values, wt.values]):.3f}")
    print(f"  overlap: {POS_CLASS} below 0.5 = {(ce<0.5).sum()}/{len(ce)}, "
          f"{NEG_CLASS} above 0.5 = {(wt>0.5).sum()}/{len(wt)}")

    print("\nP(CE) by curated label (incl. embryos never trained on):")
    print(emb.groupby(LABEL_COL)["p_homo_embryo"]
          .agg(["count", "mean", "median", "std"]).round(3).to_string())

    print("\nP(CE) by zygosity:")
    emb_z = emb.assign(z=emb["genotype"].map(ZYG_LABEL))
    print(emb_z.groupby("z")["p_homo_embryo"]
          .agg(["count", "mean", "median"]).round(3).to_string())

    print("\nretention by threshold (embryos with P(CE) >= t):")
    tot = emb_z["z"].value_counts()
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        kept = emb_z[emb_z["p_homo_embryo"] >= t]["z"].value_counts()
        s = " | ".join(f"{z}: {kept.get(z,0):3d}/{tot.get(z,0):3d}"
                       for z in ["wildtype", "heterozygous", "homozygous", "unknown"])
        print(f"  thr={t:.2f}  {s}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roster = (emb[[ID_COL, "experiment_id", "genotype", LABEL_COL,
                   "p_homo_embryo", "scoring"]]
              .rename(columns={"p_homo_embryo": "p_ce"})
              .sort_values(["genotype", "p_ce"]))
    roster_out = OUTPUT_DIR / "p_ce_scores.csv"
    roster.to_csv(roster_out, index=False)
    print(f"\nSaved: {roster_out.relative_to(RUN_DIR)}  ({len(roster)} embryos)")

    payload = build_payload(df)
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = (html.replace("b9d2 &mdash; P(homozygous)", "b9d2 &mdash; P(CE), curated-phenotype model")
                .replace("b9d2 — P(homozygous)", "b9d2 — P(CE), curated-phenotype model")
                .replace("P(homozygous)", "P(CE)")
                .replace("P(homo)", "P(CE)")
                .replace("mutant-like", "CE-like")
                .replace("wildtype-like", "not-CE"))
    out = OUTPUT_DIR / "threshold_explorer.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out.relative_to(RUN_DIR)}  "
          f"({len(payload['embryos'])} embryos, {out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
