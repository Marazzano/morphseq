"""cep290 version of the P(homozygous) severity score + threshold explorer.

Same protocol as scripts 26/27 (b9d2), pointed at the cep290 reference table:

    train  : binary logistic, homozygous vs wildtype ONLY (hets excluded)
    score  : out-of-fold P(homozygous) via random k-fold GROUPED BY EMBRYO
    apply  : hets / unknowns scored by the full model
    output : scores CSV + self-contained threshold-explorer HTML

Why this is the better test: cep290 has 635 embryos over 7 experiments (255 homo / 102 WT for
training) against b9d2's 187 over 2 (42 homo / 40 WT). If the b9d2 score was weak because of
sample size or because cep290's transition phenotypes are cleaner to separate, it should show
up here as a wider homo-vs-WT gap.

The curated phenotype column here is `phenotype_clean` (High_to_Low / Low_to_High /
Not Penetrant), not b9d2's `cluster_categories`.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/28_cep290_homo_vs_wt_probability.py
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

# Reuse the explorer's HTML template so the two genes get an identical UI.
_expl = __import__("27_build_threshold_explorer")
HTML = _expl.HTML
MAX_POINTS = _expl.MAX_POINTS

SOURCE = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_cep290_clean.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "cep290_homo_vs_wt_probability"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
PHENO_COL = "phenotype_clean"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]
FEATURE_LABEL = {
    "baseline_deviation_normalized": "curvature (baseline deviation, normalized)",
    "total_length_um": "total length (µm)",
}

N_FOLDS = 5
RANDOM_STATE = 42

# This table uses a plain `zygosity` column (wildtype/heterozygous/homozygous/unknown).
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous", "unknown"]
ZYG_COLOR = {
    "wildtype": "#7F7F7F",
    "heterozygous": "#F7B267",
    "homozygous": "#B2182B",
    "unknown": "#4C9F70",
}


def load() -> tuple[pd.DataFrame, list[str]]:
    feature_cols_probe = pd.read_csv(SOURCE, nrows=0).columns
    z_cols = sorted(
        [c for c in feature_cols_probe if c.startswith("z_mu_b_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    keep = {ID_COL, TIME_COL, PHENO_COL, "zygosity", "experiment_id",
            "use_embryo_flag", *FEATURES}
    df = pd.read_csv(
        SOURCE,
        usecols=lambda c: c in keep or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    if "use_embryo_flag" in df.columns:
        flag = df["use_embryo_flag"]
        mask = flag if flag.dtype == bool else flag.astype(str).str.lower().isin(
            {"1", "true", "t", "yes", "y"})
        df = df[mask.fillna(False)]
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *FEATURES, *z_cols])
    df = df[df["zygosity"].isin(ZYG_ORDER)].copy()
    return df, z_cols


def score_embryos(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    train_mask = df["zygosity"].isin(["homozygous", "wildtype"])
    train = df[train_mask]
    y = (train["zygosity"] == "homozygous").astype(int).to_numpy()
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
    df["p_homo"] = np.nan
    df.loc[train.index, "p_homo"] = oof
    other = df.index[~train_mask]
    if len(other):
        df.loc[other, "p_homo"] = full.predict_proba(
            df.loc[other, feature_cols].to_numpy())[:, 1]

    df["scoring"] = np.where(train_mask, "out-of-fold", "applied (not in training)")
    df["p_homo_embryo"] = df[ID_COL].map(df.groupby(ID_COL)["p_homo"].mean())
    return df


def build_payload(df: pd.DataFrame) -> dict:
    embryos = []
    for eid, grp in df.groupby(ID_COL):
        grp = grp.sort_values(TIME_COL)
        if len(grp) > MAX_POINTS:
            grp = grp.iloc[np.linspace(0, len(grp) - 1, MAX_POINTS).astype(int)]
        embryos.append({
            "id": eid,
            "zyg": grp["zygosity"].iloc[0],
            "p": round(float(grp["p_homo_embryo"].iloc[0]), 4),
            "pheno": str(grp[PHENO_COL].iloc[0]) if PHENO_COL in grp else "",
            "t": [round(float(v), 2) for v in grp[TIME_COL]],
            "y0": [round(float(v), 4) for v in grp[FEATURES[0]]],
            "y1": [round(float(v), 1) for v in grp[FEATURES[1]]],
        })
    return {
        "embryos": embryos,
        "zygOrder": [z for z in ZYG_ORDER if any(e["zyg"] == z for e in embryos)],
        "zygColor": ZYG_COLOR,
        "featureLabels": [FEATURE_LABEL[f] for f in FEATURES],
    }


def main() -> None:
    df, feature_cols = load()
    print(f"loaded {df[ID_COL].nunique()} embryos, {len(feature_cols)} latent features")
    print(df.drop_duplicates(ID_COL)["zygosity"].value_counts().to_string())

    df = score_embryos(df, feature_cols)
    emb = df.drop_duplicates(ID_COL)

    print("\nP(homozygous) by zygosity (embryo-level):")
    print(emb.groupby("zygosity")["p_homo_embryo"]
          .agg(["count", "mean", "std"]).round(3).to_string())

    print("\ndeciles:")
    for z in ZYG_ORDER:
        sub = emb[emb["zygosity"] == z]["p_homo_embryo"]
        if len(sub) == 0:
            continue
        q = sub.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict()
        print(f"  {z:14s} n={len(sub):3d}  {q}")

    homo = emb[emb["zygosity"] == "homozygous"]
    het = emb[emb["zygosity"] == "heterozygous"]
    print(f"\nhomozygotes P(homo) < 0.5 (candidate NON-PENETRANT): "
          f"{(homo['p_homo_embryo'] < 0.5).sum()} / {len(homo)}")
    print(f"hets P(homo) > 0.5 (candidate PENETRANT): "
          f"{(het['p_homo_embryo'] > 0.5).sum()} / {len(het)}")

    # Separation summary, directly comparable to the b9d2 run.
    wt = emb[emb["zygosity"] == "wildtype"]["p_homo_embryo"]
    hz = homo["p_homo_embryo"]
    pooled_sd = np.sqrt((wt.var() + hz.var()) / 2)
    print(f"\nhomo-vs-WT mean gap: {hz.mean() - wt.mean():.3f}  "
          f"(Cohen's d = {(hz.mean() - wt.mean()) / pooled_sd:.2f})")

    print("\nretention by threshold:")
    tot = emb["zygosity"].value_counts()
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        kept = emb[emb["p_homo_embryo"] >= t]["zygosity"].value_counts()
        s = " | ".join(f"{z}: {kept.get(z, 0):3d}/{tot.get(z, 0):3d}" for z in ZYG_ORDER)
        print(f"  thr={t:.2f}  {s}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roster = (emb[[ID_COL, "experiment_id", "zygosity", PHENO_COL,
                   "p_homo_embryo", "scoring"]]
              .sort_values(["zygosity", "p_homo_embryo"]))
    roster_out = OUTPUT_DIR / "p_homo_scores.csv"
    roster.to_csv(roster_out, index=False)
    print(f"\nSaved: {roster_out.relative_to(RUN_DIR)}  ({len(roster)} embryos)")

    payload = build_payload(df)
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("b9d2 &mdash; P(homozygous)", "cep290 &mdash; P(homozygous)")
    html = html.replace("b9d2 — P(homozygous)", "cep290 — P(homozygous)")
    out = OUTPUT_DIR / "threshold_explorer.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out.relative_to(RUN_DIR)}  "
          f"({len(payload['embryos'])} embryos, {out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
