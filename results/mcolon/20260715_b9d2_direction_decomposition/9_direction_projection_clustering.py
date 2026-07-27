"""
9_direction_projection_clustering.py
------------------------------------
Failure-mode demonstration + benchmark for direction-projection clustering on b9d2.

b9d2 has TWO phenotypes (CE, HTA) that we showed live on DIFFERENT latent
directions. Question: if we cluster embryos by their projection onto phenotype
directions, do the two phenotypes separate?

  Condition A -- FAILURE CASE (1 pooled direction):
    project onto the SINGLE pooled b9d2-homozygous(CE+HTA)-vs-WT direction -> 1-D.
    Because CE and HTA were composed into ONE axis, projecting onto it cannot
    separate them: they collapse onto the same coordinate. Expect a split
    (affected vs WT / severity) but CE and HTA overlapping. This is the point --
    a single composed direction can't distinguish two phenotypes on different axes.

  Condition B -- BENCHMARK (2 split directions):
    project onto CE-vs-WT AND HTA-vs-WT as two axes -> 2-D. The classifier
    recovered both directions at the split, so this SHOULD resolve CE vs HTA.

Both: embryos = CE + HTA + wildtype; z-scored (WT-referenced per bin); direction
vectors fit per time bin; condense; multiview time-slice HTML colored by
phenotype_clean {CE, HTA, wildtype}.

The slide message: you must have FOUND the two directions to separate two
phenotypes; a pooled direction fails.

Outputs (figures/b9d2_projection/<cond>/):
  condensed_positions.npz, x0_init.npz, projection_scores.csv, multiview_time_slice.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260715_b9d2proj_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE / "numba"))
for _d in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
    Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.trajectory_condensation import init_embedding
from analyze.trajectory_condensation.condensation import (
    CondensationConfig, StoppingConfig, run_condensation,
)
import analyze.trajectory_condensation as tc
from analyze.classification.directions.extract import extract_classifier_directions

_BASELINE = _HERE.parent / "20260703_raw_latent_clustering_baseline"
import importlib.util
_spec = importlib.util.spec_from_file_location("_b", _BASELINE / "1_cluster_raw_latent.py")
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)

B9D2_CSV = (
    _REPO / "results" / "mcolon" / "20260607_sci_cilia_gene14_imaging_qc"
    / "tables" / "reference_b9d2_clean.csv"
)
OUT_ROOT = _HERE / "figures" / "b9d2_projection"
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
RANDOM_STATE = 42
N_ITER = 500
SAVE_EVERY = 25

PHENO_COLORS = {"CE": "#1b7837", "HTA": "#762a83", "wildtype": "#2166AC"}


def load_and_zscore():
    df = pd.read_csv(B9D2_CSV, low_memory=False)
    z_cols = sorted([c for c in df.columns if c.startswith("z_mu_b")],
                    key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")))
    df = df.copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    # keep CE + HTA + wildtype embryos
    df = df[((df["zygosity"] == "wildtype") | df["phenotype_clean"].isin(["CE", "HTA"]))].copy()
    df["pheno"] = np.where(df["zygosity"] == "wildtype", "wildtype", df["phenotype_clean"])
    # WT-referenced z-score per bin
    out = df.copy()
    for tb, idx in df.groupby("time_bin").groups.items():
        sub = df.loc[idx]
        wt = sub[sub["pheno"] == "wildtype"]
        mu, sd = (wt[z_cols].mean(), wt[z_cols].std(ddof=1)) if len(wt) >= 2 \
            else (sub[z_cols].mean(), sub[z_cols].std(ddof=1))
        out.loc[idx, z_cols] = (sub[z_cols] - mu) / sd.replace(0, np.nan)
    out[z_cols] = out[z_cols].fillna(0.0)
    return out, z_cols


def fit_dirs(dfz, z_cols, comparisons):
    fit = extract_classifier_directions(
        dfz, class_col="grp", id_col="embryo_id", time_col=TIME_COL,
        comparisons=comparisons, features={"emb": z_cols}, bin_width=BIN_WIDTH,
        min_samples_per_group=2, min_samples_per_member=2, verbose=False,
    )
    out = {}
    for _, r in fit.metadata.iterrows():
        vec = fit.vectors[r["vector_id"]]
        names = fit.feature_names[r["feature_set"]]
        idx = {n: i for i, n in enumerate(names)}
        out[(r["comparison_id"], float(r["time_bin_center"]))] = np.array(
            [float(vec[idx[c]]) for c in z_cols])
    return out


def build_features(dfz, z_cols, dir_vecs, comp_ids):
    embryo_ids = np.array(sorted(dfz["embryo_id"].unique()))
    # per-embryo pheno label + collapse per (embryo, bin) mean of z (embryo=unit)
    pheno = dfz.groupby("embryo_id")["pheno"].agg(lambda s: s.value_counts().index[0])
    time_values = np.array(sorted(dfz["time_bin"].unique()), dtype=float)
    eidx = {e: i for i, e in enumerate(embryo_ids)}
    tidx = {t: i for i, t in enumerate(time_values)}
    K = len(comp_ids)
    features = np.full((len(embryo_ids), len(time_values), K), np.nan)
    mask = np.zeros((len(embryo_ids), len(time_values)), dtype=bool)
    labels = np.array([pheno[e] for e in embryo_ids], dtype=object)

    zmean = dfz.groupby(["embryo_id", "time_bin"])[z_cols].mean()
    for (e, tb), zrow in zmean.iterrows():
        ei, ti = eidx[e], tidx[float(tb)]
        z = zrow.to_numpy(dtype=float)
        bc = float(tb) + BIN_WIDTH / 2.0
        for k, cid in enumerate(comp_ids):
            w = dir_vecs.get((cid, bc))
            if w is not None:
                features[ei, ti, k] = float(np.dot(w, z))
        mask[ei, ti] = True
    mask = mask & ~np.isnan(features).all(axis=2)
    return features, mask, embryo_ids, time_values, labels


def condense_and_render(features, mask, embryo_ids, time_values, labels, out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = init_embedding.aligned_umap_init(features, mask, n_neighbors=15, min_dist=0.1,
                                          random_state=RANDOM_STATE)
    np.savez(out_dir / "x0_init.npz", x0=x0, time_values=time_values)
    profile = _b._relative_profile_from_margin_reference()
    config = CondensationConfig(
        **profile, temporal_cohere_window=3, elastic_strength=16.0, elastic_mix=0.25,
        fidelity_half_life=_b._gamma_from_half_life_iters(70.0), void_strength=0.014,
        outlier_strength=16.0, outlier_cutoff_mode="robust", outlier_cutoff_value=3.0,
        attract_k=20, solver_lr=1e-4, solver_momentum=0.9, solver_max_iter=N_ITER,
    )
    stopping = StoppingConfig(disp_max_rel_threshold=None, disp_rms_rel_threshold=None,
                              energy_change_rel_threshold=None, coherence_change_rel_threshold=None)
    result = run_condensation(x0=x0, mask=mask, config=config, stopping=stopping,
                              log_every=max(1, N_ITER // 20), save_every=SAVE_EVERY, verbose=True)
    np.savez(out_dir / "condensed_positions.npz",
             positions=result.positions, x0=x0, mask=mask, time_values=time_values,
             embryo_ids=embryo_ids, labels=labels)
    views = [{"name": "phenotype", "labels": labels, "color_map": PHENO_COLORS}]
    tc.time_slice_html(result.positions, mask, time_values, embryo_ids=embryo_ids,
                       views=views, title=title,
                       output_path=out_dir / "multiview_time_slice.html")
    print(f"  rendered -> {out_dir / 'multiview_time_slice.html'}")


def save_scores(features, mask, embryo_ids, time_values, labels, comp_ids, out_dir):
    rows = []
    for i, e in enumerate(embryo_ids):
        for t, tv in enumerate(time_values):
            if mask[i, t]:
                d = dict(embryo_id=e, time_bin=tv, phenotype=labels[i])
                for k, cid in enumerate(comp_ids):
                    d[f"proj_{cid}"] = features[i, t, k]
                rows.append(d)
    pd.DataFrame(rows).to_csv(out_dir / "projection_scores.csv", index=False)


def main():
    dfz, z_cols = load_and_zscore()
    print(f"z-scored {len(dfz)} frames; embryos: {dfz.embryo_id.nunique()}")
    print(dfz.drop_duplicates('embryo_id')['pheno'].value_counts().to_dict())

    # ── Condition A: pooled b9d2-vs-WT (1 direction) -- the failure case ──
    dfA = dfz.copy()
    dfA["grp"] = np.where(dfA["pheno"] == "wildtype", "WT", "GRP")
    vecsA = fit_dirs(dfA, z_cols, [{"positive": "GRP", "negative": "WT"}])
    compA = ["GRP__vs__WT"]
    featA, maskA, eA, tA, labA = build_features(dfz, z_cols, vecsA, compA)
    print(f"\n[A pooled] features {featA.shape}, mask cov {maskA.mean():.1%}")
    outA = OUT_ROOT / "A_pooled_1dir"; outA.mkdir(parents=True, exist_ok=True)
    save_scores(featA, maskA, eA, tA, labA, compA, outA)
    condense_and_render(featA, maskA, eA, tA, labA, outA,
                        "b9d2 clustered on 1 POOLED direction (CE+HTA vs WT) — FAILURE MODE: CE/HTA collapse")

    # ── Condition B: 2 split directions (CE-vs-WT, HTA-vs-WT) -- benchmark ──
    dfB = dfz.copy()
    # grp column = the phenotype for CE/HTA rows, WT for wildtype; fit two comparisons
    dfB["grp"] = dfB["pheno"].map({"CE": "CE", "HTA": "HTA", "wildtype": "WT"})
    vecsB = fit_dirs(dfB, z_cols, [{"positive": "CE", "negative": "WT"},
                                   {"positive": "HTA", "negative": "WT"}])
    compB = ["CE__vs__WT", "HTA__vs__WT"]
    featB, maskB, eB, tB, labB = build_features(dfz, z_cols, vecsB, compB)
    print(f"\n[B split] features {featB.shape}, mask cov {maskB.mean():.1%}")
    outB = OUT_ROOT / "B_split_2dir"; outB.mkdir(parents=True, exist_ok=True)
    save_scores(featB, maskB, eB, tB, labB, compB, outB)
    condense_and_render(featB, maskB, eB, tB, labB, outB,
                        "b9d2 clustered on 2 SPLIT directions (CE-vs-WT, HTA-vs-WT) — BENCHMARK: should resolve CE/HTA")

    print("\nDone. figures/b9d2_projection/{A_pooled_1dir,B_split_2dir}/")


if __name__ == "__main__":
    main()
