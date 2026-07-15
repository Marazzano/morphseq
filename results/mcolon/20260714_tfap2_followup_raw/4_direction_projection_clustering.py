"""
4_direction_projection_clustering.py
------------------------------------
Apply the phenotype-direction projection clustering (PBX scripts 35/36) to the
larger tfap2 dataset (16 genotypes, 597 embryos, 39 bins). Projecting embryos
onto classifier directions is batch-orthogonal by construction, so this tests
whether we get clean genotype structure with experiment effects removed on tfap2.

Recipe:
  1. z-score embeddings (inj_ctrl-referenced, per time bin) -> even footing.
  2. per genotype-vs-inj_ctrl (15 directions), per time bin: fit logistic w.
  3. project each embryo onto all 15 directions: score = w . z (one number per
     direction per bin) -> 15-dim feature per embryo x bin.
  4. cluster (aligned UMAP init -> condensation) on that 15-dim space.

Direction set: 15 genotypes vs inj_ctrl (well-supported; mirrors the PBX
3-direction run). Filter level: 'all' (no coefficient filtering).

This is a LARGE run -> submit to the cluster (run_direction_projection.qsub, 24G).

Outputs (figures/direction_projection/all/):
  condensed_positions.npz, x0_init.npz, projection_scores.csv, multiview_time_slice.html
colored by experiment + genotype.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260714_dirproj_cache"
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
from analyze.trajectory_condensation.condensation.geometry_refs import estimate_geometry_refs
import analyze.trajectory_condensation as tc
from analyze.classification.directions.extract import extract_classifier_directions
from analyze.viz.styling.color_utils import build_genotype_color_lookup

# reuse the baseline condensation helpers from the PBX raw-latent script
_BASELINE = _HERE.parent / "20260703_raw_latent_clustering_baseline"
import importlib.util
_spec = importlib.util.spec_from_file_location("_b", _BASELINE / "1_cluster_raw_latent.py")
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)

TABLES = _HERE / "tables"
OUT_ROOT = _HERE / "figures" / "direction_projection"
RANDOM_STATE = 42
N_ITER = 500
SAVE_EVERY = 25
BIN_WIDTH = 4.0
CONTROL = "inj_ctrl"

EXP_COLORS = {
    "20260213": "#1B9E77", "20260223": "#D95F02", "20260224": "#7570B3",
    "20260319": "#E7298A", "20260320": "#66A61E",
}


def wt_zscore(binned, z_cols):
    """Center/scale each dim by inj_ctrl mean/std PER TIME BIN (WT-referenced)."""
    out = binned.copy()
    for tb, idx in out.groupby("time_bin").groups.items():
        sub = out.loc[idx]
        ctrl = sub[sub.genotype == CONTROL]
        if len(ctrl) < 2:
            mu, sd = sub[z_cols].mean(), sub[z_cols].std(ddof=1)
        else:
            mu, sd = ctrl[z_cols].mean(), ctrl[z_cols].std(ddof=1)
        sd = sd.replace(0, np.nan)
        out.loc[idx, z_cols] = (sub[z_cols] - mu) / sd
    out[z_cols] = out[z_cols].fillna(0.0)
    return out


def fit_direction_vectors(binned_z, z_cols, directions):
    df = binned_z[binned_z.genotype.isin((*directions, CONTROL))].copy()
    comparisons = [{"positive": g, "negative": CONTROL} for g in directions]
    fit = extract_classifier_directions(
        df, class_col="genotype", id_col="embryo_id", time_col="time_bin",
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


def build_projection_tensor(binned_z, z_cols, dir_vecs, directions):
    embryo_ids = np.array(sorted(binned_z["embryo_id"].unique()))
    time_values = np.array(sorted(binned_z["time_bin"].unique()), dtype=float)
    eidx = {e: i for i, e in enumerate(embryo_ids)}
    tidx = {t: i for i, t in enumerate(time_values)}
    K = len(directions)
    features = np.full((len(embryo_ids), len(time_values), K), np.nan)
    mask = np.zeros((len(embryo_ids), len(time_values)), dtype=bool)
    labels = np.full(len(embryo_ids), "", dtype=object)
    comp_ids = [f"{g}__vs__{CONTROL}" for g in directions]
    for _, row in binned_z.iterrows():
        ei, ti = eidx[row["embryo_id"]], tidx[float(row["time_bin"])]
        z = row[z_cols].to_numpy(dtype=float)
        bc = float(row["time_bin"]) + BIN_WIDTH / 2.0
        for k, cid in enumerate(comp_ids):
            w = dir_vecs.get((cid, bc))
            if w is not None:
                features[ei, ti, k] = float(np.dot(w, z))
        mask[ei, ti] = True
        labels[ei] = str(row["genotype"])
    mask = mask & ~np.isnan(features).all(axis=2)
    return features, mask, embryo_ids, time_values, labels


def main():
    OUT_DIR = OUT_ROOT / "all"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    binned = pd.read_csv(TABLES / "tfap2_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    genotypes = sorted(binned.genotype.unique())
    directions = [g for g in genotypes if g != CONTROL]
    print(f"Loaded {len(binned)} rows, {len(z_cols)} dims, {len(genotypes)} genotypes")
    print(f"{len(directions)} directions vs {CONTROL}")

    binned_z = wt_zscore(binned, z_cols)
    dir_vecs = fit_direction_vectors(binned_z, z_cols, directions)
    print(f"Fit {len(dir_vecs)} (direction,bin) vectors")

    features, mask, embryo_ids, time_values, labels = build_projection_tensor(
        binned_z, z_cols, dir_vecs, directions)
    print(f"features {features.shape}, mask cov {mask.mean():.1%}")

    emap = binned.drop_duplicates("embryo_id").set_index("embryo_id")["experiment_id"].astype(str)
    exp_labels = np.array([str(emap.get(str(e), "")) for e in embryo_ids], dtype=object)

    # save projection scores
    rows = []
    for i, e in enumerate(embryo_ids):
        for t, tv in enumerate(time_values):
            if mask[i, t]:
                d = dict(embryo_id=e, time_bin=tv, genotype=labels[i], experiment=exp_labels[i])
                for k, g in enumerate(directions):
                    d[f"proj_{g}"] = features[i, t, k]
                rows.append(d)
    pd.DataFrame(rows).to_csv(OUT_DIR / "projection_scores.csv", index=False)

    # condense
    x0 = init_embedding.aligned_umap_init(features, mask, n_neighbors=15, min_dist=0.1,
                                          random_state=RANDOM_STATE)
    np.savez(OUT_DIR / "x0_init.npz", x0=x0, time_values=time_values)
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
    np.savez(OUT_DIR / "condensed_positions.npz",
             positions=result.positions, x0=x0, mask=mask, time_values=time_values,
             embryo_ids=embryo_ids, labels=labels, experiments=exp_labels)

    gen_cmap = build_genotype_color_lookup(sorted(set(labels.tolist())))
    views = [
        {"name": "experiment", "labels": exp_labels, "color_map": EXP_COLORS},
        {"name": "genotype", "labels": labels, "color_map": gen_cmap},
    ]
    tc.time_slice_html(result.positions, mask, time_values, embryo_ids=embryo_ids,
                       views=views,
                       title="tfap2 phenotype-direction projection (15 vs inj_ctrl, z-scored)",
                       output_path=OUT_DIR / "multiview_time_slice.html")
    print(f"rendered -> {OUT_DIR / 'multiview_time_slice.html'}")
    print("Done.")


if __name__ == "__main__":
    main()
