"""
36_all_pairwise_direction_projection.py
---------------------------------------
Script 35 (3 crispant-vs-inj_ctrl directions) mixed experiments while preserving
genotype -- projecting onto classifier directions is batch-orthogonal by
construction. Now use ALL PAIRWISE directions among the 4 PBX genotypes so we get
full pairwise resolution AND batch removal ("have our cake and eat it too"), since
the classification removes experimental effects.

4 PBX genotypes {inj_ctrl, pbx1b, pbx4, pbx1b_pbx4} -> C(4,2) = 6 pairwise
directions. Each embryo x bin is projected onto all 6 direction axes -> a 6-dim
feature vector. Cluster (aligned UMAP init -> condensation) on that.

Recipe (identical to script 35, generalized to all pairs):
  1. z-score embeddings (inj_ctrl-referenced, per time bin).
  2. per pair, per time bin: fit logistic direction w (points A->B).
  3. project each embryo: score = w_filtered . z  (one number per pair per bin).
Three coefficient filter levels (per bin, per pair): all / clf90 / clf50.

Outputs (figures/pairwise_direction/<level>/):
  condensed_positions.npz, x0_init.npz, projection_scores.csv, multiview_time_slice.html
colored by experiment + genotype.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_pairdir_cache"
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

import importlib.util
_spec = importlib.util.spec_from_file_location("_b", _HERE / "1_cluster_raw_latent.py")
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)
_spec34 = importlib.util.spec_from_file_location("_z", _HERE / "34_zscore_classifier_select_clustering.py")
_z = importlib.util.module_from_spec(_spec34); _spec34.loader.exec_module(_z)

TABLES = _HERE / "tables"
OUT_ROOT = _HERE / "figures" / "pairwise_direction"
RANDOM_STATE = 42
N_ITER = 500
SAVE_EVERY = 25
BIN_WIDTH = 4.0
CONTROL = "inj_ctrl"
GENOTYPES = ("inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
# all 6 unordered pairs; convention positive=second, negative=first (alpha order)
PAIRS = [tuple(sorted(p)) for p in combinations(GENOTYPES, 2)]

EXP = _z.EXP
GEN = _z.GEN


def _filter_coef(w, frac):
    if frac is None:
        return w
    w2 = w ** 2
    order = np.argsort(w2)[::-1]
    cum = np.cumsum(w2[order]) / w2.sum()
    keep_n = int(np.searchsorted(cum, frac) + 1)
    m = np.zeros_like(w, dtype=bool); m[order[:keep_n]] = True
    return np.where(m, w, 0.0)


def fit_pairwise_vectors(binned_z, z_cols):
    """Per (pair comparison_id, bin_center) -> coefficient vector over z_cols."""
    df = binned_z[binned_z.genotype.isin(GENOTYPES)].copy()
    comparisons = [{"positive": b, "negative": a} for (a, b) in PAIRS]
    directions = extract_classifier_directions(
        df, class_col="genotype", id_col="embryo_id", time_col="time_bin",
        comparisons=comparisons, features={"emb": z_cols}, bin_width=BIN_WIDTH,
        min_samples_per_group=2, min_samples_per_member=2, verbose=False,
    )
    out = {}
    for _, r in directions.metadata.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        idx = {n: i for i, n in enumerate(names)}
        w = np.array([float(vec[idx[c]]) for c in z_cols])
        out[(r["comparison_id"], float(r["time_bin_center"]))] = w
    return out


def build_projection_tensor(binned_z, z_cols, pair_vecs, frac):
    """Project each embryo x bin onto all 6 pairwise directions -> (N_e, T, 6)."""
    embryo_ids = np.array(sorted(binned_z["embryo_id"].unique()))
    time_values = np.array(sorted(binned_z["time_bin"].unique()), dtype=float)
    eidx = {e: i for i, e in enumerate(embryo_ids)}
    tidx = {t: i for i, t in enumerate(time_values)}
    K = len(PAIRS)
    features = np.full((len(embryo_ids), len(time_values), K), np.nan)
    mask = np.zeros((len(embryo_ids), len(time_values)), dtype=bool)
    labels = np.full(len(embryo_ids), "", dtype=object)

    comp_ids = [f"{b}__vs__{a}" for (a, b) in PAIRS]
    for _, row in binned_z.iterrows():
        ei, ti = eidx[row["embryo_id"]], tidx[float(row["time_bin"])]
        z = row[z_cols].to_numpy(dtype=float)
        bc = float(row["time_bin"]) + BIN_WIDTH / 2.0
        for k, cid in enumerate(comp_ids):
            w = pair_vecs.get((cid, bc))
            if w is None:
                continue
            features[ei, ti, k] = float(np.dot(_filter_coef(w, frac), z))
        mask[ei, ti] = True
        labels[ei] = str(row["genotype"])

    # NaN-off-support ok; keep a bin unless ALL directions missing
    mask = mask & ~np.isnan(features).all(axis=2)
    # prune extreme (embryo, bin) cells that spread randomly and distort the layout
    mask = prune_outlier_cells(features, mask, k_mad=3.0)
    return features, mask, embryo_ids, time_values, labels


def prune_outlier_cells(features, mask, k_mad=4.0):
    """Drop (embryo, bin) cells whose projection is an extreme outlier within its bin.

    For each time bin, compute each observed point's robust radius = distance from
    the per-bin median projection, then flag points beyond ``k_mad`` median-absolute-
    deviations of that radius. Those cells are removed from the mask so they never
    enter UMAP / condensation — surgical: an embryo keeps its good bins, only its
    extreme slices are dropped. Missing directions (NaN) are treated as 0 for the
    radius so a partially-defined cell isn't judged an outlier on that basis alone.
    """
    mask = mask.copy()
    N_e, T, K = features.shape
    n_pruned = 0
    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        if obs.size < 5:
            continue
        pts = np.nan_to_num(features[obs, t, :], nan=0.0)
        center = np.median(pts, axis=0)
        radius = np.linalg.norm(pts - center, axis=1)
        med_r = np.median(radius)
        mad = np.median(np.abs(radius - med_r))
        if mad <= 0:
            continue
        cutoff = med_r + k_mad * mad
        drop = obs[radius > cutoff]
        mask[drop, t] = False
        n_pruned += drop.size
    # a bin with <2 survivors can't inform UMAP; leave as-is (init handles small slices)
    print(f"  pruned {n_pruned} outlier (embryo,bin) cells (>{k_mad}·MAD per bin)")
    return mask


def condense_and_render(features, mask, embryo_ids, time_values, labels, exp_labels, out_dir,
                        title, render_gif=False):
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

    payload = dict(positions=result.positions, x0=x0, mask=mask, time_values=time_values,
                   embryo_ids=embryo_ids, labels=labels, experiments=exp_labels)
    # keep the per-iteration snapshots so we can animate the condensation
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    npz_out = out_dir / "condensed_positions.npz"
    np.savez(npz_out, **payload)

    views = [
        {"name": "experiment", "labels": exp_labels, "color_map": EXP},
        {"name": "genotype", "labels": labels, "color_map": GEN},
    ]
    tc.time_slice_html(result.positions, mask, time_values, embryo_ids=embryo_ids,
                       views=views, title=title,
                       output_path=out_dir / "multiview_time_slice.html")
    print(f"  rendered -> {out_dir / 'multiview_time_slice.html'}")

    if render_gif and result.position_history is not None:
        # iterations.gif: watch the condensation move point-by-point over solver iters
        run = tc.load_run(npz_out, title=title, color_map=GEN)
        tc.render_run(run, str(out_dir), skip_animations=False)
        print(f"  iterations.gif -> {out_dir / 'iterations.gif'}")


def main():
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    binned_z = _z.wt_zscore(binned, z_cols)
    print(f"Loaded {len(binned)} rows, {len(z_cols)} dims; {len(PAIRS)} pairwise directions:")
    for (a, b) in PAIRS:
        print(f"    {b} vs {a}")

    pair_vecs = fit_pairwise_vectors(binned_z, z_cols)
    n_bins = len(set(bc for (_, bc) in pair_vecs))
    print(f"Fit {len(pair_vecs)} (pair,bin) vectors across {n_bins} bins")

    emap = binned.drop_duplicates("embryo_id").set_index("embryo_id")["experiment_id"]

    for level, frac in {"all": None, "clf90": 0.9, "clf50": 0.5}.items():
        print(f"\n=== level '{level}' (frac={frac}) ===")
        feats, mask, eids, tvals, labels = build_projection_tensor(binned_z, z_cols, pair_vecs, frac)
        exp_labels = np.array([str(emap.get(str(e), "")) for e in eids], dtype=object)
        print(f"  features {feats.shape}, mask cov {mask.mean():.1%}")
        out_dir = OUT_ROOT / level
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        cols = [f"proj_{b}_vs_{a}" for (a, b) in PAIRS]
        for i, e in enumerate(eids):
            for t, tv in enumerate(tvals):
                if mask[i, t]:
                    d = dict(embryo_id=e, time_bin=tv, genotype=labels[i], experiment=exp_labels[i])
                    for k, cn in enumerate(cols):
                        d[cn] = feats[i, t, k]
                    rows.append(d)
        pd.DataFrame(rows).to_csv(out_dir / "projection_scores.csv", index=False)
        condense_and_render(feats, mask, eids, tvals, labels, exp_labels, out_dir,
                            f"PBX all-pairwise direction projection ({level})",
                            render_gif=(level == "all"))

    print("\nDone. HTMLs under figures/pairwise_direction/{all,clf90,clf50}/")


if __name__ == "__main__":
    main()
