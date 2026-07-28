"""
35_phenotype_direction_projection_clustering.py
-----------------------------------------------
z-scoring alone did NOT remove the PBX batch effects (script 34). Next idea:
represent each embryo purely by its PROJECTION onto the phenotype DIRECTION(S) --
the pure supervised axis, which is batch-orthogonal by construction (the same
reason the classifier margin space mixed experiments). If we cluster on those
projections instead of the raw embedding, the batch effect may finally vanish
while genotype separation holds.

Per phenotype direction (each crispant vs inj_ctrl):
  1. z-score the embeddings (inj_ctrl-referenced, per time bin) -> even footing.
  2. fit logistic GRP-vs-WT per time bin -> coefficient vector w (the phenotype
     direction: points from WT toward that phenotype).
  3. project each embryo onto w: score = w . z  (one number per embryo per bin).
     Equivalent on RAW embeddings via the scale-folded coefficient w/sigma:
     w . (x-mu)/sigma = (w/sigma) . x - const. We project the z-scored embedding,
     which is identical.

THREE directions: pbx1b, pbx4, pbx1b_pbx4 (each vs inj_ctrl). Each embryo x bin =
a 3-vector of direction scores. We cluster (aligned UMAP init -> condensation) in
that 3-D phenotype-coordinate space.

THREE coefficient filter levels (zero out low-|coef| dims BEFORE projecting):
  all  : all 80 dims of w
  clf90: keep dims for top 90% of w's coef^2 mass  (per direction, per bin)
  clf50: keep dims for top 50%

Outputs (figures/pheno_direction/<level>/):
  condensed_positions.npz, x0_init.npz, projection_scores.csv, multiview_time_slice.html
colored by experiment + genotype.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_projdir_cache"
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
OUT_ROOT = _HERE / "figures" / "pheno_direction"
RANDOM_STATE = 42
N_ITER = 500
SAVE_EVERY = 25
BIN_WIDTH = 4.0
CONTROL = "inj_ctrl"
DIRECTIONS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")

EXP = _z.EXP
GEN = _z.GEN


def _filter_coef(w: np.ndarray, frac: float | None) -> np.ndarray:
    """Zero out dims outside the top-`frac` of coef^2 mass. frac=None -> unchanged."""
    if frac is None:
        return w
    w2 = w ** 2
    order = np.argsort(w2)[::-1]
    cum = np.cumsum(w2[order]) / w2.sum()
    keep_n = int(np.searchsorted(cum, frac) + 1)
    mask = np.zeros_like(w, dtype=bool)
    mask[order[:keep_n]] = True
    return np.where(mask, w, 0.0)


def fit_direction_vectors(binned_z, z_cols):
    """Per (direction, time_bin_center) -> coefficient vector w over z_cols.
    Returns dict[(comparison_id, bin_center)] = np.array(len z_cols)."""
    df = binned_z[binned_z.genotype.isin((*DIRECTIONS, CONTROL))].copy()
    comparisons = [{"positive": g, "negative": CONTROL} for g in DIRECTIONS]
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


def build_projection_tensor(binned_z, z_cols, dir_vecs, frac):
    """For each embryo x time_bin, project z-scored embedding onto each of the 3
    filtered direction vectors -> features (N_e, T, 3). Mask = embryo observed at bin."""
    embryo_ids = np.array(sorted(binned_z["embryo_id"].unique()))
    time_values = np.array(sorted(binned_z["time_bin"].unique()), dtype=float)
    eidx = {e: i for i, e in enumerate(embryo_ids)}
    tidx = {t: i for i, t in enumerate(time_values)}
    N_e, T, K = len(embryo_ids), len(time_values), len(DIRECTIONS)

    features = np.full((N_e, T, K), np.nan)
    mask = np.zeros((N_e, T), dtype=bool)
    labels = np.full(N_e, "", dtype=object)

    # bin_center used in dir_vecs keys = raw time_bin + BIN_WIDTH/2
    Zmat = binned_z.set_index(["embryo_id", "time_bin"])
    for _, row in binned_z.iterrows():
        ei, ti = eidx[row["embryo_id"]], tidx[float(row["time_bin"])]
        z = row[z_cols].to_numpy(dtype=float)
        bin_center = float(row["time_bin"]) + BIN_WIDTH / 2.0
        for k, g in enumerate(DIRECTIONS):
            w = dir_vecs.get((f"{g}__vs__{CONTROL}", bin_center))
            if w is None:
                continue
            wf = _filter_coef(w, frac)
            features[ei, ti, k] = float(np.dot(wf, z))
        mask[ei, ti] = True
        labels[ei] = str(row["genotype"])

    # NaN-off-support is fine (the init/condensation handle it): keep an embryo-bin
    # observed as long as it's in the embedding; leave NaN for any direction
    # undefined at that bin. Only drop cells where ALL 3 directions are missing.
    all_missing = np.isnan(features).all(axis=2)
    mask = mask & ~all_missing
    return features, mask, embryo_ids, time_values, labels


def condense_and_render(features, mask, embryo_ids, time_values, labels, exp_labels,
                        out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = init_embedding.aligned_umap_init(
        features, mask, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
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
             embryo_ids=embryo_ids, labels=labels, experiments=exp_labels)
    views = [
        {"name": "experiment", "labels": exp_labels, "color_map": EXP},
        {"name": "genotype", "labels": labels, "color_map": GEN},
    ]
    tc.time_slice_html(result.positions, mask, time_values, embryo_ids=embryo_ids,
                       views=views, title=title,
                       output_path=out_dir / "multiview_time_slice.html")
    print(f"  rendered -> {out_dir / 'multiview_time_slice.html'}")


def main():
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    binned_z = _z.wt_zscore(binned, z_cols)
    print(f"Loaded {len(binned)} rows, {len(z_cols)} dims")

    dir_vecs = fit_direction_vectors(binned_z, z_cols)
    n_bins = len(set(b for (_, b) in dir_vecs))
    print(f"Fit {len(dir_vecs)} (direction,bin) vectors across {n_bins} bins")

    emap = binned.drop_duplicates("embryo_id").set_index("embryo_id")["experiment_id"]

    levels = {"all": None, "clf90": 0.9, "clf50": 0.5}
    for level, frac in levels.items():
        print(f"\n=== level '{level}' (coef filter frac={frac}) ===")
        features, mask, embryo_ids, time_values, labels = build_projection_tensor(
            binned_z, z_cols, dir_vecs, frac)
        exp_labels = np.array([str(emap.get(str(e), "")) for e in embryo_ids], dtype=object)
        print(f"  features {features.shape}, mask coverage {mask.mean():.1%}")
        out_dir = OUT_ROOT / level
        # save the projection scores too
        rows = []
        for i, e in enumerate(embryo_ids):
            for t, tv in enumerate(time_values):
                if mask[i, t]:
                    rows.append(dict(embryo_id=e, time_bin=tv, genotype=labels[i],
                                     experiment=exp_labels[i],
                                     proj_pbx1b=features[i, t, 0],
                                     proj_pbx4=features[i, t, 1],
                                     proj_double=features[i, t, 2]))
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "projection_scores.csv", index=False)
        condense_and_render(features, mask, embryo_ids, time_values, labels, exp_labels,
                            out_dir, f"PBX phenotype-direction projection ({level})")

    print("\nDone. HTMLs under figures/pheno_direction/{all,clf90,clf50}/")


if __name__ == "__main__":
    main()
