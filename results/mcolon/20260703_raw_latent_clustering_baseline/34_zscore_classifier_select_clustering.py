"""
34_zscore_classifier_select_clustering.py
------------------------------------------
Hypothesis (from the b9d2 direction-decomposition work, 20260715): the batch
effects that plagued the raw z_mu_b PBX clustering (see SEAM_INVESTIGATION_SUMMARY.md
-- batch "columns", islands, batch-entry seams) are driven by a few NUISANCE
HIGH-VARIANCE dims dominating the unstandardized geometry. The classifier removed
batch effects previously because it downweights those batch-orthogonal high-variance
directions. So: z-scoring the embeddings (and/or restricting to the dims the
classifier deems important, on that even footing) should reproduce the batch
cleanup WITHOUT the classifier margin space.

NOTE on "top X% of variance": after z-scoring every dim has variance 1, so
variance-based selection is undefined. On a level playing field the meaningful
selector is the CLASSIFIER COEFFICIENT -- let the classifier tell us which dims
matter. So the two selected conditions keep the dims making up the top 90% / 50%
of classifier coef^2 mass (pooled crispant-vs-inj_ctrl, fit on z-scored dims).

Three conditions, each run through the SAME condensation pipeline as
1_cluster_raw_latent.py (aligned UMAP init -> condensation), then rendered as a
multi-view time-slice HTML coloured by experiment + genotype:
  z_all      : z-score all 80 dims (WT/inj_ctrl-referenced), keep all
  z_clf90    : z-score, keep dims for top 90% of classifier coef^2 mass
  z_clf50    : z-score, keep dims for top 50% of classifier coef^2 mass

Z-scoring is WT-referenced: each dim centered/scaled by inj_ctrl mean/std
(control = reference geometry), computed PER TIME BIN so developmental drift in
the control is removed too.

Outputs (figures/zscore_select/<cond>/):
  condensed_positions.npz, x0_init.npz, multiview_time_slice.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_zscore_cache"
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

# reuse the proven helpers from the baseline clustering script
import importlib.util
_spec = importlib.util.spec_from_file_location("_b", _HERE / "1_cluster_raw_latent.py")
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)

TABLES = _HERE / "tables"
OUT_ROOT = _HERE / "figures" / "zscore_select"
RANDOM_STATE = 42
N_ITER = 500
SAVE_EVERY = 25
BIN_WIDTH = 4.0
CONTROL = "inj_ctrl"
CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")

EXP = {"20251207_pbx": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GEN = {"inj_ctrl": "#2166AC", "pbx1b_crispant": "#9467bd",
       "pbx4_crispant": "#F7B267", "pbx1b_pbx4_crispant": "#B2182B"}


def wt_zscore(binned: pd.DataFrame, z_cols: list[str]) -> pd.DataFrame:
    """Center/scale each dim by inj_ctrl mean/std PER TIME BIN (WT-referenced)."""
    out = binned.copy()
    for tb, idx in out.groupby("time_bin").groups.items():
        sub = out.loc[idx]
        ctrl = sub[sub.genotype == CONTROL]
        if len(ctrl) < 2:
            # too few controls this bin: fall back to all-embryo stats
            mu, sd = sub[z_cols].mean(), sub[z_cols].std(ddof=1)
        else:
            mu, sd = ctrl[z_cols].mean(), ctrl[z_cols].std(ddof=1)
        sd = sd.replace(0, np.nan)
        out.loc[idx, z_cols] = (sub[z_cols] - mu) / sd
    # dims with zero control std everywhere -> fill NaN with 0 (no info)
    out[z_cols] = out[z_cols].fillna(0.0)
    return out


def classifier_coef_rank(binned_z: pd.DataFrame, z_cols: list[str]) -> pd.Series:
    """Mean coef^2 per dim, pooled crispant-vs-inj_ctrl on z-scored dims, over time.
    Returns a Series indexed by dim, descending."""
    df = binned_z[binned_z.genotype.isin((*CRISPANTS, CONTROL))].copy()
    df["grp"] = np.where(df.genotype == CONTROL, CONTROL, "GRP")
    directions = extract_classifier_directions(
        df, class_col="grp", id_col="embryo_id", time_col="time_bin",
        comparisons=[{"positive": "GRP", "negative": CONTROL}],
        features={"emb": z_cols}, bin_width=BIN_WIDTH,
        min_samples_per_group=2, min_samples_per_member=2, verbose=False,
    )
    acc = {c: [] for c in z_cols}
    for _, r in directions.metadata.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        for name, w in zip(names, vec):
            acc[name].append(float(w) ** 2)
    coef2 = pd.Series({c: np.mean(acc[c]) if acc[c] else 0.0 for c in z_cols})
    return coef2.sort_values(ascending=False)


def top_by_coverage(coef2_ranked: pd.Series, frac: float) -> list[str]:
    cum = np.cumsum(coef2_ranked.values) / coef2_ranked.values.sum()
    n = int(np.searchsorted(cum, frac) + 1)
    return list(coef2_ranked.index[:n])


def condense_and_render(features, mask, embryo_ids, time_values, labels, exp_labels,
                        out_dir: Path, title: str):
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
    print(f"Loaded {len(binned)} rows, {len(z_cols)} dims, "
          f"{binned.embryo_id.nunique()} embryos")

    # WT-referenced z-score (shared by all three conditions)
    binned_z = wt_zscore(binned, z_cols)

    # classifier coef ranking on the z-scored dims (for the selected conditions)
    coef2 = classifier_coef_rank(binned_z, z_cols)
    dims90 = top_by_coverage(coef2, 0.9)
    dims50 = top_by_coverage(coef2, 0.5)
    print(f"classifier coef^2 ranking (z-scored): top5={list(coef2.index[:5])}")
    print(f"  top 90% coef^2 mass -> {len(dims90)} dims")
    print(f"  top 50% coef^2 mass -> {len(dims50)} dims")

    conditions = {
        "z_all": (z_cols, "z-scored (all 80 dims)"),
        "z_clf90": (dims90, f"z-scored + top-90% classifier dims ({len(dims90)})"),
        "z_clf50": (dims50, f"z-scored + top-50% classifier dims ({len(dims50)})"),
    }

    for cond, (feat_cols, title) in conditions.items():
        print(f"\n=== condition '{cond}': {len(feat_cols)} dims ===")
        features, mask, embryo_ids, time_values, labels = _b._pivot_to_tensor(binned_z, feat_cols)
        # per-embryo experiment labels aligned to embryo_ids
        emap = binned.drop_duplicates("embryo_id").set_index("embryo_id")["experiment_id"]
        exp_labels = np.array([str(emap.get(str(e), "")) for e in embryo_ids], dtype=object)
        condense_and_render(features, mask, embryo_ids, time_values, labels, exp_labels,
                            OUT_ROOT / cond, f"PBX {title}")

    print("\nDone. Three multiview HTMLs under figures/zscore_select/{z_all,z_clf90,z_clf50}/")


if __name__ == "__main__":
    main()
