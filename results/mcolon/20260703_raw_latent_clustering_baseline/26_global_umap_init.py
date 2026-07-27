"""
26_global_umap_init.py
----------------------
Alignment-fix experiment #1: GLOBAL UMAP initializer (no per-bin Procrustes).

Root cause of the 48/72 hpf seam (established): the aligned initializer fits an
independent UMAP per time bin and stitches consecutive bins with Procrustes using
SAME-EMBRYO anchors. A batch entering mid-timeline (20251207 @48hpf, 20260306
@72hpf) has NO anchors in the previous bin, so its bundle is placed purely by the
old batch's rotation -> a cliff. This is a lack-of-bridge-support problem, not a
feature batch effect.

This runner replaces the aligned init with ONE global UMAP over all observed
(embryo, time) points (init_embedding.global_umap_init), so every batch lands in
one shared frame with no stitching. Condensation config is otherwise identical to
1_cluster_raw_latent.py. We score the ridge on x0 and on condensed output and
compare against the aligned baseline.

Outputs
=======
figures/condensed_raw_zmub_global_umap/condensed_positions.npz
figures/condensed_raw_zmub_global_umap/x0_init.npz
figures/condensed_raw_zmub_global_umap/ridge_comparison.txt
figures/condensed_raw_zmub_global_umap/time_slice_by_experiment.html
figures/condensed_raw_zmub_global_umap/time_slice_by_genotype.html
"""
from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
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

import analyze.trajectory_condensation as tc
from analyze.trajectory_condensation import init_embedding
from analyze.trajectory_condensation.condensation import (
    CondensationConfig, StoppingConfig, run_condensation,
)
from analyze.trajectory_condensation.seam_bridge_init import (
    ridge_score, estimate_attraction_bandwidth,
)

_spec = importlib.util.spec_from_file_location(
    "_baseline_runner", _HERE / "1_cluster_raw_latent.py"
)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

N_ITER = _baseline.N_ITER
SAVE_EVERY = _baseline.SAVE_EVERY
RANDOM_STATE = _baseline.RANDOM_STATE
TABLES = _baseline.TABLES

RAW_DIR = _HERE / "figures" / "condensed_raw_zmub"
OUT_DIR = _HERE / "figures" / "condensed_raw_zmub_global_umap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_COLORS = {"20251207_pbx": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC", "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd", "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def _experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if len(parts) > 1 and parts[1].isalpha() else parts[0]


def _report_ridge(fh, tag, rs, time_values):
    print(f"\n=== {tag} ===", file=fh)
    print(f"  frac_one_sided={rs['frac_one_sided']:.3f}  "
          f"frac_two_sided={rs['frac_two_sided']:.3f}  h={rs['attraction_bandwidth']:.4f}",
          file=fh)
    for t in sorted(rs["per_time"]):
        v = rs["per_time"][t]
        frac = v["n_one_sided"] / max(v["n"], 1)
        bar = "#" * int(frac * 40)
        flag = "  <-- SEAM" if frac > 0.3 and t > 0 else ""
        print(f"    t={t:2d} ({time_values[t]:5.1f}hpf) n={v['n']:3d} "
              f"one_sided={frac:.2f} {bar}{flag}", file=fh)


def main() -> None:
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    features, mask, embryo_ids, time_values, labels_arr = _baseline._pivot_to_tensor(
        binned, z_cols
    )
    batch = np.array([str(e).split("_")[0] for e in embryo_ids], dtype=object)
    print(f"Tensor {features.shape}, {int(mask.sum())} obs, batches={sorted(set(batch))}")

    # ── Global UMAP initialization (one shared frame, no Procrustes) ──────────
    x0_path = OUT_DIR / "x0_init.npz"
    if x0_path.exists():
        print(f"Loading cached global UMAP init from {x0_path}")
        x0 = np.load(x0_path)["x0"]
    else:
        print("Computing GLOBAL UMAP init over all observed points...")
        x0 = init_embedding.global_umap_init(
            features, mask, n_neighbors=15, min_dist=0.1,
            random_state=RANDOM_STATE, time_weight=0.0,
        )
        np.savez(x0_path, x0=x0, time_values=time_values)
        print(f"  Saved -> {x0_path}")

    # ── Condensation (identical config to baseline) ───────────────────────────
    relative_profile = _baseline._relative_profile_from_margin_reference()
    config = CondensationConfig(
        **relative_profile,
        temporal_cohere_window=3,
        elastic_strength=16.0,
        elastic_mix=0.25,
        fidelity_half_life=_baseline._gamma_from_half_life_iters(70.0),
        void_strength=0.014,
        outlier_strength=16.0,
        outlier_cutoff_mode="robust",
        outlier_cutoff_value=3.0,
        attract_k=20,
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=N_ITER,
    )
    stopping = StoppingConfig(
        disp_max_rel_threshold=None, disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None, coherence_change_rel_threshold=None,
    )

    print(f"\nRunning condensation ({N_ITER} iters) on global-UMAP init...")
    result = run_condensation(
        x0=x0, mask=mask, config=config, stopping=stopping,
        log_every=max(1, N_ITER // 20), save_every=SAVE_EVERY, verbose=True,
    )

    payload = {
        "positions": result.positions, "x0": x0, "mask": mask,
        "time_values": time_values, "embryo_ids": embryo_ids, "labels": labels_arr,
    }
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    np.savez(OUT_DIR / "condensed_positions.npz", **payload)
    print(f"Saved -> {OUT_DIR / 'condensed_positions.npz'}")

    # ── Ridge comparison vs aligned baseline ─────────────────────────────────
    # h fixed from the ALIGNED baseline x0 so all runs share one ruler.
    aligned_x0 = np.load(RAW_DIR / "x0_init.npz")["x0"]
    h = estimate_attraction_bandwidth(aligned_x0, mask, labels=None, quantile=0.5,
                                      match_within_label=False)
    aligned_cond = np.load(RAW_DIR / "condensed_positions.npz", allow_pickle=True)["positions"]

    rs_aligned_x0 = ridge_score(aligned_x0, mask, batch=batch, attraction_bandwidth=h)
    rs_aligned_cond = ridge_score(aligned_cond, mask, batch=batch, attraction_bandwidth=h)
    rs_global_x0 = ridge_score(x0, mask, batch=batch, attraction_bandwidth=h)
    rs_global_cond = ridge_score(result.positions, mask, batch=batch, attraction_bandwidth=h)

    report_path = OUT_DIR / "ridge_comparison.txt"
    with open(report_path, "w") as fh:
        print(f"fixed h = {h:.4f} (median within-bin NN dist at ALIGNED x0)", file=fh)
        _report_ridge(fh, "ALIGNED x0 (baseline init)", rs_aligned_x0, time_values)
        _report_ridge(fh, "ALIGNED condensed (baseline)", rs_aligned_cond, time_values)
        _report_ridge(fh, "GLOBAL-UMAP x0 (this init)", rs_global_x0, time_values)
        _report_ridge(fh, "GLOBAL-UMAP condensed (this run)", rs_global_cond, time_values)

        def seam(rs, t):
            v = rs["per_time"].get(t)
            return (v["n_one_sided"] / v["n"]) if v else float("nan")

        print("\n--- headline (t=7=48hpf, t=13=72hpf) ---", file=fh)
        for tag, rs in [("ALIGNED x0", rs_aligned_x0), ("ALIGNED cond", rs_aligned_cond),
                        ("GLOBAL   x0", rs_global_x0), ("GLOBAL   cond", rs_global_cond)]:
            print(f"  {tag:14s} overall={rs['frac_one_sided']:.3f}  "
                  f"t7={seam(rs,7):.2f}  t13={seam(rs,13):.2f}", file=fh)

    print(report_path.read_text())

    # ── Time-slice HTML (experiment + genotype) ──────────────────────────────
    experiments = np.array([_experiment_of(str(e)) for e in embryo_ids])
    tc.time_slice_html(
        result.positions, mask, time_values,
        labels=experiments, color_map=EXPERIMENT_COLORS, embryo_ids=embryo_ids,
        title="global-UMAP init + condensation | experiment",
        output_path=OUT_DIR / "time_slice_by_experiment.html",
    )
    tc.time_slice_html(
        result.positions, mask, time_values,
        labels=labels_arr, color_map=GENOTYPE_COLORS, embryo_ids=embryo_ids,
        title="global-UMAP init + condensation | genotype",
        output_path=OUT_DIR / "time_slice_by_genotype.html",
    )
    print("Saved experiment + genotype time-slice HTML")


if __name__ == "__main__":
    main()
