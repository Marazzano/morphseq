"""
29_condense_drop_20251207.py
----------------------------
Raw z_mu_b condensation with the 20251207_pbx experiment DROPPED entirely.

Isolates whether the pipeline is clean without the problem batch: only 20260304
(enters 20hpf) and 20260306 (enters 72hpf) remain. Question: does the 48hpf seam
disappear (it was 20251207 entering) and how does the remaining 72hpf entry
(20260306) behave with the full baseline config?

Everything else identical to 1_cluster_raw_latent.py (fresh UMAP init on the
filtered tensor, same condensation config).

Outputs (figures/condensed_raw_zmub_no1207/):
  condensed_positions.npz, x0_init.npz, ridge_comparison.txt,
  time_slice_by_experiment.html, time_slice_by_genotype.html
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

_spec = importlib.util.spec_from_file_location("_baseline", _HERE / "1_cluster_raw_latent.py")
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

N_ITER = _baseline.N_ITER
SAVE_EVERY = _baseline.SAVE_EVERY
RANDOM_STATE = _baseline.RANDOM_STATE
TABLES = _baseline.TABLES

OUT_DIR = _HERE / "figures" / "condensed_raw_zmub_no1207"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_BATCH = "20251207"
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
    print(f"  frac_one_sided={rs['frac_one_sided']:.3f}  frac_two_sided={rs['frac_two_sided']:.3f}  "
          f"h={rs['attraction_bandwidth']:.4f}", file=fh)
    for t in sorted(rs["per_time"]):
        v = rs["per_time"][t]
        frac = v["n_one_sided"] / max(v["n"], 1)
        bar = "#" * int(frac * 40)
        flag = "  <-- SEAM" if frac > 0.3 and t > 0 else ""
        print(f"    t={t:2d} ({time_values[t]:5.1f}hpf) n={v['n']:3d} one_sided={frac:.2f} {bar}{flag}",
              file=fh)


def main() -> None:
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    features, mask, embryo_ids, time_values, labels_arr = _baseline._pivot_to_tensor(binned, z_cols)
    batch = np.array([str(e).split("_")[0] for e in embryo_ids], dtype=object)

    # ── Drop 20251207 rows ────────────────────────────────────────────────────
    keep = batch != DROP_BATCH
    features, mask = features[keep], mask[keep]
    embryo_ids, labels_arr, batch = embryo_ids[keep], labels_arr[keep], batch[keep]
    # drop time bins that are now empty
    nonempty_t = mask.any(axis=0)
    features, mask, time_values = features[:, nonempty_t, :], mask[:, nonempty_t], time_values[nonempty_t]
    print(f"After dropping {DROP_BATCH}: {features.shape}, {int(mask.sum())} obs, "
          f"batches={sorted(set(batch))}, {mask.shape[1]} time bins")

    # ── Fresh UMAP init on the filtered tensor ────────────────────────────────
    x0_path = OUT_DIR / "x0_init.npz"
    if x0_path.exists():
        print(f"Loading cached init {x0_path}")
        x0 = np.load(x0_path)["x0"]
    else:
        print("Computing aligned UMAP init on filtered tensor...")
        x0 = init_embedding.aligned_umap_init(
            features, mask, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE,
        )
        np.savez(x0_path, x0=x0, time_values=time_values)

    # ── Condensation (baseline config) ───────────────────────────────────────
    relative_profile = _baseline._relative_profile_from_margin_reference()
    config = CondensationConfig(
        **relative_profile, temporal_cohere_window=3, elastic_strength=16.0,
        elastic_mix=0.25, fidelity_half_life=_baseline._gamma_from_half_life_iters(70.0),
        void_strength=0.014, outlier_strength=16.0, outlier_cutoff_mode="robust",
        outlier_cutoff_value=3.0, attract_k=20, solver_lr=1e-4, solver_momentum=0.9,
        solver_max_iter=N_ITER,
    )
    stopping = StoppingConfig(
        disp_max_rel_threshold=None, disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None, coherence_change_rel_threshold=None,
    )
    print(f"\nRunning condensation ({N_ITER} iters) without {DROP_BATCH}...")
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

    # ── Ridge (x0 and condensed); h from this run's x0 ───────────────────────
    h = estimate_attraction_bandwidth(x0, mask, labels=None, quantile=0.5, match_within_label=False)
    rs_x0 = ridge_score(x0, mask, batch=batch, attraction_bandwidth=h)
    rs_cd = ridge_score(result.positions, mask, batch=batch, attraction_bandwidth=h)

    report_path = OUT_DIR / "ridge_comparison.txt"
    with open(report_path, "w") as fh:
        print(f"Dropped {DROP_BATCH}. Remaining batches enter: 20260304@20hpf, 20260306@72hpf.",
              file=fh)
        print(f"fixed h = {h:.4f}", file=fh)
        _report_ridge(fh, "x0 (init, no 20251207)", rs_x0, time_values)
        _report_ridge(fh, "condensed (no 20251207)", rs_cd, time_values)
        print("\n--- headline ---", file=fh)
        print(f"  x0        overall={rs_x0['frac_one_sided']:.3f}", file=fh)
        print(f"  condensed overall={rs_cd['frac_one_sided']:.3f}", file=fh)
    print(report_path.read_text())

    # ── HTML (experiment + genotype) ─────────────────────────────────────────
    experiments = np.array([_experiment_of(str(e)) for e in embryo_ids])
    tc.time_slice_html(
        result.positions, mask, time_values, labels=experiments, color_map=EXPERIMENT_COLORS,
        embryo_ids=embryo_ids, title=f"raw condensation, {DROP_BATCH} dropped | experiment",
        output_path=OUT_DIR / "time_slice_by_experiment.html",
    )
    tc.time_slice_html(
        result.positions, mask, time_values, labels=labels_arr, color_map=GENOTYPE_COLORS,
        embryo_ids=embryo_ids, title=f"raw condensation, {DROP_BATCH} dropped | genotype",
        output_path=OUT_DIR / "time_slice_by_genotype.html",
    )
    print("Saved experiment + genotype HTML")


if __name__ == "__main__":
    main()
