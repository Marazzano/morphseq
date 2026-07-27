"""
25_condense_no_fidelity.py
--------------------------
Step-0 experiment: run trajectory condensation with the INITIAL-PLACEMENT
CONSTRAINT TURNED OFF (fidelity_init_strength = 0), everything else identical to
the raw-latent baseline (1_cluster_raw_latent.py).

Question
========
The 48 hpf batch-entry ridge survives the standard solve. Is that because the
fidelity (anchor-to-x0) term is holding the incoming batch on its broken initial
placement? If we free the solver from x0 entirely, does condensation close the
seam on its own?

Method
======
- Reuse the SAME cached UMAP init x0 as the baseline (figures/condensed_raw_zmub/
  x0_init.npz) so the only change is the solver constraint, not the layout.
- Rebuild the exact baseline config via the runner's own helpers, then override
  fidelity_init_strength = 0.0.
- Score the ridge (batch/genotype-agnostic manifold-continuity metric) on x0 and
  on the resulting condensed positions, and compare against the fidelity-ON
  baseline condensed output already on disk.

Output
======
figures/condensed_raw_zmub_no_fidelity/condensed_positions.npz
figures/condensed_raw_zmub_no_fidelity/ridge_comparison.txt
"""
from __future__ import annotations

import os
import sys
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
from analyze.trajectory_condensation.condensation import (
    CondensationConfig, StoppingConfig, run_condensation,
)
from analyze.trajectory_condensation.seam_bridge_init import (
    ridge_score, estimate_attraction_bandwidth,
)

# Match the coloring used by 23_render_global_harmony_by_experiment.py
EXPERIMENT_COLORS = {"20251207_pbx": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC", "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd", "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def _experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if len(parts) > 1 and parts[1].isalpha() else parts[0]

# Reuse the baseline runner's constants + helpers so config stays identical.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_baseline_runner", _HERE / "1_cluster_raw_latent.py"
)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

N_ITER = _baseline.N_ITER
SAVE_EVERY = _baseline.SAVE_EVERY
TABLES = _baseline.TABLES

RAW_DIR = _HERE / "figures" / "condensed_raw_zmub"
OUT_DIR = _HERE / "figures" / "condensed_raw_zmub_no_fidelity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_tensor_and_x0():
    """Load the same features/mask and reuse the cached baseline x0."""
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    features, mask, embryo_ids, time_values, labels_arr = _baseline._pivot_to_tensor(
        binned, z_cols
    )
    x0 = np.load(RAW_DIR / "x0_init.npz")["x0"]
    assert x0.shape[:2] == mask.shape, "cached x0 does not match rebuilt tensor"
    return features, mask, embryo_ids, time_values, labels_arr, x0


def _report_ridge(fh, tag, rs, time_values):
    print(f"\n=== {tag} ===", file=fh)
    print(f"  frac_one_sided={rs['frac_one_sided']:.3f}  "
          f"frac_two_sided={rs['frac_two_sided']:.3f}  h={rs['attraction_bandwidth']:.4f}",
          file=fh)
    print("  per-time one-sided fraction:", file=fh)
    for t in sorted(rs["per_time"]):
        v = rs["per_time"][t]
        frac = v["n_one_sided"] / max(v["n"], 1)
        bar = "#" * int(frac * 40)
        flag = "  <-- SEAM" if frac > 0.3 and t > 0 else ""
        print(f"    t={t:2d} ({time_values[t]:5.1f}hpf) n={v['n']:3d} "
              f"one_sided={frac:.2f} {bar}{flag}", file=fh)


def main() -> None:
    features, mask, embryo_ids, time_values, labels_arr, x0 = _load_tensor_and_x0()
    batch = np.array([str(e).split("_")[0] for e in embryo_ids], dtype=object)
    print(f"Tensor {features.shape}, {int(mask.sum())} obs, batches={sorted(set(batch))}")

    # Exact baseline config, then override the initial-placement constraint to 0.
    relative_profile = _baseline._relative_profile_from_margin_reference()
    relative_profile["fidelity_init_strength"] = 0.0  # <-- TURN OFF x0 anchoring
    print("Config profile (fidelity_init_strength forced to 0):", relative_profile)

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
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    print(f"\nRunning condensation ({N_ITER} iters, NO fidelity)...")
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

    # ── Ridge comparison: x0 | baseline (fidelity ON) | this run (fidelity OFF) ──
    h = estimate_attraction_bandwidth(x0, mask, labels=None, quantile=0.5,
                                      match_within_label=False)
    rs_x0 = ridge_score(x0, mask, batch=batch, attraction_bandwidth=h)
    rs_off = ridge_score(result.positions, mask, batch=batch, attraction_bandwidth=h)

    baseline_pos = np.load(RAW_DIR / "condensed_positions.npz", allow_pickle=True)["positions"]
    rs_on = ridge_score(baseline_pos, mask, batch=batch, attraction_bandwidth=h)

    report_path = OUT_DIR / "ridge_comparison.txt"
    with open(report_path, "w") as fh:
        print(f"fixed h = {h:.4f} (median within-bin NN dist at x0)", file=fh)
        _report_ridge(fh, "x0 (initializer)", rs_x0, time_values)
        _report_ridge(fh, "condensed, fidelity ON (baseline)", rs_on, time_values)
        _report_ridge(fh, "condensed, fidelity OFF (this run)", rs_off, time_values)
        print("\n--- headline ---", file=fh)
        print(f"  x0            frac_one_sided={rs_x0['frac_one_sided']:.3f}", file=fh)
        print(f"  fidelity ON   frac_one_sided={rs_on['frac_one_sided']:.3f}  "
              f"t=7(48hpf) one_sided="
              f"{rs_on['per_time'][7]['n_one_sided']/rs_on['per_time'][7]['n']:.2f}", file=fh)
        print(f"  fidelity OFF  frac_one_sided={rs_off['frac_one_sided']:.3f}  "
              f"t=7(48hpf) one_sided="
              f"{rs_off['per_time'][7]['n_one_sided']/rs_off['per_time'][7]['n']:.2f}", file=fh)

    print(report_path.read_text())

    # ── Time-slice HTML colored by experiment (matches global-Harmony render) ──
    experiments = np.array([_experiment_of(str(e)) for e in embryo_ids])
    tc.time_slice_html(
        result.positions, mask, time_values,
        labels=experiments, color_map=EXPERIMENT_COLORS, embryo_ids=embryo_ids,
        title="raw z_mu_b condensation, fidelity OFF (no x0 anchor) | experiment",
        output_path=OUT_DIR / "time_slice_by_experiment.html",
    )
    print(f"Saved -> {OUT_DIR / 'time_slice_by_experiment.html'}")
    tc.time_slice_html(
        result.positions, mask, time_values,
        labels=labels_arr, color_map=GENOTYPE_COLORS, embryo_ids=embryo_ids,
        title="raw z_mu_b condensation, fidelity OFF (no x0 anchor) | genotype",
        output_path=OUT_DIR / "time_slice_by_genotype.html",
    )
    print(f"Saved -> {OUT_DIR / 'time_slice_by_genotype.html'}")


if __name__ == "__main__":
    main()
