"""
1_condense_raw_tfap2.py
-----------------------
Run trajectory condensation on the raw z_mu_b tfap2 tensor (all 16 genotypes),
using the current relative-forces condensation API. This is the raw-latent
analogue of 20260703_raw_latent_clustering_baseline/1_cluster_raw_latent.py.

Pipeline: features -> aligned UMAP init -> condensation dynamics. No k-means /
Leiden. The geometry over (embryo, time) is the deliverable, rendered by
2_make_multiview.py.

Seam handling: NO correction on this first pass (no bridge_seams, no zipper
force). ridge_score is computed on x0 AND the condensed positions with a fixed
bandwidth h, written to ridge_report.txt as honest batch-entry QC. Escalate to
bridge_seams(mode="init") only if those numbers show the seams hurt clustering.

Outputs (figures/condensed_raw_tfap2/):
  condensed_positions.npz  — positions,x0,mask,time_values,embryo_ids,labels,experiments
  x0_init.npz              — cached UMAP init (re-runs skip UMAP)
  metrics.csv              — per-iter solver diagnostics
  ridge_report.txt         — seam QC (x0 vs condensed, fixed h)
  time_slice.html          — standard single-view viewer
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Cache dirs to avoid permission issues (matches sibling raw-latent script).
_CACHE = Path("/tmp") / "morphseq_20260714_tfap2_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE / "numba"))
for _d in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
    Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.trajectory_condensation import init_embedding  # noqa: E402
from analyze.trajectory_condensation.condensation import (  # noqa: E402
    CondensationConfig,
    StoppingConfig,
    run_condensation,
)
from analyze.trajectory_condensation.condensation.geometry_refs import (  # noqa: E402
    estimate_geometry_refs,
)
from analyze.trajectory_condensation.seam_bridge_init import (  # noqa: E402
    estimate_attraction_bandwidth,
    ridge_score,
)
import analyze.trajectory_condensation as tc  # noqa: E402
from analyze.viz.styling.color_utils import build_genotype_color_lookup  # noqa: E402

from common import RANDOM_STATE  # noqa: E402

TABLES = _HERE / "tables"
OUT_DIR = _HERE / "figures" / "condensed_raw_tfap2"

N_ITER = 500
SAVE_EVERY = 25
MARGIN_REFERENCE_NPZ = (
    _HERE.parent / "20260407_pbx_analysis_cont" / "results" / "positioning" / "trajectory"
    / "combined_raw_condensation_5class_bin4_perm500" / "condensed_positions.npz"
)


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def _pivot_to_tensor(
    binned: pd.DataFrame, z_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot (embryo_id, time_bin, z_cols) → (N_e,T,K), mask, embryo_ids,
    time_values, genotype labels, and per-embryo experiment labels.

    Copied from 20260703_.../1_cluster_raw_latent.py:_pivot_to_tensor, extended to
    also carry a per-embryo experiment array for the viewer's experiment view.
    """
    embryo_ids = np.array(sorted(binned["embryo_id"].unique()))
    time_values = np.array(sorted(binned["time_bin"].unique()), dtype=float)
    N_e, T, K = len(embryo_ids), len(time_values), len(z_cols)

    eid_idx = {e: i for i, e in enumerate(embryo_ids)}
    t_idx = {t: i for i, t in enumerate(time_values)}

    features = np.full((N_e, T, K), np.nan)
    mask = np.zeros((N_e, T), dtype=bool)
    labels_arr = np.full(N_e, "", dtype=object)
    experiment_arr = np.full(N_e, "", dtype=object)

    for _, row in binned.iterrows():
        ei = eid_idx[row["embryo_id"]]
        ti = t_idx[float(row["time_bin"])]
        features[ei, ti, :] = row[z_cols].values.astype(float)
        mask[ei, ti] = True
        labels_arr[ei] = str(row.get("genotype", ""))
        experiment_arr[ei] = str(row.get("experiment_id", ""))

    return features, mask, embryo_ids, time_values, labels_arr, experiment_arr


def _relative_profile_from_margin_reference() -> dict[str, float]:
    """Express the known-good legacy PBX margin force balance in public units.

    Copied verbatim from 20260703_.../1_cluster_raw_latent.py so the raw tfap2 run
    uses the same force calibration for apples-to-apples geometry.
    """
    ref = np.load(MARGIN_REFERENCE_NPZ, allow_pickle=True)
    geometry = estimate_geometry_refs(ref["x0"], ref["mask"])
    return {
        "attract_bandwidth_mult": 0.5 / geometry.s_global,
        "temporal_cohere_bandwidth_mult": 0.5 / geometry.s_local,
        "repulsion_strength": 5e-4 / geometry.s_local**2,
        "repulsion_softening_mult": 1e-4 / geometry.s_local**4,
        "fidelity_init_strength": 0.25 * geometry.s_local**2,
        "void_bandwidth_mult": 0.5 / geometry.s_global,
    }


def _write_ridge_report(path: Path, x0, positions, mask, time_values) -> None:
    """Score seam severity on x0 and condensed positions with a FIXED bandwidth.

    h is estimated once (from x0, genotype-agnostic per the ridge MVP definition)
    and held constant across both arrays so the numbers are comparable.
    """
    h = estimate_attraction_bandwidth(x0, mask, labels=None, match_within_label=False)
    r_x0 = ridge_score(x0, mask, attraction_bandwidth=h)
    r_cond = ridge_score(positions, mask, attraction_bandwidth=h)

    lines = [
        "TFAP2 raw-latent condensation — seam QC (ridge_score)",
        "=" * 56,
        f"fixed bandwidth h = {h:.4g}  (held constant across x0 and condensed)",
        "",
        "frac_one_sided = obs with no neighbor within h at t-1 (broad seam)",
        "frac_two_sided = obs isolated at BOTH t-1 and t+1 (honest ridge)",
        "",
        f"{'array':<12}{'frac_one_sided':>16}{'frac_two_sided':>16}",
        f"{'x0':<12}{r_x0['frac_one_sided']:>16.4f}{r_x0['frac_two_sided']:>16.4f}",
        f"{'condensed':<12}{r_cond['frac_one_sided']:>16.4f}{r_cond['frac_two_sided']:>16.4f}",
        "",
        "per-time frac_one_sided (condensed) — batch-entry bins should spike:",
    ]
    per_time = r_cond.get("per_time", {})
    # per_time[idx] = {"n", "n_one_sided", "n_two_sided"}; compute the fraction.
    for k in sorted(per_time, key=lambda x: float(x)):
        idx = int(float(k))
        hpf = time_values[idx] if 0 <= idx < len(time_values) else float("nan")
        v = per_time[k]
        n = v.get("n", 0)
        one = (v.get("n_one_sided", 0) / n) if n else float("nan")
        two = (v.get("n_two_sided", 0) / n) if n else float("nan")
        lines.append(
            f"  t_idx={idx:<3} hpf={hpf:<6.0f} n={n:<4} "
            f"frac_one_sided={one:.3f} frac_two_sided={two:.3f}"
        )

    path.write_text("\n".join(lines) + "\n")
    print(f"Saved ridge report -> {path}")
    print(f"  x0:        one_sided={r_x0['frac_one_sided']:.3f} two_sided={r_x0['frac_two_sided']:.3f}")
    print(f"  condensed: one_sided={r_cond['frac_one_sided']:.3f} two_sided={r_cond['frac_two_sided']:.3f}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    binned = pd.read_csv(TABLES / "tfap2_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    print(f"Loaded: {len(binned)} rows, {len(z_cols)} z_mu_b dims")

    print("Building (N_e, T, K) feature tensor...")
    features, mask, embryo_ids, time_values, labels_arr, experiment_arr = _pivot_to_tensor(
        binned, z_cols
    )
    print(f"  Shape: {features.shape}, mask coverage: {mask.mean():.1%}")
    print(f"  Embryos: {len(embryo_ids)}, Time bins: {len(time_values)}")

    # ── UMAP init (cached) ────────────────────────────────────────────────────
    x0_path = OUT_DIR / "x0_init.npz"
    if x0_path.exists():
        print(f"\nLoading cached UMAP init from {x0_path}")
        x0 = np.load(x0_path)["x0"]
    else:
        print("\nComputing aligned UMAP initialization (may take a few minutes)...")
        x0 = init_embedding.aligned_umap_init(
            features, mask, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE,
        )
        np.savez(x0_path, x0=x0, time_values=time_values)
        print(f"  Saved UMAP init -> {x0_path}")

    # ── Condensation — relative-forces API, PBX margin calibration ────────────
    print(f"\nRunning trajectory condensation ({N_ITER} iters)...")
    relative_profile = _relative_profile_from_margin_reference()
    print("Relative profile (from known-good margin geometry):", relative_profile)
    config = CondensationConfig(
        **relative_profile,
        temporal_cohere_window=3,
        elastic_strength=16.0,
        elastic_mix=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
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

    result = run_condensation(
        x0=x0, mask=mask, config=config, stopping=stopping,
        log_every=max(1, N_ITER // 20), save_every=SAVE_EVERY, verbose=True,
    )

    # ── Save condensed positions (npz contract read by 2_make_multiview.py) ───
    payload = {
        "positions": result.positions,
        "x0": x0,
        "mask": mask,
        "time_values": time_values,
        "embryo_ids": embryo_ids,
        "labels": labels_arr,
        "experiments": experiment_arr,
    }
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    npz_out = OUT_DIR / "condensed_positions.npz"
    np.savez(npz_out, **payload)
    print(f"\nSaved -> {npz_out}")

    # ── metrics.csv ───────────────────────────────────────────────────────────
    pd.DataFrame(result.metrics_history).to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"Saved metrics -> {OUT_DIR / 'metrics.csv'}")

    # ── Seam QC (no correction; report only) ──────────────────────────────────
    _write_ridge_report(OUT_DIR / "ridge_report.txt", x0, result.positions, mask, time_values)

    # ── Standard single-view viewer (multi-view is 2_make_multiview.py) ───────
    color_map = build_genotype_color_lookup(sorted(set(labels_arr.tolist())))
    title = "raw z_mu_b tfap2 condensation (16 genotypes)"
    run = tc.load_run(npz_out, title=title, color_map=color_map)
    tc.time_slice_html(
        run.positions, run.mask, run.time_values,
        labels=run.labels, color_map=color_map, embryo_ids=run.embryo_ids,
        title=f"{title} | final condensation",
        output_path=OUT_DIR / "time_slice.html",
    )
    print(f"Saved viz bundle -> {OUT_DIR}")
    print("\nDone. Run 2_make_multiview.py next.")


if __name__ == "__main__":
    main()
