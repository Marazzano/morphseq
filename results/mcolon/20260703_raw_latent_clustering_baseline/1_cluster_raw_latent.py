"""
1_cluster_raw_latent.py
-----------------------
Run trajectory condensation on the raw z_mu_b features (biological latent block)
using the same solver parameters that worked for PBX in 20260407.

The clustering method is fixed: features -> UMAP init -> condensation dynamics.
There is no k-means/Leiden step. This arm only changes what goes INTO UMAP (raw
z_mu_b instead of classifier margins), producing a geometry over (embryo, time)
that can be compared to the margin-space condensation on an apples-to-apples basis.

Outputs (raw, default):
  figures/condensed_raw_zmub/condensed_positions.npz  — solver output
  figures/condensed_raw_zmub/x0_init.npz              — UMAP initialization
  figures/condensed_raw_zmub/plot_trajectories.png    — standard viz bundle

With --harmony-theta <value>, reads the Harmony-corrected tensor produced by
4_harmony_correction.py (figures/harmony_corrected/z_corrected_theta{value}.npz)
instead of the raw pivot, and writes to a sibling
figures/condensed_harmony_theta{value}/ directory so the raw arm's outputs
are never overwritten. All condensation config is identical to the raw run
for an apples-to-apples comparison via 3_cross_experiment_diagnostic.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cache dirs to avoid permission issues
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

from analyze.trajectory_condensation import init_embedding
from analyze.trajectory_condensation.condensation import CondensationConfig, StoppingConfig, run_condensation
from analyze.trajectory_condensation.condensation.geometry_refs import estimate_geometry_refs
import analyze.trajectory_condensation as tc

TABLES = _HERE / "tables"

RANDOM_STATE = 42
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot (embryo_id, time_bin, z_cols) → (N_e, T, K), mask, embryo_ids, time_values, labels."""
    embryo_ids = np.array(sorted(binned["embryo_id"].unique()))
    time_values = np.array(sorted(binned["time_bin"].unique()), dtype=float)
    N_e, T, K = len(embryo_ids), len(time_values), len(z_cols)

    eid_idx = {e: i for i, e in enumerate(embryo_ids)}
    t_idx = {t: i for i, t in enumerate(time_values)}

    features = np.full((N_e, T, K), np.nan)
    mask = np.zeros((N_e, T), dtype=bool)
    labels_arr = np.full(N_e, "", dtype=object)

    for _, row in binned.iterrows():
        ei = eid_idx[row["embryo_id"]]
        ti = t_idx[row["time_bin"]]
        features[ei, ti, :] = row[z_cols].values.astype(float)
        mask[ei, ti] = True
        labels_arr[ei] = str(row.get("genotype", ""))

    return features, mask, embryo_ids, time_values, labels_arr


def _relative_profile_from_margin_reference() -> dict[str, float]:
    """Express the known-good legacy margin force balance in public units."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--harmony-theta", type=float, default=None,
        help="If set, condense the Harmony-corrected tensor at this theta "
             "(from 4_harmony_correction.py) instead of the raw pivot.",
    )
    parser.add_argument(
        "--harmony-global-theta", type=float, default=None,
        help="Use one shared global Harmony correction from 22_harmony_global_correction.py.",
    )
    args = parser.parse_args()

    if args.harmony_theta is not None and args.harmony_global_theta is not None:
        raise ValueError("Choose either --harmony-theta or --harmony-global-theta, not both.")

    if args.harmony_global_theta is not None:
        harmony_path = (
            _HERE / "figures" / "harmony_corrected_global"
            / f"z_corrected_global_theta{args.harmony_global_theta:g}.npz"
        )
        print(f"Loading global Harmony-corrected tensor (theta={args.harmony_global_theta:g}) from {harmony_path}")
        data = np.load(harmony_path, allow_pickle=True)
        features = data["features"]
        mask = data["mask"]
        embryo_ids = data["embryo_ids"]
        time_values = data["time_values"]
        labels_arr = data["labels"]
        OUT_DIR = _HERE / "figures" / f"condensed_harmony_global_theta{args.harmony_global_theta:g}"
    elif args.harmony_theta is not None:
        harmony_path = _HERE / "figures" / "harmony_corrected" / f"z_corrected_theta{args.harmony_theta:g}.npz"
        print(f"Loading Harmony-corrected tensor (theta={args.harmony_theta:g}) from {harmony_path}")
        data = np.load(harmony_path, allow_pickle=True)
        features = data["features"]
        mask = data["mask"]
        embryo_ids = data["embryo_ids"]
        time_values = data["time_values"]
        labels_arr = data["labels"]
        OUT_DIR = _HERE / "figures" / f"condensed_harmony_theta{args.harmony_theta:g}"
    else:
        binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
        z_cols = [c for c in binned.columns if "z_mu_b" in c]
        print(f"Loaded: {len(binned)} rows, {len(z_cols)} z_mu_b dims")

        # ── Build (N_e, T, K) tensor ──────────────────────────────────────────
        print("Building (N_e, T, K) feature tensor...")
        features, mask, embryo_ids, time_values, labels_arr = _pivot_to_tensor(binned, z_cols)
        OUT_DIR = _HERE / "figures" / "condensed_raw_zmub_relative_api"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Shape: {features.shape}, mask coverage: {mask.mean():.1%}")
    print(f"  Embryos: {len(embryo_ids)}, Time bins: {len(time_values)}")
    print(f"  Output dir: {OUT_DIR}")

    # ── UMAP initialization (same as PBX condensation) ───────────────────────
    x0_path = OUT_DIR / "x0_init.npz"
    legacy_x0_path = _HERE / "figures" / "condensed_raw_zmub" / "x0_init.npz"
    if x0_path.exists():
        print(f"\nLoading cached UMAP init from {x0_path}")
        x0 = np.load(x0_path)["x0"]
    elif args.harmony_theta is None and args.harmony_global_theta is None and legacy_x0_path.exists():
        print(f"\nReusing fixed raw UMAP initialization from {legacy_x0_path}")
        x0 = np.load(legacy_x0_path)["x0"]
        np.savez(x0_path, x0=x0, time_values=time_values)
    else:
        print("\nComputing aligned UMAP initialization (this may take a few minutes)...")
        x0 = init_embedding.aligned_umap_init(
            features, mask,
            n_neighbors=15,
            min_dist=0.1,
            alignment_regularisation=1e-2,
            alignment_window_size=3,
            random_state=RANDOM_STATE,
        )
        np.savez(x0_path, x0=x0, time_values=time_values)
        print(f"  Saved UMAP init -> {x0_path}")

    # ── Condensation — public relative API, calibrated from PBX margin ───────
    print(f"\nRunning trajectory condensation ({N_ITER} iters)...")
    relative_profile = _relative_profile_from_margin_reference()
    print("Relative profile (derived once from known-good margin geometry):", relative_profile)
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
        x0=x0,
        mask=mask,
        config=config,
        stopping=stopping,
        log_every=max(1, N_ITER // 20),
        save_every=SAVE_EVERY,
        verbose=True,
    )

    # ── Save condensed positions ──────────────────────────────────────────────
    payload = {
        "positions": result.positions,
        "x0": x0,
        "mask": mask,
        "time_values": time_values,
        "embryo_ids": embryo_ids,
        "labels": labels_arr,
    }
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)

    npz_out = OUT_DIR / "condensed_positions.npz"
    np.savez(npz_out, **payload)
    print(f"\nSaved -> {npz_out}")
    title = (
        f"global-Harmony z_mu_b condensation (theta={args.harmony_global_theta:g})"
        if args.harmony_global_theta is not None else (
            f"harmony-corrected z_mu_b condensation (theta={args.harmony_theta:g})"
            if args.harmony_theta is not None else "raw z_mu_b condensation"
        )
    )

    if result.position_history is not None:
        inspection_dir = OUT_DIR / "iteration_inspection"
        tc.render_iteration_inspection_bundle(
            position_history=result.position_history,
            snapshot_iters=result.snapshot_iters,
            mask=mask,
            time_values=time_values,
            labels=labels_arr,
            output_dir=inspection_dir,
            metrics_history=result.metrics_history,
            color_map={
                "inj_ctrl": "#2166AC", "wik_ab": "#808080",
                "pbx1b_crispant": "#9467bd", "pbx4_crispant": "#F7B267",
                "pbx1b_pbx4_crispant": "#B2182B",
            },
            title_prefix=title,
            n_select=6,
            config_payload={
                "relative_profile": relative_profile,
                "resolved_force_balance": tc.describe_force_balance(x0, mask, config),
            },
        )
        print(f"Saved iteration inspection -> {inspection_dir}")

    # ── Standard viz bundle ───────────────────────────────────────────────────
    GENOTYPE_COLORS = {
        "inj_ctrl": "#2166AC",
        "wik_ab": "#808080",
        "pbx1b_crispant": "#9467bd",
        "pbx4_crispant": "#F7B267",
        "pbx1b_pbx4_crispant": "#B2182B",
    }
    run = tc.load_run(npz_out, title=title, color_map=GENOTYPE_COLORS)
    tc.render_run(run, str(OUT_DIR), skip_animations=True)
    tc.time_slice_html(
        run.positions,
        run.mask,
        run.time_values,
        labels=run.labels,
        color_map=run.color_map,
        embryo_ids=run.embryo_ids,
        title=f"{title} | final condensation",
        output_path=OUT_DIR / "time_slice.html",
    )
    print(f"Saved viz bundle -> {OUT_DIR}")
    print("\nDone. Run 2_combined_html.py next.")


if __name__ == "__main__":
    main()
