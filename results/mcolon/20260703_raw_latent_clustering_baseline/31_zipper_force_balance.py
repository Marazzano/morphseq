"""
31_zipper_force_balance.py
--------------------------
Instrumented diagnostic for the temporal-zipper force (per analyst): don't just
watch the seam metric — measure the DIRECTIONAL FORCE BALANCE at the seam.

For seam points at t7 (and t13), each checkpoint records the ACTUAL UPDATE
vectors (not raw gradients) projected onto the zipper direction u_i = (c_i-p_i)/||..||:

  Delta_zip  = -lr * (momentum-carried) zipper-only gradient
  Delta_base = -lr * base-stack-only gradient
  m_zip  = Delta_zip . u   (>0 => moving toward temporal target)
  m_base = Delta_base . u   (<0 => base stack pushes away from target)

Interpretations (analyst's 7 mechanisms):
  m_zip>0, m_base<0, sum~0        -> genuine tug-of-war equilibrium (strength problem)
  m_zip, m_base both >0, seam flat -> island+trunk moving together (metric invariance)
  m_zip ~ 0 despite high gate      -> target/impl wrong (target problem)
  |c_i - trunk ridge| large        -> centroid target on wrong side (target problem)

Runs W in {0, 10, 50} from the SAME init and logs every 10 iters:
  seam_dist, mean gate@t7, mean ||c-p||@t7, ||g_zip||, ||g_base||,
  net motion toward target, island centroid, target centroid.

Output: figures/zipper_force_balance/balance_log.txt
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

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.trajectory_condensation.condensation import CondensationConfig
from analyze.trajectory_condensation.condensation.forces.total import total_energy_and_grad
from analyze.trajectory_condensation.condensation.coherence.compute import compute_coherence
from analyze.trajectory_condensation.condensation.engine.run import resolve_force_balance
from analyze.trajectory_condensation.condensation.geometry_refs import (
    build_local_scale_refs, build_slice_outlier_refs,
)

_spec = importlib.util.spec_from_file_location("_zip", _HERE / "30_temporal_coalesce_force.py")
_zip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_zip)
temporal_coalesce_grad = _zip.temporal_coalesce_grad

_specb = importlib.util.spec_from_file_location("_baseline", _HERE / "1_cluster_raw_latent.py")
_baseline = importlib.util.module_from_spec(_specb)
_specb.loader.exec_module(_baseline)

RAW_DIR = _HERE / "figures" / "condensed_raw_zmub"
OUT_DIR = _HERE / "figures" / "zipper_force_balance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ITER = 120
LOG_EVERY = 15
COH_EVERY = 25          # cache coherence (barely changes; the per-iter recompute was the bottleneck)
SEAM_BINS = [7, 13]
K_TEMPORAL = _zip.K_TEMPORAL


def zipper_target_and_gate(positions, mask, s_local, t, k=K_TEMPORAL,
                           tau=_zip.GATE_TAU, z_mid=_zip.GATE_ZMID):
    """For each obs point at bin t, return (target centroid at t-1, gate w, ||c-p||),
    computed toward the t-1 side (the entry-seam side). Mirrors the force internals
    exactly: s_local ruler + real GATE_ZMID / GATE_TAU."""
    T = positions.shape[1]
    obs = np.flatnonzero(mask[:, t])
    t_ref = t - 1
    ref = np.flatnonzero(mask[:, t_ref]) if t_ref >= 0 else np.array([], int)
    if obs.size == 0 or ref.size == 0:
        return obs, None, None, None
    p = positions[obs, t, :]
    pr = positions[ref, t_ref, :]
    d = np.linalg.norm(p[:, None, :] - pr[None, :, :], axis=-1)
    nearest = d.min(axis=1)
    z = nearest / s_local
    w = 1.0 / (1.0 + np.exp(-np.clip((z - z_mid) / tau, -60, 60)))
    k_eff = min(k, ref.size)
    nn = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
    target = pr[nn].mean(axis=1)
    return obs, target, w, np.linalg.norm(target - p, axis=1)


def run_one(W, x0, mask, config, rf, local_scale_refs, outlier_refs, s_local, fh):
    positions = x0.copy()
    # Real shared momentum + separate diagnostic accumulators (linear => v = vb+vz).
    velocities = np.zeros_like(positions)
    v_base = np.zeros_like(positions)
    v_zip = np.zeros_like(positions)
    coherence = None
    coh_last = -1
    lr, mom = config.solver_lr, config.solver_momentum

    print(f"\n===== W_COALESCE = {W} =====", file=fh)
    print(" iter  seam_d  gate@7  ||c-p||  gz.u    gb.u    Dz.u     Db.u     "
          "Dnet.u    consist   island_c        target_c", file=fh)

    for n in range(N_ITER):
        if coherence is None or (n - coh_last) >= COH_EVERY:
            coherence = compute_coherence(positions, mask, sigma=rf.sigma_coh,
                                          delta=config.temporal_cohere_window)
            coh_last = n
        mu = rf.fidelity_strength * (config.fidelity_half_life ** n)

        _, g_base = total_energy_and_grad(
            positions=positions, x0=x0, mask=mask, coherence=coherence,
            sigma=rf.sigma_att, epsilon_r=rf.epsilon_r, eta=rf.repulsion_eta,
            lambda_stretch=rf.lambda_stretch, lambda_bend=rf.lambda_bend,
            elasticity_kernel=config.elastic_kernel, s_step_ref=rf.geometry_s_step,
            s_bend_ref=rf.geometry_s_bend, mu=mu, k_attract=config.attract_k,
            subtract_mean_attraction=False, sigma_attract_local=None,
            epsilon_void=rf.void_strength, sigma_void=rf.void_bandwidth,
            outlier_strength=rf.outlier_strength, outlier_refs=outlier_refs,
            r_cut=0.0, lambda_scale=rf.local_scale_strength,
            neighborhood_info=local_scale_refs, w_attract=config.attract_weight,
            w_repel=1.0, w_fidelity=1.0, w_elastic=1.0, w_void=1.0, w_scale=1.0,
        )
        _, g_zip_raw = temporal_coalesce_grad(positions, mask, s_local=s_local, k=K_TEMPORAL)
        g_zip = W * g_zip_raw

        # advance real + diagnostic momenta (plain momentum SGD: no clip/normalize
        # after summation, so v == v_base + v_zip exactly; we log 'consist' to prove it)
        velocities = mom * velocities - lr * (g_base + g_zip)
        v_base = mom * v_base - lr * g_base
        v_zip = mom * v_zip - lr * g_zip

        # ── instrument at t7 using THIS iter's force direction, BEFORE update ──
        if n % LOG_EVERY == 0:
            t = 7
            obs, target, w_gate, cp = zipper_target_and_gate(positions, mask, s_local, t)
            if target is not None:
                p = positions[obs, t, :]
                u = (target - p)
                u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
                # raw-gradient directions (as UPDATES: -lr*g projected on u)
                gz_u = float((-lr * g_zip[obs, t, :] * u).sum(1).mean())   # instant zip
                gb_u = float((-lr * g_base[obs, t, :] * u).sum(1).mean())  # instant base
                # momentum-carried contributions
                Dz_u = float((v_zip[obs, t, :] * u).sum(1).mean())
                Db_u = float((v_base[obs, t, :] * u).sum(1).mean())
                Dnet_u = float((velocities[obs, t, :] * u).sum(1).mean())  # what point did
                consist = float(np.abs(velocities[obs, t, :]
                                       - (v_base[obs, t, :] + v_zip[obs, t, :])).max())
                island_c = p.mean(0); target_c = target.mean(0)
                seam_d = float(np.linalg.norm(island_c - target_c))
                print(f"{n:4d}  {seam_d:6.3f}  {float(w_gate.mean()):5.2f}  "
                      f"{float(cp.mean()):6.3f}  {gz_u:+.4f} {gb_u:+.4f} "
                      f"{Dz_u:+.4f} {Db_u:+.4f} {Dnet_u:+.4f}  {consist:.1e}  "
                      f"[{island_c[0]:5.2f},{island_c[1]:5.2f}]  "
                      f"[{target_c[0]:5.2f},{target_c[1]:5.2f}]", file=fh)
                fh.flush()

        positions = positions + velocities

    return positions


def main():
    d = np.load(RAW_DIR / "condensed_positions.npz", allow_pickle=True)
    x0 = d["x0"]; mask = d["mask"]
    rel = _baseline._relative_profile_from_margin_reference()
    rel["fidelity_init_strength"] = 0.0  # drop_fidelity, matches last run
    config = CondensationConfig(
        **rel, temporal_cohere_window=3, elastic_strength=16.0, elastic_mix=0.25,
        fidelity_half_life=_baseline._gamma_from_half_life_iters(70.0),
        void_strength=0.014, outlier_strength=16.0, outlier_cutoff_mode="robust",
        outlier_cutoff_value=3.0, attract_k=20, solver_lr=1e-4, solver_momentum=0.9,
        solver_max_iter=N_ITER,
    )
    rf = resolve_force_balance(x0, mask, config)
    s_local = rf.geometry_s_local
    local_scale_refs = build_local_scale_refs(x0, mask, k_local=5)
    outlier_refs = build_slice_outlier_refs(x0, mask, cutoff_mode="robust",
                                            robust_k=float(config.outlier_cutoff_value))
    print(f"s_local={s_local:.4f}  z_mid={_zip.GATE_ZMID}  tau={_zip.GATE_TAU}  "
          f"lr={config.solver_lr}  mom={config.solver_momentum}")

    report = OUT_DIR / "balance_log.txt"
    with open(report, "w") as fh:
        print(f"s_local={s_local:.4f}  z_mid={_zip.GATE_ZMID}  tau={_zip.GATE_TAU}  "
              f"lr={config.solver_lr}  mom={config.solver_momentum}  N_ITER={N_ITER}", file=fh)
        print("Dz.u>0 = zipper moves toward target; Db.u<0 = base opposes; "
              "Dnet~0 despite Dz>0 = base cancels (tug-of-war); "
              "island_c & target_c both moving = joint motion (metric invariant)", file=fh)
        for W in (0.0, 50.0):
            run_one(W, x0, mask, config, rf, local_scale_refs, outlier_refs, s_local, fh)
            fh.flush()
    print(report.read_text())


if __name__ == "__main__":
    main()
