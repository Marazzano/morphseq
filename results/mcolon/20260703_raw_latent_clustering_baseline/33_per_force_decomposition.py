"""
33_per_force_decomposition.py
-----------------------------
Answer the user's question directly: at the t7 seam, decompose EVERY force into its
projection onto the desired merge direction u = (target - p)/||.||, and its relative
magnitude. Which force is preventing the strand from being incorporated into the
graph topology, and by how much?

Unlike 31 (which lumped all base forces into g_base), this calls each force module
directly and projects each onto u:
  attraction, repulsion, void, elasticity, fidelity, local_scale, slice_outlier,
  and the strand-propagated zipper.

For each checkpoint at t7 (mean over the unsupported/seam points), reports:
  <force>.u  = mean projection of (-lr * g_force) onto u  (>0 = helps merge, <0 = opposes)
  |<force>|  = mean gradient magnitude of that force at t7 (relative scale)
Plus seam_d and the strand's forward-bin activation.

Runs the ACTUAL Fix-B configuration (strand propagation, drop_fidelity) so the
decomposition reflects the real dynamics that plateaued at t7~0.6.

Output: figures/per_force_decomp/decomp.txt
"""
from __future__ import annotations

import os, sys, importlib.util
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.trajectory_condensation.condensation import CondensationConfig
from analyze.trajectory_condensation.condensation.coherence.compute import compute_coherence
from analyze.trajectory_condensation.condensation.engine.run import resolve_force_balance
from analyze.trajectory_condensation.condensation.geometry_refs import (
    build_local_scale_refs, build_slice_outlier_refs,
)
from analyze.trajectory_condensation.condensation.forces.attraction import attraction
from analyze.trajectory_condensation.condensation.forces.repulsion import repulsion
from analyze.trajectory_condensation.condensation.forces.void import void_repulsion
from analyze.trajectory_condensation.condensation.forces.elasticity import elasticity
from analyze.trajectory_condensation.condensation.forces.fidelity import fidelity
from analyze.trajectory_condensation.condensation.forces.local_scale import local_scale_preservation
from analyze.trajectory_condensation.condensation.forces.slice_outlier import slice_outlier

_spec = importlib.util.spec_from_file_location("_zip", _HERE / "30_temporal_coalesce_force.py")
_zip = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_zip)
_specb = importlib.util.spec_from_file_location("_baseline", _HERE / "1_cluster_raw_latent.py")
_baseline = importlib.util.module_from_spec(_specb); _specb.loader.exec_module(_baseline)

RAW_DIR = _HERE / "figures" / "condensed_raw_zmub"
OUT_DIR = _HERE / "figures" / "per_force_decomp"; OUT_DIR.mkdir(parents=True, exist_ok=True)
N_ITER, LOG_EVERY, T7 = 120, 20, 7
K = _zip.K_TEMPORAL


def merge_dir_and_seam(positions, mask, s_local, t=T7):
    """u (per t7 point), gate w, target, seam_d, and the unsupported-point subset."""
    obs = np.flatnonzero(mask[:, t]); ref = np.flatnonzero(mask[:, t - 1])
    p = positions[obs, t, :]; pr = positions[ref, t - 1, :]
    d = np.linalg.norm(p[:, None, :] - pr[None, :, :], axis=-1)
    nearest = d.min(1); z = nearest / s_local
    w = 1.0 / (1.0 + np.exp(-np.clip((z - _zip.GATE_ZMID) / _zip.GATE_TAU, -60, 60)))
    nn = np.argpartition(d, kth=min(K, ref.size) - 1, axis=1)[:, :min(K, ref.size)]
    target = pr[nn].mean(1)
    u = target - p; u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    seam_d = float(np.linalg.norm(p.mean(0) - target.mean(0)))
    return obs, u, w, seam_d


def main():
    d = np.load(RAW_DIR / "condensed_positions.npz", allow_pickle=True)
    x0, mask, ids = d["x0"], d["mask"], d["embryo_ids"]
    N_e = x0.shape[0]
    erbi = {i: np.flatnonzero(mask[i]).tolist() for i in range(N_e)}
    rel = _baseline._relative_profile_from_margin_reference()
    rel["fidelity_init_strength"] = 0.0
    config = CondensationConfig(**rel, temporal_cohere_window=3, elastic_strength=16.0,
        elastic_mix=0.25, fidelity_half_life=_baseline._gamma_from_half_life_iters(70.0),
        void_strength=0.014, outlier_strength=16.0, outlier_cutoff_mode="robust",
        outlier_cutoff_value=3.0, attract_k=20, solver_lr=1e-4, solver_momentum=0.9,
        solver_max_iter=N_ITER)
    rf = resolve_force_balance(x0, mask, config)
    s_local = rf.geometry_s_local
    lsr = build_local_scale_refs(x0, mask, k_local=5)
    orf = build_slice_outlier_refs(x0, mask, cutoff_mode="robust", robust_k=3.0)

    positions = x0.copy(); vel = np.zeros_like(positions); coh = None; cl = -1
    lr = config.solver_lr

    forces = ["attract", "repel", "void", "elastic", "fidelity", "scale", "outlier", "zip"]
    report = OUT_DIR / "decomp.txt"
    with open(report, "w") as fh:
        print(f"s_local={s_local:.4f}  Fix B strand-prop  W_zip={_zip.W_COALESCE}  "
              f"sigma_T={_zip.STRAND_SIGMA_T}", file=fh)
        print(f"At t7: proj of (-lr*g) on merge dir u (>0 helps merge, <0 OPPOSES); "
              f"averaged over unsupported seam pts.", file=fh)
        hdr = " iter seam_d  " + "  ".join(f"{f:>8}" for f in forces) + "   dominant_opposer"
        print(hdr, file=fh); fh.flush()

        for n in range(N_ITER):
            if coh is None or (n - cl) >= config.coherence_cache_every:
                coh = compute_coherence(positions, mask, sigma=rf.sigma_coh,
                                        delta=config.temporal_cohere_window); cl = n
            mu = rf.fidelity_strength * (config.fidelity_half_life ** n)

            _, g_att = attraction(positions, mask, coh, rf.sigma_att, k_attract=config.attract_k)
            _, g_rep = repulsion(positions, mask, rf.epsilon_r, rf.repulsion_eta)
            _, g_void = void_repulsion(positions, mask, rf.void_strength, rf.void_bandwidth)
            _, g_ela = elasticity(positions, mask, rf.lambda_stretch, rf.lambda_bend,
                                  elasticity_kernel=config.elastic_kernel,
                                  s_step_ref=rf.geometry_s_step, s_bend_ref=rf.geometry_s_bend)
            _, g_fid = fidelity(positions, x0, mask, mu)
            _, g_scl = local_scale_preservation(positions, mask, lsr, rf.local_scale_strength)
            _, g_out = slice_outlier(positions, mask, orf, rf.outlier_strength)
            _, g_zip = _zip.strand_propagated_grad(positions, mask, erbi, s_local=s_local,
                                                   k=K, sigma_T=_zip.STRAND_SIGMA_T)
            g_zip = _zip.W_COALESCE * g_zip

            gmap = {"attract": config.attract_weight * g_att, "repel": g_rep, "void": g_void,
                    "elastic": g_ela, "fidelity": g_fid, "scale": g_scl, "outlier": g_out,
                    "zip": g_zip}
            g_total = sum(gmap.values())

            if n % LOG_EVERY == 0:
                obs, u, w, seam_d = merge_dir_and_seam(positions, mask, s_local)
                sel = obs[w > 0.05] if np.any(w > 0.05) else obs  # unsupported seam pts
                locmask = np.isin(obs, sel)
                projs = {}
                for f, g in gmap.items():
                    dvec = -lr * g[obs, T7, :][locmask]     # actual update contribution
                    projs[f] = float((dvec * u[locmask]).sum(1).mean())
                opp = min(projs, key=projs.get)  # most negative = dominant opposer
                row = f"{n:4d}  {seam_d:5.3f}  " + "  ".join(f"{projs[f]:+8.5f}" for f in forces)
                row += f"   {opp}({projs[opp]:+.4f})"
                print(row, file=fh); fh.flush()

            vel = config.solver_momentum * vel - lr * g_total
            positions = positions + vel

    print(report.read_text())


if __name__ == "__main__":
    main()
