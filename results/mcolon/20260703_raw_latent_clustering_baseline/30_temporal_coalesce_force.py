"""
30_temporal_coalesce_force.py
-----------------------------
PROTOTYPE (results-side, no src/ changes) of the temporal-coalescence force.

Idea (user MVP): purely in the (x,y,T) embedding grid, each point is attracted
toward the CENTROID of its k nearest neighbors in the ADJACENT time bins (t-1 and
t+1). Islands that UMAP/Procrustes stranded at a batch entry get pulled toward the
mass of the previous/next bin (the trunk), so isolated branches coalesce onto the
main connected branch. No unconnected branches left behind. Label-free; pure
embedding geometry (no feature tensor).

This runs a fresh condensation solve from x0 that mirrors run_dynamics' force stack
AND adds the new coalescence force, then scores the ridge vs the baseline. If it
does not close the t13=72hpf seam, the fallback is to drop the fidelity (init)
constraint (which decays anyway).

Outputs (figures/condensed_raw_zmub_coalesce/):
  condensed_positions.npz, ridge_comparison.txt,
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

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import analyze.trajectory_condensation as tc
from analyze.trajectory_condensation.condensation import CondensationConfig
from analyze.trajectory_condensation.condensation.geometry_refs import (
    estimate_geometry_refs, build_local_scale_refs, build_slice_outlier_refs,
)
from analyze.trajectory_condensation.condensation.forces.total import total_energy_and_grad
from analyze.trajectory_condensation.condensation.coherence.compute import compute_coherence
from analyze.trajectory_condensation.condensation.engine.run import resolve_force_balance
from analyze.trajectory_condensation.seam_bridge_init import (
    ridge_score, estimate_attraction_bandwidth,
)

_spec = importlib.util.spec_from_file_location("_baseline", _HERE / "1_cluster_raw_latent.py")
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

N_ITER = _baseline.N_ITER
SAVE_EVERY = _baseline.SAVE_EVERY
_K_LOCAL = 5  # matches _K_LOCAL_SCALE_DEFAULT in run.py

RAW_DIR = _HERE / "figures" / "condensed_raw_zmub"
OUT_DIR = _HERE / "figures" / "condensed_raw_zmub_coalesce"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_COLORS = {"20251207_pbx": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC", "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd", "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}

# ── Force knobs (support-conditioned temporal zipper) ─────────────────────────
K_TEMPORAL = 8         # k nearest at t_ref to average into the pull target
GATE_TAU = 0.5         # logistic softness on severity z = d / s_local
GATE_ZMID = 3.5        # gate midpoint (units of s_local). Set from the separation
                       # test (32_...): healthy q90 ~1.2, seam tail 3-14. z_mid=3.5
                       # (~off below 2.5, ~on above 4.5) sits in the clean gap:
                       # fires on the seam's exposed "teeth", silent on healthy
                       # bins (only ~0.3% of healthy points exceed z=3).
R_MAX_MULT = 3.0       # cap displacement at R_MAX_MULT * s_local (no trebuchet)
DROP_FIDELITY = True   # remove init anchor so exposed strand-ends can move
# ANNEALED zipper (user): strong at iter 0, decays STEEPLY, so the geometry is
# shifted into the right basin while everything is still molten (base forces have
# not locked the isolated island yet), then the zipper fades and base forces
# settle the ALREADY-MERGED configuration instead of re-separating it.
# W(n) = W_COALESCE * exp(-n / ANNEAL_TAU).  Steep decay = small ANNEAL_TAU.
W_COALESCE = 50.0      # zipper strength (guides un-pinned island home)
ANNEAL_TAU = 1e12      # ~no anneal now: with suppression the basin is gone, so the
                       # zipper need not out-muscle it; keep steady, releases via gate.
# CONDITIONAL SUPPRESSION (the actual fix, forced by elimination): while a point is
# unsupported (gate w), damp the attraction/coherence that pins it in its isolated
# island. Removes the attractor so ordinary trunk forces + gentle zipper merge it;
# restores smoothly as support forms (w->0). Prior temporary-force approaches all
# failed because the basin re-formed once the force stopped.
SUPPRESS_STRENGTH = 0.0  # Fix A (coherence suppression) did NOT close seam => off.
# Fix B: STRAND PROPAGATION. Fix A proved coherence isn't the jailer -- ELASTICITY
# is (it transmits the endpoint pull as a restoring force). So translate a SEGMENT
# of the strand coherently (shared direction, temporal kernel along same embryo),
# so elasticity sees ~no stretch change and stops fighting.
USE_STRAND_PROP = True    # use strand-propagated force instead of pointwise zipper
STRAND_SIGMA_T = 3.0      # temporal bandwidth (bins) of the propagation kernel
# REPULSION DAMPING (the fix, from per-force decomposition #33): repulsion is the
# jailer preventing condensation at the merge zone. Damp it for unsupported points.
DAMP_REPULSION = 1.0      # soften the repulsion barrier at the jam (decomp #33)
# JAMMING / NOISE hypothesis (user): the solve is zero-temperature gradient descent;
# two dense groups meeting edge-to-edge JAM (tectonic plates) and cannot interdigitate
# because interleaving requires transiently climbing the repulsion barrier (going
# uphill). The plateau at a STABLE seam_d is the jammed-minimum signature. Add annealed
# stochastic jiggle (simulated annealing) to unsupported points so they can hop the
# barrier and slot between trunk points, then cool. Tests: does noise close the seam
# where deterministic forces couldn't, AND does it INTERLEAVE (not just abut)?
NOISE_T0 = 2.0            # initial jiggle amplitude in units of s_local (0 = off)
NOISE_ANNEAL_TAU = 60.0   # cooling time constant (iters)
# CRITICAL: the support/closure ruler is s_LOCAL (within-bin nearest-neighbor
# spacing), NOT s_step. s_step is a TRAJECTORY scale (how far a strand moves per
# time bin) and is an artifact of the binning grid; it answers "can this make the
# temporal jump", the wrong question. The zipper asks a SUPPORT question -- "has
# this strand re-entered the local ridge" -- whose ruler is local point spacing.
# s_local is exactly what estimate_attraction_bandwidth / ridge_score use to
# DETECT the seam, so detection, rescue, and closure now share one ruler.
# s_local is FIXED from x0 (trusted initial geometry): if recomputed each iter it
# shrinks as the embedding contracts and the finish line retreats from the runner.


def _experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if len(parts) > 1 and parts[1].isalpha() else parts[0]


def unsupported_gate(positions, mask, s_local, tau=GATE_TAU, z_mid=GATE_ZMID):
    """Per-point 'unsupportedness' w in [0,1] (N_e, T), max over the two temporal
    sides of the same severity gate the zipper uses: w~1 if the point lacks a
    cross-time neighbor within local ridge spacing, ~0 if supported. Used to
    SUPPRESS the attraction/coherence that pins an unsupported island (the basin
    the base forces keep re-forming), so ordinary trunk forces can pull it home."""
    N_e, T, _ = positions.shape
    w = np.zeros((N_e, T))
    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        if obs.size == 0:
            continue
        p = positions[obs, t, :]
        side_w = np.zeros(obs.size)
        for t_ref in (t - 1, t + 1):
            if t_ref < 0 or t_ref >= T:
                continue
            ref = np.flatnonzero(mask[:, t_ref])
            if ref.size == 0:
                continue
            dmin = np.linalg.norm(p[:, None, :] - positions[ref, t_ref, :][None], axis=-1).min(1)
            z = dmin / s_local
            wr = 1.0 / (1.0 + np.exp(-np.clip((z - z_mid) / tau, -60, 60)))
            side_w = np.maximum(side_w, wr)
        w[obs, t] = side_w
    return w


def damped_repulsion(positions, mask, epsilon_r, eta, w_unsup, damp_strength=1.0):
    """Soft-core repulsion (E=eps/(r^4+eta)) with per-point damping for UNSUPPORTED
    points. The per-force decomposition (33) proved REPULSION is what prevents the
    island from condensing onto the trunk: it grows monotonically to become the
    dominant force at the merge zone ('the space won't compress'). For a pair (i,j)
    the repulsion weight is (1 - damp*w_i)(1 - damp*w_j): an unsupported island point
    feels REDUCED repulsion from the trunk while merging, so the space can condense;
    once supported (w->0) full repulsion restores, preventing collapse. Returns
    (energy, grad), a drop-in replacement for the base repulsion term.
    """
    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)
    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        if obs.size < 2:
            continue
        pos_obs = positions[obs, t, :]
        diff = pos_obs[:, None, :] - pos_obs[None, :, :]
        sq = (diff ** 2).sum(-1)
        valid = np.ones_like(sq); np.fill_diagonal(valid, 0.0)
        # per-point damping factor, then symmetric pair weight
        f = np.clip(1.0 - damp_strength * w_unsup[obs, t], 0.0, 1.0)  # (n_obs,)
        pair_w = valid * f[:, None] * f[None, :]
        denom = sq ** 2 + eta
        energy += (epsilon_r / denom * pair_w).sum()
        coeff = -4.0 * epsilon_r * sq / (denom ** 2) * pair_w
        coeff_sym = coeff + coeff.T
        grad[obs, t, :] += (coeff_sym[:, :, None] * diff).sum(1)
    return energy, grad


def suppress_coherence(coherence, w_unsup, suppress_strength=1.0):
    """Damp coherence bonds of unsupported points: for point i at bin t,
    multiply coherence[i,:,t] and [:,i,t] by (1 - s*w_i). Un-pins the island
    from its self-reinforcing attraction basin. Returns a suppressed COPY."""
    C = coherence.copy()
    T = C.shape[2]
    for t in range(T):
        factor = 1.0 - suppress_strength * w_unsup[:, t]  # (N_e,)
        C[:, :, t] *= factor[:, None]
        C[:, :, t] *= factor[None, :]
    return C


def temporal_coalesce_grad(positions, mask, s_local, k=K_TEMPORAL, tau=GATE_TAU,
                           z_mid=GATE_ZMID, r_max_mult=R_MAX_MULT):
    """Support-conditioned temporal zipper force (returns energy, grad).

    First principles: every point should have support (a neighbor within LOCAL
    RIDGE SPACING) at t-1 AND t+1. The defect is a strand end with no such
    neighbor on one side (a batch entry lacks t-1; an exit lacks t+1). For each
    EXISTING side s of each point i:

      d_{i,s}  = distance to NEAREST neighbor in bin t+s            (support probe)
      z        = d_{i,s} / s_local                                 (LOCAL ruler)
      w        = logistic((z - z_mid) / tau)                       # ~0 when z<~1
      c        = centroid of i's k nearest neighbors in bin t+s    (pull target)
      F       += W * w * cap(c - p_i)                              # toward support

    CRUCIAL (analyst): the ruler is ``s_local`` = within-bin nearest-neighbor
    spacing, NOT s_step. s_step is a TRAJECTORY scale (per-bin displacement) and
    an artifact of the binning grid; normalizing by it answers "can this make the
    temporal jump" -- the wrong question, and it made the gate release ~5x too
    early (empirically: seam stalled at ~0.75 while gate collapsed to 0.05). The
    zipper asks a SUPPORT question -- "has this strand re-entered the local ridge"
    -- whose correct ruler is local point spacing. s_local is exactly what
    estimate_attraction_bandwidth / ridge_score use to DETECT the seam, so
    detection, rescue, and closure share one ruler. z~1 => within local ridge
    spacing (supported, gate off); z>>1 => stranded (gate on). z_mid ~ 2 keeps the
    gate active until the strand is within ~1 local spacing of the ridge.

    s_local must be FIXED (from x0), not recomputed each iter: a shrinking ruler
    retreats the finish line as the embedding contracts.

    Displacement capped at ``r_max_mult * s_local`` (no trebuchet); gate w not
    capped. Both sides gated independently -> supported side contributes nothing.
    A strand's t+1 neighbors are itself, so pulling the exposed t-1 end drags the
    whole strand along via elasticity -- the zipper -- releasing as z->1.
    """
    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)
    if not np.isfinite(s_local) or s_local <= 0:
        return energy, grad
    r_max = r_max_mult * s_local  # displacement cap (self-calibrated to local spacing)

    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        if obs.size == 0:
            continue
        p = positions[obs, t, :]
        for t_ref in (t - 1, t + 1):
            if t_ref < 0 or t_ref >= T:
                continue
            ref = np.flatnonzero(mask[:, t_ref])
            if ref.size == 0:
                continue
            pr = positions[ref, t_ref, :]
            d = np.linalg.norm(p[:, None, :] - pr[None, :, :], axis=-1)  # (n_obs, n_ref)
            nearest = d.min(axis=1)                                      # d_{i,s}

            z = nearest / s_local
            gate_arg = (z - z_mid) / tau
            w = 1.0 / (1.0 + np.exp(-np.clip(gate_arg, -60.0, 60.0)))    # (n_obs,)

            k_eff = min(k, ref.size)
            nn = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
            target = pr[nn].mean(axis=1)                                 # (n_obs, 2)

            pull = target - p                                           # toward support
            pull_norm = np.linalg.norm(pull, axis=1)
            scale_cap = np.minimum(1.0, r_max / np.maximum(pull_norm, 1e-12))
            pull_capped = pull * scale_cap[:, None]

            energy += 0.5 * (w * (pull_capped ** 2).sum(axis=1)).sum()
            grad[obs, t, :] += -w[:, None] * pull_capped  # descent moves toward target

    return energy, grad


def strand_propagated_grad(positions, mask, embryo_rows_by_id, s_local,
                           k=K_TEMPORAL, tau=GATE_TAU, z_mid=GATE_ZMID,
                           r_max_mult=R_MAX_MULT, sigma_T=STRAND_SIGMA_T):
    """Strand-propagated rescue (Fix B): translate a SEGMENT of the orphan strand,
    not just its exposed endpoint. Returns (energy, grad).

    The pointwise zipper lost the tug-of-war because it pulls a few exposed points
    while the WHOLE strand's elasticity resists (endpoint towing a body on a short
    leash). Here, for each unsupported point i (gate w_i, attachment direction
    v_i = target_centroid - p_i), the SAME direction v_i is applied to i AND to i's
    same-embryo future points, kernel-weighted K(Delta)=exp(-Delta^2/2 sigma_T^2).
    The segment translates COHERENTLY, so elasticity sees ~no stretch change and
    has nothing to restore -> the strand glides toward support instead of snapping
    back. Direction is SHARED (not each point re-choosing its own centroid), which
    prevents the strand folding/compressing.

    Only the t-1 (entry) side is propagated forward along the strand; the exposed
    end of a batch ENTRY is at t-1, and its body extends into t+1, t+2, ...
    """
    N_e, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)
    if not np.isfinite(s_local) or s_local <= 0:
        return energy, grad
    r_max = r_max_mult * s_local
    half = int(np.ceil(3 * sigma_T))  # propagate up to ~3 sigma along the strand

    for t in range(T):
        t_ref = t - 1
        if t_ref < 0:
            continue
        obs = np.flatnonzero(mask[:, t])
        ref = np.flatnonzero(mask[:, t_ref])
        if obs.size == 0 or ref.size == 0:
            continue
        p = positions[obs, t, :]
        pr = positions[ref, t_ref, :]
        d = np.linalg.norm(p[:, None, :] - pr[None, :, :], axis=-1)
        nearest = d.min(axis=1)
        z = nearest / s_local
        w = 1.0 / (1.0 + np.exp(-np.clip((z - z_mid) / tau, -60.0, 60.0)))

        k_eff = min(k, ref.size)
        nn = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
        target = pr[nn].mean(axis=1)
        pull = target - p
        pnorm = np.linalg.norm(pull, axis=1)
        pull = pull * np.minimum(1.0, r_max / np.maximum(pnorm, 1e-12))[:, None]

        # For each unsupported seam point, propagate its (gated) direction along
        # its OWN embryo's future strand with a temporal Gaussian kernel.
        for li, ei in enumerate(obs):
            wi = w[li]
            if wi < 1e-3:
                continue
            v = wi * pull[li]                       # shared attachment direction
            rows = embryo_rows_by_id.get(int(ei))   # this embryo's (t_j) presence
            if rows is None:
                # fall back: apply to the seam point only
                grad[ei, t, :] += -v
                energy += 0.5 * (v ** 2).sum()
                continue
            for (tj) in rows:
                dlt = tj - t
                if dlt < 0 or dlt > half:            # forward along strand only
                    continue
                if not mask[ei, tj]:
                    continue
                kd = float(np.exp(-(dlt ** 2) / (2 * sigma_T ** 2)))
                grad[ei, tj, :] += -kd * v
                energy += 0.5 * kd * (v ** 2).sum()

    return energy, grad


def main() -> None:
    d = np.load(RAW_DIR / "condensed_positions.npz", allow_pickle=True)
    x0 = d["x0"]
    mask = d["mask"]
    time_values = d["time_values"]
    embryo_ids = d["embryo_ids"]
    labels_arr = d["labels"]
    batch = np.array([str(e).split("_")[0] for e in embryo_ids], dtype=object)
    N_e, T, _ = x0.shape
    # strand map: embryo row -> list of observed time bins (the same-embryo strand)
    embryo_rows_by_id = {i: np.flatnonzero(mask[i]).tolist() for i in range(N_e)}
    print(f"x0 {x0.shape}, {int(mask.sum())} obs, batches={sorted(set(batch))}")

    # Baseline config (same as 1_cluster_raw_latent.py)
    rel = _baseline._relative_profile_from_margin_reference()
    if DROP_FIDELITY:
        rel["fidelity_init_strength"] = 0.0
    config = CondensationConfig(
        **rel, temporal_cohere_window=3, elastic_strength=16.0, elastic_mix=0.25,
        fidelity_half_life=_baseline._gamma_from_half_life_iters(70.0),
        void_strength=0.014, outlier_strength=16.0, outlier_cutoff_mode="robust",
        outlier_cutoff_value=3.0, attract_k=20, solver_lr=1e-4, solver_momentum=0.9,
        solver_max_iter=N_ITER,
    )

    rf = resolve_force_balance(x0, mask, config)
    local_scale_refs = build_local_scale_refs(x0, mask, k_local=_K_LOCAL)
    outlier_refs = build_slice_outlier_refs(
        x0, mask, cutoff_mode="robust", robust_k=float(config.outlier_cutoff_value)
    )

    positions = x0.copy()
    velocities = np.zeros_like(positions)
    coherence = None
    coh_last = -1
    s_local = rf.geometry_s_local
    rng = np.random.default_rng(0)

    print(f"Solve: strand_prop={USE_STRAND_PROP} damp_repel={DAMP_REPULSION} "
          f"noise_T0={NOISE_T0} W0={W_COALESCE} drop_fidelity={DROP_FIDELITY}...")
    for n in range(N_ITER):
        if coherence is None or (n - coh_last) >= config.coherence_cache_every:
            coherence = compute_coherence(
                positions, mask, sigma=rf.sigma_coh, delta=config.temporal_cohere_window,
            )
            coh_last = n

        # per-point unsupportedness gate (drives suppression, damped-repulsion, noise)
        w_unsup = unsupported_gate(positions, mask, s_local)

        coh_eff = (suppress_coherence(coherence, w_unsup, SUPPRESS_STRENGTH)
                   if SUPPRESS_STRENGTH > 0 else coherence)
        w_repel_base = 0.0 if DAMP_REPULSION > 0 else 1.0  # replace base repel if damping

        mu = rf.fidelity_strength * (config.fidelity_half_life ** n)
        energies, grad = total_energy_and_grad(
            positions=positions, x0=x0, mask=mask, coherence=coh_eff,
            sigma=rf.sigma_att, epsilon_r=rf.epsilon_r, eta=rf.repulsion_eta,
            lambda_stretch=rf.lambda_stretch, lambda_bend=rf.lambda_bend,
            elasticity_kernel=config.elastic_kernel, s_step_ref=rf.geometry_s_step,
            s_bend_ref=rf.geometry_s_bend, mu=mu, k_attract=config.attract_k,
            subtract_mean_attraction=False, sigma_attract_local=None,
            epsilon_void=rf.void_strength, sigma_void=rf.void_bandwidth,
            outlier_strength=rf.outlier_strength, outlier_refs=outlier_refs,
            r_cut=0.0, lambda_scale=rf.local_scale_strength,
            neighborhood_info=local_scale_refs, w_attract=config.attract_weight,
            w_repel=w_repel_base, w_fidelity=1.0, w_elastic=1.0, w_void=1.0, w_scale=1.0,
        )
        if DAMP_REPULSION > 0:
            _, g_rep_d = damped_repulsion(positions, mask, rf.epsilon_r,
                                          rf.repulsion_eta, w_unsup, DAMP_REPULSION)
            grad = grad + g_rep_d

        # ── Rescue force: strand-propagated (Fix B) or pointwise zipper ────────
        w_anneal = W_COALESCE * float(np.exp(-n / ANNEAL_TAU))
        if USE_STRAND_PROP:
            e_co, g_co = strand_propagated_grad(
                positions, mask, embryo_rows_by_id, s_local=rf.geometry_s_local,
                k=K_TEMPORAL, sigma_T=STRAND_SIGMA_T,
            )
        else:
            e_co, g_co = temporal_coalesce_grad(
                positions, mask, s_local=rf.geometry_s_local, k=K_TEMPORAL
            )
        grad = grad + w_anneal * g_co

        grad *= mask[:, :, None].astype(float)
        velocities = config.solver_momentum * velocities - config.solver_lr * grad
        positions = positions + velocities

        # ── ANNEALED NOISE (jamming test): jiggle unsupported points so they can hop
        # the repulsion barrier and interdigitate with the trunk, then cool. Scaled
        # by per-point unsupportedness so only the jammed seam is agitated. ─────────
        if NOISE_T0 > 0:
            temp = NOISE_T0 * s_local * float(np.exp(-n / NOISE_ANNEAL_TAU))
            if temp > 1e-9:
                jiggle = rng.normal(0.0, temp, size=positions.shape)
                jiggle *= (w_unsup[:, :, None] * mask[:, :, None])  # only unsupported obs
                positions = positions + jiggle

        if n % max(1, N_ITER // 20) == 0:
            print(f"  iter {n:4d}  W_zip={w_anneal:7.2f}  E_total={energies['total']:.3e}  "
                  f"E_coalesce={e_co:.3e}")

    # ── Score ridge vs baseline ──────────────────────────────────────────────
    h = estimate_attraction_bandwidth(x0, mask, labels=None, quantile=0.5,
                                      match_within_label=False)
    base_pos = d["positions"]
    rs_base = ridge_score(base_pos, mask, batch=batch, attraction_bandwidth=h)
    rs_new = ridge_score(positions, mask, batch=batch, attraction_bandwidth=h)

    def seam(rs, t):
        v = rs["per_time"].get(t)
        return v["n_one_sided"] / v["n"] if v else float("nan")

    lines = [f"Annealed-zipper prototype (W0={W_COALESCE}, anneal_tau={ANNEAL_TAU}, "
             f"z_mid={GATE_ZMID}, k={K_TEMPORAL}, drop_fidelity={DROP_FIDELITY}). "
             f"h={h:.4f}", ""]
    lines.append(" t   hpf   n   ridge_base  ridge_coalesce")
    for t in range(T):
        v = rs_new["per_time"].get(t)
        n_t = v["n"] if v else int(mask[:, t].sum())
        fl = "  <-ENTRY" if t in (7, 13) else ""
        lines.append(f"{t:2d}  {time_values[t]:4.0f}  {n_t:3d}   {seam(rs_base,t):5.2f}"
                     f"       {seam(rs_new,t):5.2f}{fl}")
    lines.append("")
    lines.append(f"OVERALL  base={rs_base['frac_one_sided']:.3f}  "
                 f"coalesce={rs_new['frac_one_sided']:.3f}")
    lines.append(f"  t7(48hpf):  base={seam(rs_base,7):.2f}  coalesce={seam(rs_new,7):.2f}")
    lines.append(f"  t13(72hpf): base={seam(rs_base,13):.2f}  coalesce={seam(rs_new,13):.2f}")

    # ── INTERLEAVING metric: at t7, are 20251207 & 20260304 MERGED (share neighbors)
    # or just ABUTTED as two jammed plates? Fraction of each 20251207@t7 point's k
    # nearest SAME-BIN neighbors that are the OTHER batch. ~0.5 = interleaved/merged;
    # ~0 = jammed plates (island points only neighbor their own kind). ────────────
    def interleave(pos, t=7, kk=10):
        obs = np.flatnonzero(mask[:, t]); b = batch[obs]
        if len(set(b)) < 2:
            return float("nan")
        p = pos[obs, t, :]
        dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        fr = []
        for i in range(len(obs)):
            nn = np.argsort(dd[i])[:kk]
            fr.append(np.mean(b[nn] != b[i]))
        return float(np.mean(fr))
    il_base = interleave(base_pos); il_new = interleave(positions)
    lines.append("")
    lines.append(f"INTERLEAVE @t7 (frac cross-batch in 10nn; 0=jammed plates, "
                 f"0.5=merged):  base={il_base:.3f}  new={il_new:.3f}")
    # Descriptive run tag from the active knobs, so runs don't overwrite & the
    # filename tells you the config (e.g. strandprop_damprepel_noise2).
    parts = ["strandprop" if USE_STRAND_PROP else "zipper"]
    if DAMP_REPULSION > 0:
        parts.append(f"damprepel{DAMP_REPULSION:g}")
    if NOISE_T0 > 0:
        parts.append(f"noise{NOISE_T0:g}")
    if SUPPRESS_STRENGTH > 0:
        parts.append(f"suppress{SUPPRESS_STRENGTH:g}")
    if not DROP_FIDELITY:
        parts.append("fidelity")
    tag = "_".join(parts)
    lines.insert(1, f"RUN TAG: {tag}")
    report = "\n".join(lines)

    (OUT_DIR / f"{tag}__ridge.txt").write_text(report)
    print("\n" + report)

    np.savez(OUT_DIR / f"{tag}__positions.npz", positions=positions, x0=x0, mask=mask,
             time_values=time_values, embryo_ids=embryo_ids, labels=labels_arr)

    experiments = np.array([_experiment_of(str(e)) for e in embryo_ids])
    tc.time_slice_html(positions, mask, time_values, labels=experiments,
                       color_map=EXPERIMENT_COLORS, embryo_ids=embryo_ids,
                       title=f"{tag} | experiment",
                       output_path=OUT_DIR / f"{tag}__by_experiment.html")
    tc.time_slice_html(positions, mask, time_values, labels=labels_arr,
                       color_map=GENOTYPE_COLORS, embryo_ids=embryo_ids,
                       title=f"{tag} | genotype",
                       output_path=OUT_DIR / f"{tag}__by_genotype.html")
    print(f"Saved HTMLs with tag: {tag}")


if __name__ == "__main__":
    main()
