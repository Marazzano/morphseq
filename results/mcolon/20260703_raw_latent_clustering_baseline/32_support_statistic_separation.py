"""
32_support_statistic_separation.py
----------------------------------
Decide whether the zipper's support statistic can separate a genuine seam from
ordinary temporal ridge motion, BEFORE tuning any threshold (per analyst).

Two failed rulers bracket the problem:
  d / s_step   -> false NEGATIVES (gate releases before closure)
  d / s_local  -> false POSITIVES (fires on healthy bins: normal ridge motion
                  already exceeds local spacing)

Candidate statistics for each observed point i at bin t, side s in {-1,+1}:
  d_i          = min_{j in t+s} ||p_i - p_j||            (raw cross-time gap)
  A) sev_local = d_i / s_local                           (current, false-positive prone)
  B) sev_peer  = (d_i - median(d_.,s)) / MAD(d_.,s)      (peer-relative robust z;
                 normal ridge motion cancels via the per-bin-pair median; the
                 orphan tail can't hide because ref is the population, and MAD is
                 robust to the tail inflating its own scale)

The decisive quantity is DISTRIBUTION SEPARATION between seam and healthy bins:
  for each statistic, compare its values at SEAM bins (t7=48hpf entry of 20251207,
  t13=72hpf entry of 20260306; measured on the t-1 side, the exposed side) vs at
  HEALTHY bins. If seam and healthy overlap heavily, the feature is insufficient
  and no z_mid works. If B separates and A doesn't, B is the fix.

Runs on the CONDENSED baseline positions (where t7=0.60 seam persists) so the
statistic is judged on the geometry the force must actually fix.

Output: figures/support_stat_separation/separation.txt
"""
from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.trajectory_condensation.condensation.geometry_refs import estimate_geometry_refs

RAW_DIR = _HERE / "figures" / "condensed_raw_zmub"
OUT_DIR = _HERE / "figures" / "support_stat_separation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEAM_BINS = {7, 13}   # t7=48hpf (20251207 enters), t13=72hpf (20260306 enters)


def cross_time_nearest(positions, mask, t, t_ref):
    """For each observed point at bin t, distance to nearest observed point at t_ref.
    Returns (embryo_rows, d array) or (None, None) if a side is empty."""
    T = positions.shape[1]
    if t_ref < 0 or t_ref >= T:
        return None, None
    obs = np.flatnonzero(mask[:, t])
    ref = np.flatnonzero(mask[:, t_ref])
    if obs.size == 0 or ref.size == 0:
        return None, None
    p = positions[obs, t, :]
    pr = positions[ref, t_ref, :]
    d = np.linalg.norm(p[:, None, :] - pr[None, :, :], axis=-1).min(axis=1)
    return obs, d


def robust_z(d):
    """(d - median) / (1.4826 * MAD), robust per-bin-pair standardization."""
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    spread = 1.4826 * mad if mad > 1e-12 else (np.std(d) if np.std(d) > 1e-12 else 1.0)
    return (d - med) / spread, med, spread


def summarize(vals):
    a = np.asarray(vals)
    return (f"n={a.size:4d}  med={np.median(a):6.2f}  q25={np.quantile(a,.25):6.2f}  "
            f"q75={np.quantile(a,.75):6.2f}  q90={np.quantile(a,.90):6.2f}  max={a.max():6.2f}")


def main():
    d = np.load(RAW_DIR / "condensed_positions.npz", allow_pickle=True)
    positions, mask, tv = d["positions"], d["mask"], d["time_values"]
    refs = estimate_geometry_refs(positions, mask)
    s_local = refs.s_local
    T = mask.shape[1]

    lines = [f"Support-statistic separation test (condensed baseline). s_local={s_local:.4f}",
             "Side = t-1 (the exposed/entry side). Seam bins: t7(48hpf), t13(72hpf).", ""]

    # Collect per-point statistics, tagged seam vs healthy, using the t-1 side.
    A_seam, A_healthy = [], []   # d / s_local
    B_seam, B_healthy = [], []   # peer-relative robust z within this bin-pair
    per_bin = []                 # (t, is_seam, medA, q90A, medB, q90B, frac_B_gt2)

    for t in range(T):
        obs, dd = cross_time_nearest(positions, mask, t, t - 1)
        if obs is None:
            continue
        A = dd / s_local
        B, med, spread = robust_z(dd)
        is_seam = t in SEAM_BINS
        (A_seam if is_seam else A_healthy).extend(A.tolist())
        (B_seam if is_seam else B_healthy).extend(B.tolist())
        per_bin.append((t, is_seam, float(np.median(A)), float(np.quantile(A, .90)),
                        float(np.median(B)), float(np.quantile(B, .90)),
                        float(np.mean(B > 2.0))))

    lines.append("Per-bin (t-1 side):  A=d/s_local   B=peer robust-z")
    lines.append(" t  hpf  seam  medA   q90A    medB   q90B   frac(B>2)")
    for (t, seam, mA, qA, mB, qB, fB) in per_bin:
        lines.append(f"{t:2d}  {tv[t]:3.0f}   {'S' if seam else '.'}   "
                     f"{mA:5.2f}  {qA:5.2f}   {mB:+5.2f}  {qB:+5.2f}   {fB:5.2f}")

    lines += ["", "=== DISTRIBUTION SEPARATION (seam vs healthy) ===",
              "A) d / s_local:",
              f"   seam:    {summarize(A_seam)}",
              f"   healthy: {summarize(A_healthy)}",
              "B) peer-relative robust-z:",
              f"   seam:    {summarize(B_seam)}",
              f"   healthy: {summarize(B_healthy)}", ""]

    # Separation metric: how distinguishable are seam points from healthy?
    def sep(seam, healthy):
        s, h = np.asarray(seam), np.asarray(healthy)
        d_med = np.median(s) - np.median(h)
        pooled = np.median(np.abs(s - np.median(s))) + np.median(np.abs(h - np.median(h)))
        pooled = pooled if pooled > 1e-9 else 1.0
        # fraction of seam points above healthy q90 (clean-separation proxy)
        frac_above = float(np.mean(s > np.quantile(h, 0.90)))
        return d_med, d_med / pooled, frac_above

    dA, nA, fA = sep(A_seam, A_healthy)
    dB, nB, fB = sep(B_seam, B_healthy)
    lines += ["Separation (median gap; median-gap/pooled-MAD; frac seam > healthy-q90):",
              f"  A d/s_local:   gap={dA:+.2f}  norm={nA:+.2f}  frac_sep={fA:.2f}",
              f"  B peer-z:      gap={dB:+.2f}  norm={nB:+.2f}  frac_sep={fB:.2f}", "",
              "VERDICT: higher norm & frac_sep = cleaner separation. If B >> A, the",
              "peer-relative statistic is the fix; gate on B. If neither separates,",
              "the cross-time-distance feature itself is insufficient."]

    report = "\n".join(lines)
    (OUT_DIR / "separation.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
