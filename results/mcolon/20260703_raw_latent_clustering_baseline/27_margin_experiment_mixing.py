"""
27_margin_experiment_mixing.py
------------------------------
Reference check: does the MARGIN/classifier-initialized space have the 48hpf
batch-entry seam, and does it mix experiments where raw z_mu_b does not?

The margin condensation (combined_raw_condensation_5class_bin4_perm500) is the
known-good reference the raw-latent baseline is compared against. If it does NOT
show the seam, its mechanism for mixing experiments is the answer.

Outputs (figures/margin_reference_experiment/):
  time_slice_by_experiment_x0.html         — margin initializer, colored by experiment
  time_slice_by_experiment_condensed.html  — margin condensation, colored by experiment
  time_slice_by_genotype_condensed.html    — margin condensation, colored by genotype
  mixing_by_bin.txt                        — cross-batch kNN overlap + ridge, per bin
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib
matplotlib.use("Agg")

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import analyze.trajectory_condensation as tc
from analyze.trajectory_condensation.seam_bridge_init import (
    ridge_score, estimate_attraction_bandwidth,
)

MARGIN_NPZ = (
    _HERE.parent / "20260407_pbx_analysis_cont" / "results" / "positioning" / "trajectory"
    / "combined_raw_condensation_5class_bin4_perm500" / "condensed_positions.npz"
)
OUT_DIR = _HERE / "figures" / "margin_reference_experiment"
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


def cross_batch_overlap(positions, mask, batch, t, k=10):
    """Fraction of each point's k nearest same-bin neighbors that are a DIFFERENT
    batch, averaged over points at bin t. High => batches mixed; ~0 => separated."""
    obs = np.flatnonzero(mask[:, t])
    if obs.size < k + 2:
        return float("nan"), obs.size
    pos = positions[obs, t, :]
    b = batch[obs]
    if len(set(b.tolist())) < 2:
        return 0.0, obs.size  # only one batch present -> no cross-batch possible
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    fracs = []
    for i in range(len(obs)):
        nn = np.argsort(d[i])[:k]
        fracs.append(np.mean(b[nn] != b[i]))
    return float(np.mean(fracs)), obs.size


def main() -> None:
    d = np.load(MARGIN_NPZ, allow_pickle=True)
    positions, x0, mask = d["positions"], d["x0"], d["mask"]
    time_values, embryo_ids, labels = d["time_values"], d["embryo_ids"], d["labels"]
    batch = np.array([str(e).split("_")[0] for e in embryo_ids], dtype=object)
    experiments = np.array([_experiment_of(str(e)) for e in embryo_ids])
    print(f"margin ref: {positions.shape}, batches={sorted(set(batch))}")

    # ── HTMLs colored by experiment (x0 and condensed) + genotype ────────────
    tc.time_slice_html(
        x0, mask, time_values, labels=experiments, color_map=EXPERIMENT_COLORS,
        embryo_ids=embryo_ids, title="MARGIN initializer (x0) | experiment",
        output_path=OUT_DIR / "time_slice_by_experiment_x0.html",
    )
    tc.time_slice_html(
        positions, mask, time_values, labels=experiments, color_map=EXPERIMENT_COLORS,
        embryo_ids=embryo_ids, title="MARGIN condensation | experiment",
        output_path=OUT_DIR / "time_slice_by_experiment_condensed.html",
    )
    tc.time_slice_html(
        positions, mask, time_values, labels=labels, color_map=GENOTYPE_COLORS,
        embryo_ids=embryo_ids, title="MARGIN condensation | genotype",
        output_path=OUT_DIR / "time_slice_by_genotype_condensed.html",
    )
    print("Saved 3 HTMLs")

    # ── Per-bin mixing + ridge, margin x0 and condensed ──────────────────────
    h_x0 = estimate_attraction_bandwidth(x0, mask, labels=None, quantile=0.5,
                                         match_within_label=False)
    h_cd = estimate_attraction_bandwidth(positions, mask, labels=None, quantile=0.5,
                                         match_within_label=False)
    rs_x0 = ridge_score(x0, mask, batch=batch, attraction_bandwidth=h_x0)
    rs_cd = ridge_score(positions, mask, batch=batch, attraction_bandwidth=h_cd)

    def seam(rs, t):
        v = rs["per_time"].get(t)
        return v["n_one_sided"] / v["n"] if v else float("nan")

    T = mask.shape[1]
    lines = []
    lines.append(f"MARGIN reference: cross-batch mixing + ridge per bin")
    lines.append(f"  h_x0={h_x0:.3f}  h_condensed={h_cd:.3f}")
    lines.append("")
    lines.append(" t   hpf   n   batches            mix_x0  mix_cond   ridge_x0  ridge_cond")
    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        bp = sorted(set(batch[obs].tolist()))
        mx0, _ = cross_batch_overlap(x0, mask, batch, t)
        mcd, _ = cross_batch_overlap(positions, mask, batch, t)
        entering = ""
        if t > 0:
            prev = set(batch[np.flatnonzero(mask[:, t - 1])].tolist())
            new = [b for b in bp if b not in prev]
            if new:
                entering = "  <-ENTER:" + ",".join(new)
        lines.append(
            f"{t:2d}  {time_values[t]:4.0f}  {obs.size:3d}  {str(bp):28s} "
            f"{mx0:5.2f}   {mcd:5.2f}    {seam(rs_x0,t):5.2f}     {seam(rs_cd,t):5.2f}{entering}"
        )
    lines.append("")
    lines.append(f"OVERALL ridge: x0={rs_x0['frac_one_sided']:.3f}  "
                 f"condensed={rs_cd['frac_one_sided']:.3f}")
    lines.append(f"  t7(48hpf):  x0={seam(rs_x0,7):.2f}  cond={seam(rs_cd,7):.2f}")
    lines.append(f"  t13(72hpf): x0={seam(rs_x0,13):.2f}  cond={seam(rs_cd,13):.2f}")

    report = "\n".join(lines)
    (OUT_DIR / "mixing_by_bin.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
