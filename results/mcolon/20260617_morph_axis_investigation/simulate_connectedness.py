"""
Validate the connectedness statistic (`connectedness.py`) on simulated data BEFORE
trusting it on real cep290/b9d2 anchors.

This is the spec's step-4 gate: dial a controlled 2-D distribution from
fully-continuous -> stretched-wide-but-still-continuous (the variance-only control,
the thing that must NOT read as discrete) -> clearly bimodal (must read as discrete).

The generators + assertions here double as regression tests we can later lift into
a `tests/` module.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/simulate_connectedness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(RUN_DIR))

from connectedness import (  # noqa: E402
    ConnectednessResult,
    connectedness_pvalue,
    normalize_shape,
    relative_valley_depth,
)

RNG = np.random.default_rng(42)
N_PER_GROUP = 60
N_RESAMPLE = 60


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def sample_unimodal(n: int, sigma: float = 1.0, rng: np.random.Generator = RNG) -> np.ndarray:
    """A single 2-D gaussian blob -- the reference continuum."""
    return rng.normal(loc=0.0, scale=sigma, size=(n, 2))


def sample_wide_continuum(n: int, stretch: float, rng: np.random.Generator = RNG) -> np.ndarray:
    """Variance-only control: same single blob, stretched wider. Still ONE connected
    mode -- must NOT read as discrete regardless of stretch factor."""
    return rng.normal(loc=0.0, scale=1.0 * stretch, size=(n, 2))


def sample_bimodal(n: int, separation: float, mode_sigma: float = 0.6,
                    rng: np.random.Generator = RNG) -> np.ndarray:
    """Two blobs separated along x by `separation`, each of width `mode_sigma`."""
    n0 = n // 2
    n1 = n - n0
    blob0 = rng.normal(loc=[-separation / 2, 0.0], scale=mode_sigma, size=(n0, 2))
    blob1 = rng.normal(loc=[separation / 2, 0.0], scale=mode_sigma, size=(n1, 2))
    return np.vstack([blob0, blob1])


# ---------------------------------------------------------------------------
# Sweep 1: variance-only control must stay flat / continuous
# ---------------------------------------------------------------------------

print("=" * 60)
print("Sweep 1: variance-only stretch (must stay CONTINUOUS)")
print("=" * 60)

reference_pool = sample_unimodal(2000, sigma=1.0, rng=np.random.default_rng(1))

N_DRAWS_PER_LEVEL = 3  # average over several independent draws -- a single n=60 KDE
                        # draw can look bimodal by chance; the sweep must judge the
                        # STATISTIC's behavior, not one unlucky seed.

stretch_factors = np.linspace(1.0, 4.0, 6)
stretch_results = []
for stretch in stretch_factors:
    draw_results = []
    for draw_i in range(N_DRAWS_PER_LEVEL):
        seed = int(stretch * 1000) + draw_i
        grp = sample_wide_continuum(N_PER_GROUP, stretch=stretch, rng=np.random.default_rng(seed))
        draw_results.append(connectedness_pvalue(grp, reference_pool, n_resample=N_RESAMPLE,
                                                   rng=np.random.default_rng(7)))
    mean_stat = float(np.mean([r.stat for r in draw_results]))
    mean_pval = float(np.mean([r.pvalue for r in draw_results]))
    ref_stat = draw_results[0].reference_stat
    stretch_results.append(ConnectednessResult(stat=mean_stat, reference_stat=ref_stat,
                                                pvalue=mean_pval, null_dist=draw_results[0].null_dist))
    print(f"  stretch={stretch:.2f}  mean_stat={mean_stat:.3f}  ref_null_median={ref_stat:.3f}"
          f"  mean_p={mean_pval:.3f}  (avg of {N_DRAWS_PER_LEVEL} draws)")

# All stretch factors should be statistically indistinguishable from the reference
# null on average (mean p not small) -- the whole point of shape-normalization
# killing variance as a confound. A single draw can look spuriously bimodal by
# chance at small n; averaging over draws is what separates real variance-vs-mode
# confusion from ordinary sampling noise.
stretch_pvals = np.array([r.pvalue for r in stretch_results])
assert np.all(stretch_pvals > 0.05), (
    f"FAIL: variance-only stretch triggered a significant 'discrete' call "
    f"(min mean-p={stretch_pvals.min():.3f}). Variance is being confused with modality."
)
print("PASS: variance-only stretch never reads as significantly discrete.\n")


# ---------------------------------------------------------------------------
# Sweep 2: separation must eventually flip to DISCRETE
# ---------------------------------------------------------------------------

print("=" * 60)
print("Sweep 2: bimodal separation (must flip to DISCRETE)")
print("=" * 60)

separations = np.linspace(0.5, 6.0, 8)
sep_results = []
for sep in separations:
    draw_results = []
    for draw_i in range(N_DRAWS_PER_LEVEL):
        seed = int(sep * 1000) + draw_i + 1
        grp = sample_bimodal(N_PER_GROUP, separation=sep, rng=np.random.default_rng(seed))
        draw_results.append(connectedness_pvalue(grp, reference_pool, n_resample=N_RESAMPLE,
                                                   rng=np.random.default_rng(7)))
    mean_stat = float(np.mean([r.stat for r in draw_results]))
    mean_pval = float(np.mean([r.pvalue for r in draw_results]))
    ref_stat = draw_results[0].reference_stat
    res = ConnectednessResult(stat=mean_stat, reference_stat=ref_stat,
                               pvalue=mean_pval, null_dist=draw_results[0].null_dist)
    sep_results.append(res)
    print(f"  sep={sep:.2f}  mean_stat={mean_stat:.3f}  ref_null_median={ref_stat:.3f}"
          f"  mean_p={mean_pval:.3f}  (avg of {N_DRAWS_PER_LEVEL} draws)")

sep_pvals = np.array([r.pvalue for r in sep_results])
assert sep_pvals[-1] < 0.05, (
    f"FAIL: even the most separated bimodal case (sep={separations[-1]:.2f}) "
    f"did not read as significantly discrete (p={sep_pvals[-1]:.3f})."
)
assert sep_pvals[0] > sep_pvals[-1], (
    "FAIL: p-value did not decrease as separation increased -- statistic doesn't "
    "track the continuous->bimodal transition."
)
print("PASS: statistic flips to discrete as separation grows, "
      "and does NOT flip for variance alone.\n")


# ---------------------------------------------------------------------------
# Figure: sweep summary
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.plot(stretch_factors, [r.stat for r in stretch_results], "o-", color="#1f78b4",
        label="variance-only stretch")
ax.axhline([r.reference_stat for r in stretch_results][0], color="gray", linestyle="--",
           label="reference null (median)")
ax.set_xlabel("stretch factor")
ax.set_ylabel("relative valley depth")
ax.set_title("Sweep 1: variance control\n(must stay flat, near reference)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(separations, [r.stat for r in sep_results], "o-", color="#e66101",
        label="bimodal separation")
ax.axhline([r.reference_stat for r in sep_results][0], color="gray", linestyle="--",
           label="reference null (median)")
ax.set_xlabel("mode separation")
ax.set_ylabel("relative valley depth")
ax.set_title("Sweep 2: bimodal separation\n(must rise above reference)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(stretch_factors, stretch_pvals, "o-", color="#1f78b4", label="variance-only (p)")
ax.plot(separations, sep_pvals, "o-", color="#e66101", label="bimodal separation (p)")
ax.axhline(0.05, color="k", linestyle=":", label="p=0.05")
ax.set_xlabel("stretch factor  /  separation")
ax.set_ylabel("p-value (vs. reference null)")
ax.set_yscale("log")
ax.set_title("p-value: variance stays non-significant,\nseparation becomes significant")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Connectedness statistic validation on simulated data", fontsize=13, y=1.03)
fig.tight_layout()
out_path = PLOT_DIR / "simulation_validation.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Example scatter panel: representative continuous vs. discrete cases
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
examples = [
    ("unimodal\n(reference)", sample_unimodal(N_PER_GROUP, sigma=1.0, rng=np.random.default_rng(2))),
    ("wide continuum\n(stretch=4, variance only)",
     sample_wide_continuum(N_PER_GROUP, stretch=4.0, rng=np.random.default_rng(3))),
    ("borderline bimodal\n(sep=3)", sample_bimodal(N_PER_GROUP, separation=3.0, rng=np.random.default_rng(4))),
    ("clearly bimodal\n(sep=6)", sample_bimodal(N_PER_GROUP, separation=6.0, rng=np.random.default_rng(5))),
]
for ax, (title, pts) in zip(axes2, examples):
    norm_pts = normalize_shape(pts)
    stat = relative_valley_depth(norm_pts)
    ax.scatter(norm_pts[:, 0], norm_pts[:, 1], s=20, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_title(f"{title}\nvalley_depth={stat:.2f}", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

fig2.suptitle("Example point clouds (normalized shape)", fontsize=12, y=1.05)
fig2.tight_layout()
out_path2 = PLOT_DIR / "simulation_examples.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out_path2.name}")

print("\nAll simulation gates passed.")
