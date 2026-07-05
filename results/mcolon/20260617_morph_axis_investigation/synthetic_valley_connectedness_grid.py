"""
synthetic_valley_connectedness_grid.py
---------------------------------------
Merge the valley-visualization (real KDE + significance ring) with the
connectedness-panel vote table (valley / MST / Fiedler / conductance vs WT null),
rendered across EVERY synthetic scenario in `synthetic_scenarios.py`.

Goal: see the density-vs-graph disagreement directly, scenario by scenario --
which cases are density-yes/graph-no (b9d2-like), which are graph-yes/density-no
(crescent/spiral-like), and which agree.

Per scenario, one column with two rows:
  ROW 1  KDE of the scenario's mutant cloud (soft blue) + WT reference points
         overlaid (light gray), + a significance ring if valley_depth fires.
  ROW 2  vote table: valley / MST / Fiedler / conductance, each p vs a matched-N
         WT null, plus the tally call and the disagreement type
         (density-yes/graph-no, graph-yes/density-no, agree, or none-fire).

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/synthetic_valley_connectedness_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as ndi_label

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import compute_support_geometry, normalize_shape  # noqa: E402
from synthetic_scenarios import SCENARIOS, wt_reference  # noqa: E402

PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

N = 40                  # sample size per scenario (mid-range of SAMPLE_SIZES)
N_RESAMPLE = 200
RNG_SEED = 7
GRID = 60
MIN_MODE_MASS_FRAC = 0.03
DENS_CMAP = "Blues"
DENS_CMAP_HI = 0.62
RING_COLOR = "#B8860B"
WT_POINT_COLOR = "#9a9a9a"
TARGET_POINT_COLOR = "#1a1a1a"
DISCRETE_P_THRESHOLDS = {"valley_depth": 0.05, "mst_max_edge": 0.05,
                         "fiedler": 0.05, "conductance": 0.05}
TALLY_EXCLUDED = {"mst_max_edge"}
_STAT_ROWS = [("valley_depth", "valley"), ("mst_max_edge", "MST"),
              ("fiedler", "Fiedler"), ("conductance", "conduct")]
_FIRED_COLOR = "#B2182B"
_QUIET_COLOR = "#7f7f7f"
CONTINUOUS_COLOR = "#2166AC"


def _votes_discrete(name, pvalue):
    return pvalue < DISCRETE_P_THRESHOLDS.get(name, 0.05)


def _tally_call(bundle):
    votes = [_votes_discrete(n, sr.pvalue) for n, sr in bundle.results.items()
             if sr is not None and n not in TALLY_EXCLUDED]
    if not votes:
        return "n/a"
    return "discrete" if sum(votes) > len(votes) / 2 else "continuous"


def _disagreement_type(bundle):
    """Classify density-vs-graph disagreement for this scenario.

    density-yes/graph-no : valley fires but no graph stat corroborates (b9d2-like)
    graph-yes/density-no : a graph stat fires but valley does not (crescent/spiral-like)
    agree-discrete       : valley AND >=1 graph stat fire
    agree-continuous     : nothing fires
    """
    vd = bundle.results.get("valley_depth")
    fied = bundle.results.get("fiedler")
    mst = bundle.results.get("mst_max_edge")
    cond = bundle.results.get("conductance")
    valley_fires = vd is not None and _votes_discrete("valley_depth", vd.pvalue)
    graph_fires = any(
        s is not None and _votes_discrete(n, s.pvalue)
        for n, s in (("fiedler", fied), ("mst_max_edge", mst), ("conductance", cond))
    )
    if valley_fires and not graph_fires:
        return "density-YES / graph-NO", "#B2182B"
    if graph_fires and not valley_fires:
        return "graph-YES / density-NO", "#6A3D9A"
    if valley_fires and graph_fires:
        return "agree: DISCRETE", "#B2182B"
    return "agree: continuous", CONTINUOUS_COLOR


def _kde_grid_padded(pts, grid=GRID, pad_frac=0.35):
    from scipy.stats import gaussian_kde
    xr = np.percentile(pts[:, 0], [1, 99])
    yr = np.percentile(pts[:, 1], [1, 99])
    padx, pady = (xr[1] - xr[0]) * pad_frac, (yr[1] - yr[0]) * pad_frac
    xs = np.linspace(xr[0] - padx, xr[1] + padx, grid)
    ys = np.linspace(yr[0] - pady, yr[1] + pady, grid)
    xx, yy = np.meshgrid(xs, ys)
    dens = gaussian_kde(pts.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, dens


def _count_modes(dens):
    peak = float(dens.max()); total = float(dens.sum())
    if peak <= 0 or total <= 0:
        return 0, None
    for frac in np.linspace(0.95, 0.05, 25):
        level = frac * peak
        labels, n = ndi_label(dens >= level)
        if n < 2:
            continue
        masses = np.array([dens[labels == k].sum() / total for k in range(1, n + 1)])
        if int((masses >= MIN_MODE_MASS_FRAC).sum()) >= 2:
            return int((masses >= MIN_MODE_MASS_FRAC).sum()), level
    return 1, 0.5 * peak


def _render_kde_cell(ax, pts_norm, wt_norm, sig):
    xx, yy, dens = _kde_grid_padded(pts_norm)
    light_blues = ListedColormap(plt.get_cmap(DENS_CMAP)(np.linspace(0.0, DENS_CMAP_HI, 256)))
    peak = float(dens.max())
    levels = np.linspace(0.06 * peak, peak, 16)
    ax.contourf(xx, yy, dens, levels=levels, cmap=light_blues, extend="max")
    ax.scatter(wt_norm[:, 0], wt_norm[:, 1], s=10, alpha=0.35, facecolors="none",
               edgecolors=WT_POINT_COLOR, linewidths=0.6, zorder=2)
    ax.scatter(pts_norm[:, 0], pts_norm[:, 1], s=14, alpha=0.9,
               facecolors=TARGET_POINT_COLOR, edgecolors="white", linewidths=0.3, zorder=3)
    if sig:
        _, level = _count_modes(dens)
        if level is not None:
            ax.contour(xx, yy, dens, levels=[level], colors=RING_COLOR,
                       linewidths=2.0, linestyles="--", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#ccc")


def _render_vote_table(ax, bundle):
    x_label, x_discr, x_cont, x_p = 0.02, 0.42, 0.60, 0.78
    y, dy = 0.94, 0.155
    ax.axis("off")
    ax.text(x_label, y, "metric", transform=ax.transAxes, fontsize=7.2, va="top",
            ha="left", color="#333", fontstyle="italic", fontfamily="monospace")
    ax.text(x_discr, y, "DISCR", transform=ax.transAxes, fontsize=7.2, va="top",
            ha="center", color=_FIRED_COLOR, fontweight="bold", fontfamily="monospace")
    ax.text(x_cont, y, "CONT", transform=ax.transAxes, fontsize=7.2, va="top",
            ha="center", color=CONTINUOUS_COLOR, fontweight="bold", fontfamily="monospace")
    ax.text(x_p, y, "p", transform=ax.transAxes, fontsize=7.2, va="top",
            ha="left", color="#333", fontstyle="italic", fontfamily="monospace")
    for i, (name, label) in enumerate(_STAT_ROWS):
        yy = y - (i + 1) * dy
        sr = bundle.results.get(name)
        excluded = name in TALLY_EXCLUDED
        lc = "#aaa" if excluded else "#222"
        ax.text(x_label, yy, label, transform=ax.transAxes, fontsize=7.2, va="top",
                ha="left", color=lc, fontfamily="monospace")
        if sr is None:
            continue
        if excluded:
            ax.text((x_discr + x_cont) / 2, yy, "(excl)", transform=ax.transAxes,
                    fontsize=6.4, va="top", ha="center", color="#aaa",
                    fontfamily="monospace", fontstyle="italic")
            ax.text(x_p, yy, f"{sr.pvalue:.2f}", transform=ax.transAxes, fontsize=7.0,
                    va="top", ha="left", color="#aaa", fontfamily="monospace")
            continue
        disc = _votes_discrete(name, sr.pvalue)
        ax.text(x_discr, yy, "●" if disc else "", transform=ax.transAxes, fontsize=9,
                va="top", ha="center", color=_FIRED_COLOR)
        ax.text(x_cont, yy, "" if disc else "●", transform=ax.transAxes, fontsize=9,
                va="top", ha="center", color=CONTINUOUS_COLOR)
        ax.text(x_p, yy, f"{sr.pvalue:.2f}", transform=ax.transAxes, fontsize=7.0,
                va="top", ha="left", color=_FIRED_COLOR if disc else _QUIET_COLOR,
                fontfamily="monospace", fontweight="bold" if disc else "normal")

    call = _tally_call(bundle)
    dis_label, dis_color = _disagreement_type(bundle)
    ax.text(x_label, y - (len(_STAT_ROWS) + 0.9) * dy,
            f"tally => {call.upper()}", transform=ax.transAxes,
            fontsize=7.4, va="top", ha="left",
            color=_FIRED_COLOR if call == "discrete" else CONTINUOUS_COLOR,
            fontweight="bold", fontfamily="monospace")
    ax.text(x_label, y - (len(_STAT_ROWS) + 1.9) * dy, dis_label, transform=ax.transAxes,
            fontsize=7.6, va="top", ha="left", color=dis_color, fontweight="bold")


def main():
    rng_master = np.random.default_rng(RNG_SEED)
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(2, n_scen, figsize=(2.7 * n_scen, 6.3), squeeze=False,
                              gridspec_kw={"height_ratios": [2.0, 1.15]})

    for col, scen in enumerate(SCENARIOS):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        pts_raw = scen.generator(N, rng)
        wt_raw = wt_reference(N, rng)

        bundle = compute_support_geometry(pts_raw, wt_raw, n_resample=N_RESAMPLE,
                                          rng=np.random.default_rng(RNG_SEED + col))
        vp = bundle.results["valley_depth"].pvalue
        sig = vp < 0.05

        pts_norm = normalize_shape(pts_raw)
        wt_norm = normalize_shape(wt_raw)

        ax0 = axes[0][col]
        _render_kde_cell(ax0, pts_norm, wt_norm, sig)
        expected = f"exp:{scen.expected_support}"
        ax0.set_title(f"{scen.name}\n({expected})", fontsize=8.6, fontweight="bold")

        ax1 = axes[1][col]
        _render_vote_table(ax1, bundle)

    axes[0][0].set_ylabel("KDE + ring if\nvalley significant", fontsize=9)
    axes[1][0].text(-0.35, 0.5, "vote table\n(p vs WT null)", transform=axes[1][0].transAxes,
                    fontsize=9, rotation=90, va="center", ha="center")

    fig.suptitle(
        "Synthetic scenarios — density (KDE/valley) vs graph (MST/Fiedler/conductance) disagreement\n"
        "ring = valley significant (p<0.05, matched-N WT null); vote table shows each statistic's own call",
        fontsize=11, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.03, wspace=0.18, hspace=0.08)
    out = PLOT_DIR / "synthetic_valley_connectedness_grid.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
