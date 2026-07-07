"""
valley_visualization.py  (b9d2 + cep290)
-----------------------------------------
Make valley_depth -- and its SIGNIFICANCE -- visible. Both genes genuinely HAVE
density valleys; they only count as "discrete" if DEEPER than the wildtype null.

We ALWAYS show the real KDE -- never a synthetic single-mode Gaussian. Significance
changes only the ANNOTATION, not the density: a ring is a POSITIVE claim of two
statistically-supported modes, so it is drawn only when the valley is significant.
Not significant -> no ring (the honest KDE still shows whatever lumps exist, we just
don't claim they are discrete).

Per hpf bin, three stacked views (top -> bottom):

  ROW 1  TARGET KDE + decision ring: the real KDE of the target genotype. Two island
         rings ONLY if the valley is significant; no ring otherwise.

  ROW 2  OVERLAP MAP: WT-null and target KDEs thresholded into occupied regions on a
         SHARED grid, then colored by set membership -- target-only, WT-only, and the
         overlap region get three distinct colors. Shows how the two distributions
         sit relative to each other.

  ROW 3  RAW POINTS, side by side: the actual scatter -- WT points (left) and target
         points (right) -- the honest raw geometry the decision is made from.

Significance = valley_depth p < VALLEY_SIG_P vs a matched-N wildtype null.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/valley_visualization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RUN_DIR))

GENE14_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
REF_DIR = GENE14_DIR / "tables"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(GENE14_DIR))
from plot_config import PHENOTYPE_COLORS  # noqa: E402

from support_geometry import (  # noqa: E402
    MIN_COMPONENT_MASS_FRAC, VALLEY_SWEEP_STEPS, compute_support_geometry,
    normalize_shape, valley_detection_detail,
)
from morphseq_investigation.core.peak_counting import count_mass_significant_modes  # noqa: E402
from morphseq_investigation.plotting.modal_distribution_plotting import (  # noqa: E402
    build_distribution_overlay,
    format_density_axis,
    plot_density_overlap,
)

GENES = {
    "b9d2":   {"csv": REF_DIR / "reference_b9d2_clean.csv",
               "phenotype_labels": ["CE", "HTA"]},
    "cep290": {"csv": REF_DIR / "reference_cep290_clean.csv",
               "phenotype_labels": ["High_to_Low", "Low_to_High"]},
}
X_FEAT, Y_FEAT = "total_length_um", "baseline_deviation_normalized"
TIME_COL, BIN_WIDTH = "predicted_stage_hpf", 4.0
TARGET_DESIGN_HPF = [14, 18, 24, 30, 48]
_bin_center = lambda h: float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)
BIN_CENTER_TO_DESIGN_HPF = {_bin_center(h): h for h in TARGET_DESIGN_HPF}
MIN_EMBRYOS = 10
N_RESAMPLE = 80
VALLEY_SIG_P = 0.05
UNLABELED_COLOR = "#BBBBBB"
RING_COLOR = "#B8860B"          # dark gold, dashed -- valley: circles significant modes
AREA_RING_COLOR = "#1B7837"     # green, solid  -- HDR-area: densest-mass concentration
HDR_MASS_FRAC = 0.5             # HDR-area concentration ring at densest 50% of KDE mass
DENS_CMAP = "Blues"            # soft-blue density field (row 1)
DENS_CMAP_HI = 0.62            # cap the Blues ramp here so the peak stays light
TARGET_DENS_COLOR = "#2166AC"   # target overlap density   (blue)
WT_DENS_COLOR = "#808080"       # reference overlap density (gray)
WT_POINT_COLOR = "#808080"      # WT reference raw points (gray)
GRID = 40
LABEL_FS = 11                  # bigger labels throughout


def load_bins(cfg):
    df = pd.read_csv(cfg["csv"], low_memory=False)
    needed = [TIME_COL, X_FEAT, Y_FEAT, "phenotype_clean", "zygosity"]
    df = df.dropna(subset=needed).copy()
    df["tb"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2
    ec = "embryo_id" if df["physical_embryo_id"].isna().all() else "physical_embryo_id"
    agg = {X_FEAT: "mean", Y_FEAT: "mean",
           "phenotype_clean": lambda x: x.mode().iloc[0],
           "zygosity": lambda x: x.mode().iloc[0]}
    edf = df.groupby([ec, "tb"]).agg(agg).reset_index()
    out = {}
    for bc, hpf in BIN_CENTER_TO_DESIGN_HPF.items():
        sub = edf[edf["tb"] == bc]
        wt = sub[sub["zygosity"] == "wildtype"]
        grp = sub[sub["phenotype_clean"].isin(cfg["phenotype_labels"])]
        if len(wt) < MIN_EMBRYOS or len(grp) < MIN_EMBRYOS:
            continue
        out[hpf] = (grp[[X_FEAT, Y_FEAT]].values.astype(float),
                    grp["phenotype_clean"].values,
                    wt[[X_FEAT, Y_FEAT]].values.astype(float))
    return out

def _target_ring(ax, xx, yy, dens):
    """Draw the two island rings at the valley waterline. Caller only invokes this when
    the valley is SIGNIFICANT, so a ring is always a positive two-mode claim."""
    _, level = count_mass_significant_modes(
        dens,
        min_component_mass_frac=MIN_COMPONENT_MASS_FRAC,
        sweep_steps=VALLEY_SWEEP_STEPS,
    )
    if level is not None:
        ax.contour(xx, yy, dens, levels=[level], colors=RING_COLOR,
                   linewidths=2.2, linestyles="--", zorder=4)


def _hdr_mass_level(dens, mass_frac=HDR_MASS_FRAC):
    """Density value bounding the densest `mass_frac` of KDE mass -- the HDR contour.

    This is the geometric locus of the `hdr_area_concentration` statistic: the region
    inside this contour is the area the densest `mass_frac` of the mass occupies. A
    tighter contour => mass more concentrated => more discrete vs the WT null."""
    flat = np.sort(np.asarray(dens, dtype=float).ravel())[::-1]
    csum = np.cumsum(flat)
    total = float(csum[-1])
    if total <= 0:
        return None
    idx = int(np.searchsorted(csum / total, mass_frac, side="left"))
    idx = min(idx, len(flat) - 1)
    return float(flat[idx])


def _area_ring(ax, xx, yy, dens):
    """Draw the HDR densest-mass concentration ring. Caller invokes only when the
    HDR-area statistic is SIGNIFICANT vs the WT null, so this ring is a positive claim
    that the target's mass concentrates into less area than matched-N wildtype."""
    level = _hdr_mass_level(dens)
    if level is not None:
        ax.contour(xx, yy, dens, levels=[level], colors=AREA_RING_COLOR,
                   linewidths=2.0, linestyles="-", zorder=5)


def _null_fit_strip(ax, valley_sr, area_sr):
    """Inset at the bottom-left of a row-1 cell: for each density witness, show where
    the OBSERVED statistic falls within its matched-N WT null distribution.

    Two stacked mini-rows (valley on top, HDR-area below). Each draws the null's
    5-95 percentile band (gray bar), its median (tick), and the observed value (colored
    triangle). Observed sitting to the RIGHT of the null band = more discrete than
    wildtype = the significant direction. This is the visual of the p-value: it makes
    'deeper/tighter than the WT reference null' literal rather than a number in a title."""
    inset = ax.inset_axes([0.02, 0.02, 0.42, 0.20])
    inset.set_xticks([]); inset.set_yticks([])
    inset.patch.set_alpha(0.82)
    for s in inset.spines.values():
        s.set_visible(False)
    rows = [("valley", valley_sr, RING_COLOR), ("HDR-area", area_sr, AREA_RING_COLOR)]
    y_positions = [0.72, 0.28]
    for (label, sr, color), y in zip(rows, y_positions):
        if sr is None:
            continue
        null = np.asarray(sr.null_dist, dtype=float)
        lo, hi = np.percentile(null, [5, 95])
        med = float(np.median(null))
        span = hi - lo if hi > lo else (abs(med) + 1e-9)
        # map [lo-0.2span, obs or hi +0.2span] -> [0.28, 0.98] within the inset
        left = min(lo, sr.stat) - 0.2 * span
        right = max(hi, sr.stat) + 0.2 * span
        rng = right - left if right > left else 1.0
        to_x = lambda v: 0.28 + 0.70 * (v - left) / rng
        inset.plot([to_x(lo), to_x(hi)], [y, y], color="#999", lw=3, solid_capstyle="butt")
        inset.plot([to_x(med)], [y], marker="|", color="#555", ms=8, mew=1.4)
        sig_dir = sr.pvalue < VALLEY_SIG_P
        inset.plot([to_x(sr.stat)], [y], marker=">", color=color, ms=7,
                   mec="k", mew=0.4 if sig_dir else 0.0)
        inset.text(0.0, y, label, fontsize=6.0, va="center", ha="left",
                   color=color, fontweight="bold", fontfamily="monospace")
    inset.text(0.63, 0.99, "obs vs WT null", fontsize=5.4, va="top", ha="center",
               color="#666", fontstyle="italic")
    inset.set_xlim(0, 1); inset.set_ylim(0, 1)


def render_gene(gene, cfg, *, kde=None):
    bins = load_bins(cfg)
    hpfs = [h for h in TARGET_DESIGN_HPF if h in bins]
    n = len(hpfs)
    if n == 0:
        print(f"  {gene}: no usable bins"); return

    fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 9.6), squeeze=False)
    # fix positions BEFORE _pair_cell reads them via get_position()
    fig.subplots_adjust(bottom=0.08, top=0.87, left=0.07, right=0.99,
                        hspace=0.30, wspace=0.10)

    for col, hpf in enumerate(hpfs):
        grp_raw, phenos, wt_raw = bins[hpf]
        grp = normalize_shape(grp_raw)
        wt = normalize_shape(wt_raw)

        bundle = compute_support_geometry(grp_raw, wt_raw, n_resample=N_RESAMPLE,
                                          rng=np.random.default_rng(42), kde=kde)
        valley_sr = bundle.results["valley_depth"]
        area_sr = bundle.results.get("hdr_area_concentration")
        vp = valley_sr.pvalue
        sig = vp < VALLEY_SIG_P                      # valley witness
        ap = area_sr.pvalue if area_sr is not None else 1.0
        area_sig = area_sr is not None and ap < VALLEY_SIG_P   # HDR-area witness

        overlay = build_distribution_overlay(grp, wt, grid=GRID, kde=kde)
        box = overlay.box
        gx, gy, gd = overlay.target_grid.xx, overlay.target_grid.yy, overlay.target_grid.density

        # ── ROW 1: TARGET KDE (soft blue) + decision ring (ring ONLY if sig) ─
        ax = axes[0][col]
        from matplotlib.colors import ListedColormap
        light_blues = ListedColormap(
            plt.get_cmap(DENS_CMAP)(np.linspace(0.0, DENS_CMAP_HI, 256)))
        # start the lowest band just above zero so the tail fades out instead of
        # painting a flat block to the grid edge
        peak = float(gd.max())
        levels = np.linspace(0.06 * peak, peak, 16)
        ax.contourf(gx, gy, gd, levels=levels, cmap=light_blues, extend="max")
        for pheno in cfg["phenotype_labels"] + ["unlabeled"]:
            m = phenos == pheno
            if not m.any():
                continue
            ax.scatter(grp[m, 0], grp[m, 1], s=20, alpha=0.9,
                       facecolors=PHENOTYPE_COLORS.get(pheno, UNLABELED_COLOR),
                       edgecolors="k", linewidths=0.4, zorder=3, label=pheno)
        format_density_axis(ax, box)
        # Title carries BOTH witnesses; each colored by its own significance so the
        # density-yes-here / no-there split is legible per bin.
        v_col = "#B2182B" if sig else "#2166AC"
        a_col = AREA_RING_COLOR if area_sig else "#2166AC"
        ax.set_title(f"{hpf} hpf", fontsize=LABEL_FS, fontweight="bold", color="#222")
        ax.text(0.5, 1.11, f"valley p={vp:.2f}", transform=ax.transAxes,
                fontsize=LABEL_FS - 2, fontweight="bold", color=v_col,
                ha="right", va="bottom")
        ax.text(0.5, 1.11, f"   HDR-area p={ap:.2f}", transform=ax.transAxes,
                fontsize=LABEL_FS - 2, fontweight="bold", color=a_col,
                ha="left", va="bottom")
        # Where each observed stat sits against its WT null (the "fit to reference
        # nulls" the fused figure is meant to show): a compact bracket per witness.
        _null_fit_strip(ax, valley_sr, area_sr)

        # ── ROW 2: smooth OVERLAP of the two density distributions ──────────
        # framed to fit BOTH distributions so neither density is cut off
        plot_density_overlap(axes[1][col], overlay.target_grid, overlay.reference_grid)
        format_density_axis(axes[1][col], box)

        # ── ROW 3: RAW POINTS + each cloud's own KDE, target (L) vs WT (R) ──
        _points_pair_cell(fig, axes[2][col], grp, phenos, wt, cfg, gene,
                          overlay.target_grid, overlay.reference_grid, box)

    # row labels
    axes[0][0].set_ylabel("TARGET KDE\n+ ring if significant", fontsize=LABEL_FS)
    r1 = axes[1][0].get_position(); r2 = axes[2][0].get_position()
    fig.text(0.014, r1.y0 + r1.height / 2, "DENSITY OVERLAP\n(target vs WT)",
             fontsize=LABEL_FS, rotation=90, va="center", ha="center")
    fig.text(0.014, r2.y0 + r2.height / 2, "RAW POINTS\n(target | WT)", fontsize=LABEL_FS,
             rotation=90, va="center", ha="center")

    # combined legend: phenotype markers (rows 1/3) + overlap density hues (row 2)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = handles + [
        Patch(facecolor=TARGET_DENS_COLOR, alpha=0.55, label=f"{gene} density"),
        Patch(facecolor=WT_DENS_COLOR, alpha=0.55, label="WT density"),
        Line2D([0], [0], color=RING_COLOR, lw=2.2, ls="--", label="valley ring (sig.)"),
        Line2D([0], [0], color=AREA_RING_COLOR, lw=2.0, ls="-", label="HDR-area ring (sig.)"),
    ]
    labels_ = labels_ + [f"{gene} density", "WT density",
                         "valley ring (sig.)", "HDR-area ring (sig.)"]
    fig.legend(handles, labels_, loc="lower center", ncol=len(handles),
               fontsize=LABEL_FS - 1, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"{gene} — density-discreteness visualization  (axis: {X_FEAT} x {Y_FEAT}, normalized)\n"
        f"real KDE always shown; TWO parallel density witnesses vs the matched-N wildtype null "
        f"(p < {VALLEY_SIG_P}): dashed gold valley ring (gap between modes) + solid green "
        f"HDR-area ring (mass concentration).  Inset per cell = obs vs WT null.",
        fontsize=LABEL_FS + 1, fontweight="bold", y=0.996)
    out = PLOT_DIR / f"{gene}_valley_visualization.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.name}")


def _points_pair_cell(fig, host_ax, grp_pts, phenos, wt_pts, cfg, gene,
                      grp_grid, wt_grid, box):
    """Row 3: TARGET (LEFT) then reference WT (RIGHT) -- as two mini-axes inside
    host_ax, on the shared frame so spreads compare directly. Each panel overlays its
    OWN KDE (target blue / WT gray) behind the raw points; target keeps its phenotype
    colors, WT points are gray."""
    from matplotlib.colors import to_rgba
    host_ax.axis("off")
    pos = host_ax.get_position()
    w = pos.width / 2
    axL = fig.add_axes([pos.x0, pos.y0, w * 0.94, pos.height])
    axR = fig.add_axes([pos.x0 + w * 1.06, pos.y0, w * 0.94, pos.height])

    def _bg_kde(ax, grid, color):
        dens = np.asarray(grid.density, dtype=float)
        peak = float(dens.max())
        colors = [to_rgba(color, a) for a in np.linspace(0.10, 0.40, 12)]
        levels = np.linspace(0.06 * peak, peak, 13)
        ax.contourf(grid.xx, grid.yy, dens, levels=levels, colors=colors,
                    antialiased=True, zorder=0)

    _bg_kde(axL, grp_grid, TARGET_DENS_COLOR)
    _bg_kde(axR, wt_grid, WT_DENS_COLOR)

    for pheno in cfg["phenotype_labels"] + ["unlabeled"]:
        m = phenos == pheno
        if not m.any():
            continue
        axL.scatter(grp_pts[m, 0], grp_pts[m, 1], s=16, alpha=0.85,
                    facecolors=PHENOTYPE_COLORS.get(pheno, UNLABELED_COLOR),
                    edgecolors="k", linewidths=0.3, zorder=3)
    axR.scatter(wt_pts[:, 0], wt_pts[:, 1], s=16, alpha=0.85,
                facecolors=WT_POINT_COLOR, edgecolors="k", linewidths=0.3, zorder=3)
    axL.set_title(gene, fontsize=LABEL_FS - 2, color=TARGET_DENS_COLOR)
    axR.set_title("WT (reference)", fontsize=LABEL_FS - 2, color=WT_POINT_COLOR)
    for ax in (axL, axR):
        format_density_axis(ax, box)


def main():
    for gene, cfg in GENES.items():
        print(f"\n=== {gene} ===")
        render_gene(gene, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
