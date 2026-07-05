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

from scipy.ndimage import label as ndi_label  # noqa: E402

from support_geometry import (  # noqa: E402
    MIN_COMPONENT_MASS_FRAC, VALLEY_SWEEP_STEPS, compute_support_geometry,
    evaluate_kde_on_grid, normalize_shape, valley_detection_detail,
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
RING_COLOR = "#B8860B"          # dark gold, dashed -- circles significant modes
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


SUPPORT_FRAC = 0.02   # combined KDE >= this * peak defines the "where there's mass" box


def _data_box(pts_list, support_frac=SUPPORT_FRAC, margin=0.06, scan_pct=2.0, *, kde=None):
    """THE single per-column box, defined by DENSITY not by point extremes.

    A lone outlier contributes almost no KDE mass, so instead of a bounding box of
    points we take the bounding box of the region where the COMBINED (target+WT) kernel
    density exceeds `support_frac` of its peak, then add a margin. This frames the mass
    of both distributions and ignores fliers.

    IMPORTANT: the density-support SCAN grid is built on a robust (percentile-clipped)
    range, NOT raw min/max. Otherwise a couple of far outliers stretch the coarse scan
    grid, and an isolated flier can still clear the support threshold on a sparse cell --
    blowing the box out (the 48 hpf bug: a WT point ~15 sigma out dragged the frame so
    the real cloud sat in a tiny corner). Clipping the scan range keeps outliers out of
    the search entirely.

    Every row in the column -- KDE grid AND axis limits -- uses this exact box, so it is
    the single coordinate frame all rows map into (row 3 just renders the same box at a
    different aspect ratio). Returns (xlo, xhi, ylo, yhi)."""
    allpts = np.vstack(pts_list)
    # robust scan range: percentile-clip so far fliers never enter the support search
    xr = np.percentile(allpts[:, 0], [scan_pct, 100 - scan_pct])
    yr = np.percentile(allpts[:, 1], [scan_pct, 100 - scan_pct])
    padx, pady = (xr[1] - xr[0]) * 0.35, (yr[1] - yr[0]) * 0.35
    xs = np.linspace(xr[0] - padx, xr[1] + padx, 80)
    ys = np.linspace(yr[0] - pady, yr[1] + pady, 80)
    xx, yy = np.meshgrid(xs, ys)
    dens = evaluate_kde_on_grid(allpts, xx, yy, kde=kde)
    mask = dens >= support_frac * float(dens.max())
    xsel, ysel = xx[mask], yy[mask]
    xlo, xhi, ylo, yhi = xsel.min(), xsel.max(), ysel.min(), ysel.max()
    mx, my = (xhi - xlo) * margin, (yhi - ylo) * margin
    return xlo - mx, xhi + mx, ylo - my, yhi + my


def _kde_in_box(pts, box, grid=GRID, *, kde=None):
    """Evaluate a gaussian KDE of `pts` on the column box. The density is rendered with
    its lowest contour band starting above zero (see callers) so it fades to background
    before the edge -- no hard fill rectangle, even though the grid == the visible box."""
    xlo, xhi, ylo, yhi = box
    xs = np.linspace(xlo, xhi, grid)
    ys = np.linspace(ylo, yhi, grid)
    xx, yy = np.meshgrid(xs, ys)
    dens = evaluate_kde_on_grid(pts, xx, yy, kde=kde)
    return xx, yy, dens


def count_modes(dens):
    """Number of mass-carrying density modes, and the super-level threshold at which
    they FIRST separate -- so the ring sits exactly where the modes split (matches the
    valley_depth logic), not at an arbitrary fixed height.

    Sweep the super-level threshold down from the peak; the first level whose
    super-level set has >=2 components each holding >= MIN_MODE_MASS_FRAC of the mass
    gives the mode count + ring level. If it never splits -> 1 mode, ring at half-peak.
    Returns (n_modes, level)."""
    peak = float(dens.max()); total = float(dens.sum())
    if peak <= 0 or total <= 0:
        return 0, None
    for frac in np.linspace(0.97, 0.02, VALLEY_SWEEP_STEPS):
        level = frac * peak
        labels, n = ndi_label(dens >= level)
        if n < 2:
            continue
        masses = np.array([dens[labels == k].sum() / total for k in range(1, n + 1)])
        n_real = int((masses >= MIN_COMPONENT_MASS_FRAC).sum())
        if n_real >= 2:
            return n_real, level
    return 1, 0.5 * peak  # never splits -> single mode, ring at half-peak


def _target_ring(ax, xx, yy, dens):
    """Draw the two island rings at the valley waterline. Caller only invokes this when
    the valley is SIGNIFICANT, so a ring is always a positive two-mode claim."""
    _, level = count_modes(dens)
    if level is not None:
        ax.contour(xx, yy, dens, levels=[level], colors=RING_COLOR,
                   linewidths=2.2, linestyles="--", zorder=4)


def render_gene(gene, cfg, *, kde=None):
    bins = load_bins(cfg)
    hpfs = [h for h in TARGET_DESIGN_HPF if h in bins]
    n = len(hpfs)
    if n == 0:
        print(f"  {gene}: no usable bins"); return

    fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 9.6), squeeze=False)
    # fix positions BEFORE _pair_cell reads them via get_position()
    fig.subplots_adjust(bottom=0.08, top=0.89, left=0.07, right=0.99,
                        hspace=0.26, wspace=0.10)

    for col, hpf in enumerate(hpfs):
        grp_raw, phenos, wt_raw = bins[hpf]
        grp = normalize_shape(grp_raw)
        wt = normalize_shape(wt_raw)

        bundle = compute_support_geometry(grp_raw, wt_raw, n_resample=N_RESAMPLE,
                                          rng=np.random.default_rng(42), kde=kde)
        vp = bundle.results["valley_depth"].pvalue
        sig = vp < VALLEY_SIG_P

        # ONE box per column: fits all target+WT data (robust), + margin. Every row --
        # KDE grid and axis limits -- uses this exact box, so nothing is re-cropped.
        box = _data_box([grp, wt], kde=kde)
        gx, gy, gd = _kde_in_box(grp, box, kde=kde)  # target density on the box
        wx, wy, wd = _kde_in_box(wt, box, kde=kde)   # WT density on the SAME box -> comparable

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
        if sig:
            _target_ring(ax, gx, gy, gd)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_xticks([]); ax.set_yticks([])
        sig_txt, sig_col = ("SIGNIF", "#B2182B") if sig else ("n.s.", "#2166AC")
        ax.set_title(f"{hpf} hpf   valley p={vp:.2f} [{sig_txt}]",
                     fontsize=LABEL_FS, fontweight="bold", color=sig_col)

        # ── ROW 2: smooth OVERLAP of the two density distributions ──────────
        # framed to fit BOTH distributions so neither density is cut off
        _overlap_cell(axes[1][col], gx, gy, gd, wd, gene, box)

        # ── ROW 3: RAW POINTS + each cloud's own KDE, target (L) vs WT (R) ──
        _points_pair_cell(fig, axes[2][col], grp, phenos, wt, cfg, gene,
                          (gx, gy, gd), (wx, wy, wd), box)

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
    handles = handles + [
        Patch(facecolor=TARGET_DENS_COLOR, alpha=0.55, label=f"{gene} density"),
        Patch(facecolor=WT_DENS_COLOR, alpha=0.55, label="WT density"),
    ]
    labels_ = labels_ + [f"{gene} density", "WT density"]
    fig.legend(handles, labels_, loc="lower center", ncol=len(handles),
               fontsize=LABEL_FS - 1, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"{gene} — valley significance visualization  (axis: {X_FEAT} x {Y_FEAT}, normalized)\n"
        f"real KDE always shown; a valley ring is drawn ONLY when significant "
        f"(deeper than the wildtype null, p < {VALLEY_SIG_P})",
        fontsize=LABEL_FS + 1, fontweight="bold", y=0.996)
    out = PLOT_DIR / f"{gene}_valley_visualization.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.name}")


def _overlap_cell(ax, xx, yy, target_dens, wt_dens, gene, vlim):
    """Row 2: two smooth probability distributions overlaid, seaborn `kdeplot(hue=...)`
    style -- a FEW translucent filled contour bands per distribution in two distinct
    hues. Low alpha means the intersection blends the two hues, which is the standard,
    least-busy way to read overlap (per seaborn / ggplot practice)."""
    from matplotlib.colors import to_rgba

    def _fill(dens, color):
        # a handful of iso-proportion bands (not a dense ramp); translucent so overlap blends
        peak = float(dens.max())
        levels = [f * peak for f in (0.20, 0.45, 0.70, 0.90)] + [peak]
        colors = [to_rgba(color, a) for a in (0.16, 0.24, 0.32, 0.42)]
        ax.contourf(xx, yy, dens, levels=levels, colors=colors, antialiased=True)

    _fill(wt_dens, WT_DENS_COLOR)
    _fill(target_dens, TARGET_DENS_COLOR)
    ax.set_xlim(vlim[0], vlim[1]); ax.set_ylim(vlim[2], vlim[3])
    ax.set_xticks([]); ax.set_yticks([])


def _points_pair_cell(fig, host_ax, grp_pts, phenos, wt_pts, cfg, gene,
                      grp_field, wt_field, vlim):
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

    def _bg_kde(ax, field, color):
        xx, yy, dens = field
        peak = float(dens.max())
        colors = [to_rgba(color, a) for a in np.linspace(0.10, 0.40, 12)]
        levels = np.linspace(0.06 * peak, peak, 13)
        ax.contourf(xx, yy, dens, levels=levels, colors=colors,
                    antialiased=True, zorder=0)

    _bg_kde(axL, grp_field, TARGET_DENS_COLOR)
    _bg_kde(axR, wt_field, WT_DENS_COLOR)

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
        ax.set_xlim(vlim[0], vlim[1]); ax.set_ylim(vlim[2], vlim[3])
        ax.set_xticks([]); ax.set_yticks([])


def main():
    for gene, cfg in GENES.items():
        print(f"\n=== {gene} ===")
        render_gene(gene, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
