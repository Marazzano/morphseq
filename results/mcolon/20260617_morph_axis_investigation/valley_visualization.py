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

from dataclasses import dataclass
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
    isotropic_geometry_kde_spec, normalize_shape,
)
from morphseq_investigation.core.density_composition import CanonicalGrid  # noqa: E402
from morphseq_investigation.core.distribution_records import (  # noqa: E402
    DistributionComparison,
    DistributionRecord,
    add_density,
    add_observed_metrics,
    add_peak_detection,
    add_peak_membership,
    add_resolved_peak_distribution,
    add_resolved_peak_summary,
)
from morphseq_investigation.core.resolved_peak_analysis import (  # noqa: E402
    ResolvedPeakAnalysisSpec, resolve_points_with_analysis_spec,
)
from morphseq_investigation.plotting.modal_distribution_plotting import (  # noqa: E402
    derive_shared_grid,
    format_density_axis,
    plot_density_overlap,
)
from resolved_peak_reference_readout import (  # noqa: E402
    READOUT_METRICS, compute_reference_readout, render_readout_cell,
    _SIG_COLOR, _NS_COLOR, _INVALID_COLOR,
)

# Tuned geometry-bandwidth methods to render. Each is a
# (KDESpec, ResolvedPeakAnalysisSpec, label, filename-slug). The MST-edge rule is
# the current default; median-kNN is kept as a conservative fallback diagnostic.
_PEAK_METHOD = "kde_peak_basins_sample_support"
def _analysis_spec(rule, mult):
    return ResolvedPeakAnalysisSpec(bandwidth_rule=rule, bandwidth_multiplier=mult,
                                    peak_detector_method=_PEAK_METHOD, min_sample_fraction=0.10)
DEFAULT_KDE = isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75)
DEFAULT_ANALYSIS_SPEC = _analysis_spec("longest_non_outlier_MST_edge", 0.75)
METHODS = [
    (DEFAULT_KDE, DEFAULT_ANALYSIS_SPEC, "MST_edge x0.75", "MST_edge"),
    (isotropic_geometry_kde_spec("median_kNN_distance", 1.0),
     _analysis_spec("median_kNN_distance", 1.0), "median_kNN x1.0", "median_kNN"),
]
PEAK_OVERLAY_COLOR = "#666666"   # grey dashed peak-basin contour (row 1)
N_READOUT_DRAWS = 200
N_MODE_RESAMPLE_DRAWS = 80
MODE_DOWNSAMPLE_FRACTION = 0.80
MODE_RESAMPLE_MIN_FREQ = 0.80

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


@dataclass(frozen=True)
class ResampledModeCount:
    modal_count: int
    count: int | None
    frequency: float
    counts: dict[int, int]

    @property
    def stable(self):
        return self.count is not None


def _mode_count_label(n_modes, *, frequency=None):
    """Compact label for downsample-supported mode counts in row titles."""
    if n_modes is None:
        return "mode count unstable"
    n = int(n_modes)
    word = "mode" if n == 1 else "modes"
    suffix = f" ({frequency:.0%})" if frequency is not None else ""
    return f"{n} {word}{suffix}"


def _resampled_mode_count(points, canonical_grid, analysis_spec, *, n_draws, sample_fraction, min_freq, rng):
    """Subsample one distribution and return its consensus resolved mode count."""
    n = len(points)
    sample_n = min(n, max(MIN_EMBRYOS, int(np.ceil(float(sample_fraction) * n))))
    counts = []
    for _ in range(n_draws):
        sample = points[rng.choice(n, size=sample_n, replace=False)]
        counts.append(_resolved_peaks(sample, canonical_grid, analysis_spec).number_of_peaks)
    values, freqs = np.unique(np.asarray(counts, dtype=int), return_counts=True)
    best_i = int(np.argmax(freqs))
    best_count = int(values[best_i])
    best_freq = float(freqs[best_i]) / float(n_draws)
    return ResampledModeCount(
        modal_count=best_count,
        count=best_count if best_freq >= min_freq else None,
        frequency=best_freq,
        counts={int(v): int(c) for v, c in zip(values, freqs)},
    )


def _displayable_resampled_mode_count(mode_count, observed_count):
    """Do not display a resampled count that the observed KDE cannot outline."""
    if mode_count.count is None or int(mode_count.count) <= int(observed_count):
        return mode_count
    return ResampledModeCount(
        modal_count=mode_count.modal_count,
        count=None,
        frequency=mode_count.frequency,
        counts=mode_count.counts,
    )


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

def _resolved_peaks(points, canonical_grid, analysis_spec):
    """Resolve peaks for a point set on a shared CanonicalGrid; return the
    ResolvedPeakDistribution (whose .peaks carry center + R80 radius)."""
    return resolve_points_with_analysis_spec(
        distribution_id="fig", points=points, canonical_grid=canonical_grid,
        analysis_spec=analysis_spec,
    )


def _build_distribution_record(distribution_id, points, canonical_grid, analysis_spec, *, kde):
    record = DistributionRecord(
        distribution_id=distribution_id,
        points=np.asarray(points, dtype=float),
        canonical_grid=canonical_grid,
        metadata={
            "bandwidth_rule": analysis_spec.bandwidth_rule,
            "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
            "peak_detector_method": analysis_spec.peak_detector_method,
        },
    )
    record = add_density(record, name="primary", kde=kde)
    record = add_peak_detection(
        record,
        name="primary",
        density_name="primary",
        method=analysis_spec.peak_detector_method,
        min_component_mass_frac=analysis_spec.min_component_mass_frac,
        min_sample_fraction=analysis_spec.min_sample_fraction,
        min_prominence_ratio=analysis_spec.min_prominence_ratio,
        outlier_density_floor_fraction=analysis_spec.outlier_density_floor_fraction,
    )
    record = add_peak_membership(record, name="primary", detection_name="primary")
    record = add_resolved_peak_distribution(
        record,
        name="primary",
        density_name="primary",
        detection_name="primary",
        membership_name="primary",
        outlier_density_floor_fraction=analysis_spec.outlier_density_floor_fraction,
    )
    record = add_resolved_peak_summary(record, name="primary", resolved_name="primary")
    return record


def _hdr_contour(ax, xx, yy, mode_dens, color, lw, hdr_mass):
    """Draw one smooth HDR iso-density loop of `mode_dens` enclosing `hdr_mass`
    of its mass -- a closed curve that follows the KDE bump's shape."""
    if mode_dens.max() <= 0:
        return
    flat = np.sort(mode_dens.ravel())[::-1]
    csum = np.cumsum(flat)
    total = float(csum[-1])
    if total <= 0:
        return
    idx = min(int(np.searchsorted(csum / total, hdr_mass, side="left")), len(flat) - 1)
    level = float(flat[idx])
    if level <= 0:
        level = float(flat[flat > 0].min()) if np.any(flat > 0) else 0.0
    ax.contour(xx, yy, mode_dens, levels=[level], colors=[color],
               linewidths=lw, zorder=5, alpha=0.95)


def _draw_basins(ax, dist, *, color, lw=2.0, hdr_mass=0.60, n_modes=None):
    """Outline the modes with smooth KDE HDR loops that hug each bump's shape.

    `n_modes` is the downsample-supported number of modes to display.
    - n_modes is None -> do not draw a basin count claim.
    - n_modes >= detected count -> draw every detected basin.
    - n_modes <  detected count -> the extra detected modes aren't statistically
      supported; collapse to ONE outer loop of the whole distribution (for
      n_modes==1) rather than drawing unsupported sub-modes. This keeps the
      drawing consistent with the count shown in the title.

    Basin masking uses `empirical_basin_labels` so adjacent supported modes stay
    separate loops."""
    dg = dist.density_grid
    dens = np.asarray(dg.density, dtype=float)
    labels = getattr(dist, "empirical_basin_labels", None)
    labels = np.asarray(labels, dtype=int) if labels is not None else None
    peaks = list(dist.peaks)
    detected = len(peaks)
    if n_modes is None:
        return
    show = int(n_modes)

    # Not enough support for the detected sub-structure: draw ONE loop for the
    # whole distribution (the honest "one mode" view). Only the single-mode
    # collapse is handled specially; 1 <= show < detected with show>1 is rare and
    # falls through to drawing the `show` strongest basins.
    if show <= 1 or detected <= 1:
        # center mark(s): the strongest peak
        if peaks:
            cx, cy = peaks[0].geometry.center_coordinate
            ax.plot([cx], [cy], marker="+", color=color, ms=7, mew=1.6, zorder=6)
        _hdr_contour(ax, dg.xx, dg.yy, dens, color, lw, hdr_mass)
        return

    # Show the `show` strongest detected modes as separate basin loops.
    peaks_by_height = sorted(peaks, key=lambda p: p.geometry.total_support_fraction, reverse=True)
    for pk in peaks_by_height[:show]:
        cx, cy = pk.geometry.center_coordinate
        ax.plot([cx], [cy], marker="+", color=color, ms=7, mew=1.6, zorder=6)
        if labels is not None and pk.geometry.peak_id in np.unique(labels):
            mode_dens = np.where(labels == pk.geometry.peak_id, dens, 0.0)
        else:
            mode_dens = dens
        _hdr_contour(ax, dg.xx, dg.yy, mode_dens, color, lw, hdr_mass)


def render_gene(gene, cfg, *, kde=DEFAULT_KDE, analysis_spec=DEFAULT_ANALYSIS_SPEC,
                method_label="MST_edge x0.75", method_slug="MST_edge"):
    bins = load_bins(cfg)
    hpfs = [h for h in TARGET_DESIGN_HPF if h in bins]
    n = len(hpfs)
    if n == 0:
        print(f"  {gene}: no usable bins"); return

    # 4 rows: (1) target KDE + peaks, (2) reference readout, (3) density overlap,
    # (4) raw points with circled peaks. Row 2 (readout) is shorter than the map rows.
    fig, axes = plt.subplots(4, n, figsize=(3.3 * n, 12.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.0, 0.72, 1.0, 1.0]})
    fig.subplots_adjust(bottom=0.06, top=0.88, left=0.09, right=0.99,
                        hspace=0.34, wspace=0.10)

    from matplotlib.colors import ListedColormap
    light_blues = ListedColormap(plt.get_cmap(DENS_CMAP)(np.linspace(0.0, DENS_CMAP_HI, 256)))

    for col, hpf in enumerate(hpfs):
        grp_raw, phenos, wt_raw = bins[hpf]
        grp = normalize_shape(grp_raw)
        wt = normalize_shape(wt_raw)

        canonical_grid = derive_shared_grid(grp, wt, grid=GRID, kde=kde)
        box = (canonical_grid.x_min, canonical_grid.x_max, canonical_grid.y_min, canonical_grid.y_max)
        grp_record = _build_distribution_record(
            f"{gene}_{hpf}_target", grp, canonical_grid, analysis_spec, kde=kde
        )
        wt_record = _build_distribution_record(
            f"{gene}_{hpf}_reference", wt, canonical_grid, analysis_spec, kde=kde
        )
        comparison = DistributionComparison(
            comparison_id=f"{gene}_{hpf}_comparison",
            members={"reference": wt_record, "target": grp_record},
            metadata={"stage_hpf": hpf, "gene": gene},
        )
        comparison = add_observed_metrics(comparison, name="primary")
        grp_dist = grp_record.resolved_peak_distributions["primary"]
        wt_dist = wt_record.resolved_peak_distributions["primary"]
        grp_density = grp_record.densities["primary"]
        wt_density = wt_record.densities["primary"]
        gx, gy, gd = grp_density.xx, grp_density.yy, grp_density.density

        # Downsample each distribution independently. Row 1 and row 4 only claim
        # a mode count if the same count appears in >= MODE_RESAMPLE_MIN_FREQ of
        # subsampled draws; otherwise the panel is labeled unstable.
        seed_base = 42 + sum(ord(c) for c in f"{gene}:{method_slug}:{hpf}")
        seed_seq = np.random.SeedSequence(seed_base)
        grp_mode_boot = _resampled_mode_count(
            grp, canonical_grid, analysis_spec,
            n_draws=N_MODE_RESAMPLE_DRAWS,
            sample_fraction=MODE_DOWNSAMPLE_FRACTION,
            min_freq=MODE_RESAMPLE_MIN_FREQ,
            rng=np.random.default_rng(seed_seq.spawn(1)[0]),
        )
        wt_mode_boot = _resampled_mode_count(
            wt, canonical_grid, analysis_spec,
            n_draws=N_MODE_RESAMPLE_DRAWS,
            sample_fraction=MODE_DOWNSAMPLE_FRACTION,
            min_freq=MODE_RESAMPLE_MIN_FREQ,
            rng=np.random.default_rng(seed_seq.spawn(1)[0]),
        )
        grp_mode_boot = _displayable_resampled_mode_count(grp_mode_boot, grp_dist.number_of_peaks)
        wt_mode_boot = _displayable_resampled_mode_count(wt_mode_boot, wt_dist.number_of_peaks)
        print(
            f"{gene} {method_slug} {hpf}hpf: "
            f"target subsample modal={grp_mode_boot.modal_count}, display={grp_mode_boot.count} "
            f"(freq={grp_mode_boot.frequency:.2f}, observed={grp_dist.number_of_peaks}, "
            f"counts={grp_mode_boot.counts}); "
            f"WT subsample modal={wt_mode_boot.modal_count}, display={wt_mode_boot.count} "
            f"(freq={wt_mode_boot.frequency:.2f}, observed={wt_dist.number_of_peaks}, "
            f"counts={wt_mode_boot.counts})"
        )

        # Comparative target-vs-WT readout. This does not decide the displayed
        # mode count; it only says whether the target differs significantly from WT.
        cells = compute_reference_readout(
            target_points=grp, wt_points=wt, canonical_grid=canonical_grid,
            n_draws=N_READOUT_DRAWS, seed=42, stage_id=f"{hpf}hpf", gene=gene,
            analysis_spec=analysis_spec)
        observed_metrics = comparison.observed_metrics["primary"]
        _ = observed_metrics  # retained for parity with the new typed comparison path

        # ── ROW 1: TARGET KDE + points + supported modes + count ─────────────
        ax = axes[0][col]
        peak = float(gd.max())
        levels = np.linspace(0.06 * peak, peak, 16)
        ax.contourf(gx, gy, gd, levels=levels, cmap=light_blues, extend="max")
        _draw_basins(ax, grp_dist, color=PEAK_OVERLAY_COLOR, lw=1.8, n_modes=grp_mode_boot.count)
        for pheno in cfg["phenotype_labels"] + ["unlabeled"]:
            m = phenos == pheno
            if not m.any():
                continue
            ax.scatter(grp[m, 0], grp[m, 1], s=20, alpha=0.9,
                       facecolors=PHENOTYPE_COLORS.get(pheno, UNLABELED_COLOR),
                       edgecolors="k", linewidths=0.4, zorder=4, label=pheno)
        format_density_axis(ax, box)
        ax.set_title(f"{hpf} hpf  ·  {_mode_count_label(grp_mode_boot.count, frequency=grp_mode_boot.frequency)}",
                     fontsize=LABEL_FS, fontweight="bold", color="#222")

        # ── ROW 2: reference readout (resolved-peak metrics, target vs WT) ───
        render_readout_cell(axes[1][col], cells, label_fs=LABEL_FS)

        # ── ROW 3: DENSITY OVERLAP (lightened background) ────────────────────
        plot_density_overlap(axes[2][col], grp_density, wt_density,
                             alpha_scale=0.62)
        format_density_axis(axes[2][col], box)

        # ── ROW 4: RAW POINTS + supported modes (target | WT) ────────────────
        # Target and WT each show their own downsample-supported mode count.
        _points_pair_cell(fig, axes[3][col], grp, phenos, wt, cfg, gene,
                          grp_density, wt_density, box,
                          grp_dist=grp_dist, wt_dist=wt_dist,
                          grp_mode_boot=grp_mode_boot, wt_mode_boot=wt_mode_boot)

    # row labels (row 2 readout has no map, so label it plainly)
    axes[0][0].set_ylabel("TARGET KDE\n+ stable modes", fontsize=LABEL_FS)
    metric_order = " · ".join(f"{i+1}.{s.label}" for i, s in enumerate(READOUT_METRICS))
    for r, txt in [(1, "REF READOUT\n(vs WT)"), (2, "DENSITY OVERLAP\n(target vs WT)"),
                   (3, "RAW POINTS\n(target | WT)")]:
        pos = axes[r][0].get_position()
        fig.text(0.016, pos.y0 + pos.height / 2, txt, fontsize=LABEL_FS,
                 rotation=90, va="center", ha="center")

    handles, labels_ = axes[0][0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles += [
        Patch(facecolor=TARGET_DENS_COLOR, alpha=0.55, label=f"{gene} density"),
        Patch(facecolor=WT_DENS_COLOR, alpha=0.55, label="WT density"),
        Line2D([0], [0], color=PEAK_OVERLAY_COLOR, lw=1.8, label="resolved mode basin"),
    ]
    labels_ += [f"{gene} density", "WT density", "resolved mode basin"]
    fig.legend(handles, labels_, loc="lower center", ncol=len(handles),
               fontsize=LABEL_FS - 1, frameon=False, bbox_to_anchor=(0.5, 0.004))

    fig.suptitle(
        f"{gene} — mode-structure visualization  ·  bandwidth: {method_label}\n"
        f"axis: {X_FEAT} x {Y_FEAT} (normalized).  Row 1 = target KDE + resolved mode basins "
        f"(count accepted if {MODE_DOWNSAMPLE_FRACTION:.0%}-downsample consensus >= "
        f"{MODE_RESAMPLE_MIN_FREQ:.0%}).  Row 2 = resolved-peak readout vs matched-N WT "
        f"({N_READOUT_DRAWS} draws), top→bottom: {metric_order}.\n"
        f"Row 3 = target-vs-WT density overlap.  Row 4 = raw points + resolved mode basins/counts (target | WT).",
        fontsize=LABEL_FS, fontweight="bold", y=0.997)
    out = PLOT_DIR / f"{gene}_valley_{method_slug}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.name}")


def _points_pair_cell(fig, host_ax, grp_pts, phenos, wt_pts, cfg, gene,
                      grp_grid, wt_grid, box, *, grp_dist=None, wt_dist=None,
                      grp_mode_boot=None, wt_mode_boot=None):
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
    # Draw each detected peak basin on its own panel: target modes (blue) on the
    # left, WT modes (dark) on the right, so the mode counts are explicit.
    if grp_dist is not None:
        _draw_basins(axL, grp_dist, color=TARGET_DENS_COLOR,
                     n_modes=None if grp_mode_boot is None else grp_mode_boot.count)
    if wt_dist is not None:
        _draw_basins(axR, wt_dist, color="#333333",
                     n_modes=None if wt_mode_boot is None else wt_mode_boot.count)
    grp_count = None if grp_mode_boot is None else grp_mode_boot.count
    grp_freq = None if grp_mode_boot is None else grp_mode_boot.frequency
    wt_count = None if wt_mode_boot is None else wt_mode_boot.count
    wt_freq = None if wt_mode_boot is None else wt_mode_boot.frequency
    axL.set_title(f"{gene} · {_mode_count_label(grp_count, frequency=grp_freq)}",
                  fontsize=LABEL_FS - 2, color=TARGET_DENS_COLOR)
    axR.set_title(f"WT · {_mode_count_label(wt_count, frequency=wt_freq)}",
                  fontsize=LABEL_FS - 2, color=WT_POINT_COLOR)
    for ax in (axL, axR):
        format_density_axis(ax, box)


def main():
    for gene, cfg in GENES.items():
        for kde, analysis_spec, label, slug in METHODS:
            print(f"\n=== {gene} · {label} ===")
            render_gene(gene, cfg, kde=kde, analysis_spec=analysis_spec,
                        method_label=label, method_slug=slug)
    print("\nDone.")


if __name__ == "__main__":
    main()
