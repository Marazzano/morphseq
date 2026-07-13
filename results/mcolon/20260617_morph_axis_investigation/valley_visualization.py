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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
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
    DistributionAnalysisContext,
    DistributionComparison,
    DistributionRecord,
    PeakResolutionConfig,
    compute_observed_metrics,
    compute_peak_stats,
    compute_resolved_peaks,
)
from morphseq_investigation.core.peak_stability import (  # noqa: E402
    PeakCountRobustnessPolicy,
    PeakVotingSpec,
)
from morphseq_investigation.core.resolved_peak_analysis import (  # noqa: E402
    ResolvedPeakAnalysisSpec,
)
from morphseq_investigation.plotting.modal_distribution_plotting import (  # noqa: E402
    derive_shared_grid,
    draw_resolved_peak_basins,
    format_density_axis,
    mode_count_label,
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
MODE_MIN_VALID_DRAWS = int(np.ceil(0.80 * N_MODE_RESAMPLE_DRAWS))

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

# Stage 2a: the bootstrap mode-count vote is folded INTO compute_resolved_peaks
# (core/distribution_records.py). This config's defaults already match the
# pre-refactor numbers (N_MODE_RESAMPLE_DRAWS/MODE_DOWNSAMPLE_FRACTION/
# MODE_RESAMPLE_MIN_FREQ below), but each render_gene call needs its own seed
# per (gene, hpf, distribution) to preserve the old per-distribution
# independent-RNG-stream behavior -- see render_gene for the derivation.
def _resolution_config(seed):
    return PeakResolutionConfig(
        voting_spec=PeakVotingSpec(
            n_draws=N_MODE_RESAMPLE_DRAWS,
            sample_fraction=MODE_DOWNSAMPLE_FRACTION,
            min_valid_draws=MODE_MIN_VALID_DRAWS,
        ),
        min_bootstrap_sample_size=MIN_EMBRYOS,
        robustness_policy=PeakCountRobustnessPolicy(
            min_mode_frequency=MODE_RESAMPLE_MIN_FREQ
        ),
        seed=seed,
    )


def _build_distribution_record(distribution_id, points, canonical_grid, analysis_spec, *, seed):
    """Build + resolve a typed DistributionRecord in one call: points -> the
    ONE resolve engine (resolve_points_with_analysis_spec + the Stage-2a
    bootstrap vote, reached via compute_resolved_peaks) -> scalar peak_stats.
    Density is derived from `analysis_spec` alone inside the engine -- no
    separate `kde=` parameter (Stage 0: ResolvedPeakAnalysisSpec is the sole
    bandwidth authority)."""
    record = DistributionRecord(
        distribution_id=distribution_id,
        points=np.asarray(points, dtype=float),
        analysis_context=DistributionAnalysisContext(grid=canonical_grid, spec=analysis_spec),
        metadata={
            "bandwidth_rule": analysis_spec.bandwidth_rule,
            "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
            "peak_detector_method": analysis_spec.peak_detector_method,
        },
    )
    record = compute_resolved_peaks(record, _resolution_config(seed))
    record = compute_peak_stats(record)
    return record


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

        canonical_grid = derive_shared_grid(grp, wt, grid=GRID)
        box = (canonical_grid.x_min, canonical_grid.x_max, canonical_grid.y_min, canonical_grid.y_max)

        # Each distribution's bootstrap vote (folded into compute_resolved_peaks,
        # Stage 2a) gets its own independent RNG stream -- same SeedSequence.spawn
        # pattern as the pre-refactor _resampled_mode_count calls, so target and
        # WT draws stay decorrelated per (gene, method, hpf).
        seed_base = 42 + sum(ord(c) for c in f"{gene}:{method_slug}:{hpf}")
        seed_seq = np.random.SeedSequence(seed_base)
        grp_seed = int(seed_seq.spawn(1)[0].generate_state(1)[0])
        wt_seed = int(seed_seq.spawn(1)[0].generate_state(1)[0])

        grp_record = _build_distribution_record(
            f"{gene}_{hpf}_target", grp, canonical_grid, analysis_spec, seed=grp_seed
        )
        wt_record = _build_distribution_record(
            f"{gene}_{hpf}_reference", wt, canonical_grid, analysis_spec, seed=wt_seed
        )
        comparison = DistributionComparison(
            comparison_id=f"{gene}_{hpf}_comparison",
            members={"reference": wt_record, "target": grp_record},
            metadata={"stage_hpf": hpf, "gene": gene},
        )
        comparison = compute_observed_metrics(comparison, name="primary")
        grp_dist = grp_record.resolved_peaks
        wt_dist = wt_record.resolved_peaks
        grp_density = grp_record.resolved_peaks.density_grid
        wt_density = wt_record.resolved_peaks.density_grid
        gx, gy, gd = grp_density.xx, grp_density.yy, grp_density.density

        # Row 1 and row 4 read the vote-supported mode count + reliability
        # straight off the resolved distribution -- no resampling here anymore
        # (Stage 2a: robustness lives in compute_resolved_peaks, plotting only
        # reads fields).
        grp_count = grp_dist.resolved_peak_count
        wt_count = wt_dist.resolved_peak_count
        grp_evidence = grp_dist.resolution_evidence
        wt_evidence = wt_dist.resolution_evidence
        grp_summary = grp_evidence.count_stability if grp_evidence is not None else None
        wt_summary = wt_evidence.count_stability if wt_evidence is not None else None
        grp_freq = grp_summary.mode_frequency if grp_summary is not None else float("nan")
        wt_freq = wt_summary.mode_frequency if wt_summary is not None else float("nan")
        print(
            f"{gene} {method_slug} {hpf}hpf: "
            f"target resolved_peak_count={grp_count}, is_robust={grp_summary.is_robust if grp_summary else False} "
            f"(freq={grp_freq:.2f}, "
            f"valid_draws={grp_summary.peak_count_vote.n_draws_valid if grp_summary else 0}, "
            f"counts={dict(grp_summary.peak_count_vote.peak_count_frequencies) if grp_summary else {}}); "
            f"WT resolved_peak_count={wt_count}, is_robust={wt_summary.is_robust if wt_summary else False} "
            f"(freq={wt_freq:.2f}, "
            f"valid_draws={wt_summary.peak_count_vote.n_draws_valid if wt_summary else 0}, "
            f"counts={dict(wt_summary.peak_count_vote.peak_count_frequencies) if wt_summary else {}})"
        )

        # Comparative target-vs-WT readout. This does not decide the displayed
        # mode count; it only says whether the target differs significantly from WT.
        cells = compute_reference_readout(
            target_points=grp, wt_points=wt, canonical_grid=canonical_grid,
            n_draws=N_READOUT_DRAWS, seed=42, stage_id=f"{hpf}hpf", gene=gene,
            analysis_spec=analysis_spec)

        # ── ROW 1: TARGET KDE + points + supported modes + count ─────────────
        ax = axes[0][col]
        peak = float(gd.max())
        levels = np.linspace(0.06 * peak, peak, 16)
        ax.contourf(gx, gy, gd, levels=levels, cmap=light_blues, extend="max")
        draw_resolved_peak_basins(ax, grp_dist, color=PEAK_OVERLAY_COLOR, linewidth=1.8, n_modes=grp_count)
        for pheno in cfg["phenotype_labels"] + ["unlabeled"]:
            m = phenos == pheno
            if not m.any():
                continue
            ax.scatter(grp[m, 0], grp[m, 1], s=20, alpha=0.9,
                       facecolors=PHENOTYPE_COLORS.get(pheno, UNLABELED_COLOR),
                       edgecolors="k", linewidths=0.4, zorder=4, label=pheno)
        format_density_axis(ax, box)
        ax.set_title(f"{hpf} hpf  ·  {mode_count_label(grp_count, frequency=grp_freq)}",
                     fontsize=LABEL_FS, fontweight="bold", color="#222")

        # ── ROW 2: reference readout (resolved-peak metrics, target vs WT) ───
        render_readout_cell(axes[1][col], cells, label_fs=LABEL_FS)

        # ── ROW 3: DENSITY OVERLAP (lightened background) ────────────────────
        plot_density_overlap(axes[2][col], grp_density, wt_density,
                             alpha_scale=0.62)
        format_density_axis(axes[2][col], box)

        # ── ROW 4: RAW POINTS + supported modes (target | WT) ────────────────
        # Target and WT each show their own vote-supported mode count.
        _points_pair_cell(fig, axes[3][col], grp, phenos, wt, cfg, gene,
                          grp_density, wt_density, box,
                          grp_dist=grp_dist, wt_dist=wt_dist,
                          grp_count=grp_count, grp_freq=grp_freq,
                          wt_count=wt_count, wt_freq=wt_freq)

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
                      grp_count=None, grp_freq=None, wt_count=None, wt_freq=None):
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
        draw_resolved_peak_basins(axL, grp_dist, color=TARGET_DENS_COLOR, n_modes=grp_count)
    if wt_dist is not None:
        draw_resolved_peak_basins(axR, wt_dist, color="#333333", n_modes=wt_count)
    axL.set_title(f"{gene} · {mode_count_label(grp_count, frequency=grp_freq)}",
                  fontsize=LABEL_FS - 2, color=TARGET_DENS_COLOR)
    axR.set_title(f"WT · {mode_count_label(wt_count, frequency=wt_freq)}",
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
