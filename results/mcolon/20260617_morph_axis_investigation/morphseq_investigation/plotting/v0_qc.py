"""V0 distribution quality-control layouts and compatibility adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from morphseq_investigation.core.peak_counting import PeakCountDetail
from morphseq_investigation.plotting.modal_distribution_plotting import (
    HDR_COLOR,
    POINT_COLOR,
    POINT_COLOR_BY_LABEL,
    POINT_EDGE_COLOR,
    TRUE_DENS_COLOR,
    DistributionVisualSpec,
    RowOverlayFn,
    canonical_box_from_specs,
    density_box,
    format_density_axis,
    plot_hdr_contour,
    plot_kde_field,
    plot_metric_summary_row,
    plot_peak_count_summary_row,
    plot_raw_points,
)
from morphseq_investigation.plotting.v0_analysis import (
    compute_v0_metric_summary,
    compute_v0_peak_count_summary,
)


def render_distribution_qc_grid(
    specs: list[DistributionVisualSpec],
    out_path: str | Path,
    *,
    title: str = "V0 modal distribution visual QA",
    show_hdr: bool = True,
    auto_scale: bool = True,
    include_peak_row: bool = True,
    include_metric_row: bool = False,
    observed_details_by_method: dict[str, dict[str, PeakCountDetail | None]] | None = None,
    observed_method_order: tuple[str, ...] | None = None,
    primary_observed_method: str | None = None,
    row_overlays: dict[int, Sequence[RowOverlayFn]] | None = None,
    peak_summaries_by_id: dict[str, dict[str, PeakCountDetail | None]] | None = None,
    metric_summaries_by_id: dict[str, dict[str, float | None]] | None = None,
) -> Path:
    """Render a distribution QA grid from precomputed densities and analyses.

    Rows:
      1. True density if provided, otherwise sampled KDE.
      2. Sampled points on sampled KDE.
      3. Densest-50% HDR contour on sampled KDE.
    """
    if not specs:
        raise ValueError("plot_v0_distribution_qc_grid requires at least one distribution")

    n = len(specs)
    n_rows = 3 + int(include_peak_row) + int(include_metric_row)
    extra_peak_detail_rows = 0
    if observed_details_by_method:
        extra_peak_detail_rows = max(0, len(observed_method_order or tuple(next(iter(observed_details_by_method.values())).keys())) - 1)
    fig_height = 6.95 + (1.05 if include_peak_row else 0.0) + (1.35 if include_metric_row else 0.0) + 0.45 * extra_peak_detail_rows
    fig, axes = plt.subplots(n_rows, n, figsize=(2.45 * n, fig_height), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.12, top=0.84, wspace=0.12, hspace=0.18)

    shared_box = None
    if not auto_scale:
        shared_box = canonical_box_from_specs(specs)

    for col, spec in enumerate(specs):
        points = np.asarray(spec.points, dtype=float)
        if auto_scale:
            grids_for_box = [points]
            if spec.composed_grid is not None:
                true_pts = np.column_stack([spec.composed_grid.xx.ravel(), spec.composed_grid.yy.ravel()])
                finite = np.isfinite(spec.composed_grid.density.ravel()) & (spec.composed_grid.density.ravel() > 0)
                if finite.any():
                    grids_for_box.append(true_pts[finite])
            box = density_box(grids_for_box)
        else:
            box = shared_box
        sample_grid = spec.sampled_grid
        if sample_grid is None:
            raise ValueError(
                f"DistributionVisualSpec {spec.distribution_id!r} requires a precomputed sampled_grid"
            )
        peak_summary = (peak_summaries_by_id or {}).get(spec.distribution_id, {})
        truth_detail = peak_summary.get("truth")
        observed_detail = peak_summary.get("observed")
        observed_method_details = None
        if observed_details_by_method is not None:
            observed_method_details = observed_details_by_method.get(spec.distribution_id)
            if observed_method_details is None:
                observed_method_details = {}
        primary_detail = observed_detail
        if observed_method_details:
            if primary_observed_method and primary_observed_method in observed_method_details:
                primary_detail = observed_method_details[primary_observed_method]
            elif observed_method_order:
                primary_detail = observed_method_details.get(observed_method_order[0], observed_detail)
            else:
                primary_detail = next(iter(observed_method_details.values()))

        ax = axes[0][col]
        if spec.composed_grid is not None:
            plot_kde_field(
                ax,
                spec.composed_grid,
                color=TRUE_DENS_COLOR,
                cmap=None,
                alpha_max=0.30,
            )
        else:
            plot_kde_field(ax, sample_grid)
        format_density_axis(ax, box)
        for overlay_fn in (row_overlays or {}).get(0, ()):
            overlay_fn(ax, spec, sample_grid, box)
        ax.set_title(spec.distribution_id.replace("_", "\n"), fontsize=8.0, fontweight="bold")
        if spec.note:
            ax.text(0.02, 0.03, spec.note, transform=ax.transAxes, fontsize=6.2, color="#555", va="bottom")

        ax = axes[1][col]
        plot_kde_field(ax, sample_grid)
        plot_raw_points(ax, points, spec.component_labels, s=14)
        format_density_axis(ax, box)
        for overlay_fn in (row_overlays or {}).get(1, ()):
            overlay_fn(ax, spec, sample_grid, box)

        ax = axes[2][col]
        plot_kde_field(ax, sample_grid)
        if show_hdr:
            plot_hdr_contour(ax, sample_grid)
        if primary_detail is not None and primary_detail.n_modes >= 2 and primary_detail.split_level is not None:
            ax.contour(
                sample_grid.xx,
                sample_grid.yy,
                sample_grid.density,
                levels=[primary_detail.split_level],
                colors="#B8860B",
                linewidths=1.8,
                linestyles="--",
                zorder=5,
            )
        plot_raw_points(ax, points, spec.component_labels, s=11)
        format_density_axis(ax, box)
        for overlay_fn in (row_overlays or {}).get(2, ()):
            overlay_fn(ax, spec, sample_grid, box)

        row_idx = 3
        if include_peak_row:
            plot_peak_count_summary_row(
                axes[row_idx][col],
                truth_detail,
                observed_method_details if observed_method_details else observed_detail,
                title="peak count audit",
                observed_method_order=observed_method_order,
                primary_method=primary_observed_method,
            )
            row_idx += 1
        if include_metric_row:
            metric_summary = (metric_summaries_by_id or {}).get(spec.distribution_id)
            if metric_summary is None:
                raise ValueError(
                    "include_metric_row=True requires a precomputed metric summary "
                    f"for {spec.distribution_id!r}"
                )
            plot_metric_summary_row(axes[row_idx][col], metric_summary, title="metric probes")

    axes[0][0].set_ylabel("true density\nor sample KDE", fontsize=9)
    axes[1][0].set_ylabel("sampled points\n+ KDE", fontsize=9)
    axes[2][0].set_ylabel("sample KDE\n+ HDR 50% / valley", fontsize=9)
    if include_peak_row:
        axes[3][0].set_ylabel("peak count\n(truth vs detected)", fontsize=9)
    if include_metric_row:
        axes[3 + int(include_peak_row)][0].set_ylabel("metrics\n(normalized)", fontsize=9)
    all_labels: list[str] = []
    for spec in specs:
        if spec.component_labels is None:
            continue
        all_labels.extend([str(label) for label in np.unique(spec.component_labels)])
    preferred = [
        "mode_0",
        "mode_1",
        "mode_2",
        "mode_left",
        "mode_right",
        "bridge_left_right",
        "bridge",
        "artifact",
    ]
    legend_labels: list[str] = []
    seen = set()
    for label in preferred + sorted(set(all_labels) - set(preferred)):
        if label in seen or label not in all_labels:
            continue
        seen.add(label)
        legend_labels.append(label)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=POINT_COLOR_BY_LABEL.get(label, POINT_COLOR),
            markeredgecolor=POINT_EDGE_COLOR,
            markersize=6,
            label=label,
        )
        for label in legend_labels
    ]
    handles.append(Line2D([0], [0], color="#B8860B", lw=1.8, ls="--", label="valley split floor"))
    handles.append(Line2D([0], [0], color=HDR_COLOR, lw=1.8, label="HDR 50% contour"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.5,
        0.015,
        "Point colors are known generator component labels for visual QA; they are not inferred modes.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555",
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path


def plot_v0_distribution_qc_grid(
    specs: list[DistributionVisualSpec],
    out_path: str | Path,
    *,
    title: str = "V0 modal distribution visual QA",
    show_hdr: bool = True,
    auto_scale: bool = True,
    include_peak_row: bool = True,
    include_metric_row: bool = False,
    observed_details_by_method: dict[str, dict[str, PeakCountDetail | None]] | None = None,
    observed_method_order: tuple[str, ...] | None = None,
    primary_observed_method: str | None = None,
    row_overlays: dict[int, Sequence[RowOverlayFn]] | None = None,
) -> Path:
    """Legacy composition wrapper outside the pure-renderer boundary.

    It computes V0 summaries for historical callers, then delegates all
    drawing to :func:`render_distribution_qc_grid`.
    """

    peak_summaries = {
        spec.distribution_id: compute_v0_peak_count_summary(
            truth_density=None if spec.composed_grid is None else spec.composed_grid.density,
            observed_density=None if spec.sampled_grid is None else spec.sampled_grid.density,
        )
        for spec in specs
    }
    metric_summaries = None
    if include_metric_row:
        metric_summaries = {
            spec.distribution_id: compute_v0_metric_summary(
                spec.points,
                distribution_id=spec.distribution_id,
            )
            for spec in specs
        }
    return render_distribution_qc_grid(
        specs,
        out_path,
        title=title,
        show_hdr=show_hdr,
        auto_scale=auto_scale,
        include_peak_row=include_peak_row,
        include_metric_row=include_metric_row,
        observed_details_by_method=observed_details_by_method,
        observed_method_order=observed_method_order,
        primary_observed_method=primary_observed_method,
        row_overlays=row_overlays,
        peak_summaries_by_id=peak_summaries,
        metric_summaries_by_id=metric_summaries,
    )
