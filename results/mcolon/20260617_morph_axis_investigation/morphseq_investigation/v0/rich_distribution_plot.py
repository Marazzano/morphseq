"""Rich faceted distribution plots for the b9d2 worked example.

This is the "rich plot" follow-up to TASK_E (see docs/tech_debt/next_steps.md
"True KDE ridge plot over time"). The plain emergence strip in
``b9d2_worked_example.py`` stacks one target curve per time bin — it is NOT the
right way to *look* at these distributions. This module builds the real thing:

    columns = time bins (hpf)          rows = three LABEL VIEWS
      one 1-D KDE marginal per group, overlaid within each (view, time) cell.

The three label views (rows), all on ONE shared per-column 1-D grid so the
curves in a cell are directly comparable (ontology §1b):

  row 0  derived   : target peak-finding SampleSets — the label groups we get
                     from UNSUPERVISED mode discovery. Modes are found in 2-D
                     (both features jointly, in ``run_bin``); here we only
                     PROJECT each discovered group onto one feature axis and
                     render its marginal. We never re-discover modes in 1-D.
  row 1  provided  : phenotype-column SampleSets (CE / HTA) — labels we PROVIDE.
  row 2  genotype  : no sub-labels — all b9d2 (target) vs its WT controls
                     (reference). Two curves, the coarsest view.

Feature handling (per the repo owner): mode discovery is 2-D, but we render each
feature's 1-D marginal in its OWN figure. So this emits, per layout:
  - one figure for total_length_um
  - one figure for baseline_deviation_normalized

Two layouts, each written to its own file (the owner wants to see both):
  - panel  : a true (view x time) panel grid via the faceting engine's
             key=(row,col) path — comparisons live WITHIN each cell.
  - ridge  : a joyplot/ridgeline — within each row, time bins are stacked as
             vertically-offset baselines so you watch a distribution march /
             split across time.

So the full matrix of outputs is 2 features x 2 layouts = 4 PNGs.

Run:
  cd .../20260617_morph_axis_investigation
  PYTHONPATH=.:src:$PYTHONPATH conda run -n segmentation_grounded_sam \
      --no-capture-output python -m morphseq_investigation.v0.rich_distribution_plot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze.viz.plotting.faceting_engine import (
    FacetSpec,
    FigureData,
    SubplotData,
    TraceStyle,
    render,
)
from analyze.viz.styling.genotype_colors import get_color_for_genotype

from ..engine.grid import build_grid, evaluate_density
from ..engine.plotting import overlay_strip_subplot
from .b9d2_worked_example import (
    FEATURE_NAMES,
    TARGET_DESIGN_HPF,
    load_binned,
    run_bin,
)

# --------------------------------------------------------------------------- #
# Label-view definitions. Each view is a list of (label, sample_id -> value)
# groups; the actual per-feature values are resolved per feature below.
# --------------------------------------------------------------------------- #
VIEW_ORDER = ("derived", "provided", "genotype")
VIEW_TITLES = {
    "derived": "unsupervised modes (2-D discovery)",
    "provided": "provided phenotype label (CE / HTA)",
    "genotype": "genotype only (b9d2 vs WT)",
}

GRID_RESOLUTION = 200  # smooth 1-D marginals

# The comparison principle: EVERY cell overlays the TARGET distribution(s) on the
# REFERENCE (WT) distribution(s). Reference is ALWAYS dashed (line + fill outline)
# so it reads as the baseline you compare the target against.
#
# Row 0 (modes): HUE ENCODES ROLE. Target modes are shades of RED (warm),
# reference modes are shades of BLUE (cool) and dashed — so warm-vs-cool tells
# you target-vs-reference at a glance, and the shade within each family
# distinguishes the modes (by prominence rank). This reads faster than pairing
# the same hue across roles.
_TARGET_MODE_COLORS = ("#b2182b", "#ef8a62", "#d6604d", "#fddbc7")  # crimson -> salmon
_REFERENCE_MODE_COLORS = ("#2166ac", "#67a9cf", "#4393c3", "#d1e5f0")  # navy -> sky
# Row 1 (phenotype): the canonical project phenotype colors (CE=green, HTA=orange).
from analyze.viz.styling.color_mapping_config import B9D2_PHENOTYPE_COLORS
# Reference / WT is the canonical wildtype soft blue, used as the dashed baseline
# in every row.
_WT_COLOR = get_color_for_genotype("wildtype")  # '#2166AC' soft blue
_TARGET_GENOTYPE_COLOR = get_color_for_genotype("homo")  # b9d2 crimson


@dataclass(frozen=True)
class Group:
    """One overlaid curve: a label + the member sample_ids that form it.

    ``is_reference`` marks the WT baseline — reference groups are drawn dashed
    (line and fill outline) in every view, so target-vs-reference reads the same
    way across all three rows.
    """

    label: str
    sample_ids: tuple[str, ...]
    color: str
    is_reference: bool = False

    @property
    def linestyle(self) -> str:
        return "--" if self.is_reference else "-"


def _rank_ordered_sets(sample_sets, label_group):
    """Return peak SampleSets ordered by prominence rank (1 = most-supported).

    Uses the LabelGroup.per_sample_set_metrics 'prominence_rank' the peak labeler
    records; falls back to descending member count if metrics are absent.
    """
    metrics = getattr(label_group, "per_sample_set_metrics", {}) or {}

    def rank(sset):
        m = metrics.get(sset.sample_set_id, {})
        if "prominence_rank" in m:
            return float(m["prominence_rank"])
        return -float(len(sset.sample_ids))  # bigger set first

    return sorted((s for s in sample_sets if s.sample_ids), key=rank)


# --------------------------------------------------------------------------- #
# Turn one per-bin result (from run_bin) into the three label views. Each view is
# a list of Groups; every view overlays target group(s) on reference group(s).
# --------------------------------------------------------------------------- #
def _groups_for_views(res: dict[str, Any]) -> dict[str, list[Group]]:
    views: dict[str, list[Group]] = {}

    # --- row 0 derived: target modes (solid) vs reference modes (dashed),
    #     color-paired by prominence rank on a shared base palette. ------------
    derived: list[Group] = []
    tgt_modes = _rank_ordered_sets(res["target_peak_sets"], res["target_peak_lg"])
    ref_modes = _rank_ordered_sets(res["reference_peak_sets"], res["reference_peak_lg"])
    for rank_i, sset in enumerate(tgt_modes):
        derived.append(
            Group(
                label=f"target mode {rank_i} (n={len(sset.sample_ids)})",
                sample_ids=tuple(str(s) for s in sset.sample_ids),
                color=_TARGET_MODE_COLORS[rank_i % len(_TARGET_MODE_COLORS)],
            )
        )
    for rank_i, sset in enumerate(ref_modes):
        derived.append(
            Group(
                label=f"WT mode {rank_i} (n={len(sset.sample_ids)})",
                sample_ids=tuple(str(s) for s in sset.sample_ids),
                color=_REFERENCE_MODE_COLORS[rank_i % len(_REFERENCE_MODE_COLORS)],
                is_reference=True,
            )
        )
    views["derived"] = derived

    # --- row 1 provided: target phenotype categories (canonical colors, solid)
    #     overlaid on the WT reference baseline (soft blue, dashed). -----------
    provided: list[Group] = []
    for sset in res["phenotype_sets"]:
        if not sset.sample_ids:
            continue
        name = sset.sample_set_name
        provided.append(
            Group(
                label=f"{name} (n={len(sset.sample_ids)})",
                sample_ids=tuple(str(s) for s in sset.sample_ids),
                color=B9D2_PHENOTYPE_COLORS.get(name, "#666666"),
            )
        )
    provided.append(
        Group(
            label=f"WT (n={res['n_reference']})",
            sample_ids=tuple(str(s) for s in res["reference_sample_ids"]),
            color=_WT_COLOR,
            is_reference=True,
        )
    )
    views["provided"] = provided

    # --- row 2 genotype: all-target (b9d2, solid) vs all-reference (WT, dashed).
    views["genotype"] = [
        Group(
            label=f"b9d2 (n={len(res['target_sample_ids'])})",
            sample_ids=tuple(str(s) for s in res["target_sample_ids"]),
            color=_TARGET_GENOTYPE_COLOR,
        ),
        Group(
            label=f"WT (n={res['n_reference']})",
            sample_ids=tuple(str(s) for s in res["reference_sample_ids"]),
            color=_WT_COLOR,
            is_reference=True,
        ),
    ]
    return views


def _silverman_bandwidth(values: np.ndarray) -> float:
    n = max(len(values), 2)
    spread = float(np.std(values)) or 1.0
    return 1.06 * spread * n ** (-1.0 / 5.0)


@dataclass
class CellDensities:
    """Per-(view, time) cell: a shared 1-D grid + one (label,color,ls,density)
    per group, ready for either renderer."""

    grid: Any
    # (label, color, linestyle, is_reference, DensityGrid)
    curves: list[tuple[str, str, str, bool, Any]]


def _build_cell(
    groups: Sequence[Group],
    sample_value: dict[str, float],
    feature_name: str,
    axis_bounds: tuple[float, float],
) -> CellDensities:
    """Build one cell's shared 1-D grid + each group's marginal DensityGrid.

    ``axis_bounds`` is shared across ALL views in a time column (union of every
    sample in the bin) so the three rows line up on one x-axis per column. The
    grid's fit set is every sample in the bin's groups.
    """
    all_ids: list[str] = []
    for g in groups:
        all_ids.extend(g.sample_ids)
    grid = build_grid(
        feature_names=(feature_name,),
        pooled_values=np.array([sample_value[s] for s in all_ids], dtype=float).reshape(-1, 1),
        fit_sample_ids=all_ids,
        method="fixed_bounds",
        params={"resolution": GRID_RESOLUTION, "bounds": [axis_bounds]},
    )
    curves: list[tuple[str, str, str, bool, Any]] = []
    for g in groups:
        vals = np.array([sample_value[s] for s in g.sample_ids], dtype=float)
        if vals.size == 0:
            continue
        density = evaluate_density(
            grid, vals.reshape(-1, 1), bandwidth_spec=_silverman_bandwidth(vals)
        )
        curves.append((g.label, g.color, g.linestyle, g.is_reference, density))
    return CellDensities(grid=grid, curves=curves)


# --------------------------------------------------------------------------- #
# Assemble all cells for one feature: {view: {design_hpf: CellDensities}}.
# --------------------------------------------------------------------------- #
def _cells_for_feature(
    results: list[dict[str, Any]],
    feature_name: str,
) -> tuple[dict[str, dict[int, CellDensities]], list[int]]:
    feature_idx = FEATURE_NAMES.index(feature_name)
    cells: dict[str, dict[int, CellDensities]] = {v: {} for v in VIEW_ORDER}
    design_hpfs: list[int] = []

    for res in results:
        design_hpf = res["design_hpf"]
        design_hpfs.append(design_hpf)
        # Per-sample feature value for this bin (target + reference members).
        sample_value: dict[str, float] = {}
        for role in ("target", "reference"):
            ids = res[f"{role}_sample_ids"]
            col = res[f"{role}_feature_values"][:, feature_idx]
            for sid, v in zip(ids, col):
                sample_value[str(sid)] = float(v)
        # Shared x-bounds for the whole column (all samples in the bin).
        all_vals = np.array(list(sample_value.values()), dtype=float)
        axis_bounds = (float(all_vals.min()), float(all_vals.max()))

        views = _groups_for_views(res)
        for view in VIEW_ORDER:
            cells[view][design_hpf] = _build_cell(
                views[view], sample_value, feature_name, axis_bounds
            )
    return cells, design_hpfs


# --------------------------------------------------------------------------- #
# LAYOUT 1: panel grid via the faceting engine (key = (view_row, time_col)).
# --------------------------------------------------------------------------- #
def render_panel_grid(
    cells: dict[str, dict[int, CellDensities]],
    design_hpfs: list[int],
    feature_name: str,
    out_path: Path,
) -> None:
    subplots: list[SubplotData] = []
    for r, view in enumerate(VIEW_ORDER):
        for c, hpf in enumerate(design_hpfs):
            cell = cells[view][hpf]
            labels = [lbl for (lbl, _c, _ls, _r, _d) in cell.curves]
            styles = [
                TraceStyle(color=col, alpha=1.0, width=2.0, linestyle=ls)
                for (_lbl, col, ls, _r, _d) in cell.curves
            ]
            densities = [d for (_lbl, _c, _ls, _r, d) in cell.curves]
            title = f"{hpf} hpf" if r == 0 else None
            sub = overlay_strip_subplot(
                cell.grid,
                densities,
                labels=labels,
                styles=styles,
                key=(view, hpf),
                title=title,
                x_label=feature_name if r == len(VIEW_ORDER) - 1 else None,
                y_label=VIEW_TITLES[view] if c == 0 else None,
            )
            subplots.append(sub)

    fig = FigureData(
        title=f"b9d2 distributions over time — {feature_name}",
        subtitle="rows = label views · columns = hpf · solid = target, dashed = WT reference (1-D marginals of 2-D groups)",
        subplots=subplots,
        row_labels=[VIEW_TITLES[v] for v in VIEW_ORDER],
        col_labels=[f"{h} hpf" for h in design_hpfs],
    )
    render(
        fig,
        backend="matplotlib",
        facet=FacetSpec(sharex=False, sharey=False),
        style=_wide_style(),
        output_path=out_path,
    )
    print(f"panel grid written: {out_path}")


def _wide_style():
    from analyze.viz.plotting.faceting_engine import default_style

    style = default_style()
    style.legend_loc = "per-panel"
    style.legend_fontsize = 6
    return style


# --------------------------------------------------------------------------- #
# LAYOUT 2: ridgeline / joyplot. Within each view (row), stack the time bins as
# vertically-offset baselines so the distribution's march across time is legible.
# Self-contained matplotlib (the faceting IR has no offset-baseline band-stack).
# --------------------------------------------------------------------------- #
def render_ridgeline(
    cells: dict[str, dict[int, CellDensities]],
    design_hpfs: list[int],
    feature_name: str,
    out_path: Path,
    *,
    variant: str = "overlaid",
) -> None:
    """Ridgeline / joyplot. ``variant`` controls how target and reference are
    placed WITHIN each hpf bin:

      overlaid : both share the bin's baseline (fills overlap). Best for reading
                 COINCIDENCE — do target and reference sit on top of each other?
      stacked  : reference on the bin baseline, target on a small sub-offset just
                 ABOVE it. Best for reading each SHAPE without fill collision; a
                 vertical gap = "target moved off its reference".
      mirror   : reference mirrored DOWNWARD (negative) from the bin baseline,
                 target upward — a back-to-back raincloud. Symmetric divergence.
    """
    if variant not in ("overlaid", "stacked", "mirror"):
        raise ValueError(f"unknown ridge variant {variant!r}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_views = len(VIEW_ORDER)
    fig, axes = plt.subplots(1, n_views, figsize=(6.0 * n_views, 7.0), squeeze=False)
    axes = axes[0]

    for ax, view in zip(axes, VIEW_ORDER):
        peak = 0.0
        for hpf in design_hpfs:
            for (_lbl, _c, _ls, _r, d) in cells[view][hpf].curves:
                peak = max(peak, float(np.asarray(d.density).max()))
        peak = peak or 1.0
        # Row spacing must clear whatever the variant stacks within a bin.
        if variant == "overlaid":
            step = peak * 0.7
        elif variant == "stacked":
            step = peak * 1.3          # room for ref (baseline) + target (sub-offset)
        else:  # mirror
            step = peak * 1.6          # room for +target above and -reference below
        sub = peak * 0.55              # within-bin sub-offset for the 'stacked' variant

        seen_labels: set[str] = set()
        for row_i, hpf in enumerate(reversed(design_hpfs)):
            offset = row_i * step
            cell = cells[view][hpf]
            x = np.asarray(cell.grid.axis_values[0], dtype=float)
            for (lbl, color, ls, is_ref, d) in cell.curves:
                dens = np.asarray(d.density, dtype=float).reshape(-1)
                base_label = lbl.split(" (n=")[0]
                show = base_label not in seen_labels
                if show:
                    seen_labels.add(base_label)

                # Place the curve per variant. baseline = where the fill sits;
                # sign flips the reference downward in 'mirror'.
                if variant == "overlaid":
                    baseline, y = offset, offset + dens
                elif variant == "stacked":
                    baseline = offset + (0.0 if is_ref else sub)
                    y = baseline + dens
                else:  # mirror: target up, reference down
                    if is_ref:
                        baseline, y = offset, offset - dens
                    else:
                        baseline, y = offset, offset + dens

                if is_ref:
                    # Reference = dashed outline, no solid fill (the baseline).
                    ax.fill_between(
                        x, baseline, y, facecolor="none", edgecolor=color,
                        linestyle="--", linewidth=1.2, alpha=0.9, zorder=row_i,
                    )
                else:
                    ax.fill_between(x, baseline, y, color=color, alpha=0.25, zorder=row_i)
                ax.plot(
                    x, y, color=color, lw=1.8, ls=ls, zorder=row_i,
                    label=base_label if show else None,
                )
            ax.axhline(offset, color="#cccccc", lw=0.6, zorder=row_i - 0.5)
            ax.text(
                x.min(), offset, f"{hpf} hpf  ",
                ha="right", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_title(VIEW_TITLES[view], fontsize=11, fontweight="bold")
        ax.set_xlabel(feature_name)
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.85)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)

    _VARIANT_BLURB = {
        "overlaid": "target & WT share each bin's baseline (read coincidence)",
        "stacked": "WT on baseline, target lifted just above (read each shape)",
        "mirror": "target up, WT mirrored down (read divergence)",
    }
    fig.suptitle(
        f"b9d2 distributions over time (ridge · {variant}) — {feature_name}\n"
        f"columns = label views · each ridge = one hpf bin · earliest at bottom · "
        f"{_VARIANT_BLURB[variant]}",
        fontsize=13, fontweight="bold", linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ridge ({variant}) written: {out_path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    binned = load_binned()
    results: list[dict[str, Any]] = []
    for design_hpf in TARGET_DESIGN_HPF:
        res = run_bin(binned, design_hpf)
        if res is None:
            print(f"[skip] {design_hpf} hpf — too few embryos")
            continue
        results.append(res)

    out_dir = Path(__file__).resolve().parent / "outputs"
    for feature_name in FEATURE_NAMES:
        cells, design_hpfs = _cells_for_feature(results, feature_name)
        render_panel_grid(
            cells, design_hpfs, feature_name,
            out_dir / f"b9d2_rich_panel_{feature_name}.png",
        )
        for variant in ("overlaid", "stacked", "mirror"):
            render_ridgeline(
                cells, design_hpfs, feature_name,
                out_dir / f"b9d2_rich_ridge_{variant}_{feature_name}.png",
                variant=variant,
            )


if __name__ == "__main__":
    main()
