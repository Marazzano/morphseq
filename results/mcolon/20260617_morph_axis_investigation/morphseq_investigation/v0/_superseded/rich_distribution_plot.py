"""RETIRED (superseded) — its ridgeline offset math + target/reference styling
were re-homed onto the grid IR in ``engine/ridge.py::plot_1d_ridgeline``; the
canonical driver is ``../b9d2_catalog_example.py``. Kept for historical
reference; will NOT run on ``main`` (imports the removed pre-migration API).

Rich faceted distribution plots for the b9d2 worked example.

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
# LAYOUT 2: ridgeline / joyplot. RETIRED as bespoke matplotlib — the offset math
# now lives in the shared Tier-2 verb ``engine.ridge.plot_1d_ridgeline`` (TASK_D),
# which consumes the same DistributionGrid IR as ``plot_1d_density_grid``. This
# thin shim repackages the prototype's per-(view, hpf) CellDensities into that IR
# (view -> row facet, hpf -> col facet / time bin), preserving the per-curve
# colors via ``color_lookup`` and marking WT reference curves with the reserved
# ``"reference"`` style_group so they degrade to the dashed/unfilled baseline.
# --------------------------------------------------------------------------- #
def _cells_to_distribution_grid(
    cells: dict[str, dict[int, CellDensities]],
    design_hpfs: list[int],
    feature_name: str,
) -> tuple["DistributionGrid", dict[tuple[Any, str], str]]:
    """Repackage the prototype cells into a DistributionGrid + a color_lookup.

    ``style_group`` is ``"reference"`` for WT curves (so the verb applies the
    reserved dashed/unfilled baseline role) and the curve's own display label
    otherwise. ``color_lookup`` keys are ``(style_group, sample_set_name)`` — the
    exact key ``engine.ridge`` resolves colors on — so the prototype's bespoke
    per-mode/per-phenotype palette is preserved through the shared verb.
    """
    from ..engine.plotting import DistributionCurve, DistributionGrid

    curves: list[DistributionCurve] = []
    color_lookup: dict[tuple[Any, str], str] = {}
    for view in VIEW_ORDER:
        for hpf in design_hpfs:
            cell = cells[view][hpf]
            for (lbl, color, _ls, is_ref, density) in cell.curves:
                name = lbl.split(" (n=")[0]
                style_group = "reference" if is_ref else name
                key = (style_group, name)
                color_lookup[key] = color
                curves.append(
                    DistributionCurve(
                        cell=(VIEW_TITLES[view], hpf),
                        sample_set_name=name,
                        style_group=style_group,
                        grid=cell.grid,
                        density=density,
                        sample_count=0,
                    )
                )
    from ..engine.facets import CoordinateFacet, LabelGroupFacet

    grid = DistributionGrid(
        feature_name=feature_name,
        row=LabelGroupFacet(),
        col=CoordinateFacet("design_hpf"),
        curves=tuple(curves),
    )
    return grid, color_lookup


def render_ridgeline(
    cells: dict[str, dict[int, CellDensities]],
    design_hpfs: list[int],
    feature_name: str,
    out_path: Path,
    *,
    variant: str = "overlaid",
) -> None:
    """Ridgeline / joyplot — now a thin wrapper over the shared TASK_D verb
    ``engine.ridge.plot_1d_ridgeline``. ``variant`` (overlaid/stacked/mirror)
    is passed straight through; the offset math + target/reference styling that
    used to live here is the reviewed logic that TASK_D re-homed onto the IR.
    """
    from ..engine.ridge import plot_1d_ridgeline

    grid, color_lookup = _cells_to_distribution_grid(cells, design_hpfs, feature_name)
    fig = plot_1d_ridgeline(
        grid,
        variant=variant,
        color_lookup=color_lookup,
        reference_role="reference",
        title=f"b9d2 distributions over time (ridge · {variant}) — {feature_name}",
        output_path=out_path,
    )
    import matplotlib.pyplot as plt

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
