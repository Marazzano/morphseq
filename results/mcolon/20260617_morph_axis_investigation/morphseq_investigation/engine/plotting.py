"""Pure IR emitters — turn frozen ``Grid``/``DensityGrid``/``SampleSet`` objects
into faceting-engine IR (``TraceData`` / ``SubplotData`` / ``FigureData``).

Source of truth: ``docs/DISTRIBUTION_CATALOG_API.md`` ("Typed facet keys",
"PATH A", "PATH B", "CurveKey") + ``docs/VISUALIZATION_TAXONOMY.md`` (Tier-1
primitives vs Tier-2 verbs) + ``docs/tasks_catalog/TASK_C_plotting.md``.

**This module does NOT draw.** It only builds pure dataclasses defined in the
real faceting engine (``src/analyze/viz/plotting/faceting_engine/ir.py``) and
hands them to that engine's own ``render()``. No new rendering backend, no
bespoke matplotlib/plotly calls live here.

The payoff from ontology §1b: a KDE strip is nothing but a 1-D ``Grid`` + a
``DensityGrid`` evaluated on it — the SAME machinery as the 2-D peak grid,
dimension = 1. So ``strip_trace`` below is a thin adapter, not a new code path.

The high-level 1-D API consumes attached ``(Distribution, LabelGroup)`` pairs
and typed ``FacetKey`` values.
Both emit the SAME renderer-neutral ``DistributionGrid`` IR;
``plot_1d_density_grid`` renders it. The retired ``DistributionGrouping`` /
``FacetCoordinate`` enum are GONE from this module (spec §"Removed vocabulary").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from analyze.viz.plotting.faceting_engine import (
    FacetSpec,
    FigureData,
    SubplotData,
    TraceData,
    TraceStyle,
    render,
)
from analyze.viz.styling import STANDARD_PALETTE
from analyze.viz.styling.genotype_colors import get_color_for_genotype

from .catalog import DistributionComparison, DistributionComparisons
from .facets import CoordinateFacet, FacetKey, LabelGroupFacet
from .objects import (
    DensityEstimate,
    DensityGrid,
    Distribution,
    Grid,
    LabelGroup,
    SampleSet,
)
from .identifiers import make_grid_id

logger = logging.getLogger(__name__)

# Reserved visual category for LabelGroup.unassigned_sample_ids (ontology §3):
# never dropped, never counted as a mode, always rendered muted.
UNASSIGNED_LABEL = "unassigned"
_UNASSIGNED_STYLE_DEFAULTS = dict(color="#B0B0B0", alpha=0.5, width=1.0, linestyle=":")


def _default_style_for_label(label: str | None) -> TraceStyle:
    """Resolve a ``TraceStyle`` for a trace label.

    ``unassigned`` always gets the reserved muted style. Any other label is
    treated as a genotype-ish string and resolved via
    ``get_color_for_genotype`` (MEMORY.md "Genotype Colors") so colors are
    never hard-coded here; unrecognized labels fall back to that helper's own
    gray default.
    """
    if label == UNASSIGNED_LABEL:
        return TraceStyle(**_UNASSIGNED_STYLE_DEFAULTS)
    color = get_color_for_genotype(label) if label else "#808080"
    return TraceStyle(color=color, alpha=1.0, width=2.0, linestyle="-")


# --------------------------------------------------------------------------- #
# 1. KDE strip: one (Grid, DensityGrid) pair -> one TraceData curve
# --------------------------------------------------------------------------- #
def strip_trace(
    grid: Grid,
    density_grid: DensityGrid,
    *,
    style: TraceStyle | None = None,
    label: str | None = None,
) -> TraceData:
    """Build a single KDE-strip ``TraceData`` from a 1-D ``Grid`` + its ``DensityGrid``.

    Signature note: a ``DensityGrid`` alone does not carry ``axis_values`` (only
    ``density`` + ``grid_id`` — ontology §1b), so this takes BOTH the owning
    ``Grid`` (for ``x = grid.axis_values[0]``) and the ``DensityGrid`` (for
    ``y = density_grid.density``), and asserts they actually correspond
    (``density_grid.grid_id == grid.grid_id``) before use.

    ``grid`` must be 1-D (``len(grid.axis_values) == 1``) — the strip case of
    §1b ("KDE strips fall out for free": a 1-feature Grid + evaluate_density
    IS a KDE strip, no special casing). N-D grids are the 2-D peak-field case,
    out of scope here (see TASK_D2 stub for the deferred 2-D panel).

    Parameters
    ----------
    grid : Grid
        The 1-D grid the density was evaluated on (supplies the x-axis).
    density_grid : DensityGrid
        The KDE field evaluated on ``grid`` (supplies the y-axis). Must share
        ``grid.grid_id``.
    style : TraceStyle, optional
        Visual style. Defaults via :func:`_default_style_for_label` (genotype
        color lookup, or the reserved muted style for ``label="unassigned"``).
    label : str, optional
        Trace label. ``"unassigned"`` is treated as the reserved category.
    """
    if len(grid.axis_values) != 1:
        raise ValueError(
            "strip_trace requires a 1-D Grid (one feature); got "
            f"{len(grid.axis_values)} axes over feature_names={grid.feature_names!r}. "
            "N-D density fields are out of scope (see TASK_D2 stub)."
        )
    if density_grid.grid_id != grid.grid_id:
        raise ValueError(
            f"density_grid.grid_id {density_grid.grid_id!r} does not match "
            f"grid.grid_id {grid.grid_id!r} — a DensityGrid must be evaluated "
            "on the Grid it is being plotted against."
        )

    x = np.asarray(grid.axis_values[0], dtype=float)
    y = np.asarray(density_grid.density, dtype=float).reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"density_grid.density has {y.shape[0]} cells but grid axis has "
            f"{x.shape[0]} — shape mismatch between grid and density_grid."
        )

    resolved_style = style if style is not None else _default_style_for_label(label)
    show_legend = label is not None
    return TraceData(x=x, y=y, style=resolved_style, label=label, show_legend=show_legend)


# --------------------------------------------------------------------------- #
# Overlay: several (grid, density_grid, label) triples on the SAME 1-D grid_id
# --------------------------------------------------------------------------- #
def overlay_strip_subplot(
    grid: Grid,
    density_grids: Sequence[DensityGrid],
    *,
    labels: Sequence[str | None] | None = None,
    styles: Sequence[TraceStyle | None] | None = None,
    key: tuple = (None, None),
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> SubplotData:
    """Overlay several ``DensityGrid``s on the SAME 1-D grid into one ``SubplotData``.

    All ``density_grids`` MUST share ``grid.grid_id`` — this is the "directly
    comparable, same grid" guarantee from ontology §1b. A mismatched
    ``grid_id`` raises rather than silently misaligning curves.

    ``labels``/``styles`` are positional, parallel to ``density_grids`` (same
    length if provided). A label of ``"unassigned"`` always resolves to the
    reserved muted style regardless of the ``styles`` override, unless an
    explicit style is passed for that position.
    """
    n = len(density_grids)
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != density_grids length {n}")
    if styles is not None and len(styles) != n:
        raise ValueError(f"styles length {len(styles)} != density_grids length {n}")

    for i, dg in enumerate(density_grids):
        if dg.grid_id != grid.grid_id:
            raise ValueError(
                f"density_grids[{i}].grid_id {dg.grid_id!r} does not match "
                f"grid.grid_id {grid.grid_id!r} — overlay requires ALL "
                "SampleSets to be evaluated on the SAME 1-D grid (ontology "
                "§1b shared-grid comparability). Re-evaluate on a shared "
                "grid rather than mixing grid_ids."
            )

    traces = []
    for i, dg in enumerate(density_grids):
        label = labels[i] if labels is not None else None
        style = styles[i] if styles is not None else None
        traces.append(strip_trace(grid, dg, style=style, label=label))

    return SubplotData(
        traces=traces,
        key=key,
        title=title,
        x_label=x_label if x_label is not None else (grid.feature_names[0] if grid.feature_names else None),
        y_label=y_label if y_label is not None else "density",
    )


# --------------------------------------------------------------------------- #
# 2. N features x M comparisons -> a FigureData grid (one row per label_group view)
# --------------------------------------------------------------------------- #
def strip_grid_figure(
    rows: Sequence[Mapping[str, object]],
    *,
    title: str = "",
) -> FigureData:
    """Build an N (features/rows) x M (comparisons/cols already inside each row)
    ``FigureData`` from per-row strip specs.

    ``rows`` is an ordered sequence of one dict per row (one row per
    ``label_group`` view / one feature's 1-D grid), each dict with keys:
      - ``"grid"``: the 1-D ``Grid`` for that row's feature.
      - ``"density_grids"``: sequence of ``DensityGrid`` (the M comparisons /
        SampleSets for that row), all sharing ``grid.grid_id``.
      - ``"labels"`` (optional): parallel labels for each density grid.
      - ``"styles"`` (optional): parallel ``TraceStyle`` overrides.
      - ``"title"`` (optional): row/subplot title (e.g. the feature name).

    Each row becomes exactly ONE ``SubplotData`` (one panel with M overlaid
    traces) — this matches "N features (rows) x M SampleSets (traces)" from
    the brief: SampleSets overlay as traces WITHIN a panel (they share a
    grid_id and are directly comparable), features are separate panels (they
    do not share a grid_id in general). ``len(subplots) == len(rows)`` i.e.
    N x 1 conceptually, with M folded into each panel's traces; callers that
    want a strict N x M grid of *separate* panels can instead call
    :func:`overlay_strip_subplot` once per (feature, comparison) cell and pass
    the resulting flat list here via ``rows`` with one density_grid apiece —
    the emitter does not force a layout, ``FacetSpec`` (a render argument, not
    a FigureData field) decides how the panel list is arranged on screen.

    Pure IR: no rendering happens here. Callers pass a ``FacetSpec`` to
    ``render()`` themselves (e.g. a wider aspect ratio for strips).
    """
    subplots = []
    for i, row in enumerate(rows):
        grid = row["grid"]
        density_grids = row["density_grids"]
        labels = row.get("labels")
        styles = row.get("styles")
        row_title = row.get("title")
        subplots.append(
            overlay_strip_subplot(
                grid,
                density_grids,
                labels=labels,
                styles=styles,
                key=(i, 0),
                title=row_title,
            )
        )
    return FigureData(title=title, subplots=subplots)


def label_group_scatter_subplot(
    distribution: Distribution,
    label_group: LabelGroup,
    feature: str,
) -> SubplotData:
    """Density-free sample view for provided or resolved label groups."""
    if distribution.label_groups.get(label_group.name) is not label_group:
        raise ValueError("LabelGroup must be attached to its Distribution")
    feature_index = distribution.feature_names.index(feature)
    positions = {sid: i for i, sid in enumerate(distribution.sample_ids)}
    traces: list[TraceData] = []
    for sample_set in distribution.sample_sets(label_group.name):
        rows = [positions[sid] for sid in sample_set.sample_ids]
        traces.append(
            TraceData(
                x=distribution.feature_values[rows, feature_index],
                y=np.zeros(len(rows)),
                label=sample_set.sample_set_name,
                show_legend=True,
                style=_default_style_for_label(sample_set.sample_set_name),
                render_as="scatter",
            )
        )
    return SubplotData(traces=traces, key=(label_group.name, feature), x_label=feature)


# =========================================================================== #
# HIGH-LEVEL 1-D DISTRIBUTION-GRID API
# ---------------------------------------------------------------------------
# The pipeline (each stage a pure function; only the middle one fits KDEs):
#
#   PATH A: (Distribution, LabelGroup) pairs -> DistributionGrid -> Figure
#   PATH B (next commit): DistributionComparisons -> the SAME DistributionGrid
#
# Ownership rule (hard-won): a marginal KDE is NOT a property of a Distribution.
# It is per-(SampleSet, feature, GRID), and the grid is a *comparison* decision
# (the cell's shared bounds), resolvable only AFTER curves are bucketed into
# cells. ``build_1d_density_grid`` does this bucketing + fitting internally
# (the ONLY place a marginal KDE is computed) and emits ONE renderer-neutral
# ``DistributionGrid``. ``plot_1d_density_grid`` READS these densities; it
# never calls ``evaluate_density`` itself.
# =========================================================================== #


class IncomparableDistributionsError(ValueError):
    """Raised when a cell would overlay curves from >1 ``distribution_id``.

    Structural one-label-group-per-cell invariant (spec §PATH A): a cell may
    only ever compare SampleSets drawn from ONE Distribution. This is
    automatically guaranteed when ``LabelGroupFacet`` is an axis (then every
    cell is single-label-group, hence single-distribution, by construction);
    the check only bites when the axes leave room for two distributions to
    land in the same cell.
    """


@dataclass(frozen=True)
class GroupStyle:
    """Orthogonal, matplotlib-like appearance flags for one curve.

    Line and area config are independent (``fill=False`` + a dashed
    ``line_style`` is the reference baseline). ``color=None`` defers to palette
    resolution at render time (role gradient / STANDARD_PALETTE).
    """

    line_style: str = "-"
    line_width: float = 1.8
    line_alpha: float = 1.0
    fill: bool = True
    fill_alpha: float = 0.25
    color: str | None = None


# Reference (baseline) role defaults: dashed, unfilled — communicates
# experimental role without spending a color dimension. Keyed by whatever
# ``style_group`` value a curve carries (PATH A: its sample_set_name).
DEFAULT_ROLE_STYLES: Mapping[Hashable, GroupStyle] = {
    "reference": GroupStyle(line_style="--", fill=False),
}
# Auto-palette BASE hue per role. Deliberate choice (not valley's blue target):
# in the 1-D grouped view the target carries a real hue and the reference
# degrades to gray — matching valley_visualization's gray WT convention
# (WT_DENS_COLOR/#808080). Shades within a role are lightness steps off the base.
DEFAULT_ROLE_PALETTES: Mapping[Hashable, str] = {
    "target": "#B2182B",     # crimson family (target)
    "reference": "#808080",  # gray family (reference / WT baseline)
}


@dataclass(frozen=True)
class MarginalDensityMethod:
    """How the 1-D marginals were estimated — provenance so the resolved IR is
    never silently dependent on an undocumented estimator setting.
    TODO(Density1DSpec): fold bandwidth/kernel/bounds knobs in here."""

    bandwidth_method: str = "silverman"
    grid_size: int = 200


# --------------------------------------------------------------------------- #
# Renderer-neutral IR (PATH A today; PATH B/TASK_D's ridge will consume the
# SAME shape — keep it self-describing and stable).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CurveKey:
    """Structured identity for a PATH B comparison curve (spec §"PATH B" /
    §"CurveKey"): the two axes stay SEPARATE — a comparison member (the
    ``across`` value, e.g. ``"wildtype"``) and a within-member SampleSet name
    (e.g. ``"peak_0"``). NEVER a concatenated ``"wildtype_peak_0"`` string, so a
    caller can facet/style on either axis. :meth:`display` renders the joined
    human label ``"wildtype · peak_0"`` only for legends/titles."""

    comparison_member: Hashable   # the across value, e.g. "wildtype"
    sample_set: str               # e.g. "peak_0"

    def display(self) -> str:
        return f"{self.comparison_member} · {self.sample_set}"


@dataclass(frozen=True)
class DistributionCurve:
    """One resolved curve in one cell — pure data, no styling decisions yet.

    ``style_group`` is the value styling/coloring keys off (PATH A: the
    SampleSet's own name; PATH B: the comparison member / ``across`` value, so
    a reference member degrades to the gray/dashed baseline) so the renderer's
    role-gradient logic does not need to know which build path produced the
    grid.

    ``curve_key`` is populated only by PATH B (the structured
    :class:`CurveKey`); PATH A leaves it ``None`` (its curve identity is fully
    carried by ``sample_set_name``). The renderer reads it, never invents it.
    """

    cell: tuple[Hashable, Hashable]   # (row_value, col_value)
    sample_set_name: str
    style_group: Hashable
    grid: Grid
    density: DensityGrid
    sample_count: int
    curve_key: "CurveKey | None" = None
    is_robust: bool | None = None


@dataclass(frozen=True)
class DistributionGrid:
    """The renderer-neutral figure IR: a flat bag of resolved curves + the two
    facet keys. Backend/appearance are applied later by the renderer.

    :func:`build_1d_density_grid` (PATH A) emits this shape today; PATH B
    (``build_1d_distribution_comparison``, next commit) emits the SAME shape.
    """

    feature_name: str
    row: FacetKey
    col: FacetKey
    curves: tuple[DistributionCurve, ...]


# --------------------------------------------------------------------------- #
# Existing-density helpers. Plotting never calculates or evaluates a KDE.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _CellMember:
    """One curve-to-be for a cell: the SampleSet + the resolved styling/identity
    tags the two build paths decide (PATH A keys ``style_group`` off the
    SampleSet name and leaves ``curve_key`` None; PATH B keys off the comparison
    member and attaches a :class:`CurveKey`). ``_fit_cell_marginals`` stays
    path-agnostic — it only fits densities on the shared grid."""

    distribution: Distribution
    label_group: LabelGroup
    sample_set: SampleSet
    style_group: Hashable
    curve_key: "CurveKey | None" = None


def marginalize_density_estimate(
    estimate: DensityEstimate,
    feature: str,
) -> DensityEstimate:
    """Analytically marginalize a retained raster density onto one feature.

    No samples or KDE implementation are consulted. Other raster axes are
    integrated using their actual grid coordinates, then the retained 1-D
    marginal is normalized to unit integral.
    """
    if feature not in estimate.feature_names:
        raise ValueError(
            f"density features {estimate.feature_names!r} do not contain {feature!r}"
        )
    selected_axis = estimate.feature_names.index(feature)
    if len(estimate.feature_names) == 1:
        return estimate

    marginal = np.asarray(estimate.density_grid.density, dtype=float)
    # Descending axes keep the selected axis index stable until axes below it
    # disappear; axis coordinates supply nonuniform grid spacing exactly.
    for axis_index in reversed(range(len(estimate.feature_names))):
        if axis_index == selected_axis:
            continue
        marginal = np.trapz(
            marginal,
            x=np.asarray(estimate.grid.axis_values[axis_index], dtype=float),
            axis=axis_index,
        )
    selected_values = np.asarray(estimate.grid.axis_values[selected_axis], dtype=float)
    total = float(np.trapz(marginal, x=selected_values))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("retained density has no finite positive marginal mass")
    marginal = np.asarray(marginal / total, dtype=float)

    construction_method = "analytical_raster_marginal"
    construction_params = {
        "source_grid_id": estimate.grid.grid_id,
        "selected_feature": feature,
    }
    grid_id = make_grid_id(
        (feature,), construction_method, construction_params,
        (selected_values,), estimate.grid.fit_sample_ids,
    )
    grid = Grid(
        grid_id=grid_id,
        feature_names=(feature,),
        axis_values=(selected_values,),
        construction_method=construction_method,
        construction_params=construction_params,
        fit_sample_ids=estimate.grid.fit_sample_ids,
    )
    return DensityEstimate(
        distribution_id=estimate.distribution_id,
        feature_names=(feature,),
        spec=estimate.spec,
        grid=grid,
        density_grid=DensityGrid(grid_id, (feature,), marginal),
    )


def _fit_cell_marginals(
    *,
    cell_key: tuple[Hashable, ...],
    members: Sequence[_CellMember],
    feature: str,
    grid_size: int = 0,
) -> list[DistributionCurve]:
    """Build curves only from already-calculated effective densities."""
    curves: list[DistributionCurve] = []
    for member in members:
        estimate = member.label_group.density or member.distribution.shared_density
        if estimate is None:
            raise ValueError(
                f"label group {member.label_group.name!r} has no effective density; "
                "calculate and supply/register density before density plotting"
            )
        marginal = marginalize_density_estimate(estimate, feature)
        curves.append(
            DistributionCurve(
                cell=cell_key,
                sample_set_name=member.sample_set.sample_set_name,
                style_group=member.style_group,
                grid=marginal.grid,
                density=marginal.density_grid,
                sample_count=len(member.sample_set.sample_ids),
                curve_key=member.curve_key,
                is_robust=member.label_group.is_robust,
            )
        )
    return curves


# --------------------------------------------------------------------------- #
# PATH A — within-population attached label groups
# --------------------------------------------------------------------------- #
def build_1d_density_grid(
    groups: Sequence[tuple[Distribution, LabelGroup]],
    feature: str,
    *,
    facet_row: FacetKey = LabelGroupFacet(),
    facet_col: FacetKey = CoordinateFacet("time_bin"),
) -> DistributionGrid:
    """PATH A (spec §"PATH A"): one distribution per cell; curves = that label
    group's SampleSets. The label group is chosen by the INPUT object — no
    ``group_by``/``overlay_across`` parameter exists here.

    Cell grid = union bounds of the CURVES SELECTED for that cell (NOT
    automatically all of the distribution's samples — an omitted/unassigned
    category does not stretch the grid, since ``Distribution.sample_sets``
    already excludes ``UNASSIGNED_LABEL`` samples).

    One-label-group-per-cell invariant (structural): raises
    :class:`IncomparableDistributionsError` if a cell would overlay SampleSets
    from more than one ``distribution_id``. Skipped when ``LabelGroupFacet`` is
    itself an axis (then every cell is single-label-group by construction).
    """
    for distribution, group in groups:
        if group.name not in distribution.label_groups or distribution.label_groups[group.name] is not group:
            raise ValueError("each LabelGroup must be attached to its paired Distribution")
        if feature not in distribution.feature_names:
            raise ValueError(
                f"distribution {distribution.distribution_id!r} lacks requested "
                f"feature {feature!r} (has {distribution.feature_names!r})"
            )

    label_group_is_axis = isinstance(facet_row, LabelGroupFacet) or isinstance(
        facet_col, LabelGroupFacet
    )

    # Bucket groups into cells first (facet resolution only — no fitting yet).
    def facet_value(distribution: Distribution, group: LabelGroup, key: FacetKey):
        if isinstance(key, LabelGroupFacet):
            return group.name
        if isinstance(key, CoordinateFacet):
            return distribution.coordinate(key.name)
        raise TypeError(f"unsupported FacetKey: {key!r}")

    cells: dict[tuple[Hashable, Hashable], list[tuple[Distribution, LabelGroup]]] = {}
    for distribution, group in groups:
        key = (
            facet_value(distribution, group, facet_row),
            facet_value(distribution, group, facet_col),
        )
        cells.setdefault(key, []).append((distribution, group))

    all_curves: list[DistributionCurve] = []
    for cell_key, members in cells.items():
        if not label_group_is_axis:
            distribution_ids = {distribution.distribution_id for distribution, _ in members}
            if len(distribution_ids) > 1:
                raise IncomparableDistributionsError(
                    f"cell {cell_key} would overlay SampleSets from >1 distribution "
                    f"({sorted(distribution_ids)}); a cell may only compare ONE "
                    "distribution's label group (facet on LabelGroupFacet, or "
                    "restrict facet_row/facet_col so cells stay single-distribution)."
                )
        flat_members: list[_CellMember] = []
        for distribution, group in members:
            for sset in distribution.sample_sets(group.name):
                # PATH A: style_group IS the SampleSet name; no CurveKey.
                flat_members.append(
                    _CellMember(
                        distribution=distribution,
                        label_group=group,
                        sample_set=sset,
                        style_group=sset.sample_set_name,
                    )
                )
        curves = _fit_cell_marginals(
            cell_key=cell_key, members=flat_members, feature=feature, grid_size=200,
        )
        all_curves.extend(curves)

    return DistributionGrid(
        feature_name=feature,
        row=facet_row,
        col=facet_col,
        curves=tuple(all_curves),
    )


# --------------------------------------------------------------------------- #
# PATH B — cross-population: build_1d_distribution_comparison(DistributionComparisons)
# --------------------------------------------------------------------------- #
def _resolve_comparison_col(comparison: DistributionComparison, facet_col: FacetKey) -> Hashable:
    """Resolve the column facet value for ONE comparison from its held-constant
    ``coordinates`` (spec §PATH B: "one cell per DistributionComparison, keyed
    by its coordinates"). Only a :class:`CoordinateFacet` is meaningful here —
    a comparison is a set of MEMBERS varying over the ``across`` axis, so it has
    no single label-group display axis; ``LabelGroupFacet`` is rejected with a
    precise error rather than silently mis-keying every cell together."""
    if isinstance(facet_col, LabelGroupFacet):
        raise TypeError(
            "build_1d_distribution_comparison faceting a comparison by "
            "LabelGroupFacet is undefined — a comparison spans multiple members "
            "(the label group is shared across all of them via label_group=). "
            "Use CoordinateFacet on a held-constant coordinate (e.g. time_bin)."
        )
    if not isinstance(facet_col, CoordinateFacet):
        raise TypeError(f"unsupported FacetKey for facet_col: {facet_col!r}")
    if facet_col.name not in comparison.coordinates:
        raise KeyError(
            f"comparison lacks coordinate {facet_col.name!r} to facet on; "
            f"held-constant coordinates are {dict(comparison.coordinates)!r}"
        )
    return comparison.coordinates[facet_col.name]


def build_1d_distribution_comparison(
    comparisons: DistributionComparisons,
    feature: str,
    *,
    label_group: str,
    facet_col: FacetKey = CoordinateFacet("time_bin"),
    reference_value: Hashable | None = None,
) -> DistributionGrid:
    """PATH B (spec §"PATH B" / §"CurveKey"): render TASK_A's matched
    ``DistributionComparisons`` as a 1-D density grid — one cell per
    ``DistributionComparison`` (keyed by its held-constant coordinates through
    ``facet_col``), curves = every member's SampleSets of the ONE shared
    ``label_group`` (like-with-like across populations).

    Shared-grid rule (spec §"Shared grid"): the cell's grid is the union bounds
    of EVERY selected curve across ALL members in the cell (wildtype's peaks +
    b9d2's peaks). The KDE is materialized per cell on that shared grid inside
    :func:`_fit_cell_marginals` — never fit on a ``Distribution``, and every
    curve in a cell carries the SAME ``grid_id`` (raster-comparability).

    Curve identity is STRUCTURED (spec §"CurveKey"): each curve carries a
    :class:`CurveKey(comparison_member, sample_set)` — the two axes stay
    separate, never a concatenated string. ``style_group`` is the comparison
    member (the ``across`` value) so per-member styling works: pass
    ``reference_value`` to route ONE member to the gray/dashed reference role
    (else members color as peers). The asymmetric per-member ``label_groups=``
    mapping is DEFERRED — a single shared ``label_group=`` only.

    Feeds the SAME :func:`plot_1d_density_grid` renderer (same
    ``DistributionGrid`` IR) as PATH A.
    """
    row_value = comparisons.across  # single constant row: the comparison axis name

    all_curves: list[DistributionCurve] = []
    for comparison in comparisons.comparisons:
        col_value = _resolve_comparison_col(comparison, facet_col)
        cell_key = (row_value, col_value)

        flat_members: list[_CellMember] = []
        # ``members`` is ORDERED by ``values`` (DistributionComparisons
        # invariant) — iterate it so curve order is the requested member order.
        for member_value, distribution in comparison.members.items():
            if label_group not in distribution.label_groups:
                raise ValueError(
                    f"member {member_value!r} (distribution "
                    f"{distribution.distribution_id!r}) lacks the shared label "
                    f"group {label_group!r} — every member of a comparison must "
                    "carry the SAME label group for a like-with-like overlay."
                )
            if feature not in distribution.feature_names:
                raise ValueError(
                    f"member {member_value!r} lacks requested feature {feature!r} "
                    f"(has {distribution.feature_names!r})"
                )
            style_group = "reference" if member_value == reference_value else member_value
            group = distribution.label_groups[label_group]
            for sset in distribution.sample_sets(label_group):
                flat_members.append(
                    _CellMember(
                        distribution=distribution,
                        label_group=group,
                        sample_set=sset,
                        style_group=style_group,
                        curve_key=CurveKey(
                            comparison_member=member_value,
                            sample_set=sset.sample_set_name,
                        ),
                    )
                )

        # Shared grid over ALL members' curves in this cell (union bounds).
        curves = _fit_cell_marginals(
            cell_key=cell_key, members=flat_members, feature=feature, grid_size=200,
        )
        all_curves.extend(curves)

    return DistributionGrid(
        feature_name=feature,
        row=CoordinateFacet(row_value),
        col=facet_col,
        curves=tuple(all_curves),
    )


# --------------------------------------------------------------------------- #
# Stage 3: appearance + output. Reads curves; never fits.
# --------------------------------------------------------------------------- #
def _lighten(hex_color: str, amount: float) -> str:
    """Lightness step toward white; ``amount`` in [0,1] (0 = base, 1 = white)."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(hex_color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return mcolors.to_hex((r, g, b))


def _resolve_colors(
    grid: DistributionGrid,
    role_palettes: Mapping[Hashable, str],
    color_lookup: Mapping[tuple[Hashable, str], str] | None,
    has_reference: bool,
    reference_role: Hashable,
) -> dict[tuple[Hashable, str], str]:
    """Assign a color to every (style_group, sample_set_name) in the figure.

    ``style_group`` generalizes the old ``role`` field (PATH A: a curve's own
    SampleSet name). Explicit ``color_lookup`` wins. Otherwise, if a reference
    style_group is present, it (and any other style_group) gets a lightness
    gradient off its base hue (target=crimson, reference=gray). With NO
    reference, fall back to the project STANDARD_PALETTE, one color per
    distinct sample_set_name.
    """
    color_lookup = dict(color_lookup or {})
    resolved: dict[tuple[Hashable, str], str] = {}

    if not has_reference:
        # Peer mode: STANDARD_PALETTE, one color per distinct curve identity.
        # PATH A curves collapse to sample_set_name (style_group == name); PATH B
        # curves keep member + peak distinct (wildtype·peak_0 != b9d2·peak_0) via
        # the composite key, so a comparison overlay never draws two members in
        # the same color.
        keys = sorted({(c.style_group, c.sample_set_name) for c in grid.curves}, key=str)
        palette = {k: STANDARD_PALETTE[i % len(STANDARD_PALETTE)] for i, k in enumerate(keys)}
        for c in grid.curves:
            key = (c.style_group, c.sample_set_name)
            resolved[key] = color_lookup.get(key, palette[key])
        return resolved

    # Role mode: gradient off each style_group's base hue, one shade per
    # distinct sample_set_name within that group (stable, sorted).
    per_group_labels: dict[Hashable, list[str]] = {}
    for c in grid.curves:
        per_group_labels.setdefault(c.style_group, [])
        if c.sample_set_name not in per_group_labels[c.style_group]:
            per_group_labels[c.style_group].append(c.sample_set_name)
    for group in per_group_labels:
        per_group_labels[group].sort()

    non_reference_groups = sorted(
        (g for g in per_group_labels if g != reference_role), key=str
    )
    peer_palette = {
        g: STANDARD_PALETTE[i % len(STANDARD_PALETTE)]
        for i, g in enumerate(non_reference_groups)
    }

    for c in grid.curves:
        key = (c.style_group, c.sample_set_name)
        if key in color_lookup:
            resolved[key] = color_lookup[key]
            continue
        base = role_palettes.get(
            c.style_group,
            peer_palette.get(c.style_group, "#666666"),
        )
        labels = per_group_labels[c.style_group]
        i = labels.index(c.sample_set_name)
        # spread shades across [0, 0.55] lightness so even 1 label stays saturated
        amount = 0.0 if len(labels) == 1 else 0.55 * i / (len(labels) - 1)
        resolved[key] = _lighten(base, amount)
    return resolved


def plot_1d_density_grid(
    grid: DistributionGrid,
    *,
    role_styles: Mapping[Hashable, GroupStyle] = DEFAULT_ROLE_STYLES,
    role_palettes: Mapping[Hashable, str] = DEFAULT_ROLE_PALETTES,
    color_lookup: Mapping[tuple[Hashable, str], str] | None = None,
    reference_role: Hashable | None = "reference",
    title: str = "",
    output_path: str | Path | None = None,
    style=None,
):
    """Plot a DistributionGrid as a faceted 1-D DENSITY-STRIP grid.

    Named for the mark family it draws (density strips), not "all 1-D distribution
    views": ECDFs, histograms, and violins would be sibling verbs, and a genuine
    stacked RIDGELINE renderer would be its own verb consuming the same
    ``DistributionGrid``. Here: one subplot per facet cell, densities overlaid
    within each cell. (The 2-D density-field grid is a separate future verb,
    ``plot_2d_density_grid``; valley_visualization stays a custom Tier-3 report.)

    Appearance only — reads precomputed densities, never fits. ``reference_role``
    selects which style_group gets ``role_styles['reference']`` (dashed/unfilled)
    and the gray gradient; pass ``None`` to treat every curve as a peer
    (STANDARD_PALETTE, no dashes).
    """
    has_reference = reference_role is not None and any(
        c.style_group == reference_role for c in grid.curves
    )
    colors = _resolve_colors(grid, role_palettes, color_lookup, has_reference, reference_role)

    # Order cells: rows then cols, preserving first-seen order.
    row_vals: list[Hashable] = []
    col_vals: list[Hashable] = []
    for c in grid.curves:
        r, cc = c.cell
        if r not in row_vals:
            row_vals.append(r)
        if cc not in col_vals:
            col_vals.append(cc)

    default_style = GroupStyle()
    subplots: list[SubplotData] = []
    for r in row_vals:
        for cc in col_vals:
            cell_curves = [c for c in grid.curves if c.cell == (r, cc)]
            if not cell_curves:
                continue
            cell_grid = cell_curves[0].grid
            densities = []
            labels = []
            styles = []
            for c in cell_curves:
                is_ref = has_reference and c.style_group == reference_role
                gs = role_styles.get(c.style_group, default_style) if is_ref else default_style
                color = colors[(c.style_group, c.sample_set_name)]
                densities.append(c.density)
                # style_group is already encoded by color (gray for
                # reference) + dash, so keep the label short: sample-set
                # name + n. Prefix reference sets so a shared name stays
                # unambiguous.
                # PATH B carries a structured CurveKey -> render its joined
                # "member · sample_set" display so a shared peak name across
                # members stays unambiguous. PATH A (no CurveKey) keeps the
                # short sample-set label, prefixing reference sets.
                if c.curve_key is not None:
                    base_label = c.curve_key.display()
                else:
                    prefix = f"{c.style_group} " if is_ref else ""
                    base_label = f"{prefix}{c.sample_set_name}"
                labels.append(f"{base_label} (n={c.sample_count})")
                styles.append(
                    TraceStyle(
                        color=color,
                        alpha=gs.line_alpha if c.is_robust is not False else min(gs.line_alpha, 0.65),
                        width=gs.line_width,
                        linestyle=gs.line_style if c.is_robust is not False else ":",
                    )
                )
            subplots.append(
                overlay_strip_subplot(
                    cell_grid,
                    densities,
                    labels=labels,
                    styles=styles,
                    key=(r, cc),
                    title=str(cc) if r == row_vals[0] else None,
                )
            )

    fig = FigureData(
        title=title or f"distribution grid — {grid.feature_name}",
        subplots=subplots,
        row_labels=[str(r) for r in row_vals] if len(row_vals) > 1 else None,
        col_labels=[str(cc) for cc in col_vals] if len(col_vals) > 1 else None,
    )
    if style is None:
        from analyze.viz.plotting.faceting_engine import default_style

        style = default_style()
        # Per-panel legend: each cell lists only the curves it actually draws,
        # instead of one giant figure-wide legend overflowing the axes.
        style.legend_loc = "per-panel"
        style.legend_fontsize = 6
    return render(
        fig,
        backend="matplotlib",
        facet=FacetSpec(sharex=False, sharey=False),
        style=style,
        output_path=output_path,
    )


def plot_distr_metric_over_time(
    metric_df,
    value: str,
    time: str,
    group_by: str | Sequence[str],
    style: str | None = "is_robust",
    *,
    title: str | None = None,
    y_label: str | None = None,
    output_path: str | Path | None = None,
    ax=None,
):
    """Plot a purpose-specific distribution metric table over time.

    The table is already an analysis result (for example catalog peak counts or
    relative comparison metrics). This function performs no aggregation,
    filtering, peak counting, or robustness inference. Robust rows use solid
    lines and filled markers; nonrobust rows remain visible with dashed lines
    and hollow markers. Callers that want filtering must filter ``metric_df``
    explicitly before calling.
    """
    import pandas as pd

    if not isinstance(metric_df, pd.DataFrame):
        raise TypeError("metric_df must be a pandas DataFrame")
    groups = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    if not groups:
        raise ValueError("group_by must name at least one column")
    required = {value, time, *groups}
    if style is not None:
        required.add(style)
    missing = sorted(required - set(metric_df.columns))
    if missing:
        raise ValueError(f"metric_df is missing required columns: {missing}")

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = ax.figure

    grouped = metric_df.groupby(list(groups), sort=False, dropna=False)
    for group_key, frame in grouped:
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        label = " · ".join(str(item) for item in key_tuple)
        ordered = frame.sort_values(time, kind="stable")
        robust_mask = (
            np.ones(len(ordered), dtype=bool)
            if style is None
            else ordered[style].fillna(False).astype(bool).to_numpy()
        )
        for robust, suffix, linestyle, marker_face in (
            (True, "", "-", "auto"),
            (False, " (nonrobust)", "--", "none"),
        ):
            selected = ordered.loc[robust_mask == robust]
            if selected.empty:
                continue
            line, = ax.plot(
                selected[time].to_numpy(),
                selected[value].to_numpy(),
                linestyle=linestyle,
                marker="o",
                label=label + suffix,
            )
            if marker_face == "none":
                line.set_markerfacecolor("none")
                line.set_markeredgecolor(line.get_color())

    ax.set_xlabel(time)
    ax.set_ylabel(y_label or value)
    if title:
        ax.set_title(title)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
