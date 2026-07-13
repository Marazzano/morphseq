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

TASK_C reshapes the high-level 1-D grid API (below the Tier-1 primitives) onto
the TASK_0 objects (``Distribution`` / ``DistributionLabelGroup`` / typed
``FacetKey``): PATH A (``build_1d_density_grid``, within-population label
groups) lands in this commit; PATH B (cross-population comparisons) follows.
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
from .grid import build_grid, evaluate_density
from .objects import (
    DensityGrid,
    Distribution,
    DistributionLabelGroup,
    Grid,
    HDR,
    LabelGroup,
    SampleSet,
)

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


# --------------------------------------------------------------------------- #
# 3. Per-SampleSet HDR overlay (1-D): shade the HDR mask region under the strip
# --------------------------------------------------------------------------- #
def hdr_band_trace(
    grid: Grid,
    density_grid: DensityGrid,
    hdr: HDR,
    *,
    style: TraceStyle | None = None,
    label: str | None = None,
) -> TraceData:
    """Shade the 1-D HDR mask region as a filled band under the strip curve.

    Uses the faceting IR's native band support (``render_as='band'``,
    ``band_lower``/``band_upper``) rather than any bespoke fill call — no new
    rendering backend, the faceting engine's own renderers already know how to
    draw a band (matplotlib ``fill_between`` / plotly filled trace).

    ``band_lower`` is 0 everywhere; ``band_upper`` is ``density_grid.density``
    where ``hdr.mask`` is True and 0 elsewhere — i.e. the shaded region is
    exactly the HDR's support on this strip, sitting under the density curve.

    All three of ``grid``, ``density_grid``, and ``hdr`` must share one
    ``grid_id`` (self-describing comparability, ontology §2) — mismatched ids
    raise.
    """
    if len(grid.axis_values) != 1:
        raise ValueError(
            f"hdr_band_trace requires a 1-D Grid; got {len(grid.axis_values)} axes."
        )
    if density_grid.grid_id != grid.grid_id:
        raise ValueError(
            f"density_grid.grid_id {density_grid.grid_id!r} != grid.grid_id {grid.grid_id!r}"
        )
    if hdr.grid_id != grid.grid_id:
        raise ValueError(f"hdr.grid_id {hdr.grid_id!r} != grid.grid_id {grid.grid_id!r}")

    x = np.asarray(grid.axis_values[0], dtype=float)
    density = np.asarray(density_grid.density, dtype=float).reshape(-1)
    mask = np.asarray(hdr.mask).reshape(-1)
    if mask.shape[0] != x.shape[0]:
        raise ValueError(
            f"hdr.mask has {mask.shape[0]} cells but grid axis has {x.shape[0]}"
        )

    band_lower = np.zeros_like(density)
    band_upper = np.where(mask, density, 0.0)

    resolved_style = style if style is not None else _default_style_for_label(label)
    resolved_style = TraceStyle(
        color=resolved_style.color,
        alpha=min(resolved_style.alpha, 0.35),
        width=resolved_style.width,
        linestyle=resolved_style.linestyle,
        zorder=resolved_style.zorder,
    )
    return TraceData(
        x=x,
        y=band_upper,
        style=resolved_style,
        label=label,
        show_legend=False,
        band_lower=band_lower,
        band_upper=band_upper,
        render_as="band",
    )


def sample_set_strip_with_hdr(
    grid: Grid,
    sample_set: SampleSet,
    density_grid: DensityGrid,
    *,
    style: TraceStyle | None = None,
    label: str | None = None,
) -> list[TraceData]:
    """Convenience: strip curve + (if present) its HDR band for one ``SampleSet``.

    Returns a list of 1 or 2 ``TraceData`` (curve, or curve + band) ready to
    drop into a ``SubplotData.traces`` list alongside other SampleSets'
    traces. ``sample_set.hdr`` is optional (ontology §2 — not every SampleSet
    carries one); when ``None`` only the curve is returned.
    """
    resolved_label = label if label is not None else sample_set.sample_set_name
    traces = [strip_trace(grid, density_grid, style=style, label=resolved_label)]
    if sample_set.hdr is not None:
        traces.append(
            hdr_band_trace(grid, density_grid, sample_set.hdr, style=style, label=resolved_label)
        )
    return traces


# =========================================================================== #
# HIGH-LEVEL 1-D DISTRIBUTION-GRID API
# ---------------------------------------------------------------------------
# The pipeline (each stage a pure function; only the middle one fits KDEs):
#
#   PATH A: DistributionLabelGroup(s)  ->  DistributionGrid  ->  Figure
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
# Shared numerical helpers (the ONLY place KDEs are fit).
# --------------------------------------------------------------------------- #
def _silverman_bandwidth(values: np.ndarray) -> float:
    n = max(len(values), 2)
    spread = float(np.std(values)) or 1.0
    return 1.06 * spread * n ** (-1.0 / 5.0)


def _sample_set_feature_values(
    distribution: Distribution, sample_set: SampleSet, feature_idx: int
) -> np.ndarray:
    """Pull one SampleSet's members' values for ``feature_idx`` from its
    owning Distribution (join on sample_ids — the durable key)."""
    pos = {sid: i for i, sid in enumerate(distribution.sample_ids)}
    idx = [pos[s] for s in sample_set.sample_ids if s in pos]
    if not idx:
        return np.empty((0,), dtype=float)
    return distribution.feature_values[np.asarray(idx, dtype=int), feature_idx]


@dataclass(frozen=True)
class _CellMember:
    """One curve-to-be for a cell: the SampleSet + the resolved styling/identity
    tags the two build paths decide (PATH A keys ``style_group`` off the
    SampleSet name and leaves ``curve_key`` None; PATH B keys off the comparison
    member and attaches a :class:`CurveKey`). ``_fit_cell_marginals`` stays
    path-agnostic — it only fits densities on the shared grid."""

    distribution: Distribution
    sample_set: SampleSet
    style_group: Hashable
    curve_key: "CurveKey | None" = None


def _fit_cell_marginals(
    *,
    cell_key: tuple[Hashable, ...],
    members: Sequence[_CellMember],
    feature: str,
    grid_size: int,
) -> list[DistributionCurve]:
    """Shared-grid-then-fit for ONE cell: union bounds of the SELECTED curves
    only (spec §PATH A/B: an omitted/unassigned category must not stretch the
    grid; PATH B's shared grid spans EVERY selected curve across ALL members).

    ``members`` is a flat list of :class:`_CellMember` already filtered to
    exactly what this cell will draw. The KDE is materialized here — the ONLY
    place a marginal density is fit — per facet cell on the shared grid, never
    on a ``Distribution``. Order matches ``members`` minus any empty SampleSets.
    """
    pooled: list[np.ndarray] = []
    per_member_values: list[np.ndarray] = []
    for member in members:
        fi = member.distribution.feature_names.index(feature)
        vals = _sample_set_feature_values(member.distribution, member.sample_set, fi)
        per_member_values.append(vals)
        if vals.size:
            pooled.append(vals)

    if not pooled:
        return []

    pooled_arr = np.concatenate(pooled)
    lo, hi = float(pooled_arr.min()), float(pooled_arr.max())
    shared_grid = build_grid(
        feature_names=(feature,),
        pooled_values=pooled_arr.reshape(-1, 1),
        fit_sample_ids=[f"cell_{cell_key}_{i}" for i in range(pooled_arr.size)],
        method="fixed_bounds",
        params={"resolution": grid_size, "bounds": [(lo, hi)]},
    )

    curves: list[DistributionCurve] = []
    for member, vals in zip(members, per_member_values):
        if vals.size == 0:
            continue
        density = evaluate_density(
            shared_grid, vals.reshape(-1, 1),
            bandwidth_spec=_silverman_bandwidth(vals),
        )
        curves.append(
            DistributionCurve(
                cell=cell_key,
                sample_set_name=member.sample_set.sample_set_name,
                style_group=member.style_group,
                grid=shared_grid,
                density=density,
                sample_count=int(vals.size),
                curve_key=member.curve_key,
            )
        )
    return curves


# --------------------------------------------------------------------------- #
# PATH A — within-population: build_1d_density_grid(DistributionLabelGroup*)
# --------------------------------------------------------------------------- #
def build_1d_density_grid(
    groups: Sequence[DistributionLabelGroup],
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
    for g in groups:
        if feature not in g.distribution.feature_names:
            raise ValueError(
                f"distribution {g.distribution.distribution_id!r} lacks requested "
                f"feature {feature!r} (has {g.distribution.feature_names!r})"
            )

    label_group_is_axis = isinstance(facet_row, LabelGroupFacet) or isinstance(
        facet_col, LabelGroupFacet
    )

    # Bucket groups into cells first (facet resolution only — no fitting yet).
    cells: dict[tuple[Hashable, Hashable], list[DistributionLabelGroup]] = {}
    for g in groups:
        key = (g.coordinate(facet_row), g.coordinate(facet_col))
        cells.setdefault(key, []).append(g)

    all_curves: list[DistributionCurve] = []
    for cell_key, members in cells.items():
        if not label_group_is_axis:
            distribution_ids = {m.distribution.distribution_id for m in members}
            if len(distribution_ids) > 1:
                raise IncomparableDistributionsError(
                    f"cell {cell_key} would overlay SampleSets from >1 distribution "
                    f"({sorted(distribution_ids)}); a cell may only compare ONE "
                    "distribution's label group (facet on LabelGroupFacet, or "
                    "restrict facet_row/facet_col so cells stay single-distribution)."
                )
        flat_members: list[_CellMember] = []
        for g in members:
            for sset in g.sample_sets():
                # PATH A: style_group IS the SampleSet name; no CurveKey.
                flat_members.append(
                    _CellMember(
                        distribution=g.distribution,
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
            if label_group not in distribution.labels:
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
            for sset in distribution.sample_sets(label_group):
                flat_members.append(
                    _CellMember(
                        distribution=distribution,
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
                        alpha=gs.line_alpha,
                        width=gs.line_width,
                        linestyle=gs.line_style,
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
