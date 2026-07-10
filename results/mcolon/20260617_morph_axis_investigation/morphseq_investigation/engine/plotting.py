"""Pure IR emitters — turn frozen ``Grid``/``DensityGrid``/``SampleSet`` objects
into faceting-engine IR (``TraceData`` / ``SubplotData`` / ``FigureData``).

Source of truth: ``docs/PRIMITIVE_ONTOLOGY.md`` §1b "KDE strips fall out for
free" + ``tasks/TASK_D_plotting.md``.

**This module does NOT draw.** It only builds pure dataclasses defined in the
real faceting engine (``src/analyze/viz/plotting/faceting_engine/ir.py``) and
hands them to that engine's own ``render()``. No new rendering backend, no
bespoke matplotlib/plotly calls live here.

The payoff from §1b: a KDE strip is nothing but a 1-D ``Grid`` + a
``DensityGrid`` evaluated on it — the SAME machinery as the 2-D peak grid,
dimension = 1. So ``strip_trace`` below is a thin adapter, not a new code path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
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

from .grid import build_grid, evaluate_density
from .objects import DensityGrid, Distribution, Grid, HDR, LabelGroup, SampleSet

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
#   scientific objects  ->  DistributionGrouping        (lightweight interpretation)
#                       ->  MaterializedDistributionGrouping   (shared-grid marginals)
#                       ->  DistributionGrid             (renderer-neutral IR)
#                       ->  Figure                       (render_distribution_grid)
#
# Ownership rule (hard-won): a marginal KDE is NOT a property of a Distribution.
# It is per-(SampleSet, feature, GRID), and the grid is a *comparison* decision
# (the cell's shared bounds), resolvable only AFTER groupings are bucketed into
# cells. So densities live on MaterializedDistributionGrouping (keyed by
# sample_set_id), never on Distribution/SampleSet — those stay pure durable atoms.
# The renderer READS these densities; it never calls evaluate_density.
# =========================================================================== #


class FacetCoordinate(str, Enum):
    """A coordinate a DistributionGrouping can be faceted on (row or column).

    Resolved off the grouping's own objects — no DataFrame, no re-derivation.
    Using an enum (not raw strings) makes typos a construction-time error.
    """

    LABEL_GROUP = "label_group"   # which labeling produced this grouping
    TIME_BIN = "time_bin"         # the distribution's time bin
    SCOPE = "scope"               # the distribution's scope_id
    ROLE = "role"                 # target / reference / ... (also the ref selector)


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
# experimental role without spending a color dimension.
DEFAULT_ROLE_STYLES: Mapping[str, GroupStyle] = {
    "reference": GroupStyle(line_style="--", fill=False),
}
# Auto-palette BASE hue per role. Deliberate choice (not valley's blue target):
# in the 1-D grouped view the target carries a real hue and the reference
# degrades to gray — matching valley_visualization's gray WT convention
# (WT_DENS_COLOR/#808080). Shades within a role are lightness steps off the base.
DEFAULT_ROLE_PALETTES: Mapping[str, str] = {
    "target": "#B2182B",     # crimson family (target)
    "reference": "#808080",  # gray family (reference / WT baseline)
}


@dataclass(frozen=True)
class DistributionGrouping:
    """One distribution interpreted through ONE grouping scheme — the plot atom.

    Carries its own facet coordinates (via ``distribution``) and its own labels
    (via ``label_group`` + ``sample_sets``). The plotter never re-labels/re-bins.
    Lightweight on purpose: no densities here (see MaterializedDistributionGrouping).
    """

    distribution: Distribution
    label_group: LabelGroup
    sample_sets: tuple[SampleSet, ...]

    def coordinate(self, coordinate: FacetCoordinate) -> Hashable:
        if coordinate is FacetCoordinate.LABEL_GROUP:
            return self.label_group.label_group_name
        if coordinate is FacetCoordinate.TIME_BIN:
            return self.distribution.time_bin
        if coordinate is FacetCoordinate.SCOPE:
            return self.distribution.scope_id
        if coordinate is FacetCoordinate.ROLE:
            return self.distribution.role
        raise ValueError(f"unknown facet coordinate {coordinate!r}")


@dataclass(frozen=True)
class MarginalDensityMethod:
    """How the 1-D marginals were estimated — provenance so the resolved IR is
    never silently dependent on an undocumented estimator setting.
    TODO(Density1DSpec): fold bandwidth/kernel/bounds knobs in here."""

    bandwidth_method: str = "silverman"
    grid_size: int = 200


@dataclass(frozen=True)
class MaterializedDistributionGrouping:
    """A grouping resolved to plot-ready 1-D marginals for ONE feature.

    Feature-specific by construction: every curve here shares ``feature_name``
    and ``grid`` (the cell's shared grid), so lookups are flat and incompatible
    objects fail immediately. ``marginal_densities`` maps sample_set_id -> the
    precomputed 1-D DensityGrid (already on ``grid``). This is the ONLY place a
    marginal KDE is stored, and it is produced once, upstream, by
    :func:`materialize_distribution_marginals`.

    ``sample_counts`` is captured here so legends/validation never reopen
    membership after numerical materialization. ``method`` records the estimator.
    """

    grouping: DistributionGrouping
    feature_name: str
    grid: Grid
    marginal_densities: Mapping[str, DensityGrid]  # sample_set_id -> density
    sample_counts: Mapping[str, int]               # sample_set_id -> n
    method: MarginalDensityMethod = field(default_factory=MarginalDensityMethod)


# --------------------------------------------------------------------------- #
# Renderer-neutral IR (between materialization and drawing)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributionCurve:
    """One resolved curve in one cell — pure data, no styling decisions yet."""

    cell: tuple[Hashable, Hashable]   # (row_value, col_value)
    role: str
    label_group_name: str
    sample_set_name: str
    grid: Grid
    density: DensityGrid
    sample_count: int


@dataclass(frozen=True)
class DistributionGrid:
    """The renderer-neutral figure IR: a flat bag of resolved curves + the two
    facet coordinates. Backend/appearance are applied later by the renderer."""

    feature_name: str
    row: FacetCoordinate
    col: FacetCoordinate
    curves: tuple[DistributionCurve, ...]


# --------------------------------------------------------------------------- #
# Stage 1 (the ONLY KDE-fitting stage): resolve shared cell grids + marginals.
# --------------------------------------------------------------------------- #
def _silverman_bandwidth(values: np.ndarray) -> float:
    n = max(len(values), 2)
    spread = float(np.std(values)) or 1.0
    return 1.06 * spread * n ** (-1.0 / 5.0)


def _cell_key(
    g: DistributionGrouping, row: FacetCoordinate, col: FacetCoordinate
) -> tuple[Hashable, Hashable]:
    return (g.coordinate(row), g.coordinate(col))


def _sample_set_feature_values(
    grouping: DistributionGrouping, sample_set: SampleSet, feature_idx: int
) -> np.ndarray:
    """Pull one SampleSet's members' values for ``feature_idx`` from the
    grouping's Distribution (join on sample_ids — the durable key)."""
    dist = grouping.distribution
    pos = {sid: i for i, sid in enumerate(dist.sample_ids)}
    idx = [pos[s] for s in sample_set.sample_ids if s in pos]
    if not idx:
        return np.empty((0,), dtype=float)
    return dist.feature_values[np.asarray(idx, dtype=int), feature_idx]


def materialize_distribution_marginals(
    groupings: Sequence[DistributionGrouping],
    feature_name: str,
    *,
    row: FacetCoordinate = FacetCoordinate.LABEL_GROUP,
    col: FacetCoordinate = FacetCoordinate.TIME_BIN,
    grid_size: int = 200,  # TODO(Density1DSpec): promote to a spec (bandwidth/kernel/bounds)
) -> tuple[MaterializedDistributionGrouping, ...]:
    """The NUMERICAL boundary: resolve shared comparison grids and evaluate
    per-SampleSet 1-D marginals for ``feature_name`` ONCE. The only place KDEs
    are fit.

    The shared grid is a *comparison* property (union bounds of everything in the
    cell), so it can only be built after bucketing into cells — which is why this
    step, not the grouping and not the plotter, owns density evaluation.
    """
    method = MarginalDensityMethod(bandwidth_method="silverman", grid_size=grid_size)
    # Every distribution must actually carry the requested feature (raise, not warn).
    for g in groupings:
        if feature_name not in g.distribution.feature_names:
            raise ValueError(
                f"distribution {g.distribution.distribution_id!r} lacks requested "
                f"feature {feature_name!r} (has {g.distribution.feature_names!r})"
            )

    # 1) bucket into cells and collect each cell's pooled feature values (bounds).
    cells: dict[tuple[Hashable, Hashable], list[DistributionGrouping]] = {}
    for g in groupings:
        cells.setdefault(_cell_key(g, row, col), []).append(g)

    materialized: list[MaterializedDistributionGrouping] = []
    for key, members in cells.items():
        # Shared grid = union of every member's feature values in this cell.
        pooled: list[np.ndarray] = []
        pooled_ids: list[str] = []
        for g in members:
            fi = g.distribution.feature_names.index(feature_name)
            col_vals = np.asarray(g.distribution.feature_values[:, fi], dtype=float)
            pooled.append(col_vals)
            pooled_ids.extend(str(s) for s in g.distribution.sample_ids)
        pooled_arr = np.concatenate(pooled) if pooled else np.zeros(1)
        lo, hi = float(pooled_arr.min()), float(pooled_arr.max())
        shared_grid = build_grid(
            feature_names=(feature_name,),
            pooled_values=pooled_arr.reshape(-1, 1),
            fit_sample_ids=pooled_ids or ["_"],
            method="fixed_bounds",
            params={"resolution": grid_size, "bounds": [(lo, hi)]},
        )

        # 2) per member, evaluate each SampleSet's marginal on the shared grid.
        for g in members:
            fi = g.distribution.feature_names.index(feature_name)
            densities: dict[str, DensityGrid] = {}
            counts: dict[str, int] = {}
            for sset in g.sample_sets:
                vals = _sample_set_feature_values(g, sset, fi)
                if vals.size == 0:
                    continue
                densities[sset.sample_set_id] = evaluate_density(
                    shared_grid, vals.reshape(-1, 1),
                    bandwidth_spec=_silverman_bandwidth(vals),
                )
                counts[sset.sample_set_id] = int(vals.size)
            materialized.append(
                MaterializedDistributionGrouping(
                    grouping=g,
                    feature_name=feature_name,
                    grid=shared_grid,
                    marginal_densities=densities,
                    sample_counts=counts,
                    method=method,
                )
            )
    return tuple(materialized)


# --------------------------------------------------------------------------- #
# Stage 2: validate + convert materialized marginals into renderer-neutral IR.
# --------------------------------------------------------------------------- #
def build_distribution_grid(
    materialized: Sequence[MaterializedDistributionGrouping],
    *,
    row: FacetCoordinate = FacetCoordinate.LABEL_GROUP,
    col: FacetCoordinate = FacetCoordinate.TIME_BIN,
) -> DistributionGrid:
    """Validate cell comparability and flatten to ``DistributionCurve``s.

    Comparability invariant: every grouping overlaid in one cell must be the
    same KIND of labeling. Today we check ``label_group_name`` equality; this is
    a display-name proxy — TODO(label-schema-id): replace with a stable schema
    identity once LabelGroup carries one (two unrelated groups can share a name).
    The check is skipped when LABEL_GROUP is itself an axis (then each cell is
    single-label-group by construction).
    """
    feature_names = {m.feature_name for m in materialized}
    if len(feature_names) > 1:
        raise ValueError(f"materialized groupings mix features: {feature_names!r}")

    by_cell: dict[tuple[Hashable, Hashable], list[MaterializedDistributionGrouping]] = {}
    for m in materialized:
        by_cell.setdefault(_cell_key(m.grouping, row, col), []).append(m)

    label_group_is_axis = FacetCoordinate.LABEL_GROUP in (row, col)

    curves: list[DistributionCurve] = []
    for key, members in by_cell.items():
        if not label_group_is_axis:
            names = {m.grouping.label_group.label_group_name for m in members}
            if len(names) > 1:
                raise ValueError(
                    f"cell {key} overlays non-comparable label groups {names!r}; "
                    "a cell may only overlay one kind of labeling "
                    "(TODO(label-schema-id): validate on a stable schema, not the name)."
                )
        # Structural grid identity: everything in a cell must share one grid_id
        # (comparability without relying on Python object identity).
        grid_ids = {m.grid.grid_id for m in members}
        if len(grid_ids) > 1:
            raise ValueError(
                f"cell {key} mixes grid_ids {grid_ids!r}; all overlaid curves "
                "must share one shared cell grid."
            )
        for m in members:
            role = m.grouping.distribution.role
            lg_name = m.grouping.label_group.label_group_name
            for sset in m.grouping.sample_sets:
                density = m.marginal_densities.get(sset.sample_set_id)
                if density is None:
                    continue
                curves.append(
                    DistributionCurve(
                        cell=key,
                        role=role,
                        label_group_name=lg_name,
                        sample_set_name=sset.sample_set_name,
                        grid=m.grid,
                        density=density,
                        sample_count=m.sample_counts.get(sset.sample_set_id, len(sset.sample_ids)),
                    )
                )

    m0 = materialized[0] if materialized else None
    return DistributionGrid(
        feature_name=m0.feature_name if m0 else "",
        row=row,
        col=col,
        curves=tuple(curves),
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
    role_palettes: Mapping[str, str],
    color_lookup: Mapping[tuple[str, str], str] | None,
    has_reference: bool,
) -> dict[tuple[str, str], str]:
    """Assign a color to every (role, sample_set_name) in the figure.

    Explicit ``color_lookup`` wins. Otherwise, if roles are distinguished
    (a reference exists), each role gets a lightness gradient off its base hue
    (target=crimson, reference=gray). With NO reference (all peers), fall back to
    the project STANDARD_PALETTE, one color per distinct sample_set_name.
    """
    color_lookup = dict(color_lookup or {})
    resolved: dict[tuple[str, str], str] = {}

    if not has_reference:
        # Peer mode: STANDARD_PALETTE by label, role ignored.
        labels = sorted({c.sample_set_name for c in grid.curves})
        palette = {lbl: STANDARD_PALETTE[i % len(STANDARD_PALETTE)] for i, lbl in enumerate(labels)}
        for c in grid.curves:
            key = (c.role, c.sample_set_name)
            resolved[key] = color_lookup.get(key, palette[c.sample_set_name])
        return resolved

    # Role mode: gradient off each role's base hue, one shade per distinct
    # sample_set_name within that role (stable, sorted).
    per_role_labels: dict[str, list[str]] = {}
    for c in grid.curves:
        per_role_labels.setdefault(c.role, [])
        if c.sample_set_name not in per_role_labels[c.role]:
            per_role_labels[c.role].append(c.sample_set_name)
    for role in per_role_labels:
        per_role_labels[role].sort()

    for c in grid.curves:
        key = (c.role, c.sample_set_name)
        if key in color_lookup:
            resolved[key] = color_lookup[key]
            continue
        base = role_palettes.get(c.role, "#666666")
        labels = per_role_labels[c.role]
        i = labels.index(c.sample_set_name)
        # spread shades across [0, 0.55] lightness so even 1 label stays saturated
        amount = 0.0 if len(labels) == 1 else 0.55 * i / (len(labels) - 1)
        resolved[key] = _lighten(base, amount)
    return resolved


def plot_1d_density_grid(
    grid: DistributionGrid,
    *,
    role_styles: Mapping[str, GroupStyle] = DEFAULT_ROLE_STYLES,
    role_palettes: Mapping[str, str] = DEFAULT_ROLE_PALETTES,
    color_lookup: Mapping[tuple[str, str], str] | None = None,
    reference_role: str | None = "reference",
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
    selects which role gets ``role_styles['reference']`` (dashed/unfilled) and the
    gray gradient; pass ``None`` to treat every curve as a peer (STANDARD_PALETTE,
    no dashes).
    """
    has_reference = reference_role is not None and any(
        c.role == reference_role for c in grid.curves
    )
    colors = _resolve_colors(grid, role_palettes, color_lookup, has_reference)

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
                is_ref = has_reference and c.role == reference_role
                gs = role_styles.get(c.role, default_style) if is_ref else default_style
                color = colors[(c.role, c.sample_set_name)]
                densities.append(c.density)
                # Role is already encoded by color (gray) + dash, so keep the
                # label short: sample-set name + n. Prefix reference sets so a
                # shared name (e.g. "peak_0" on both roles) stays unambiguous.
                prefix = f"{c.role} " if (has_reference and c.role == reference_role) else ""
                labels.append(f"{prefix}{c.sample_set_name} (n={c.sample_count})")
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
