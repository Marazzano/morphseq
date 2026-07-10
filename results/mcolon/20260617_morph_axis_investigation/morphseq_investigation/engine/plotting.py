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

from typing import Mapping, Sequence

import numpy as np

from analyze.viz.plotting.faceting_engine import (
    FigureData,
    SubplotData,
    TraceData,
    TraceStyle,
)
from analyze.viz.styling.genotype_colors import get_color_for_genotype

from .objects import DensityGrid, Grid, HDR, SampleSet

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
