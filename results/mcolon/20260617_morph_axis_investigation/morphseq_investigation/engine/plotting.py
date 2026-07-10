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

from typing import Sequence

import numpy as np

from analyze.viz.plotting.faceting_engine import SubplotData, TraceData, TraceStyle
from analyze.viz.styling.genotype_colors import get_color_for_genotype

from .objects import DensityGrid, Grid

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
