"""TASK_D — the ridgeline verb: overlaid / stacked / mirror on the shared IR.

A genuine ridgeline is offset baselines within ONE coordinate system — its own
Tier-2 verb (``docs/VISUALIZATION_TAXONOMY.md``), NOT a ``layout=`` switch on the
density-strip verb. It consumes the SAME ``DistributionGrid`` IR that
``plot_1d_density_grid`` does (so BOTH of TASK_C's build paths — within-population
``build_1d_density_grid`` AND cross-population ``build_1d_distribution_comparison``
— feed it for free).

This RE-HOMES the working prototype ``v0/rich_distribution_plot.render_ridgeline``
(its offset math + target/reference styling was reviewed against real figures)
onto the grid IR. The three variants and their offset arithmetic are ported
verbatim in spirit:

  overlaid : target & reference share each bin's baseline (read COINCIDENCE).
  stacked  : reference on the bin baseline, target lifted just above (read SHAPE).
  mirror   : target up, reference mirrored down (read DIVERGENCE).

Preserved invariants:
  - Reads densities OFF the ``DistributionGrid`` — never fits a KDE (that is
    TASK_C's numerical stage, done inside ``build_1d_*``).
  - Reference role -> dashed line + dashed unfilled outline (ported fill_between
    edgecolor/linestyle logic).
  - The faceting IR has no offset-baseline band-stack, so — exactly as the
    prototype did — the ridge draws self-contained matplotlib. This is
    ``engine/`` code (not ``viz/phase0.py`` / ``viz/qc.py``), so the
    contract-wrapper ban does not apply here; it matches the prototype's
    surrounding style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Hashable, Mapping

import numpy as np

from .plotting import (
    DEFAULT_ROLE_PALETTES,
    DEFAULT_ROLE_STYLES,
    DistributionGrid,
    GroupStyle,
    _resolve_colors,
)

_VARIANTS = ("overlaid", "stacked", "mirror")

_VARIANT_BLURB = {
    "overlaid": "target & reference share each bin's baseline (read coincidence)",
    "stacked": "reference on baseline, target lifted just above (read each shape)",
    "mirror": "target up, reference mirrored down (read divergence)",
}


def _ordered_unique(values):
    """First-seen-order unique — same cell ordering discipline as the density
    grid renderer (rows/cols preserve first appearance in ``grid.curves``)."""
    seen: list = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def plot_1d_ridgeline(
    grid: DistributionGrid,
    *,
    variant: str = "overlaid",
    role_styles: Mapping[Hashable, GroupStyle] = DEFAULT_ROLE_STYLES,
    role_palettes: Mapping[Hashable, str] = DEFAULT_ROLE_PALETTES,
    color_lookup: Mapping[tuple[Hashable, str], str] | None = None,
    reference_role: Hashable | None = "reference",
    title: str = "",
    output_path: str | Path | None = None,
):
    """Render a ``DistributionGrid`` as a genuine offset-baseline ridgeline.

    Faceting maps onto the ridge exactly as it does on the density grid:
      - each distinct ``row`` value (``cell[0]``) becomes one axes column
        (a "view"): the coarsest facet, side-by-side subplots;
      - each distinct ``col`` value (``cell[1]``) is a TIME BIN that stacks as a
        vertically-offset row WITHIN a view, EARLIEST AT BOTTOM (``col`` values
        are drawn bottom-up in first-seen order, matching the prototype's
        ``reversed(design_hpfs)`` with earliest-first hpf input).

    ``variant`` controls placement of target vs reference within each bin
    (offset math ported from ``v0/rich_distribution_plot.render_ridgeline``):
      overlaid : both on the bin baseline (fills overlap).
      stacked  : reference on baseline, target on a sub-offset just above.
      mirror   : reference mirrored downward, target upward.

    Reference curves (``style_group == reference_role``) draw dashed + unfilled;
    every other curve draws solid + light-filled. Colors resolve through the SAME
    :func:`_resolve_colors` the density grid uses, so the two verbs are color-
    consistent off one IR.

    Densities are READ off ``grid`` — never re-fit.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown ridge variant {variant!r}; expected one of {_VARIANTS}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_reference = reference_role is not None and any(
        c.style_group == reference_role for c in grid.curves
    )
    colors = _resolve_colors(
        grid, role_palettes, color_lookup, has_reference, reference_role
    )
    default_style = GroupStyle()

    row_vals = _ordered_unique(c.cell[0] for c in grid.curves)
    col_vals = _ordered_unique(c.cell[1] for c in grid.curves)

    n_views = max(len(row_vals), 1)
    fig, axes = plt.subplots(
        1, n_views, figsize=(6.0 * n_views, 7.0), squeeze=False
    )
    axes = axes[0]

    for ax, row_val in zip(axes, row_vals):
        row_curves = [c for c in grid.curves if c.cell[0] == row_val]

        # Peak density across the whole view sets the inter-bin row spacing.
        peak = 0.0
        for c in row_curves:
            peak = max(peak, float(np.asarray(c.density.density).max()))
        peak = peak or 1.0
        if variant == "overlaid":
            step = peak * 0.7
        elif variant == "stacked":
            step = peak * 1.3      # room for ref (baseline) + target (sub-offset)
        else:  # mirror
            step = peak * 1.6      # room for +target above and -reference below
        sub = peak * 0.55          # within-bin sub-offset for the 'stacked' variant

        seen_labels: set[str] = set()
        # Earliest at bottom: col_vals is first-seen order (earliest first); the
        # bottom-most ridge (row_i=0) is the FIRST col value.
        for row_i, col_val in enumerate(col_vals):
            cell_curves = [c for c in row_curves if c.cell[1] == col_val]
            if not cell_curves:
                continue
            offset = row_i * step
            x = np.asarray(cell_curves[0].grid.axis_values[0], dtype=float)

            for c in cell_curves:
                is_ref = has_reference and c.style_group == reference_role
                gs = role_styles.get(c.style_group, default_style) if is_ref else default_style
                color = colors[(c.style_group, c.sample_set_name)]
                dens = np.asarray(c.density.density, dtype=float).reshape(-1)

                if c.curve_key is not None:
                    base_label = c.curve_key.display()
                else:
                    prefix = f"{c.style_group} " if is_ref else ""
                    base_label = f"{prefix}{c.sample_set_name}"
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
                    line_ls = gs.line_style if gs.line_style != "-" else "--"
                else:
                    ax.fill_between(
                        x, baseline, y, color=color, alpha=default_style.fill_alpha,
                        zorder=row_i,
                    )
                    line_ls = "-"
                ax.plot(
                    x, y, color=color, lw=1.8, ls=line_ls, zorder=row_i,
                    label=base_label if show else None,
                )

            ax.axhline(offset, color="#cccccc", lw=0.6, zorder=row_i - 0.5)
            ax.text(
                x.min(), offset, f"{col_val}  ",
                ha="right", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_title(str(row_val), fontsize=11, fontweight="bold")
        ax.set_xlabel(grid.feature_name)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.85)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        (title or f"distribution ridge ({variant}) — {grid.feature_name}")
        + f"\ncolumns = views · each ridge = one bin · earliest at bottom · "
        + _VARIANT_BLURB[variant],
        fontsize=13, fontweight="bold", linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
    return fig
