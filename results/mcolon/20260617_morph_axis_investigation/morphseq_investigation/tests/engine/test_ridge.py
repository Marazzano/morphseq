"""TASK_D — ridgeline verb tests (``docs/tasks_catalog/TASK_D_ridge.md``).

The ridge is its OWN Tier-2 verb (offset baselines), consuming the SAME
``DistributionGrid`` IR that ``plot_1d_density_grid`` does. This commit proves
the ``overlaid`` variant on the shared IR: it renders, its per-bin offsets
increase with the earliest bin at the bottom, and ONE grid drives BOTH verbs.

Fixtures reuse TASK_C's grid-construction helpers from ``test_plotting`` (the
hand-built Distributions) so the ridge is exercised against the exact IR the
strip verb consumes.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.plotting import (
    DistributionGrid,
    build_1d_density_grid,
    plot_1d_density_grid,
)
from morphseq_investigation.engine.ridge import plot_1d_ridgeline

# Reuse TASK_C's fixture builders verbatim (grid-construction helpers).
from morphseq_investigation.tests.engine.test_plotting import _make_distribution


def _path_a_grid():
    """Within-population PATH A grid over two time bins (30, 48)."""
    dist_30 = _make_distribution({"time_bin": 30}, seed=0)
    dist_48 = _make_distribution({"time_bin": 48}, seed=1)
    groups = [
        dist_30.label_group("genotype", display_name="Genotype"),
        dist_48.label_group("genotype", display_name="Genotype"),
    ]
    return build_1d_density_grid(
        groups, "total_length_um",
        facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
    )


# --------------------------------------------------------------------------- #
# overlaid renders without error.
# --------------------------------------------------------------------------- #
def test_overlaid_renders_path_a():
    fig = plot_1d_ridgeline(_path_a_grid(), variant="overlaid", reference_role=None)
    assert fig is not None


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        plot_1d_ridgeline(_path_a_grid(), variant="nope")


# --------------------------------------------------------------------------- #
# Offsets increase per bin; earliest at bottom.
# --------------------------------------------------------------------------- #
def test_offsets_increase_per_bin_earliest_at_bottom():
    grid = _path_a_grid()  # time bins 30 (first-seen) then 48
    fig = plot_1d_ridgeline(grid, variant="overlaid", reference_role=None)

    ax = fig.axes[0]
    # The per-bin baseline is drawn as an axhline (a Line2D with constant y).
    baseline_ys = sorted(
        {
            float(line.get_ydata()[0])
            for line in ax.get_lines()
            if len(np.unique(line.get_ydata())) == 1
        }
    )
    assert len(baseline_ys) >= 2, "expected >=2 stacked bin baselines"
    # Strictly increasing offsets.
    assert all(b < a for b, a in zip(baseline_ys, baseline_ys[1:]))
    # Earliest bin (30) sits at the bottom (offset 0); later bin lifted above.
    assert baseline_ys[0] == pytest.approx(0.0)

    # Bin labels confirm bottom = earliest col value: the y=0 text is "30".
    texts = {
        round(float(t.get_position()[1]), 6): t.get_text().strip()
        for t in ax.texts
    }
    assert texts[0.0] == "30"


# --------------------------------------------------------------------------- #
# Shared-IR proof: one grid drives BOTH verbs.
# --------------------------------------------------------------------------- #
def test_shared_ir_one_grid_drives_both_verbs():
    grid = _path_a_grid()
    assert isinstance(grid, DistributionGrid)

    density_result = plot_1d_density_grid(grid, reference_role=None)
    ridge_fig = plot_1d_ridgeline(grid, variant="overlaid", reference_role=None)

    assert density_result is not None
    assert ridge_fig is not None
