"""TASK_D — ridgeline verb tests (``docs/tasks_catalog/TASK_D_ridge.md``).

The ridge is its OWN Tier-2 verb (offset baselines), consuming the SAME
``DistributionGrid`` IR that ``plot_1d_density_grid`` does. These tests:
  - render each variant (overlaid/stacked/mirror) from a grid fixture w/o error;
  - assert reference curves draw dashed, offsets increase per bin, earliest at
    bottom;
  - prove ONE grid drives BOTH verbs (shared-IR).

Fixtures reuse TASK_C's grid-construction helpers from ``test_plotting`` (the
hand-built Distributions + the real DistributionComparisons assembly) so the
ridge is exercised against the exact IR the strip verb consumes.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.plotting import (
    DistributionGrid,
    build_1d_density_grid,
    build_1d_distribution_comparison,
    plot_1d_density_grid,
)
from morphseq_investigation.engine.ridge import plot_1d_ridgeline

# Reuse TASK_C's fixture builders verbatim (grid-construction helpers).
from morphseq_investigation.tests.engine.test_plotting import (
    _make_distribution,
    _two_member_comparisons,
)


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


def _path_b_grid(reference_value="wildtype"):
    """Cross-population PATH B grid; wildtype routed to the reference role."""
    comparisons = _two_member_comparisons()
    return build_1d_distribution_comparison(
        comparisons, "total_length_um", label_group="resolved_peak",
        facet_col=CoordinateFacet("time_bin"), reference_value=reference_value,
    )


# --------------------------------------------------------------------------- #
# Each variant renders without error.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["overlaid", "stacked", "mirror"])
def test_each_variant_renders_path_a(variant):
    fig = plot_1d_ridgeline(_path_a_grid(), variant=variant, reference_role=None)
    assert fig is not None


@pytest.mark.parametrize("variant", ["overlaid", "stacked", "mirror"])
def test_each_variant_renders_path_b_with_reference(variant):
    fig = plot_1d_ridgeline(
        _path_b_grid(), variant=variant, reference_role="reference"
    )
    assert fig is not None


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        plot_1d_ridgeline(_path_a_grid(), variant="nope")


# --------------------------------------------------------------------------- #
# Reference curves are dashed (line + unfilled outline).
# --------------------------------------------------------------------------- #
def test_reference_curves_are_dashed():
    grid = _path_b_grid(reference_value="wildtype")
    fig = plot_1d_ridgeline(grid, variant="stacked", reference_role="reference")

    # Reference lines are dashed; at least one dashed Line2D exists, and the
    # reference fill is unfilled (facecolor alpha ~ 0), while target fills are
    # solid-ish. matplotlib records the requested linestyle on each Line2D.
    dashed = []
    for ax in fig.axes:
        for line in ax.get_lines():
            ls = line.get_linestyle()
            if ls in ("--", "dashed") or (isinstance(ls, tuple)):
                dashed.append(line)
    assert dashed, "expected at least one dashed reference line"

    # Reference PolyCollection (fill_between) has facecolor 'none' -> alpha 0.
    from matplotlib.collections import PolyCollection

    unfilled = []
    filled = []
    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, PolyCollection):
                fc = coll.get_facecolor()
                # facecolor="none" -> empty facecolor array (size 0) or alpha 0.
                if fc.size == 0 or fc[0, 3] == 0.0:
                    unfilled.append(coll)
                else:
                    filled.append(coll)
    assert unfilled, "reference fill_between should be unfilled (facecolor none)"
    assert filled, "target fill_between should be filled"


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
