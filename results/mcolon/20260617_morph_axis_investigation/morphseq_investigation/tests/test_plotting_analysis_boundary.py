from __future__ import annotations

import numpy as np

from morphseq_investigation.core.density_composition import CanonicalGrid, DensityGrid
from morphseq_investigation.plotting import modal_distribution_plotting as plotting
from morphseq_investigation.plotting import resolved_peaks, v0_qc
from morphseq_investigation.plotting.v0_analysis import compute_v0_peak_count_summary


def _spec() -> plotting.DistributionVisualSpec:
    grid = CanonicalGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, grid_size=9)
    density = np.exp(-(grid.xx**2 + grid.yy**2))
    field = DensityGrid(xx=grid.xx, yy=grid.yy, density=density, grid=grid)
    return plotting.DistributionVisualSpec(
        distribution_id="one_peak_test",
        points=np.array([[-0.2, 0.0], [0.0, 0.1], [0.2, -0.1]]),
        sampled_grid=field,
        composed_grid=field,
    )


def test_pure_renderer_uses_precomputed_analysis(tmp_path, monkeypatch):
    spec = _spec()
    peak_summary = compute_v0_peak_count_summary(
        truth_density=spec.composed_grid.density,
        observed_density=spec.sampled_grid.density,
    )

    def fail_if_analyzed(*args, **kwargs):
        raise AssertionError("renderer performed V0 analysis")

    monkeypatch.setattr(plotting, "compute_v0_peak_count_summary", fail_if_analyzed)
    monkeypatch.setattr(plotting, "compute_v0_metric_summary", fail_if_analyzed)
    out = plotting.render_distribution_qc_grid(
        [spec],
        tmp_path / "pure.png",
        peak_summaries_by_id={spec.distribution_id: peak_summary},
    )

    assert out.is_file()


def test_legacy_entry_point_still_renders_metric_row(tmp_path):
    out = plotting.plot_v0_distribution_qc_grid(
        [_spec()],
        tmp_path / "legacy.png",
        include_metric_row=True,
    )

    assert out.is_file()


def test_monolith_exports_delegate_to_split_modules():
    assert plotting.render_distribution_qc_grid is v0_qc.render_distribution_qc_grid
    assert plotting.plot_v0_distribution_qc_grid is v0_qc.plot_v0_distribution_qc_grid
    assert plotting.mode_count_label is resolved_peaks.mode_count_label
    assert plotting.draw_resolved_peak_basins is resolved_peaks.draw_resolved_peak_basins
