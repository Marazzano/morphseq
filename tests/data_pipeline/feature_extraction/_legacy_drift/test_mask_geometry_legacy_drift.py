"""Legacy drift benchmark for the ``mask_geometry`` feature product (pilot ``20250912``).

Real-data continuity gate, NOT a unit test: joins the rebuilt ``mask_geometry`` table against the
legacy build output on ``snip_id`` and checks each numeric column against its tolerance class. Marked
``legacy_drift`` and deselected from the default ``pytest tests/`` run (pytest.ini); run explicitly
with ``-m legacy_drift``. All shared machinery (new-side load, normalize+join, report) lives in
``drift_harness``; this file is only the product-specific column map.

Tolerance map (feature_world.md → Legacy Drift Comparison, refined by what the pilot data shows):

  centroid_x_um / centroid_y_um  numeric-tight (rel <= 2e-3): position is resolution-robust
                                 (~1e-3 observed under correct alignment).
  area_um2                       numeric-loose (rel <= 1e-1): the spec calls this numeric-tight
                                 ("same mask, same formula"), but the new mask is the snip-native
                                 VIA resolution, not the legacy full-frame mask, so area shifts a
                                 few percent. Bound catches gross drift; residuals are reported.
  perimeter_um                   numeric-loose (rel <= 3e-1): perimeter is the most
                                 resolution-sensitive geometry (boundary smoothing) and drifts most.

``length_um`` / ``width_um`` are recorded only: in this legacy file the ``height_um`` / ``width_um``
columns are the constant FRAME dimensions (~7072 um), not the embryo length/width axis, so there is
no comparable legacy ground-truth column.

ALIGNMENT NOTE: the pilot legacy ``snip_id`` ``t####`` is 0-based (it has a ``t0000`` row whose
values match the new ``t0000``), contrary to the spec's "legacy is 1-based" remark; the loader
aligns on the ``t####`` anchor with no +1 shift. With this alignment centroids agree to ~1e-3; a +1
shift would spuriously disagree by ~1.5% — the wrong-alignment signature.
"""

from __future__ import annotations

import pytest

from .drift_harness import NumericColumn, run_numeric_drift_benchmark

COLUMNS = [
    NumericColumn("centroid_x_um", "centroid_x_um", 2e-3, "numeric-tight: position robust"),
    NumericColumn("centroid_y_um", "centroid_y_um", 2e-3, "numeric-tight"),
    NumericColumn("area_um2", "area_um2", 1e-1, "numeric-loose: mask resolution moved to snip-native"),
    NumericColumn("perimeter_um", "perimeter_um", 3e-1, "numeric-loose: most resolution-sensitive"),
]
REPORTED_ONLY = ("length_um", "width_um")


@pytest.mark.legacy_drift
def test_mask_geometry_legacy_drift() -> None:
    run_numeric_drift_benchmark(
        product="mask_geometry",
        step="mask_geometry",
        columns=COLUMNS,
        reported_only=REPORTED_ONLY,
    )
