"""Legacy drift benchmark for the ``stage_predictions`` feature product (pilot ``20250912``).

Compares the rebuilt per-snip ``predicted_stage_hpf`` against the legacy ``predicted_stage_hpf``
column in ``qc_staged_20250912.csv``. Per feature_world.md this is **numeric-tight** — same Kimmel
staging formula + same acquisition timing — so a real divergence is a bug, not a method shift.

Marked ``legacy_drift`` (deselected by default; run with ``-m legacy_drift``). Shared machinery is in
``drift_harness``; the new side loads from the merged table or, if only a partial pilot run exists,
from the concatenated per-well shards.

NOTE: stage at the seed frame is anchored by acquisition time, so the very first frame can differ if
the start-age convention shifted (legacy carried ``start_age_hpf`` ~11 hpf). The tight bound below is
relative; if the seed-frame offset turns out to be a deliberate convention change rather than drift,
record it in the report and relax the seed-frame comparison — do not silently widen the bound.
"""

from __future__ import annotations

import pytest

from .drift_harness import NumericColumn, run_numeric_drift_benchmark

COLUMNS = [
    NumericColumn(
        "predicted_stage_hpf",
        "predicted_stage_hpf",
        1e-3,
        "numeric-tight: same Kimmel staging formula + acquisition timing",
    ),
]


@pytest.mark.legacy_drift
def test_stage_predictions_legacy_drift() -> None:
    run_numeric_drift_benchmark(
        product="stage_predictions",
        step="stage_predictions",
        columns=COLUMNS,
    )
