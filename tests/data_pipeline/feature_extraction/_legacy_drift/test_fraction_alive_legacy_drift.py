"""Legacy drift benchmark for the ``fraction_alive`` feature product (pilot ``20250912``).

Compares the rebuilt per-snip ``fraction_alive`` against the legacy ``fraction_alive`` column in
``qc_staged_20250912.csv``. Per feature_world.md this is **numeric-loose** (rel <= 1e-2): the VIA
mask resolution moved to snip-native, so the viability fraction shifts a known amount — assert a
generous bound and report the residual distribution; a miss above the bound is gross drift.

Marked ``legacy_drift`` (deselected by default; run with ``-m legacy_drift``). At the time of
writing the new pipeline has produced no ``fraction_alive`` output for the pilot, so the harness will
``pytest.skip`` cleanly; this benchmark is ready and will run the moment the product is built.

NOTE: legacy ``fraction_alive`` is 1.0 at the seed frame and drops as the embryo dies; confirm the
new product uses the same orientation (fraction *alive*, not fraction *dead*) before trusting a low
residual — an inverted convention would show as ``new ≈ 1 - legacy``, which the residual stats in the
report will make obvious.
"""

from __future__ import annotations

import pytest

from .drift_harness import NumericColumn, run_numeric_drift_benchmark

COLUMNS = [
    NumericColumn(
        "fraction_alive",
        "fraction_alive",
        1e-2,
        "numeric-loose: VIA mask resolution moved to snip-native",
    ),
]


@pytest.mark.legacy_drift
def test_fraction_alive_legacy_drift() -> None:
    run_numeric_drift_benchmark(
        product="fraction_alive",
        step="fraction_alive",
        columns=COLUMNS,
    )
