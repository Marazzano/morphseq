"""TASK_E acceptance — the b9d2 catalog composition, guarded in CI.

``v0/b9d2_catalog_example.py`` is the real acceptance runner but reads a
gitignored ~125MB CSV, so it cannot run in a fresh checkout / CI. This test
reproduces the SAME wiring — from_dataframe -> detect_peaks -> label_groups ->
build_1d_density_grid (PATH A) AND compare -> build_1d_distribution_comparison
(PATH B) + a ridge — on a small synthetic b9d2-LIKE frame engineered to show the
documented 1 -> 2 emergence signal, so the acceptance PATH is exercised without
the data. Peak-count parity itself is covered separately in
``test_labelers.py::test_detect_peaks_peak_count_regression_matches_old_engine_per_bin``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphseq_investigation.engine.catalog import DistributionCatalog
from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.plotting import (
    DistributionGrid,
    build_1d_density_grid,
    build_1d_distribution_comparison,
)
from morphseq_investigation.engine.ridge import plot_1d_ridgeline

FEATURE_NAMES = ("total_length_um", "baseline_deviation_normalized")
DESIGN_BINS = (14.0, 18.0, 30.0)


def _b9d2_like_frame(seed: int = 100) -> pd.DataFrame:
    """One tidy frame: per (embryo, time_bin) rows for wildtype (reference) +
    b9d2 phenotype-affected (target). Later bins split the target into two
    clusters (CE vs HTA) so the emergence signal is present."""
    rng = np.random.default_rng(seed)
    rows = []
    eid = 0
    for tb in DESIGN_BINS:
        # wildtype reference: one compact cluster at every bin.
        for _ in range(30):
            x, y = rng.normal((10.0, 0.0), 1.0)
            rows.append((f"e{eid}", tb, x, y, "wildtype", "wildtype"))
            eid += 1
        # b9d2 target: single cluster at 14hpf, two clusters later (CE/HTA).
        if tb <= 14.0:
            for _ in range(40):
                x, y = rng.normal((10.0, 0.0), 1.0)
                rows.append((f"e{eid}", tb, x, y, "b9d2", "CE"))
                eid += 1
        else:
            sep = 6.0 + 0.3 * (tb - 14.0)
            for pheno, center in (("CE", 10.0), ("HTA", 10.0 + sep)):
                for _ in range(25):
                    x, y = rng.normal((center, 0.0), 0.8)
                    rows.append((f"e{eid}", tb, x, y, "b9d2", pheno))
                    eid += 1
    return pd.DataFrame(
        rows,
        columns=["embryo_id", "time_bin", *FEATURE_NAMES, "genotype", "phenotype_clean"],
    )


def test_path_a_density_grid_has_three_label_group_rows():
    df = _b9d2_like_frame()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=FEATURE_NAMES,
        label_columns=("genotype", "phenotype_clean"),
        split_columns=("time_bin",),
    ).detect_peaks(features=FEATURE_NAMES, output_label="resolved_peak")

    groups = (
        *catalog.label_groups("resolved_peak", display_name="Resolved peaks"),
        *catalog.label_groups("phenotype_clean", display_name="Phenotype"),
        *catalog.label_groups("genotype", display_name="Genotype"),
    )
    grid = build_1d_density_grid(
        groups,
        feature="total_length_um",
        facet_row=LabelGroupFacet(),
        facet_col=CoordinateFacet("time_bin"),
    )
    assert isinstance(grid, DistributionGrid)
    # 3 rows: the three label-group display names show up as cell row keys.
    row_names = {c.cell[0] for c in grid.curves}
    assert row_names == {"Resolved peaks", "Phenotype", "Genotype"}
    # columns are the design time bins. (single-split-column groupby keys the
    # coordinate as a 1-tuple, so normalize before comparing.)
    col_bins = {c.cell[1][0] if isinstance(c.cell[1], tuple) else c.cell[1] for c in grid.curves}
    assert col_bins == set(DESIGN_BINS)
    # ridge consumes the SAME IR without re-fitting (no exception on the verb).
    plot_1d_ridgeline(grid, variant="stacked", output_path=None)


def test_path_b_compare_overlays_wt_and_b9d2_per_cell():
    df = _b9d2_like_frame()
    catalog2 = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=FEATURE_NAMES,
        label_columns=("genotype", "phenotype_clean"),
        split_columns=("time_bin", "genotype"),
    ).detect_peaks(features=FEATURE_NAMES, output_label="resolved_peak")

    comparisons = catalog2.compare(across="genotype", values=("wildtype", "b9d2"))
    # every comparison holds both members (invariant).
    for comp in comparisons.comparisons:
        assert tuple(comp.members) == ("wildtype", "b9d2")

    grid_b = build_1d_distribution_comparison(
        comparisons,
        feature="total_length_um",
        label_group="resolved_peak",
        facet_col=CoordinateFacet("time_bin"),
        reference_value="wildtype",
    )
    # each curve carries a structured CurveKey with its comparison member.
    members = {c.curve_key.comparison_member for c in grid_b.curves}
    assert members == {"wildtype", "b9d2"}
    # a later bin resolves >=2 b9d2 peaks (emergence), so b9d2 contributes
    # multiple curves in at least one cell.
    b9d2_curve_counts = {}
    for c in grid_b.curves:
        if c.curve_key.comparison_member == "b9d2":
            b9d2_curve_counts[c.cell] = b9d2_curve_counts.get(c.cell, 0) + 1
    assert max(b9d2_curve_counts.values()) >= 2


def test_emergence_peak_count_via_catalog():
    """The acceptance signal itself, through the catalog: b9d2 target peak count
    is <=1 at the earliest bin and reaches >=2 at a later bin."""
    df = _b9d2_like_frame()
    catalog2 = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=FEATURE_NAMES,
        label_columns=("genotype", "phenotype_clean"),
        split_columns=("time_bin", "genotype"),
    ).detect_peaks(features=FEATURE_NAMES, output_label="resolved_peak")

    counts = {}
    for dist in catalog2.distributions:
        if dist.coordinates.get("genotype") == "b9d2":
            counts[dist.coordinates["time_bin"]] = len(dist.sample_sets("resolved_peak"))
    ordered = [counts[tb] for tb in sorted(counts)]
    assert ordered[0] <= 1 and max(ordered) >= 2, ordered
