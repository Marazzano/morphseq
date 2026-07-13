import numpy as np
import pytest

import morphseq_investigation.engine.compare as compare_module
from morphseq_investigation.engine.catalog import DistributionCatalog, DistributionComparison
from morphseq_investigation.engine.compare import (
    NullTestResult,
    compare_distributions,
    label_group_agreement,
    summarize_label_groups,
)
from morphseq_investigation.engine.grid import build_grid
from morphseq_investigation.engine.objects import (
    DensityEstimateSpec,
    Distribution,
    UNASSIGNED_LABEL,
)
from morphseq_investigation.engine.plotting import descriptive_comparison_subplot


def _distribution(distribution_id, condition):
    return Distribution(
        distribution_id=distribution_id,
        sample_ids=("s0", "s1", "s2", "s3"),
        feature_names=("x", "y"),
        feature_values=np.zeros((4, 2)),
        coordinates={"batch": "b1", "condition": condition},
    )


def test_agreement_reads_unified_assignments_and_explicit_unassigned():
    distribution = _distribution("d", "reference")
    distribution = distribution.with_label(
        "left", {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
    )
    distribution = distribution.with_label(
        "right", {"s0": "P", "s1": "P", "s2": UNASSIGNED_LABEL, "s3": "Q"}
    )
    result = label_group_agreement(
        distribution.get_label_group("left"), distribution.get_label_group("right")
    )
    assert result.n_samples_compared == 3
    assert result.left_only_sample_ids == ("s2",)
    assert result.cross_tab["A"]["P"] == 2
    assert result.cross_tab["B"]["Q"] == 1


def test_agreement_requires_same_complete_sample_membership():
    left = _distribution("d", "reference").with_label("x", {"s0": "A"}).get_label_group("x")
    right = _distribution("other", "target").with_label("x", {"s0": "A"}).get_label_group("x")
    with pytest.raises(ValueError, match="same distribution_id"):
        label_group_agreement(left, right)


def test_summary_preserves_structured_coordinates_and_derives_sample_sets():
    reference = _distribution("ref", "reference").with_label(
        "genotype", {"s0": "wt", "s1": "wt", "s2": "mut"}
    )
    target = _distribution("tgt", "target").with_label(
        "genotype", {"s0": "wt", "s1": "mut", "s2": "mut", "s3": "mut"}
    )
    comparison = DistributionComparison(
        coordinates={"batch": "b1"}, members={"reference": reference, "target": target}
    )
    rows = summarize_label_groups(comparison, "genotype")
    assert tuple(row.member_value for row in rows) == ("reference", "target")
    assert all(dict(row.comparison_coordinates) == {"batch": "b1"} for row in rows)
    assert rows[0].distribution_coordinates["condition"] == "reference"
    assert rows[0].sample_set_count == len(reference.sample_sets("genotype")) == 2
    assert rows[0].unassigned_sample_count == 1
    assert rows[0].peak_count is None
    assert rows[0].is_robust is None


def test_summary_missing_label_raises_instead_of_dropping_member():
    reference = _distribution("ref", "reference").with_label("group", {"s0": "A"})
    target = _distribution("tgt", "target")
    comparison = DistributionComparison(coordinates={"batch": "b1"}, members={"r": reference, "t": target})
    with pytest.raises(KeyError):
        summarize_label_groups(comparison, "group")


def _numeric_distribution(name, values, features=("x",)):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return Distribution(
        distribution_id=name,
        sample_ids=tuple(f"{name}_{i}" for i in range(len(values))),
        feature_names=tuple(features),
        feature_values=values,
        coordinates={"condition": name, "time": 24},
    )


def test_raw_comparison_prepares_1d_shared_grid_without_mutating_sources():
    reference = _numeric_distribution("wt", [-2, -1, -0.5, 0, 0.5, 1])
    target = _numeric_distribution("mut", [0, 0.5, 1, 1.5, 2, 2.5])
    result = compare_distributions(reference=reference, targets=(target,), grid_size=31)
    item = result.comparisons[0]
    assert item.features == ("x",)
    assert item.reference_density.grid.grid_id == item.target_density.grid.grid_id
    assert item.reference_density.density_grid.density.shape == (31,)
    assert 0 <= item.density_overlap <= 1
    assert reference.densities == target.densities == ()
    assert item.null_test.state == "not_tested"


def test_raw_comparison_prepares_2d_and_preserves_label_overlays():
    ref = _numeric_distribution(
        "wt", [[-1, 10], [0, 11], [1, 12], [2, 13]], features=("x", "y")
    ).with_label("peaks", {"wt_0": "peak_0", "wt_1": "peak_0"})
    target = _numeric_distribution(
        "mut", [[0, 20], [1, 22], [2, 24], [3, 26]], features=("x", "y")
    ).with_label("peaks", {"mut_0": "peak_0", "mut_3": "peak_1"})
    item = compare_distributions(
        reference=ref, targets=(target,), label_group="peaks",
        features=("y", "x"), grid_size=17,
    ).comparisons[0]
    assert item.grid.feature_names == ("y", "x")
    assert item.reference_density.density_grid.density.shape == (17, 17)
    assert item.reference_label_group is ref.get_label_group("peaks")
    assert len(ref.sample_ids) == 4  # unassigned samples remained in density membership


def test_comparison_requires_features_for_multifeature_and_rejects_nd():
    ref = _numeric_distribution("r", np.zeros((5, 3)), features=("a", "b", "c"))
    target = _numeric_distribution("t", np.ones((5, 3)), features=("a", "b", "c"))
    with pytest.raises(ValueError, match="explicit ordered features"):
        compare_distributions(reference=ref, targets=(target,))
    with pytest.raises(ValueError, match="only 1-D or 2-D"):
        compare_distributions(reference=ref, targets=(target,), features=("a", "b", "c"))


def test_supplied_semantic_grid_controls_features_and_is_reused_exactly():
    ref = _numeric_distribution("r", [[0, 10], [1, 11], [2, 12]], features=("x", "y"))
    target = _numeric_distribution("t", [[2, 20], [3, 21], [4, 22]], features=("x", "y"))
    grid = build_grid(
        ("y",), np.array([[0.0], [30.0]]), ("lo", "hi"),
        "fixed_bounds", {"resolution": 9, "bounds": ((0, 30),)},
    )
    item = compare_distributions(reference=ref, targets=(target,), grid=grid).comparisons[0]
    assert item.grid is grid
    assert item.features == ("y",)


def test_catalog_directed_entry_point_matches_raw_preparation_and_context():
    ref = _numeric_distribution("wt", [-1, 0, 1, 2])
    target = _numeric_distribution("mut", [0, 1, 2, 3])
    catalog = DistributionCatalog((ref, target), coordinate_names=("condition", "time"))
    catalog_result = catalog.compare(
        across="condition", reference="wt", targets=("mut",),
        match_on=("time",), grid_size=15,
    )
    raw = compare_distributions(reference=ref, targets=(target,), grid_size=15)
    item = catalog_result.comparisons[0]
    assert item.density_overlap == pytest.approx(raw.comparisons[0].density_overlap)
    assert item.reference_value == "wt" and item.target_value == "mut"
    assert dict(item.coordinates) == {"time": 24}


def test_catalog_preparation_failure_adds_member_ids_and_matched_context():
    ref = _numeric_distribution("wt", [-1, 0, 1]).with_label(
        "peaks", {"wt_0": "peak_0"}
    )
    target = _numeric_distribution("mut", [0, 1, 2])
    catalog = DistributionCatalog((ref, target), coordinate_names=("condition", "time"))
    with pytest.raises(ValueError) as caught:
        catalog.compare(
            across="condition", reference="wt", targets=("mut",),
            match_on=("time",), label_group="peaks",
        )
    message = str(caught.value)
    assert "matched coordinates {'time': 24}" in message
    assert "'reference': 'wt'" in message and "'targets': ('mut',)" in message
    assert isinstance(caught.value.__cause__, KeyError)


def test_multi_target_failure_is_all_or_error_and_sources_remain_immutable():
    ref = _numeric_distribution("ref", [-1, 0, 1])
    valid = _numeric_distribution("valid", [0, 1, 2])
    invalid = _numeric_distribution("invalid", np.ones((3, 2)), features=("y", "z"))
    snapshots = tuple((d.densities, d.shared_density_index) for d in (ref, valid, invalid))
    with pytest.raises(ValueError, match="lacks comparison feature"):
        compare_distributions(reference=ref, targets=(valid, invalid), features=("x",))
    assert snapshots == tuple(
        (d.densities, d.shared_density_index) for d in (ref, valid, invalid)
    )


def test_comparison_preserves_distinct_effective_density_specs():
    ref = _numeric_distribution("ref", [-2, -1, 0, 1, 2])
    target = _numeric_distribution("target", [-1, 0, 1, 2, 3])
    ref_spec = DensityEstimateSpec(
        bandwidth_multiplier=0.5, grid_params={"resolution": 9}
    )
    target_spec = DensityEstimateSpec(
        bandwidth_multiplier=1.25, grid_params={"resolution": 9}
    )
    ref = ref.with_density(ref.calc_density(ref_spec), select_as_shared=True)
    target = target.with_density(target.calc_density(target_spec), select_as_shared=True)
    item = compare_distributions(reference=ref, targets=(target,), grid_size=11).comparisons[0]
    assert item.reference_density.spec is ref_spec
    assert item.target_density.spec is target_spec


def test_label_overlay_does_not_filter_unassigned_density_members(monkeypatch):
    ref = _numeric_distribution("ref", [-2, -1, 0, 1]).with_label(
        "peaks", {"ref_0": "peak_0"}
    )
    target = _numeric_distribution("target", [0, 1, 2, 3]).with_label(
        "peaks", {"target_3": "peak_0"}
    )
    sizes = []
    original_bandwidth = compare_module._bandwidth

    def recording_bandwidth(values, spec):
        sizes.append(len(values))
        return original_bandwidth(values, spec)

    monkeypatch.setattr(compare_module, "_bandwidth", recording_bandwidth)
    compare_distributions(reference=ref, targets=(target,), label_group="peaks")
    assert sizes == [len(ref.sample_ids), len(target.sample_ids)] == [4, 4]


def test_null_test_is_immutable_and_has_explicit_state():
    ref = _numeric_distribution("r", [-2, -1, 0, 1, 2])
    target = _numeric_distribution("t", [-1, 0, 1, 2, 3])
    original = compare_distributions(reference=ref, targets=(target,), grid_size=11)
    tested = original.test_nulls(n_draws=5, seed=7)
    assert original.comparisons[0].null_test.state == "not_tested"
    assert tested.comparisons[0].null_test.state in {"invalid", "nonsignificant", "significant"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"state": "not_tested", "n_draws": 1}, "not_tested"),
        ({"state": "invalid", "n_draws": 2, "observed_overlap": 0.5,
          "null_overlaps": [0.4], "valid_draws": 0}, "valid_draws"),
        ({"state": "significant", "n_draws": 2, "observed_overlap": 0.5,
          "null_overlaps": [0.4], "valid_draws": 1, "p_value": 0.5}, "state"),
    ],
)
def test_null_result_rejects_cross_field_contradictions(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NullTestResult(**kwargs)


def test_null_result_accepts_each_consistent_terminal_state():
    invalid = NullTestResult(state="invalid", observed_overlap=0.3, n_draws=4)
    nonsignificant = NullTestResult(
        state="nonsignificant", observed_overlap=0.3,
        null_overlaps=np.array([0.2, 0.4]), valid_draws=2,
        p_value=0.5, n_draws=2,
    )
    significant = NullTestResult(
        state="significant", observed_overlap=0.1,
        null_overlaps=np.array([0.5] * 20), valid_draws=20,
        p_value=1 / 21, n_draws=20,
    )
    assert (invalid.state, nonsignificant.state, significant.state) == (
        "invalid", "nonsignificant", "significant"
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [("invalid", "invalid"), ("low", "nonsignificant"), ("high", "significant")],
)
def test_null_enrichment_reaches_each_terminal_state_deterministically(
    monkeypatch, outcome, expected
):
    ref = _numeric_distribution("r", [-2, -1, 0, 1, 2])
    target = _numeric_distribution("t", [-1, 0, 1, 2, 3])
    item = compare_distributions(reference=ref, targets=(target,), grid_size=11).comparisons[0]

    def controlled_overlap(left, right):
        if outcome == "invalid":
            raise ValueError("invalid permutation")
        if outcome == "low":
            return item.density_overlap - 0.01
        return item.density_overlap + 0.01

    monkeypatch.setattr(compare_module, "_overlap", controlled_overlap)
    tested = item.test_nulls(n_draws=20, seed=4)
    assert tested.null_test.state == expected
    assert tested.null_test.valid_draws == (0 if expected == "invalid" else 20)


def test_null_test_preserves_valid_draws_when_an_individual_draw_fails(monkeypatch):
    ref = _numeric_distribution("r", [-2, -1, 0, 1, 2])
    target = _numeric_distribution("t", [-1, 0, 1, 2, 3])
    item = compare_distributions(reference=ref, targets=(target,), grid_size=11).comparisons[0]
    calls = 0

    def sometimes_fails(left, right):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("one invalid draw")
        return 0.5

    monkeypatch.setattr(compare_module, "_overlap", sometimes_fails)
    tested = item.test_nulls(n_draws=4, seed=3)
    assert tested.null_test.valid_draws == 3
    assert tested.null_test.n_draws == 4
    assert tested.null_test.state in {"nonsignificant", "significant"}


def test_plotting_consumes_untested_and_tested_comparisons_without_analysis():
    ref = _numeric_distribution("r", [-2, -1, 0, 1, 2])
    target = _numeric_distribution("t", [-1, 0, 1, 2, 3])
    original = compare_distributions(reference=ref, targets=(target,), grid_size=11)
    tested = original.test_nulls(n_draws=3, seed=11)

    untested_plot = descriptive_comparison_subplot(original.comparisons[0])
    tested_plot = descriptive_comparison_subplot(tested.comparisons[0])

    assert len(untested_plot.traces) == len(tested_plot.traces) == 2
    assert untested_plot.traces[1].style.linestyle == ":"
    assert tested_plot.traces[1].style.linestyle in {"-.", "--", "-"}
