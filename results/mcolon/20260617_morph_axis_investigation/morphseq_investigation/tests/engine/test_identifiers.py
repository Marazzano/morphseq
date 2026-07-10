"""TASK_0 — id constructor tests, incl. the locked grid_id <=> coords property."""

import numpy as np

from morphseq_investigation.engine.identifiers import (
    make_distribution_id,
    make_sample_set_id,
    make_grid_id,
)


def test_make_distribution_id_derived_and_deterministic():
    coords = {"scope_id": "b9d2", "time_bin": 30}
    did = make_distribution_id(coords)
    assert did.startswith("dist_")
    # same coordinates -> same id (deterministic, not a salted hash).
    assert did == make_distribution_id(dict(coords))


def test_make_distribution_id_order_independent():
    a = make_distribution_id({"scope_id": "b9d2", "time_bin": 30})
    b = make_distribution_id({"time_bin": 30, "scope_id": "b9d2"})
    assert a == b


def test_make_distribution_id_distinguishes_coordinates():
    base = {"scope_id": "b9d2", "time_bin": 30}
    assert make_distribution_id(base) != make_distribution_id(
        {"scope_id": "b9d2", "time_bin": 48}
    )
    # a different coordinate KEY set also forks the id.
    assert make_distribution_id(base) != make_distribution_id(
        {"scope_id": "b9d2", "time_bin": 30, "experiment_id": "exp1"}
    )


def test_make_distribution_id_numeric_float_int_agree():
    # 30 and 30.0 mean the same bin -> same id (param canonicalization).
    assert make_distribution_id({"time_bin": 30}) == make_distribution_id(
        {"time_bin": 30.0}
    )


def test_make_distribution_id_empty_coordinates_valid():
    # group-by-zero-coordinates (spec §compare step 3) -> one global id, no crash.
    did = make_distribution_id({})
    assert did.startswith("dist_")


def test_make_sample_set_id_double_underscore():
    did = make_distribution_id({"scope_id": "b9d2", "time_bin": 30})
    assert make_sample_set_id(did, "peak_0") == f"{did}__peak_0"
    # peak_0 from two different distributions never collide.
    tid = make_distribution_id({"scope_id": "b9d2", "time_bin": 48})
    assert make_sample_set_id(tid, "peak_0") != make_sample_set_id(did, "peak_0")


def _axes():
    return (np.linspace(0.0, 1.0, 11), np.linspace(-2.0, 2.0, 21))


def test_grid_id_identical_inputs_identical_id():
    kw = dict(
        feature_names=("PC1", "PC2"),
        construction_method="pooled_min_max",
        construction_params={"resolution": 11},
        axis_values=_axes(),
        fit_sample_ids=["s0", "s1", "s2"],
    )
    assert make_grid_id(**kw) == make_grid_id(**kw)


def test_grid_id_fit_ids_order_independent():
    axes = _axes()
    a = make_grid_id(("PC1", "PC2"), "pooled_min_max", {"resolution": 11}, axes, ["s0", "s1", "s2"])
    b = make_grid_id(("PC1", "PC2"), "pooled_min_max", {"resolution": 11}, axes, ["s2", "s0", "s1"])
    assert a == b


def test_grid_id_different_axis_values_different_id():
    # SAME fit ids + method + params, DIFFERENT produced coordinates -> different id.
    axes1 = _axes()
    axes2 = (np.linspace(0.0, 1.0, 11), np.linspace(-3.0, 3.0, 21))  # wider PC2 bounds
    common = dict(
        feature_names=("PC1", "PC2"),
        construction_method="pooled_min_max",
        construction_params={"resolution": 11},
        fit_sample_ids=["s0", "s1"],
    )
    assert make_grid_id(axis_values=axes1, **common) != make_grid_id(axis_values=axes2, **common)


def test_grid_id_feature_order_matters():
    axes = _axes()
    a = make_grid_id(("PC1", "PC2"), "pooled_min_max", {}, axes, [])
    b = make_grid_id(("PC2", "PC1"), "pooled_min_max", {}, axes, [])
    assert a != b


def test_grid_id_param_float_canonicalization():
    axes = _axes()
    a = make_grid_id(("PC1", "PC2"), "pooled_quantile", {"q_low": 0.01}, axes, [])
    b = make_grid_id(("PC1", "PC2"), "pooled_quantile", {"q_low": 0.0100000000004}, axes, [])
    assert a == b  # rounded to fixed precision -> same id


def test_grid_id_is_stable_string_not_salted_hash():
    # blake2b digest is deterministic across runs; format is grid_<hex>.
    gid = make_grid_id(("PC1",), "fixed_bounds", {}, (np.linspace(0, 1, 3),), [])
    assert gid.startswith("grid_")
    assert gid == make_grid_id(("PC1",), "fixed_bounds", {}, (np.linspace(0, 1, 3),), [])
