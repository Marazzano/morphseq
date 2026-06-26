"""snip_qc build + contract tests — pass / single / multi-reason / fail-loud on missing flag."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.snip_qc.build import build_snip_qc_verdict
from data_pipeline.quality_control.snip_qc.contract import (
    DEFAULT_SNIP_QC_EXCLUSION_REASONS,
    SNIP_QC_TABLE_COLUMNS,
    validate_snip_qc,
)
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"
PHYS = build_physical_embryo_id(WELL, 1)
_FLAG_COLS = list(DEFAULT_SNIP_QC_EXCLUSION_REASONS.values())


def _snip(t):
    image_id = build_image_id(WELL, CHANNEL, t)
    embryo_id = build_embryo_id(PHYS, image_id)
    return build_snip_id(embryo_id, image_id), embryo_id


def _universe(n):
    rows = []
    for t in range(n):
        snip_id, embryo_id = _snip(t)
        rows.append(
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": PHYS,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
            }
        )
    return pd.DataFrame(rows)


def _flags(universe, per_snip_true_cols):
    """per_snip_true_cols: list (len == n_snips) of lists of flag-cols that are True for that snip."""
    rows = []
    for i, snip_id in enumerate(universe["snip_id"]):
        row = {"snip_id": snip_id}
        for col in _FLAG_COLS:
            row[col] = col in per_snip_true_cols[i]
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in _FLAG_COLS:
        df[col] = df[col].astype(bool)
    return df


def test_pass_when_no_flags():
    uni = _universe(1)
    flags = _flags(uni, [[]])
    out = build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    assert out["use_snip"].tolist() == [True]
    assert out["qc_fail_reasons"].tolist() == [""]
    assert list(out.columns) == SNIP_QC_TABLE_COLUMNS
    validate_snip_qc(out)


def test_single_reason():
    uni = _universe(1)
    flags = _flags(uni, [["sa_outlier_flag"]])
    out = build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    assert out["use_snip"].tolist() == [False]
    assert out["qc_fail_reasons"].tolist() == ["surface_area_outlier"]
    validate_snip_qc(out)


def test_multi_reason_pipe_delimited_in_map_order():
    uni = _universe(1)
    flags = _flags(uni, [["persistence_dead_flag", "edge_flag"]])
    out = build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    # order follows the DEFAULT_SNIP_QC_EXCLUSION_REASONS map order: dead_persistence before edge
    assert out["qc_fail_reasons"].tolist() == ["dead_persistence|edge"]
    assert out["use_snip"].tolist() == [False]
    validate_snip_qc(out)


def test_both_death_modes_surface_as_two_reasons():
    uni = _universe(1)
    flags = _flags(uni, [["viability_dead_flag", "persistence_dead_flag"]])
    out = build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    assert out["qc_fail_reasons"].tolist() == ["dead_viability|dead_persistence"]


def test_accepts_pandas_nullable_boolean_flags():
    """REGRESSION (Tier-1 through-line, 2026-06-26): inputs.py — the canonical producer of
    qc_flags_df — coerces every flag to pandas nullable ``boolean`` (dtype="boolean"), NOT numpy
    bool. build.py's old ``dtype != bool`` check rejected exactly that, so the real
    inputs.py → build.py seam broke on real data while every unit test (which used .astype(bool))
    passed. This test feeds build.py the nullable dtype inputs.py actually emits.
    """
    uni = _universe(2)
    flags = _flags(uni, [[], ["edge_flag"]])
    for col in _FLAG_COLS:
        flags[col] = flags[col].astype("boolean")  # pandas nullable BooleanDtype, no NA
        assert flags[col].dtype != bool  # guard: this is NOT numpy bool — the exact seam dtype

    out = build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    assert out["use_snip"].tolist() == [True, False]
    assert out["qc_fail_reasons"].tolist() == ["", "edge"]
    validate_snip_qc(out)


def test_nonboolean_flag_dtype_still_fails_loud():
    """The dtype gate must still REJECT genuinely non-boolean flag columns (e.g. int 0/1 or
    object strings) — accepting nullable boolean must not loosen into accepting anything."""
    uni = _universe(1)
    flags = _flags(uni, [["edge_flag"]])
    flags["edge_flag"] = flags["edge_flag"].astype("int64")  # 0/1 ints, not boolean
    with pytest.raises(ValueError, match="edge_flag.*boolean dtype"):
        build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)


def test_missing_flag_column_fails_loud():
    uni = _universe(1)
    flags = _flags(uni, [[]]).drop(columns=["edge_flag"])
    with pytest.raises(ValueError, match="edge_flag.*not in"):
        build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)


def test_universe_flag_mismatch_fails_loud():
    uni = _universe(2)
    flags = _flags(_universe(1), [[]])  # only one snip of flags for a two-snip universe
    with pytest.raises(ValueError, match="must match the universe"):
        build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)


def test_build_starts_from_full_spine_not_just_snip_id():
    uni = _universe(2)
    flags = _flags(uni, [[], ["edge_flag"]])
    out = build_snip_qc_verdict(uni, flags, exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    for col in ("experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id"):
        assert col in out.columns


def test_contract_rejects_unknown_reason():
    uni = _universe(1)
    out = build_snip_qc_verdict(uni, _flags(uni, [[]]), exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    out.loc[0, "qc_fail_reasons"] = "not_a_real_reason"
    out.loc[0, "use_snip"] = False
    with pytest.raises(ValueError, match="unknown reason"):
        validate_snip_qc(out)


def test_contract_rejects_use_snip_disagreeing_with_reasons():
    uni = _universe(1)
    out = build_snip_qc_verdict(uni, _flags(uni, [["edge_flag"]]), exclusion_reasons=DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    out.loc[0, "use_snip"] = True  # but qc_fail_reasons is non-empty
    with pytest.raises(ValueError, match="true iff qc_fail_reasons"):
        validate_snip_qc(out)
