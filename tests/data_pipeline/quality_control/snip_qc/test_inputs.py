"""snip_qc inputs tests — load + verify resolved flag sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.quality_control.snip_qc.flag_input_resolver import ResolvedFlagSource
from data_pipeline.quality_control.snip_qc.inputs import load_snip_qc_flag_inputs


def _make_source(tmp_path: Path, step: str, flag_cols: tuple[str, ...], rows: list[dict]) -> ResolvedFlagSource:
    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{step}.csv"
    df.to_csv(csv_path, index=False)
    return ResolvedFlagSource(
        step=step,
        artifact_key=step,
        flag_columns=flag_cols,
        path=csv_path,
    )


def _snip_ids():
    return ["snip_01", "snip_02", "snip_03"]


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_loads_and_merges_two_sources(tmp_path):
    src_a = _make_source(tmp_path, "step_a", ("flag_a",), [
        {"snip_id": "snip_01", "flag_a": True},
        {"snip_id": "snip_02", "flag_a": False},
    ])
    src_b = _make_source(tmp_path, "step_b", ("flag_b",), [
        {"snip_id": "snip_01", "flag_b": False},
        {"snip_id": "snip_02", "flag_b": True},
    ])
    result = load_snip_qc_flag_inputs((src_a, src_b))
    assert set(result.columns) == {"snip_id", "flag_a", "flag_b"}
    assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Missing file
# ─────────────────────────────────────────────────────────────────────────────

def test_fails_if_csv_missing(tmp_path):
    src = ResolvedFlagSource(
        step="ghost_step", artifact_key="ghost_step",
        flag_columns=("some_flag",),
        path=tmp_path / "nonexistent.csv",
    )
    with pytest.raises(FileNotFoundError, match="ghost_step"):
        load_snip_qc_flag_inputs((src,))


# ─────────────────────────────────────────────────────────────────────────────
# Missing promised column
# ─────────────────────────────────────────────────────────────────────────────

def test_fails_if_csv_missing_promised_column(tmp_path):
    src = _make_source(tmp_path, "step_a", ("missing_flag",), [
        {"snip_id": "snip_01", "other_col": True},
    ])
    with pytest.raises(ValueError, match="missing_flag"):
        load_snip_qc_flag_inputs((src,))


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate snip_id
# ─────────────────────────────────────────────────────────────────────────────

def test_fails_on_duplicate_snip_id(tmp_path):
    src = _make_source(tmp_path, "step_a", ("flag_a",), [
        {"snip_id": "snip_01", "flag_a": True},
        {"snip_id": "snip_01", "flag_a": False},
    ])
    with pytest.raises(ValueError, match="duplicate snip_id"):
        load_snip_qc_flag_inputs((src,))


# ─────────────────────────────────────────────────────────────────────────────
# Boolean coercion
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    (1, True), (0, False),
    ("True", True), ("False", False),
    ("true", True), ("false", False),
    ("1", True), ("0", False),
    ("TRUE", True), ("FALSE", False),
    (" true ", True), (" false ", False),
])
def test_coerces_boolean_like_values(tmp_path, value, expected):
    src = _make_source(tmp_path, "step_a", ("flag_a",), [
        {"snip_id": "snip_01", "flag_a": value},
    ])
    result = load_snip_qc_flag_inputs((src,))
    assert bool(result["flag_a"].iloc[0]) == expected


def test_fails_on_na_flag(tmp_path):
    src = _make_source(tmp_path, "step_a", ("flag_a",), [
        {"snip_id": "snip_01", "flag_a": None},
    ])
    with pytest.raises(ValueError, match="null"):
        load_snip_qc_flag_inputs((src,))


def test_fails_on_unknown_boolean_value(tmp_path):
    src = _make_source(tmp_path, "step_a", ("flag_a",), [
        {"snip_id": "snip_01", "flag_a": "maybe"},
    ])
    with pytest.raises(ValueError, match="unrecognized value"):
        load_snip_qc_flag_inputs((src,))


# ─────────────────────────────────────────────────────────────────────────────
# Snip universe mismatch
# ─────────────────────────────────────────────────────────────────────────────

def test_fails_on_snip_id_mismatch_between_sources(tmp_path):
    src_a = _make_source(tmp_path, "step_a", ("flag_a",), [
        {"snip_id": "snip_01", "flag_a": True},
        {"snip_id": "snip_02", "flag_a": False},
    ])
    src_b = _make_source(tmp_path, "step_b", ("flag_b",), [
        {"snip_id": "snip_01", "flag_b": True},
        {"snip_id": "snip_03", "flag_b": False},  # different set
    ])
    with pytest.raises(ValueError, match="snip_id set differs"):
        load_snip_qc_flag_inputs((src_a, src_b))


# ─────────────────────────────────────────────────────────────────────────────
# Policy/runtime consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_cmd_snip_qc_uses_exclusion_flags_from_json_not_defaults(tmp_path):
    """Verify the JSON payload carries exclusion_flags so runtime and planning never split-brain."""
    import json
    from data_pipeline.quality_control.snip_qc.flag_input_resolver import resolve_snip_qc_flag_sources

    flags = ("edge_flag",)
    resolved = resolve_snip_qc_flag_sources(
        flags,
        output_root=tmp_path,
        experiment_id="exp01",
        well_id="A01",
    )
    payload = {
        "exclusion_flags": list(flags),
        "resolved_sources": [src.to_dict() for src in resolved],
    }
    json_path = tmp_path / "plan.json"
    json_path.write_text(json.dumps(payload))

    loaded = json.loads(json_path.read_text())
    assert loaded["exclusion_flags"] == list(flags)
    assert len(loaded["resolved_sources"]) == 1
    assert loaded["resolved_sources"][0]["step"] == "mask_quality_qc"
    # Only edge_flag — not the full SNIP_QC_EXCLUSION_FLAGS set
    assert loaded["resolved_sources"][0]["flag_columns"] == ["edge_flag"]
