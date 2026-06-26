"""Tests for snip_qc/flag_input_resolver.py.

Covers: resolver correctness, error messages, kingdom boundary enforcement.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipeline.quality_control.snip_qc.contract import DEFAULT_SNIP_QC_EXCLUSION_REASONS
from data_pipeline.quality_control.snip_qc.flag_input_resolver import (
    ResolvedFlagSource,
    _SOURCE_PAYLOADS,
    _build_flag_column_index,
    resolve_snip_qc_flag_sources,
)


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def test_default_reasons_resolve_cleanly():
    index = _build_flag_column_index(DEFAULT_SNIP_QC_EXCLUSION_REASONS)
    assert set(index) == set(DEFAULT_SNIP_QC_EXCLUSION_REASONS.values())
    assert all(step in _SOURCE_PAYLOADS for step in index.values())


def test_only_needed_steps_returned():
    reasons = {"edge": "edge_flag"}
    index = _build_flag_column_index(reasons)
    assert index == {"edge_flag": "mask_quality_qc"}


def test_unknown_flag_fails_loud_with_flag_name():
    reasons = {"unknown_reason": "nonexistent_flag"}
    with pytest.raises(ValueError, match="nonexistent_flag"):
        _build_flag_column_index(reasons)


def test_unknown_flag_error_names_eligible_steps():
    reasons = {"x": "no_such_flag"}
    with pytest.raises(ValueError, match="eligible steps"):
        _build_flag_column_index(reasons)


def test_ambiguous_flag_lists_all_claimants():
    fake_payloads = {
        "step_a": ("shared_flag",),
        "step_b": ("shared_flag",),
    }
    with patch("data_pipeline.quality_control.snip_qc.flag_input_resolver._SOURCE_PAYLOADS", fake_payloads):
        with pytest.raises(ValueError, match="ambiguous"):
            _build_flag_column_index({"r": "shared_flag"})


def test_multiple_errors_reported_together():
    reasons = {"r1": "missing_1", "r2": "missing_2"}
    with pytest.raises(ValueError) as exc_info:
        _build_flag_column_index(reasons)
    msg = str(exc_info.value)
    assert "missing_1" in msg
    assert "missing_2" in msg


# ─────────────────────────────────────────────────────────────────────────────
# resolve_snip_qc_flag_sources
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_returns_one_source_per_step():
    reasons = {"edge": "edge_flag", "disc": "discontinuous_mask_flag"}
    resolved = resolve_snip_qc_flag_sources(
        reasons,
        output_root=Path("/data"),
        experiment_id="exp01",
        well_id="A01",
    )
    assert len(resolved) == 1
    assert resolved[0].step == "mask_quality_qc"
    assert set(resolved[0].flag_columns) == {"edge_flag", "discontinuous_mask_flag"}


def test_resolve_groups_flags_by_step():
    resolved = resolve_snip_qc_flag_sources(
        DEFAULT_SNIP_QC_EXCLUSION_REASONS,
        output_root=Path("/data"),
        experiment_id="exp01",
        well_id="A01",
    )
    steps = {src.step for src in resolved}
    assert steps == {"death_detection_qc", "surface_area_qc", "mask_quality_qc"}


def test_unregistered_step_fails_loud():
    fake_payloads = {"not_a_real_step": ("some_flag",)}
    with patch("data_pipeline.quality_control.snip_qc.flag_input_resolver._SOURCE_PAYLOADS", fake_payloads):
        with pytest.raises(ValueError, match="not_a_real_step"):
            resolve_snip_qc_flag_sources(
                {"r": "some_flag"},
                output_root=Path("/data"),
                experiment_id="exp01",
                well_id="A01",
            )


# ─────────────────────────────────────────────────────────────────────────────
# ResolvedFlagSource serialization
# ─────────────────────────────────────────────────────────────────────────────

def test_resolved_flag_source_roundtrip():
    src = ResolvedFlagSource(
        step="mask_quality_qc",
        artifact_key="mask_quality_qc",
        flag_columns=("edge_flag", "discontinuous_mask_flag"),
        path=Path("/data/qc/A01_mask_quality_qc.csv"),
    )
    assert ResolvedFlagSource.from_dict(src.to_dict()) == src


def test_resolved_sources_json_reflects_exclusion_policy():
    reasons = {"edge": "edge_flag"}
    resolved = resolve_snip_qc_flag_sources(
        reasons,
        output_root=Path("/data"),
        experiment_id="exp01",
        well_id="A01",
    )
    assert len(resolved) == 1
    assert resolved[0].step == "mask_quality_qc"
    assert resolved[0].flag_columns == ("edge_flag",)


# ─────────────────────────────────────────────────────────────────────────────
# Kingdom boundary — only flag_input_resolver.py may import pipeline_orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def test_no_orchestration_import_outside_resolver():
    snip_qc_src = Path(__file__).parents[4] / "src" / "data_pipeline" / "quality_control" / "snip_qc"
    violations = []
    for py_file in snip_qc_src.glob("*.py"):
        if py_file.name == "flag_input_resolver.py":
            continue
        text = py_file.read_text()
        if "pipeline_orchestrator" in text:
            violations.append(py_file.name)
    assert violations == [], (
        f"Kingdom violation: the following snip_qc modules import pipeline_orchestrator "
        f"but only flag_input_resolver.py is permitted to do so: {violations}"
    )
