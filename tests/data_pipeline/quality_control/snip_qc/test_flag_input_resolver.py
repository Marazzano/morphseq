"""Tests for snip_qc/flag_input_resolver.py.

Covers: resolver correctness, error messages, kingdom boundary enforcement.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipeline.quality_control.snip_qc.contract import SNIP_QC_EXCLUSION_FLAGS
from data_pipeline.quality_control.snip_qc.flag_input_resolver import (
    ResolvedFlagSource,
    _SOURCE_PAYLOADS,
    _build_flag_column_index,
    resolve_snip_qc_flag_sources,
    validate_snip_qc_product_requirements,
)


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def test_default_flags_resolve_cleanly():
    index = _build_flag_column_index(SNIP_QC_EXCLUSION_FLAGS)
    assert set(index) == set(SNIP_QC_EXCLUSION_FLAGS)
    assert all(step in _SOURCE_PAYLOADS for step in index.values())


def test_only_needed_steps_returned():
    flags = ("edge_flag",)
    index = _build_flag_column_index(flags)
    assert index == {"edge_flag": "mask_quality_qc"}


def test_unknown_flag_fails_loud_with_flag_name():
    flags = ("nonexistent_flag",)
    with pytest.raises(ValueError, match="nonexistent_flag"):
        _build_flag_column_index(flags)


def test_unknown_flag_error_names_eligible_steps():
    flags = ("no_such_flag",)
    with pytest.raises(ValueError, match="eligible steps"):
        _build_flag_column_index(flags)


def test_motion_blur_flag_requires_z_stack_product():
    with pytest.raises(ValueError) as exc_info:
        validate_snip_qc_product_requirements(
            ("edge_flag", "motion_blur_flag"),
            available_product_keys=("BF__projection__focus_stack",),
        )
    message = str(exc_info.value)
    assert "motion_blur_flag" in message
    assert "BF__z_stack" in message
    assert "snip_qc.exclusion_flags" in message


def test_motion_blur_flag_accepts_configured_z_stack_product():
    validate_snip_qc_product_requirements(
        ("motion_blur_flag",),
        available_product_keys=("BF__projection__focus_stack", "BF__z_stack"),
    )


def test_removing_motion_blur_flag_does_not_require_z_stack():
    validate_snip_qc_product_requirements(
        ("edge_flag", "focus_flag"),
        available_product_keys=("BF__projection__focus_stack",),
    )


def test_ambiguous_flag_lists_all_claimants():
    fake_payloads = {
        "step_a": ("shared_flag",),
        "step_b": ("shared_flag",),
    }
    with patch("data_pipeline.quality_control.snip_qc.flag_input_resolver._SOURCE_PAYLOADS", fake_payloads):
        with pytest.raises(ValueError, match="ambiguous"):
            _build_flag_column_index(("shared_flag",))


def test_multiple_errors_reported_together():
    flags = ("missing_1", "missing_2")
    with pytest.raises(ValueError) as exc_info:
        _build_flag_column_index(flags)
    msg = str(exc_info.value)
    assert "missing_1" in msg
    assert "missing_2" in msg


# ─────────────────────────────────────────────────────────────────────────────
# resolve_snip_qc_flag_sources
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_returns_one_source_per_step():
    flags = ("edge_flag", "discontinuous_mask_flag")
    resolved = resolve_snip_qc_flag_sources(
        flags,
        output_root=Path("/data"),
        experiment_id="exp01",
        well_id="A01",
    )
    assert len(resolved) == 1
    assert resolved[0].step == "mask_quality_qc"
    assert set(resolved[0].flag_columns) == {"edge_flag", "discontinuous_mask_flag"}


def test_resolve_groups_flags_by_step():
    resolved = resolve_snip_qc_flag_sources(
        SNIP_QC_EXCLUSION_FLAGS,
        output_root=Path("/data"),
        experiment_id="exp01",
        well_id="A01",
    )
    steps = {src.step for src in resolved}
    assert steps == {
        "death_detection_qc",
        "surface_area_qc",
        "mask_quality_qc",
        "focus_qc",
        "motion_blur_qc",
    }


def test_unregistered_step_fails_loud():
    fake_payloads = {"not_a_real_step": ("some_flag",)}
    with patch("data_pipeline.quality_control.snip_qc.flag_input_resolver._SOURCE_PAYLOADS", fake_payloads):
        with pytest.raises(ValueError, match="not_a_real_step"):
            resolve_snip_qc_flag_sources(
                ("some_flag",),
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
    flags = ("edge_flag",)
    resolved = resolve_snip_qc_flag_sources(
        flags,
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
