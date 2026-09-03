"""Tests for the run-policy guard.

The values asserted here mirror profiles/default/config.yaml. If that file's policy changes on
purpose, these tests are meant to fail and be updated alongside it.
"""

from __future__ import annotations

import pytest

from data_pipeline.pipeline_orchestrator.orchestration.run_policy import check_run_policy


class FakeWorkflow:
    """Stand-in for the Snakemake Workflow object, carrying only what the guard reads."""

    def __init__(
        self,
        *,
        rerun_triggers=frozenset({"mtime"}),
        global_resources=None,
    ):
        self.rerun_triggers = rerun_triggers
        self.global_resources = (
            {"gpu": 1, "_cores": 8, "_nodes": None} if global_resources is None else global_resources
        )


def test_compliant_workflow_passes():
    assert check_run_policy(FakeWorkflow()) == []


def test_snakemake_defaults_are_rejected():
    """The exact state a run gets when the profile is not found: every value reverts."""
    defaults = FakeWorkflow(
        rerun_triggers=frozenset({"mtime", "params", "input", "code", "software-env"}),
        global_resources={"_cores": 1, "_nodes": None},
    )
    with pytest.raises(RuntimeError) as excinfo:
        check_run_policy(defaults)

    message = str(excinfo.value)
    assert "rerun_triggers" in message
    assert "gpu" in message
    # The error has to say how to fix it, not just what is wrong.
    assert "--profile" in message


def test_missing_gpu_budget_is_rejected():
    """A rule's `resources: gpu=1` is inert without this budget, so its absence must fail."""
    with pytest.raises(RuntimeError, match="gpu"):
        check_run_policy(FakeWorkflow(global_resources={"_cores": 8}))


def test_broad_rerun_triggers_are_rejected():
    """`code` in the trigger set re-runs SAM2 across every well after a docstring edit."""
    with pytest.raises(RuntimeError, match="rerun_triggers"):
        check_run_policy(FakeWorkflow(rerun_triggers=frozenset({"mtime", "code"})))


def test_strict_false_reports_without_raising():
    problems = check_run_policy(
        FakeWorkflow(rerun_triggers=frozenset({"mtime", "code"})), strict=False
    )
    assert len(problems) == 1
    assert "rerun_triggers" in problems[0]
