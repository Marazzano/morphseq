"""Repo-wide guard: a report is a DAG leaf in both the import graph and the Snakemake DAG.

See docs/refactors/streamline-snakemake/target/specs/viz/report_world.md ("Done When"):
  1. no module imports a `*/report.py` module — a report is terminal in the import graph;
  2. no non-report PIPELINE_STEPS rule's `input:` names a `*_report` artifact filename — a report
     is terminal in the DAG.

`pipeline_orchestrator/tasks.py` is exempt from rule 1: it is the thin CLI adapter every rule
shells out through (the same role it plays for every product's `entrypoint.py`), explicitly
named in report_world.md's own worked example ("the `tasks.py` subcommand... calls `report.py`").
The doctrine's target is the PRODUCT import graph — no report imported by another report, an
entrypoint, or product code — not the one designated dispatcher.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from data_pipeline.pipeline_orchestrator.orchestration import PIPELINE_STEPS

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "data_pipeline"
_RULES_DIR = _SRC_ROOT / "pipeline_orchestrator" / "rules"

_REPORT_STEPS = {step: spec for step, spec in PIPELINE_STEPS.items() if step.endswith("_report")}

# Every artifact FILENAME TEMPLATE across all report steps (e.g. "{experiment_id}_mortality_curtain.png").
_REPORT_ARTIFACT_TEMPLATES = [
    template
    for spec in _REPORT_STEPS.values()
    for template in spec["artifacts"].values()
]

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)

# The designated CLI adapter: every rule shells out through tasks.py, which resolves the rule's
# input paths and calls report.py::build_*_report(...) — the same role it plays for entrypoint.py.
_ADAPTER_PATH = _SRC_ROOT / "pipeline_orchestrator" / "tasks.py"


def _report_py_modules() -> list[str]:
    """Dotted module names of every report.py under src/data_pipeline (excluding itself when scanned)."""
    modules = []
    for path in _SRC_ROOT.rglob("report.py"):
        rel = path.relative_to(_SRC_ROOT.parent).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def test_no_module_imports_a_report_py() -> None:
    report_modules = _report_py_modules()
    assert report_modules, "expected at least one report.py under src/data_pipeline"

    violations = []
    for py_file in _SRC_ROOT.rglob("*.py"):
        if py_file.name == "report.py":
            continue  # a report.py may import its own contract/viz deps, not another report.py
        if py_file == _ADAPTER_PATH:
            continue  # the one designated CLI adapter — see module docstring
        text = py_file.read_text(encoding="utf-8")
        for imported in _IMPORT_RE.findall(text):
            if imported in report_modules or any(
                imported.startswith(m + ".") for m in report_modules
            ):
                violations.append(f"{py_file}: imports {imported!r}")

    assert not violations, "report.py is a terminal leaf; found imports:\n" + "\n".join(violations)


@pytest.mark.parametrize("smk_path", sorted(_RULES_DIR.glob("*.smk")), ids=lambda p: p.name)
def test_no_smk_rule_consumes_a_report_artifact(smk_path: Path) -> None:
    text = smk_path.read_text(encoding="utf-8")
    # Split into rule blocks; a report step's OWN rule may legitimately declare its artifacts as
    # `output:` (that's not consumption) — only check for report artifact filenames appearing
    # inside a DIFFERENT rule's `input:` block.
    rule_blocks = re.split(r"\n(?=rule \w+:)", text)
    for block in rule_blocks:
        header_match = re.match(r"rule (\w+):", block)
        rule_name = header_match.group(1) if header_match else None
        if rule_name in _REPORT_STEPS:
            continue  # a report rule's own block declares its artifacts as output, not input
        input_match = re.search(r"input:\s*\n(.*?)\n(?:output:|params:|shell:|run:)", block, re.DOTALL)
        if not input_match:
            continue
        input_block = input_match.group(1)
        for template in _REPORT_ARTIFACT_TEMPLATES:
            # Filenames are format-string templates; check the literal suffix survives (the part
            # after the last "{...}" placeholder), which is enough to catch a raw reference.
            literal_suffix = re.sub(r"\{[^}]+\}", "", template)
            if literal_suffix and literal_suffix in input_block:
                pytest.fail(
                    f"{smk_path.name}: rule {rule_name!r} input: references report artifact "
                    f"{template!r} — reports are terminal leaves, consumed by nothing"
                )
