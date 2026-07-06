"""Repo-wide guard: the merged run config is NEVER a Snakemake rule `input:`.

See docs/refactors/streamline-snakemake/target/specs/pipeline_file_philosophy.md
("Config is a `params:` value, never a file `input:`").

Under Snakemake's default `--rerun-triggers`, a file `input:`'s **mtime** is a rebuild trigger.
`merged_config.yaml` is rewritten (even content-identically) at plan time, so any rule that lists it
as an `input:` gets its whole downstream subtree invalidated on essentially every invocation — the
config-touch cascade we hit and fixed. Config must instead be passed as a `params:` value (a constant
path string never re-triggers), or resolved into a content-guarded plan artifact (the
`resolved_product_plan` pattern) that downstream rules depend on. This test pins that so the leak
cannot be silently reintroduced.

The `CONFIG_YAML` symbol (the merged config path) is defined in the Snakefile as
`RUNTIME_CONFIG_DIR / "merged_config.yaml"`; rules reference it as `str(CONFIG_YAML)`. We scan each
rule's `input:` block for that reference (and the literal `merged_config.yaml` filename, in case a
future rule hardcodes the path — itself a separate paths.py violation, but caught here too).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "data_pipeline"
_RULES_DIR = _SRC_ROOT / "pipeline_orchestrator" / "rules"

# Ways the merged config path shows up in a rule: the Snakefile symbol, or the raw filename.
_CONFIG_INPUT_MARKERS = ("CONFIG_YAML", "merged_config.yaml")


def _rule_input_blocks(text: str) -> list[tuple[str, str]]:
    """(rule_name, input_block_text) for every rule that has an `input:` section."""
    blocks = []
    for block in re.split(r"\n(?=rule \w+:)", text):
        header = re.match(r"rule (\w+):", block)
        if not header:
            continue
        input_match = re.search(
            r"\n\s*input:\s*\n(.*?)\n\s*(?:output:|params:|shell:|run:|script:)",
            block,
            re.DOTALL,
        )
        if input_match:
            blocks.append((header.group(1), input_match.group(1)))
    return blocks


@pytest.mark.parametrize("smk_path", sorted(_RULES_DIR.glob("*.smk")), ids=lambda p: p.name)
def test_no_smk_rule_takes_config_as_input(smk_path: Path) -> None:
    text = smk_path.read_text(encoding="utf-8")
    for rule_name, input_block in _rule_input_blocks(text):
        for marker in _CONFIG_INPUT_MARKERS:
            if marker in input_block:
                pytest.fail(
                    f"{smk_path.name}: rule {rule_name!r} lists the merged config ({marker!r}) in "
                    f"its `input:` block. Config must be a `params:` value (or behind a "
                    f"content-guarded resolved plan), never a file input — a config-file mtime "
                    f"bump would otherwise cascade a rebuild through the whole DAG. See "
                    f"pipeline_file_philosophy.md ('Config is a `params:` value, never a file "
                    f"`input:`')."
                )


def test_at_least_one_rules_file_scanned() -> None:
    # Guard against the glob silently matching nothing (which would make the parametrized test
    # vacuously pass and hide a real regression).
    assert list(_RULES_DIR.glob("*.smk")), f"no .smk rule files found under {_RULES_DIR}"
