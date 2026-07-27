"""Guard against experiment merges silently unioning drifted well-shard schemas."""

from __future__ import annotations

from pathlib import Path

import pytest


_RULES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "data_pipeline"
    / "pipeline_orchestrator"
    / "rules"
)


@pytest.mark.parametrize("smk_path", sorted(_RULES_DIR.glob("*.smk")), ids=lambda p: p.name)
def test_concat_well_shards_calls_supply_owner_contract(smk_path: Path) -> None:
    for line_number, line in enumerate(smk_path.read_text(encoding="utf-8").splitlines(), start=1):
        if "concat_well_shards_to_file(" not in line:
            continue
        assert "required_columns=" in line, (
            f"{smk_path.name}:{line_number}: shard merge omits required_columns. "
            "Import the product's canonical contract columns and pass them to the merge so "
            "pd.concat cannot silently union a drifted shard schema."
        )


def test_at_least_one_well_shard_merge_scanned() -> None:
    assert any(
        "concat_well_shards_to_file(" in path.read_text(encoding="utf-8")
        for path in _RULES_DIR.glob("*.smk")
    )
