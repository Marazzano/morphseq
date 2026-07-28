"""Smoke tests for well_discovery/.

Run with: PYTHONPATH=src pytest tests/data_pipeline/metadata_ingest/test_well_discovery.py
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.pipeline_orchestrator import tasks
from data_pipeline.acquisition.metadata_ingest.well_discovery.discovered_wells_contract import (
    read_discovered_wells,
    validate_discovered_wells,
    write_discovered_wells,
)
from data_pipeline.acquisition.metadata_ingest.well_discovery.discover_wells_from_scope_metadata import (
    discover_wells_from_scope_metadata,
)


# ── contract helpers ─────────────────────────────────────────────────────────


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    wells = ["20250912_A01", "20250912_B02"]
    out = tmp_path / "discovered_wells.txt"
    write_discovered_wells(out, wells)
    assert read_discovered_wells(out) == wells


def test_write_creates_parent(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "discovered_wells.txt"
    write_discovered_wells(out, ["20250912_A01"])
    assert out.exists()


def test_validate_accepts_global_ids() -> None:
    validate_discovered_wells(["20250912_A01", "20250912_B02"])  # no raise


def test_validate_rejects_bare_well_index() -> None:
    with pytest.raises(ValueError, match="bare LOCAL"):
        validate_discovered_wells(["A01"])


def test_validate_rejects_id_without_underscore() -> None:
    with pytest.raises(ValueError, match="not global"):
        validate_discovered_wells(["nounderscorehere"])


def test_validate_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_discovered_wells([""])


# ── discover_wells_from_scope_metadata ───────────────────────────────────────


def _write_mapped_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_basic_discovery(tmp_path: Path) -> None:
    csv = tmp_path / "mapped.csv"
    _write_mapped_csv(csv, [
        {"well_id": "20250912_A01", "other": 1},
        {"well_id": "20250912_B02", "other": 2},
    ])
    out = tmp_path / "discovered_wells.txt"
    discover_wells_from_scope_metadata(csv, out)
    assert read_discovered_wells(out) == ["20250912_A01", "20250912_B02"]


def test_deduplication_preserves_first_encounter_order(tmp_path: Path) -> None:
    csv = tmp_path / "mapped.csv"
    _write_mapped_csv(csv, [
        {"well_id": "20250912_B02"},
        {"well_id": "20250912_A01"},
        {"well_id": "20250912_B02"},  # duplicate — should be dropped
    ])
    out = tmp_path / "discovered_wells.txt"
    discover_wells_from_scope_metadata(csv, out)
    assert read_discovered_wells(out) == ["20250912_B02", "20250912_A01"]


def test_missing_well_id_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "mapped.csv"
    pd.DataFrame([{"not_well_id": "x"}]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing required well_id column"):
        discover_wells_from_scope_metadata(csv, tmp_path / "out.txt")


def test_null_well_ids_skipped(tmp_path: Path) -> None:
    csv = tmp_path / "mapped.csv"
    _write_mapped_csv(csv, [
        {"well_id": "20250912_A01"},
        {"well_id": None},
        {"well_id": "20250912_B02"},
    ])
    out = tmp_path / "discovered_wells.txt"
    discover_wells_from_scope_metadata(csv, out)
    assert read_discovered_wells(out) == ["20250912_A01", "20250912_B02"]


def test_bare_well_index_in_csv_raises(tmp_path: Path) -> None:
    """A01 in the mapped CSV means well_id was never promoted — should fail."""
    csv = tmp_path / "mapped.csv"
    _write_mapped_csv(csv, [{"well_id": "A01"}])
    with pytest.raises(ValueError, match="bare LOCAL"):
        discover_wells_from_scope_metadata(csv, tmp_path / "out.txt")


# ── tasks.py structural guard ─────────────────────────────────────────────────


def test_tasks_cmd_discover_wells_has_no_business_logic() -> None:
    """tasks.py handler must not contain pandas or inline well logic."""
    body = inspect.getsource(tasks.cmd_discover_wells)

    assert "import pandas" not in body, "cmd_discover_wells must not import pandas"
    assert "pd.read_csv" not in body, "cmd_discover_wells must not call pd.read_csv"
    assert "well_id" not in body, "cmd_discover_wells must not reference well_id directly"
