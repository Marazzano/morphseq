"""Contract tests for the snip image-source seam (collect_snip_inputs).

Pins: manifest-relative path resolution, the is_valid_snip gate, manifest order,
and the fail-loud paths (missing CSV / missing column / missing image). Includes a
smoke against the REAL one-well snip_inventory under tests/improvements/ when present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.feature_extraction.legacy_embeddings.snip_source import (
    collect_snip_inputs,
)

REAL_MANIFEST = (
    Path(__file__).resolve().parents[4]
    / "tests/improvements/snip_processing_smoke/output_real/snip_inventory.csv"
)


def _write_manifest(tmp_path, rows, *, make_images=True):
    """Write a snip_inventory CSV + the PNG files it points at (relative paths)."""
    df = pd.DataFrame(rows)
    csv = tmp_path / "snip_inventory.csv"
    df.to_csv(csv, index=False)
    if make_images:
        for rel in df["processed_snip_path"]:
            img = tmp_path / rel
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"\x89PNG\r\n")  # not a real PNG; existence is all we check
    return csv


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Resolution + gating — the worked examples
# ─────────────────────────────────────────────────────────────────────────────────────────────

def test_resolves_relative_paths_against_manifest_dir_in_order(tmp_path):
    csv = _write_manifest(
        tmp_path,
        [
            {"snip_id": "e01_BF_t0000", "processed_snip_path": "snips/e01/a.png", "is_valid_snip": True},
            {"snip_id": "e01_BF_t0001", "processed_snip_path": "snips/e01/b.png", "is_valid_snip": True},
        ],
    )

    inputs = collect_snip_inputs(csv)

    assert [s.snip_id for s in inputs] == ["e01_BF_t0000", "e01_BF_t0001"]
    assert inputs[0].image_path == tmp_path / "snips/e01/a.png"
    assert inputs[0].image_path.is_absolute()


def test_valid_only_gate_drops_invalid_snips(tmp_path):
    csv = _write_manifest(
        tmp_path,
        [
            {"snip_id": "good", "processed_snip_path": "snips/good.png", "is_valid_snip": True},
            {"snip_id": "bad", "processed_snip_path": "snips/bad.png", "is_valid_snip": False},
        ],
    )

    assert [s.snip_id for s in collect_snip_inputs(csv)] == ["good"]
    assert {s.snip_id for s in collect_snip_inputs(csv, valid_only=False)} == {"good", "bad"}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Fail-loud — each boundary names what's wrong AND where
# ─────────────────────────────────────────────────────────────────────────────────────────────

def test_missing_csv_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="snip_inventory CSV not found"):
        collect_snip_inputs(tmp_path / "nope.csv")


def test_missing_required_column_fails_loud(tmp_path):
    csv = tmp_path / "snip_inventory.csv"
    pd.DataFrame([{"snip_id": "x"}]).to_csv(csv, index=False)
    with pytest.raises(KeyError, match="processed_snip_path"):
        collect_snip_inputs(csv)


def test_missing_image_file_fails_loud_naming_snip(tmp_path):
    csv = _write_manifest(
        tmp_path,
        [{"snip_id": "ghost", "processed_snip_path": "snips/ghost.png", "is_valid_snip": True}],
        make_images=False,
    )
    with pytest.raises(FileNotFoundError, match="ghost"):
        collect_snip_inputs(csv)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Real one-well smoke (skipped if the fixture isn't present)
# ─────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="real snip_inventory fixture absent")
def test_real_one_well_manifest_resolves_existing_pngs():
    inputs = collect_snip_inputs(REAL_MANIFEST)

    assert len(inputs) == 3
    assert all(s.image_path.exists() for s in inputs)
    assert inputs[0].snip_id == "20250912_B01_e01_BF_t0000"
