"""Step 5 — drop-in discovery + split twins."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.metadata_ingest.well_discovery.discover_wells_from_handoff import (
    discover_wells_from_handoff,
)
from data_pipeline.metadata_ingest.well_discovery.split_dropin_inventory import (
    select_dropin_well_shard,
    split_dropin_inventory_by_well,
)
from data_pipeline.metadata_ingest.well_discovery.discovered_wells_contract import (
    read_discovered_wells,
)
from data_pipeline.shared.identifiers import build_well_id


def _manifest(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(exp: str, well_index: str, t: int) -> dict:
    return {
        "experiment_id": exp,
        "well_index": well_index,
        "channel_id": "BF",
        "time_index": t,
        "source_image_path": f"imgs/{exp}_{well_index}_BF_t{t:04d}.png",
        "source_micrometers_per_pixel": 0.75,
        "image_width_px": 16,
        "image_height_px": 16,
    }


# --- discover_wells_from_handoff ------------------------------------------------------------

def test_discover_emits_global_well_ids_in_order(tmp_path):
    manifest = tmp_path / "dropin_frame_inventory.csv"
    _manifest([_row("20250912", "A01", 0), _row("20250912", "B01", 0),
               _row("20250912", "A01", 1)]).to_csv(manifest, index=False)
    out = tmp_path / "discovered_wells.txt"
    discover_wells_from_handoff(manifest, out)
    wells = read_discovered_wells(out)
    assert wells == [build_well_id("20250912", "A01"), build_well_id("20250912", "B01")]


def test_discover_rejects_mixed_experiments(tmp_path):
    manifest = tmp_path / "dropin_frame_inventory.csv"
    _manifest([_row("20250912", "A01", 0), _row("20250101", "B01", 0)]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="exactly one experiment_id"):
        discover_wells_from_handoff(manifest, tmp_path / "out.txt")


def test_discover_missing_atom_fails(tmp_path):
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"experiment_id": ["20250912"]}).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="well_index"):
        discover_wells_from_handoff(manifest, tmp_path / "out.txt")


# --- split_dropin_inventory_by_well ---------------------------------------------------------

def test_split_writes_per_well_shards(tmp_path):
    manifest = tmp_path / "dropin_frame_inventory.csv"
    _manifest([_row("20250912", "A01", 0), _row("20250912", "A01", 1),
               _row("20250912", "B01", 0)]).to_csv(manifest, index=False)
    shards = split_dropin_inventory_by_well(manifest, tmp_path / "per_well")
    a01 = build_well_id("20250912", "A01")
    b01 = build_well_id("20250912", "B01")
    assert set(shards) == {a01, b01}
    assert shards[a01].name == f"{a01}_frame_inventory.csv"
    assert len(pd.read_csv(shards[a01])) == 2
    assert len(pd.read_csv(shards[b01])) == 1


def test_split_rejects_mixed_experiments(tmp_path):
    manifest = tmp_path / "m.csv"
    _manifest([_row("20250912", "A01", 0), _row("20250101", "B01", 0)]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="exactly one experiment_id"):
        split_dropin_inventory_by_well(manifest, tmp_path / "per_well")


# --- select_dropin_well_shard (the race-free per-well DAG producer) --------------------------

def test_select_writes_only_the_requested_well(tmp_path):
    manifest = tmp_path / "dropin_frame_inventory.csv"
    _manifest([_row("20250912", "A01", 0), _row("20250912", "A01", 1),
               _row("20250912", "B01", 0)]).to_csv(manifest, index=False)
    a01 = build_well_id("20250912", "A01")
    out = tmp_path / f"{a01}_frame_inventory.csv"
    select_dropin_well_shard(manifest, a01, out)
    written = pd.read_csv(out)
    assert len(written) == 2  # only A01's two rows — NOT the whole village
    assert set(written["well_index"]) == {"A01"}


def test_select_unknown_well_fails(tmp_path):
    manifest = tmp_path / "m.csv"
    _manifest([_row("20250912", "A01", 0)]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="resolve to well_id"):
        select_dropin_well_shard(manifest, build_well_id("20250912", "H12"), tmp_path / "out.csv")


def test_select_rejects_mixed_experiments(tmp_path):
    manifest = tmp_path / "m.csv"
    _manifest([_row("20250912", "A01", 0), _row("20250101", "B01", 0)]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="exactly one experiment_id"):
        select_dropin_well_shard(manifest, build_well_id("20250912", "A01"), tmp_path / "out.csv")
