"""Tests for the collection-classify artifact (CLASSIFY ONCE, CONSUME EVERYWHERE).

Pins the CONTRACT of the early-DAG classify fact:
  - a collection → is_collection True, its sources, and start_age_by_time_index whose time_index
    keys match the acquisition union ordering (PlateSource.sort_key: declared_hpf then date);
  - a single experiment → the inert is_collection False payload;
  - the writer round-trips through the validator.

Uses tmp_path _coll dirs (never real data). Run:
  PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_provenance.py
"""

from __future__ import annotations

import json

import pytest

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import PlateSource
from data_pipeline.acquisition.metadata_ingest.collection_provenance import (
    build_collection_provenance,
    read_collection_provenance,
    write_collection_provenance,
)
from data_pipeline.acquisition.metadata_ingest.collection_provenance_contract import (
    validate_collection_provenance,
)


def _make_collection(raw_root, coll_name, children):
    coll_dir = raw_root / coll_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    for child in children:
        (coll_dir / child).mkdir()
    return coll_dir


# ── Collection: is_collection True + sources + age map ─────────────────────────

def test_classify_collection_declares_sources_and_age_map(tmp_path):
    _make_collection(
        tmp_path,
        "chem28c_coll",
        ["20250622_plate01_t28hpf", "20250623_plate01_t52hpf",
         "20250622_plate02_t28hpf"],  # a different plate — excluded
    )
    payload = build_collection_provenance("chem28c_coll_plate01", tmp_path)
    assert payload["experiment_id"] == "chem28c_coll_plate01"
    assert payload["is_collection"] is True
    # sources are PROVENANCE RECORDS ordered by PlateSource.sort_key (declared_hpf then date):
    # 28 before 52. Each carries file / raw_path / declared_hpf / time_index.
    assert [s["file"] for s in payload["sources"]] == [
        "20250622_plate01_t28hpf", "20250623_plate01_t52hpf"
    ]
    assert [s["time_index"] for s in payload["sources"]] == [0, 1]
    assert [s["declared_hpf"] for s in payload["sources"]] == [28, 52]
    assert all(s["file"] in s["raw_path"] for s in payload["sources"])  # raw_path points at the file
    # time_index keys are the block ordinals of THAT ordering — matches the acquisition union.
    assert payload["start_age_by_time_index"] == {"0": 28, "1": 52}


def test_classify_time_index_matches_acquisition_union_ordering(tmp_path):
    """The critical correctness point: the classify time_index ordinals equal the union's.

    Both this module and collection_acquisition_union sort by PlateSource.sort_key, so the block
    ordinal a source is assigned here is the SAME time_index the union stamps. We assert the
    classify keys against an independent re-derivation via PlateSource.sort_key.
    """
    children = ["20250623_plate01_t52hpf", "20250622_plate01_t28hpf"]  # unsorted on disk
    _make_collection(tmp_path, "chem28c_coll", children)
    payload = build_collection_provenance("chem28c_coll_plate01", tmp_path)

    ordered = sorted(
        (PlateSource(source_id=c, scope="Keyence") for c in children),
        key=PlateSource.sort_key,
    )
    expected = {str(i): s.declared_hpf for i, s in enumerate(ordered)}
    assert payload["start_age_by_time_index"] == expected
    assert [s["file"] for s in payload["sources"]] == [s.source_id for s in ordered]
    assert [s["time_index"] for s in payload["sources"]] == list(range(len(ordered)))


def test_classify_undeclared_age_is_null_not_absent(tmp_path):
    # A source with no t<NN>hpf token (sci-style) declares no age → its entry is null, honestly.
    _make_collection(
        tmp_path,
        "sci_snaps_coll",
        ["20250622_plate01_t28hpf", "20250623_plate01_sci"],
    )
    payload = build_collection_provenance("sci_snaps_coll_plate01", tmp_path)
    # 28 declared sorts first (ordinal 0); undeclared sorts last (ordinal 1) with a null age.
    assert payload["start_age_by_time_index"] == {"0": 28, "1": None}


# ── Single (non-collection): inert payload ─────────────────────────────────────

def test_classify_single_experiment_is_inert(tmp_path):
    payload = build_collection_provenance("20250912", tmp_path)
    assert payload == {
        "experiment_id": "20250912",
        "is_collection": False,
        "sources": [],
        "start_age_by_source_ordinal": {},
        "start_age_by_time_index": {},
    }


# ── Writer round-trips through the validator ───────────────────────────────────

def test_write_and_read_round_trip(tmp_path):
    _make_collection(
        tmp_path, "chem28c_coll", ["20250622_plate01_t28hpf", "20250623_plate01_t52hpf"]
    )
    out = tmp_path / "out" / "collection_provenance.json"
    written = write_collection_provenance(
        experiment_id="chem28c_coll_plate01",
        raw_root=tmp_path,
        microscope="Keyence",
        output_json=out,
    )
    on_disk = json.loads(out.read_text())
    assert on_disk == written
    assert read_collection_provenance(out) == written


# ── Contract validator: fail-loud shape checks ─────────────────────────────────

def test_validator_rejects_non_bool_is_collection():
    with pytest.raises(ValueError, match="is_collection.*bool"):
        validate_collection_provenance({
            "experiment_id": "x", "is_collection": "true",
            "sources": [], "start_age_by_source_ordinal": {},
            "start_age_by_time_index": {},
        })


def test_validator_rejects_non_int_age_key():
    with pytest.raises(ValueError, match="stringified ints"):
        validate_collection_provenance({
            "experiment_id": "x", "is_collection": True,
            "sources": [{"file": "20250622_plate01_t28hpf", "raw_path": "/r/t28",
                         "declared_hpf": 28, "source_ordinal": 0, "time_index": 0}],
            "start_age_by_source_ordinal": {"first": 28},
            "start_age_by_time_index": {"first": 28},
        })


def test_validator_rejects_missing_key():
    with pytest.raises(ValueError, match="missing required key"):
        validate_collection_provenance({"experiment_id": "x", "is_collection": False})


def test_validator_rejects_noninert_single():
    with pytest.raises(ValueError, match="must be inert"):
        validate_collection_provenance({
            "experiment_id": "x", "is_collection": False,
            "sources": [{"file": "20250622_plate01_t28hpf", "raw_path": "/r/t28",
                         "declared_hpf": 28, "source_ordinal": 0, "time_index": 0}],
            "start_age_by_source_ordinal": {},
            "start_age_by_time_index": {},
        })
