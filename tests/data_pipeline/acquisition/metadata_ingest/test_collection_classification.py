"""Tests for the collection-classify artifact (CLASSIFY ONCE, CONSUME EVERYWHERE).

Pins the CONTRACT of the early-DAG classify fact:
  - a collection → is_collection True, its sources, and start_age_by_time_index whose time_index
    keys match the acquisition union ordering (SourceChild.sort_key: declared_hpf then date);
  - a single experiment → the inert is_collection False payload;
  - the writer round-trips through the validator.

Uses tmp_path _coll dirs (never real data). Run:
  PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_classification.py
"""

from __future__ import annotations

import json

import pytest

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import SourceChild
from data_pipeline.acquisition.metadata_ingest.collection_classification import (
    classify_experiment,
    read_collection_classification,
    write_collection_classification,
)
from data_pipeline.acquisition.metadata_ingest.collection_classification_contract import (
    validate_collection_classification,
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
    payload = classify_experiment("chem28c_coll_plate01", tmp_path)
    assert payload["experiment_id"] == "chem28c_coll_plate01"
    assert payload["is_collection"] is True
    # sources ordered by SourceChild.sort_key (declared_hpf then date): 28 before 52.
    assert payload["sources"] == ["20250622_plate01_t28hpf", "20250623_plate01_t52hpf"]
    # time_index keys are the block ordinals of THAT ordering — matches the acquisition union.
    assert payload["start_age_by_time_index"] == {"0": 28, "1": 52}


def test_classify_time_index_matches_acquisition_union_ordering(tmp_path):
    """The critical correctness point: the classify time_index ordinals equal the union's.

    Both this module and collection_acquisition_union sort by SourceChild.sort_key, so the block
    ordinal a source is assigned here is the SAME time_index the union stamps. We assert the
    classify keys against an independent re-derivation via SourceChild.sort_key.
    """
    children = ["20250623_plate01_t52hpf", "20250622_plate01_t28hpf"]  # unsorted on disk
    _make_collection(tmp_path, "chem28c_coll", children)
    payload = classify_experiment("chem28c_coll_plate01", tmp_path)

    ordered = sorted(
        (SourceChild(child_name=c, scope="Keyence") for c in children),
        key=SourceChild.sort_key,
    )
    expected = {str(i): s.declared_hpf for i, s in enumerate(ordered)}
    assert payload["start_age_by_time_index"] == expected
    assert payload["sources"] == [s.child_name for s in ordered]


def test_classify_undeclared_age_is_null_not_absent(tmp_path):
    # A source with no t<NN>hpf token (sci-style) declares no age → its entry is null, honestly.
    _make_collection(
        tmp_path,
        "sci_snaps_coll",
        ["20250622_plate01_t28hpf", "20250623_plate01_sci"],
    )
    payload = classify_experiment("sci_snaps_coll_plate01", tmp_path)
    # 28 declared sorts first (ordinal 0); undeclared sorts last (ordinal 1) with a null age.
    assert payload["start_age_by_time_index"] == {"0": 28, "1": None}


# ── Single (non-collection): inert payload ─────────────────────────────────────

def test_classify_single_experiment_is_inert(tmp_path):
    payload = classify_experiment("20250912", tmp_path)
    assert payload == {
        "experiment_id": "20250912",
        "is_collection": False,
        "sources": [],
        "start_age_by_time_index": {},
    }


# ── Writer round-trips through the validator ───────────────────────────────────

def test_write_and_read_round_trip(tmp_path):
    _make_collection(
        tmp_path, "chem28c_coll", ["20250622_plate01_t28hpf", "20250623_plate01_t52hpf"]
    )
    out = tmp_path / "out" / "collection_classification.json"
    written = write_collection_classification(
        experiment_id="chem28c_coll_plate01",
        raw_root=tmp_path,
        microscope="Keyence",
        output_json=out,
    )
    on_disk = json.loads(out.read_text())
    assert on_disk == written
    assert read_collection_classification(out) == written


# ── Contract validator: fail-loud shape checks ─────────────────────────────────

def test_validator_rejects_non_bool_is_collection():
    with pytest.raises(ValueError, match="is_collection.*bool"):
        validate_collection_classification({
            "experiment_id": "x", "is_collection": "true",
            "sources": [], "start_age_by_time_index": {},
        })


def test_validator_rejects_non_int_age_key():
    with pytest.raises(ValueError, match="stringified ints"):
        validate_collection_classification({
            "experiment_id": "x", "is_collection": True,
            "sources": ["20250622_plate01_t28hpf"],
            "start_age_by_time_index": {"first": 28},
        })


def test_validator_rejects_missing_key():
    with pytest.raises(ValueError, match="missing required key"):
        validate_collection_classification({"experiment_id": "x", "is_collection": False})


def test_validator_rejects_noninert_single():
    with pytest.raises(ValueError, match="must be inert"):
        validate_collection_classification({
            "experiment_id": "x", "is_collection": False,
            "sources": ["20250622_plate01_t28hpf"], "start_age_by_time_index": {},
        })
