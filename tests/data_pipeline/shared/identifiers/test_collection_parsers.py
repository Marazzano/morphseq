"""Tests for the experiment-collection parsers in shared/identifiers/.

Philosophy: exercise the public parsers only — no duplicating the child-name regex.
Identifier / collection strings are opaque outside the package; consumer code must
never string-split, so these are the sanctioned round-trip tests.

Run: PYTHONPATH=src pytest tests/data_pipeline/shared/identifiers/test_collection_parsers.py
"""

import pytest

from data_pipeline.shared.identifiers import (
    compose_collection_experiment_id,
    is_collection,
    is_collection_plate_id,
    parse_collection_name_from_plate_id,
    parse_declared_hpf,
    parse_event_label,
    parse_plate_token,
)


def test_is_collection_plate_id_detects_the_inner_marker():
    # A collection PLATE id has "_coll_" INSIDE it (distinct from is_collection's suffix test).
    assert is_collection_plate_id("cilia_snapshots_coll_plate01") is True
    assert is_collection_plate_id("chem28c_coll_plate02") is True
    # A single experiment id and a bare collection NAME are not plate ids.
    assert is_collection_plate_id("20240418") is False
    assert is_collection_plate_id("cilia_snapshots_coll") is False  # name, no inner _coll_


def test_parse_collection_name_from_plate_id_round_trips_with_compose():
    plate_id = compose_collection_experiment_id("cilia_snapshots_coll", "20260607_plate01_t45hpf")
    assert plate_id == "cilia_snapshots_coll_plate01"
    assert parse_collection_name_from_plate_id(plate_id) == "cilia_snapshots_coll"


def test_parse_collection_name_from_plate_id_rejects_non_plate_id():
    with pytest.raises(ValueError, match="not a collection plate id"):
        parse_collection_name_from_plate_id("20240418")

COLL = "cilia_snapshots_coll"


# ── is_collection ─────────────────────────────────────────────────────────────

def test_is_collection_true_on_coll_suffix():
    assert is_collection("cilia_snapshots_coll") is True


def test_is_collection_false_on_bare_id():
    assert is_collection("20260416_plate01_t02") is False


def test_is_collection_false_when_coll_is_interior_not_suffix():
    assert is_collection("coll_of_stuff") is False


def test_is_collection_ignores_surrounding_whitespace():
    assert is_collection("  cilia_snapshots_coll  ") is True


# ── parse_plate_token ─────────────────────────────────────────────────────────

def test_parse_plate_token_with_event():
    assert parse_plate_token("20260607_plate01_t45hpf") == "plate01"


def test_parse_plate_token_without_event():
    assert parse_plate_token("20260607_plate01") == "plate01"


def test_parse_plate_token_second_date_same_plate():
    # The whole point of the merge model: date is dropped, plate token is stable.
    assert parse_plate_token("20260608_plate01_t72hpf") == "plate01"


def test_parse_plate_token_rejects_missing_date():
    with pytest.raises(ValueError):
        parse_plate_token("plate01_t45hpf")


def test_parse_plate_token_rejects_short_date():
    with pytest.raises(ValueError):
        parse_plate_token("2026_plate01_t45hpf")


# ── parse_event_label ─────────────────────────────────────────────────────────

def test_parse_event_label_present():
    assert parse_event_label("20260607_plate01_t45hpf") == "t45hpf"


def test_parse_event_label_absent_is_none():
    assert parse_event_label("20260607_plate01") is None


def test_parse_event_label_non_hpf_event():
    assert parse_event_label("20260607_plate01_sci") == "sci"


def test_parse_event_label_rejects_bad_grammar():
    with pytest.raises(ValueError):
        parse_event_label("not_a_child")


# ── parse_declared_hpf ────────────────────────────────────────────────────────

def test_parse_declared_hpf_from_event_label():
    assert parse_declared_hpf("t45hpf") == 45


def test_parse_declared_hpf_from_full_child_name():
    assert parse_declared_hpf("20260607_plate01_t45hpf") == 45


def test_parse_declared_hpf_no_t_suffix_is_none():
    assert parse_declared_hpf("sci") is None


def test_parse_declared_hpf_child_without_event_is_none():
    assert parse_declared_hpf("20260607_plate01") is None


def test_parse_declared_hpf_none_input():
    assert parse_declared_hpf(None) is None


def test_parse_declared_hpf_two_digit():
    assert parse_declared_hpf("t72hpf") == 72


# ── compose_collection_experiment_id ──────────────────────────────────────────

def test_compose_drops_date_and_event():
    assert (
        compose_collection_experiment_id(COLL, "20260607_plate01_t45hpf")
        == "cilia_snapshots_coll_plate01"
    )


def test_compose_two_events_same_plate_share_id():
    # Different dates AND different t-events of one plate → ONE id (merge model).
    id_a = compose_collection_experiment_id(COLL, "20260607_plate01_t45hpf")
    id_b = compose_collection_experiment_id(COLL, "20260608_plate01_t72hpf")
    assert id_a == id_b == "cilia_snapshots_coll_plate01"


def test_compose_distinct_plates_distinct_ids():
    id1 = compose_collection_experiment_id(COLL, "20260607_plate01_t45hpf")
    id2 = compose_collection_experiment_id(COLL, "20260607_plate02_t45hpf")
    assert id1 != id2
    assert id2 == "cilia_snapshots_coll_plate02"


def test_compose_is_sanitized():
    # A dirty collection name is normalized by sanitize_experiment_id.
    out = compose_collection_experiment_id("my coll_coll", "20260607_plate01_t45hpf")
    assert " " not in out
    assert out == "my_coll_coll_plate01"


def test_compose_rejects_bad_child():
    with pytest.raises(ValueError):
        compose_collection_experiment_id(COLL, "plate01_no_date")
