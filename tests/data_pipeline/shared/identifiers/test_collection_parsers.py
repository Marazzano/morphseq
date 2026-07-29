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


# ── Descriptive-middle child names (real acquisitions) ────────────────────────────────
# Real collection children carry a free-form descriptive middle between the date and the
# plate token, e.g. the YX1 collection 20260624_2x_td_bf_pbx_coll:
#   20260624_pbx_flouresence_bf_pilot_plate01_t33hpf
# A purely POSITIONAL read takes "pbx" as the plate token and finds no age token, which
# silently mis-keys the experiment id and nulls the whole age map. The parsers therefore
# anchor on token SHAPE (plate<N> / t<NN>hpf) wherever it appears in the name.

_DESCRIPTIVE = "20260624_pbx_flouresence_bf_pilot_plate01_t33hpf"


def test_plate_token_from_descriptive_middle():
    # NOT "pbx" — the plate token is found by shape, not by position.
    assert parse_plate_token(_DESCRIPTIVE) == "plate01"


def test_event_label_from_descriptive_middle():
    assert parse_event_label(_DESCRIPTIVE) == "t33hpf"


def test_declared_hpf_from_descriptive_middle():
    assert parse_declared_hpf(_DESCRIPTIVE) == 33


def test_descriptive_sources_of_one_plate_share_id():
    # The three real sources are one plate imaged at three ages → ONE experiment_id.
    ids = {
        compose_collection_experiment_id("20260624_2x_td_bf_pbx_coll", child)
        for child in (
            "20260624_pbx_flouresence_bf_pilot_plate01_t33hpf",
            "20260625_pbx_flouresence_bf_pilot_plate01_t52hpf",
            "20260626_pbx_flouresence_bf_pilot_plate01_t77hpf",
        )
    }
    assert ids == {"20260624_2x_td_bf_pbx_coll_plate01"}


def test_descriptive_ages_are_all_distinct():
    ages = [
        parse_declared_hpf("20260624_pbx_flouresence_bf_pilot_plate01_t33hpf"),
        parse_declared_hpf("20260625_pbx_flouresence_bf_pilot_plate01_t52hpf"),
        parse_declared_hpf("20260626_pbx_flouresence_bf_pilot_plate01_t77hpf"),
    ]
    assert ages == [33, 52, 77]


def test_plate_token_shape_match_is_case_normalized():
    # Casing is not identity: PLATE01 and plate01 are the same plate.
    assert parse_plate_token("20260607_PLATE01_t45hpf") == "plate01"


def test_plate_token_not_matched_inside_a_longer_word():
    # "microplate01x" must not be mistaken for the plate token — the anchored pattern is
    # bounded by underscores/ends, so this falls back to the positional read.
    assert parse_plate_token("20260607_microplate01x") == "microplate01x"


def test_trailing_age_token_wins_over_one_in_the_middle():
    # The event label is conventionally the trailing token; a middle that happens to look
    # age-shaped must not shadow it.
    assert parse_event_label("20260607_t99hpf_pilot_plate01_t45hpf") == "t45hpf"
    assert parse_declared_hpf("20260607_t99hpf_pilot_plate01_t45hpf") == 45


# ── The simple positional forms must be COMPLETELY unchanged (strict generalization) ──


def test_simple_forms_unchanged():
    assert parse_plate_token("20250622_plate01_t28hpf") == "plate01"
    assert parse_event_label("20250622_plate01_t28hpf") == "t28hpf"
    assert parse_declared_hpf("20250622_plate01_t28hpf") == 28
    # No event at all.
    assert parse_plate_token("20260607_plate01") == "plate01"
    assert parse_event_label("20260607_plate01") is None
    assert parse_declared_hpf("20260607_plate01") is None
    # Non-age event label (e.g. sci) still parses as a label with no declared age.
    assert parse_event_label("20260607_plate01_sci") == "sci"
    assert parse_declared_hpf("20260607_plate01_sci") is None


def test_non_plate_shaped_token_still_uses_positional_fallback():
    # A plate token that is not "plate<N>"-shaped keeps the original positional meaning.
    assert parse_plate_token("20260607_dish7_t45hpf") == "dish7"


def test_missing_date_still_fails_loud():
    for parse in (parse_plate_token, parse_event_label):
        with pytest.raises(ValueError, match="8 digits"):
            parse("plate01_t45hpf")


# ── Ambiguity is REJECTED, not guessed (review follow-up) ─────────────────────────────


def test_two_distinct_plate_tokens_is_ambiguous():
    # Nothing in the name says which is the plate identity; taking the first would mis-key the
    # experiment silently.
    with pytest.raises(ValueError, match="multiple plate tokens"):
        parse_plate_token("20260624_plate01_backup_plate02_t33hpf")


def test_repeated_identical_plate_token_is_fine():
    assert parse_plate_token("20260624_plate01_rescan_plate01_t33hpf") == "plate01"


def test_non_plate_shaped_token_falls_back_positionally():
    # "pilot" is not plate<N>-shaped, so the original positional reading applies.
    assert parse_plate_token("20260624_pilot_t33hpf") == "pilot"
    assert parse_declared_hpf("20260624_pilot_t33hpf") == 33


def test_non_age_event_label_is_preserved():
    assert parse_plate_token("20260624_plate01_notes") == "plate01"
    assert parse_event_label("20260624_plate01_notes") == "notes"
    assert parse_declared_hpf("20260624_plate01_notes") is None
