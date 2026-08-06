"""Tests for resolve_experiment_ids — the plural collection expander.

Uses tmp_path fixture directories (never real data). Exercises the public wrapper
only; the wrapper CONSUMES the singular resolver + shared composer, so these tests
assert on behavior (which ids come out), not on any re-derived grammar.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_discovery.py
"""

import pytest

import pytest

from data_pipeline.acquisition.metadata_ingest.collection_discovery import (
    discover_plate_sources,
    resolve_experiment_ids,
)


def _make_collection(raw_root, coll_name, children, *, as_files=False, suffix=".nd2"):
    """Create raw_root/<coll_name>/ with each child as a folder (or .nd2 file)."""
    coll_dir = raw_root / coll_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    for child in children:
        if as_files:
            (coll_dir / f"{child}{suffix}").write_text("")
        else:
            (coll_dir / child).mkdir()
    return coll_dir


# ── Inverse: {coll}_{plate} → its source children (acquisition ingest needs this) ──

def test_discover_plate_sources_returns_that_plates_children_only(tmp_path):
    _make_collection(
        tmp_path,
        "cilia_snapshots_coll",
        [
            "20260607_plate01_t45hpf",
            "20260608_plate01_t72hpf",
            "20260607_plate02_t45hpf",  # a DIFFERENT plate — must NOT be returned
        ],
    )
    coll, children = discover_plate_sources(
        "cilia_snapshots_coll_plate01", tmp_path
    )
    assert coll == "cilia_snapshots_coll"
    assert children == ["20260607_plate01_t45hpf", "20260608_plate01_t72hpf"]


def test_discover_plate_sources_rejects_non_collection_id(tmp_path):
    with pytest.raises(ValueError, match="not a collection plate id"):
        discover_plate_sources("20240418", tmp_path)


# ── Core: 2 plates x 2 t-events → 2 experiment_ids ────────────────────────────

def test_collection_two_plates_two_events_each(tmp_path):
    _make_collection(
        tmp_path,
        "cilia_snapshots_coll",
        [
            "20260607_plate01_t45hpf",
            "20260608_plate01_t72hpf",
            "20260607_plate02_t45hpf",
            "20260608_plate02_t72hpf",
        ],
    )
    out = resolve_experiment_ids(["cilia_snapshots_coll"], tmp_path)
    assert out == [
        "cilia_snapshots_coll_plate01",
        "cilia_snapshots_coll_plate02",
    ]


def test_collection_events_merge_into_one_id_per_plate(tmp_path):
    # Four child folders (2 plates x 2 events) collapse to exactly 2 ids.
    _make_collection(
        tmp_path,
        "cilia_snapshots_coll",
        [
            "20260607_plate01_t45hpf",
            "20260608_plate01_t72hpf",
            "20260607_plate02_t45hpf",
            "20260608_plate02_t72hpf",
        ],
    )
    out = resolve_experiment_ids(["cilia_snapshots_coll"], tmp_path)
    assert len(out) == 2


# ── Passthrough: bare experiment_id ───────────────────────────────────────────

def test_bare_id_passes_through(tmp_path):
    out = resolve_experiment_ids(["20260416_plate01_t02"], tmp_path)
    assert out == ["20260416_plate01_t02"]


def test_bare_id_is_sanitized_on_passthrough(tmp_path):
    # The singular resolver sanitizes; a space becomes an underscore.
    out = resolve_experiment_ids(["my exp"], tmp_path)
    assert out == ["my_exp"]


# ── Mixed list ────────────────────────────────────────────────────────────────

def test_mixed_list_expands_and_passes_through(tmp_path):
    _make_collection(
        tmp_path,
        "cilia_snapshots_coll",
        ["20260607_plate01_t45hpf", "20260607_plate02_t45hpf"],
    )
    out = resolve_experiment_ids(
        ["20240418_something", "cilia_snapshots_coll", "20250101_other"],
        tmp_path,
    )
    assert out == [
        "20240418_something",
        "cilia_snapshots_coll_plate01",
        "cilia_snapshots_coll_plate02",
        "20250101_other",
    ]


# ── Legacy (no _coll) untouched even if such a folder exists ───────────────────

def test_legacy_folder_not_expanded(tmp_path):
    # A legacy folder without _coll is a single experiment: passthrough, NOT globbed,
    # even though it has plate-token-shaped children on disk.
    legacy = tmp_path / "20260416_legacy_plate01_t02"
    legacy.mkdir()
    (legacy / "20260416_plate01_t45hpf").mkdir()
    out = resolve_experiment_ids(["20260416_legacy_plate01_t02"], tmp_path)
    assert out == ["20260416_legacy_plate01_t02"]


# ── Dedup + order preservation ────────────────────────────────────────────────

def test_dedupe_preserves_order(tmp_path):
    _make_collection(
        tmp_path,
        "cilia_snapshots_coll",
        ["20260607_plate01_t45hpf", "20260607_plate02_t45hpf"],
    )
    out = resolve_experiment_ids(
        ["cilia_snapshots_coll", "cilia_snapshots_coll", "20240418_x", "20240418_x"],
        tmp_path,
    )
    assert out == [
        "cilia_snapshots_coll_plate01",
        "cilia_snapshots_coll_plate02",
        "20240418_x",
    ]


# ── YX1 flat: children are .nd2 files ─────────────────────────────────────────

def test_collection_children_as_nd2_files(tmp_path):
    _make_collection(
        tmp_path,
        "yx1_snaps_coll",
        ["20260607_plate01_t45hpf", "20260608_plate01_t72hpf", "20260607_plate02_t45hpf"],
        as_files=True,
    )
    out = resolve_experiment_ids(["yx1_snaps_coll"], tmp_path)
    assert out == ["yx1_snaps_coll_plate01", "yx1_snaps_coll_plate02"]


# ── Robustness ────────────────────────────────────────────────────────────────

def test_stray_sidecar_child_is_skipped(tmp_path):
    coll_dir = _make_collection(
        tmp_path, "cilia_snapshots_coll", ["20260607_plate01_t45hpf"]
    )
    (coll_dir / "README.txt").write_text("notes")  # no plate token → skipped
    out = resolve_experiment_ids(["cilia_snapshots_coll"], tmp_path)
    assert out == ["cilia_snapshots_coll_plate01"]


def test_missing_collection_dir_fails_loud(tmp_path):
    with pytest.raises(ValueError):
        resolve_experiment_ids(["nonexistent_coll"], tmp_path)


def test_empty_collection_fails_loud(tmp_path):
    (tmp_path / "empty_coll").mkdir()
    with pytest.raises(ValueError):
        resolve_experiment_ids(["empty_coll"], tmp_path)


def test_empty_entries_returns_empty(tmp_path):
    assert resolve_experiment_ids([], tmp_path) == []


# ── ONE filesystem interpretation: both public callers must AGREE ─────────────────────
# resolve_experiment_ids (run-target resolution) and discover_plate_sources (per-plate provenance)
# used to be two independent `iterdir -> parse -> group` loops. They now share one private
# authority, so they cannot diverge on child filtering, parsing, grouping, or ordering. These tests
# pin that promise rather than trusting that two siblings happen to agree today.

from data_pipeline.acquisition.metadata_ingest.collection_discovery import (
    discover_plate_sources,
    resolve_experiment_ids,
)


def _collection(tmp_path, name, children, *, files=False):
    coll = tmp_path / name
    coll.mkdir(parents=True, exist_ok=True)
    for child in children:
        if files:
            (coll / f"{child}.nd2").write_bytes(b"")
        else:
            (coll / child).mkdir()
    return coll


def _plates_via_each_caller(tmp_path, name, children, *, files=False):
    """(ids from resolve_experiment_ids, ids implied by discover_plate_sources groups)."""
    _collection(tmp_path, name, children, files=files)
    resolved = resolve_experiment_ids([name], raw_root=tmp_path, microscope="Keyence")
    # Every resolved id must be discoverable, and its sources must be non-empty.
    discovered = {eid: discover_plate_sources(eid, tmp_path)[1] for eid in resolved}
    return resolved, discovered


def test_both_callers_agree_on_a_multi_plate_collection(tmp_path):
    resolved, discovered = _plates_via_each_caller(
        tmp_path,
        "chem_coll",
        [
            "20250622_plate01_t28hpf",
            "20250623_plate01_t52hpf",
            "20250622_plate02_t28hpf",
        ],
    )
    assert sorted(resolved) == sorted(discovered)
    assert discovered["chem_coll_plate01"] == [
        "20250622_plate01_t28hpf",
        "20250623_plate01_t52hpf",
    ]
    assert discovered["chem_coll_plate02"] == ["20250622_plate02_t28hpf"]


def test_both_callers_agree_on_yx1_file_sources(tmp_path):
    resolved, discovered = _plates_via_each_caller(
        tmp_path,
        "pbx_coll",
        ["20260624_pilot_plate01_t33hpf", "20260625_pilot_plate01_t52hpf"],
        files=True,
    )
    assert resolved == ["pbx_coll_plate01"]
    assert len(discovered["pbx_coll_plate01"]) == 2


def test_malformed_children_are_skipped_by_BOTH_callers(tmp_path):
    # Stray sidecars carry no plate token; neither caller may treat them as sources.
    resolved, discovered = _plates_via_each_caller(
        tmp_path,
        "chem_coll",
        ["20250622_plate01_t28hpf", "Thumbs.db", "notes", "readme_no_date"],
    )
    assert resolved == ["chem_coll_plate01"]
    assert discovered["chem_coll_plate01"] == ["20250622_plate01_t28hpf"]


def test_same_age_sources_are_still_DISCOVERED_by_both(tmp_path):
    """Discovery groups them; rejecting same-age ordering is provenance's job, not discovery's.

    Keeps the layers honest: discovery reports what is on disk, the provenance producer applies
    assert_source_order_unambiguous.
    """
    resolved, discovered = _plates_via_each_caller(
        tmp_path, "chem_coll", ["20250622_plate01_t28hpf", "20250623_plate01_t28hpf"]
    )
    assert resolved == ["chem_coll_plate01"]
    assert len(discovered["chem_coll_plate01"]) == 2


def test_empty_collection_fails_loud_in_both_callers(tmp_path):
    _collection(tmp_path, "empty_coll", [])
    with pytest.raises(ValueError, match="Nothing to discover"):
        resolve_experiment_ids(["empty_coll"], raw_root=tmp_path, microscope="Keyence")
    with pytest.raises(ValueError, match="Nothing to discover"):
        discover_plate_sources("empty_coll_plate01", tmp_path)
