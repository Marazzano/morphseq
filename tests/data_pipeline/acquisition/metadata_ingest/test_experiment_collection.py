"""Tests for resolve_experiment_ids — the plural collection expander.

Uses tmp_path fixture directories (never real data). Exercises the public wrapper
only; the wrapper CONSUMES the singular resolver + shared composer, so these tests
assert on behavior (which ids come out), not on any re-derived grammar.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_experiment_collection.py
"""

import pytest

from data_pipeline.acquisition.metadata_ingest.experiment_collection import (
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
