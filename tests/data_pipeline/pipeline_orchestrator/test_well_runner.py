"""Tests for the well-runner orchestration layer (orchestration/well_runner.py).

These pin the PUBLIC CONTRACT of ``run_well_ids_for_experiment`` — the single definition of
"which wells run for an experiment" — not the private helpers. The contract, by concept:

  - the discover_wells CHECKPOINT is the source of truth for which wells exist;
  - config only FILTERS that set (never names a well that doesn't exist -> fail loud);
  - config entries are LENIENT at the boundary (local slug OR global well_id) but the OUTPUT is
    STRICT (global well_ids only, in discovered order);
  - ``output_root`` is a PARAMETER (the discovered-wells file is read beneath it via the registry).

Error-message assertions check for the important words, not exact prose. Resolved-path coupling
goes through ``paths.artifact_path`` so a registry filename change is caught there, not re-pinned
here.

Run with:
  PYTHONPATH=src:$PYTHONPATH pytest tests/data_pipeline/pipeline_orchestrator/test_well_runner.py
"""

from pathlib import Path

import pytest

import pandas as pd

from data_pipeline.pipeline_orchestrator.orchestration import (
    PATH_MODE_PER_WELL,
    artifact_path,
    collect_well_shard_paths,
    run_well_shard_paths,
    concat_well_shards_to_file,
    per_well_step_dir,
    run_well_ids_for_experiment,
    validated_path,
)

EXP = "20250912"
# Discovered, GLOBAL well_ids — the form the checkpoint emits (one per line).
A01 = "20250912_A01"
B01 = "20250912_B01"
C01 = "20250912_C01"

# The registered per_well_then_merge step used to exercise the merge side.
SHARD_STEP = "frame_inventory"
SHARD_ARTIFACT = "inventory"


def _write_discovered(root: Path, experiment_id: str, well_ids) -> None:
    """Materialize a discovered_wells.txt at the registry-resolved location under ``root``."""
    wells_file = artifact_path(root, "discover_wells", "wells", experiment_id)
    wells_file.parent.mkdir(parents=True, exist_ok=True)
    wells_file.write_text("\n".join(well_ids) + "\n", encoding="utf-8")


def _write_shard(root: Path, well_id: str, *, rows, validated: bool) -> Path:
    """Write a per-well frame_inventory shard CSV (and optionally its .validated sentinel)."""
    shard = artifact_path(root, SHARD_STEP, SHARD_ARTIFACT, EXP,
                          path_mode=PATH_MODE_PER_WELL, well_id=well_id)
    shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(shard, index=False)
    if validated:
        sentinel = validated_path(root, SHARD_STEP, SHARD_ARTIFACT, EXP,
                                  path_mode=PATH_MODE_PER_WELL, well_id=well_id)
        sentinel.write_text("ok\n", encoding="utf-8")
    return shard


class TestNoConfigFilterRunsAllDiscovered:
    """No config wells for the experiment -> run EVERY discovered well (the data decides)."""

    def test_empty_config_returns_all_discovered(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01, C01])
        assert run_well_ids_for_experiment(EXP, {}, output_root=tmp_path) == [A01, B01, C01]

    def test_config_without_this_experiment_returns_all_discovered(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"experiment_wells": {"some_other_experiment": ["x"]}}
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [A01, B01]

    def test_discovered_order_is_preserved(self, tmp_path):
        # Output order follows the file, not sorting.
        _write_discovered(tmp_path, EXP, [C01, A01, B01])
        assert run_well_ids_for_experiment(EXP, {}, output_root=tmp_path) == [C01, A01, B01]


class TestConfigFiltersDiscovered:
    """Config wells present -> return discovered ∩ target, in DISCOVERED order."""

    def test_filter_narrows_to_named_wells(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01, C01])
        config = {"experiment_wells": {EXP: [B01]}}
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [B01]

    def test_filter_keeps_discovered_order_not_config_order(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01, C01])
        config = {"experiment_wells": {EXP: [C01, A01]}}  # config lists C before A
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [A01, C01]

    def test_target_wells_config_key_also_works(self, tmp_path):
        # TARGET spelling (front_end doc) as well as today's `experiment_wells`.
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"target_wells": {EXP: [A01]}}
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [A01]

    def test_int_experiment_key_in_config_is_tolerated(self, tmp_path):
        # YAML may parse a numeric experiment id as an int key.
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"experiment_wells": {int(EXP): [B01]}}
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [B01]


class TestLenientBoundaryStrictInternal:
    """Config entries may be LOCAL slugs or GLOBAL well_ids; the output is always global well_ids,
    normalized exactly once at the seam."""

    def test_local_slug_is_promoted_to_global(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"experiment_wells": {EXP: ["B01"]}}  # local slug
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [B01]

    def test_global_well_id_is_kept_as_is(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"experiment_wells": {EXP: [B01]}}  # already global
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [B01]

    def test_local_and_global_entries_mix(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01, C01])
        config = {"experiment_wells": {EXP: ["A01", C01]}}  # one local, one global
        assert run_well_ids_for_experiment(EXP, config, output_root=tmp_path) == [A01, C01]

    def test_experiment_id_with_underscore_promotes_local_correctly(self, tmp_path):
        # An experiment id can itself contain '_' (e.g. 20240509_24hpf); a local slug must still
        # promote to the right global well_id and match discovered.
        exp = "20240509_24hpf"
        well_id = "20240509_24hpf_A04"
        _write_discovered(tmp_path, exp, [well_id])
        config = {"experiment_wells": {exp: ["A04"]}}
        assert run_well_ids_for_experiment(exp, config, output_root=tmp_path) == [well_id]


class TestCheckpointIsTruth:
    """Config can only filter discovered wells; naming a non-discovered well FAILS LOUD (Win 5)."""

    def test_unknown_global_well_fails_loud(self, tmp_path):
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"experiment_wells": {EXP: ["20250912_ZZ9"]}}
        with pytest.raises(ValueError) as excinfo:
            run_well_ids_for_experiment(EXP, config, output_root=tmp_path)
        message = str(excinfo.value)
        assert "20250912_ZZ9" in message
        assert "discovered" in message.lower()

    def test_unknown_local_slug_fails_loud_after_promotion(self, tmp_path):
        # A typo'd local slug promotes to a global id that isn't discovered -> fail loud.
        _write_discovered(tmp_path, EXP, [A01, B01])
        config = {"experiment_wells": {EXP: ["Z99"]}}
        with pytest.raises(ValueError) as excinfo:
            run_well_ids_for_experiment(EXP, config, output_root=tmp_path)
        # The message names the PROMOTED (global) form, proving normalization happened first.
        assert "20250912_Z99" in str(excinfo.value)


class TestOutputRootIsAParameter:
    """The discovered-wells file is read beneath the supplied output_root, never PROJECT_ROOT."""

    def test_reads_under_supplied_root(self, tmp_path):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        _write_discovered(root_a, EXP, [A01])
        _write_discovered(root_b, EXP, [B01])
        assert run_well_ids_for_experiment(EXP, {}, output_root=root_a) == [A01]
        assert run_well_ids_for_experiment(EXP, {}, output_root=root_b) == [B01]

    def test_missing_checkpoint_output_fails_loud(self, tmp_path):
        # No discovered_wells.txt written -> the checkpoint hasn't run.
        with pytest.raises(FileNotFoundError) as excinfo:
            run_well_ids_for_experiment(EXP, {}, output_root=tmp_path)
        assert "discover" in str(excinfo.value).lower()

    def test_blank_lines_in_discovered_file_are_ignored(self, tmp_path):
        wells_file = artifact_path(tmp_path, "discover_wells", "wells", EXP)
        wells_file.parent.mkdir(parents=True, exist_ok=True)
        wells_file.write_text(f"{A01}\n\n  \n{B01}\n", encoding="utf-8")
        assert run_well_ids_for_experiment(EXP, {}, output_root=tmp_path) == [A01, B01]

    def test_bare_local_label_in_discovered_file_fails_loud(self, tmp_path):
        # discovered_wells.txt is contracted to hold GLOBAL well_ids; a bare local label is
        # stale/corrupt and must not travel downstream.
        wells_file = artifact_path(tmp_path, "discover_wells", "wells", EXP)
        wells_file.parent.mkdir(parents=True, exist_ok=True)
        wells_file.write_text(f"{A01}\nB01\n", encoding="utf-8")  # B01 is a bare local label
        with pytest.raises(ValueError) as excinfo:
            run_well_ids_for_experiment(EXP, {}, output_root=tmp_path)
        message = str(excinfo.value)
        assert "B01" in message
        assert "discovered" in message.lower()


def _shard_path(root: Path, well_id: str) -> Path:
    """The registry-resolved per-well shard path for one well (what the collector returns)."""
    return artifact_path(root, SHARD_STEP, SHARD_ARTIFACT, EXP,
                         path_mode=PATH_MODE_PER_WELL, well_id=well_id)


class TestRunWellShardPathsIsDagCollector:
    """The DAG collector resolves a caller-provided RUN set to shard paths without disk checks."""

    def test_resolves_requested_run_wells_in_caller_order_without_touching_disk(self, tmp_path):
        # No shard files or sentinels exist. That is correct at DAG-planning time: returning these
        # paths declares dependencies; Snakemake owns building and validating them.
        assert run_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP, [C01, A01]) == [
            _shard_path(tmp_path, C01),
            _shard_path(tmp_path, A01),
        ]

    def test_rejects_bare_local_well_label_without_reading_disk(self, tmp_path):
        with pytest.raises(ValueError):
            run_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP, ["B01"])


class TestCollectWellShardPathsIsFilesystemDriven:
    """The merge side is decided by DISK + validation sentinels, never by config.
    collect_well_shard_paths takes NO config argument — disk + sentinels are its only inputs, so
    the whole experiment's shards are returned regardless of what any run selected (a narrow rerun
    cannot amputate the published table)."""

    def test_collects_all_validated_shards_present_on_disk(self, tmp_path):
        # All three present-and-validated shards come back, in well_id-dir order — independent of
        # how many wells a run computed.
        _write_shard(tmp_path, A01, rows=[{"x": 1}], validated=True)
        _write_shard(tmp_path, B01, rows=[{"x": 2}], validated=True)
        _write_shard(tmp_path, C01, rows=[{"x": 3}], validated=True)
        assert collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP) == \
            [_shard_path(tmp_path, A01), _shard_path(tmp_path, B01), _shard_path(tmp_path, C01)]


class TestCollectWellShardPathsFailLoudTaxonomy:
    """The four cases: bad dir name -> raise; sentinel-without-artifact -> raise;
    artifact-without-sentinel -> skip; both -> keep. Plus missing pool dir -> raise."""

    def test_artifact_without_sentinel_is_skipped(self, tmp_path):
        _write_shard(tmp_path, A01, rows=[{"x": 1}], validated=True)
        _write_shard(tmp_path, B01, rows=[{"x": 2}], validated=False)  # not validated yet
        # B01 mid-flight, skipped silently.
        assert collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP) == \
            [_shard_path(tmp_path, A01)]

    def test_sentinel_without_artifact_fails_loud(self, tmp_path):
        _write_shard(tmp_path, A01, rows=[{"x": 1}], validated=True)
        # B01: write a sentinel but delete the artifact -> corrupt shard.
        shard = _write_shard(tmp_path, B01, rows=[{"x": 2}], validated=True)
        shard.unlink()
        with pytest.raises(FileNotFoundError) as excinfo:
            collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP)
        message = str(excinfo.value)
        assert B01 in message
        assert "sentinel" in message.lower()

    def test_bad_directory_name_fails_loud(self, tmp_path):
        _write_shard(tmp_path, A01, rows=[{"x": 1}], validated=True)
        # A junk dir whose name is not a global well_id (bare local label).
        pool = per_well_step_dir(tmp_path, SHARD_STEP, EXP)
        (pool / "B01").mkdir()  # bare local slug -> not a valid global well_id
        with pytest.raises(ValueError):
            collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP)

    def test_missing_per_well_dir_fails_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP)

    def test_empty_pool_returns_empty_list(self, tmp_path):
        # Pool dir exists (a shard was written) but no validated shards remain.
        _write_shard(tmp_path, A01, rows=[{"x": 1}], validated=False)
        assert collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP) == []


class TestConcatWellShardsToFile:
    """Row-concat + per-shard required-column assertion + deterministic sort + format-by-suffix
    write."""

    def test_concatenates_shards_in_order(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        pd.DataFrame([{"well_id": A01, "v": 1}]).to_csv(a, index=False)
        pd.DataFrame([{"well_id": B01, "v": 2}]).to_csv(b, index=False)
        out = tmp_path / "merged.csv"
        concat_well_shards_to_file([a, b], out)
        merged = pd.read_csv(out)
        assert list(merged["well_id"]) == [A01, B01]
        assert len(merged) == 2

    def test_required_column_missing_fails_loud(self, tmp_path):
        a = tmp_path / "a.csv"
        pd.DataFrame([{"v": 1}]).to_csv(a, index=False)
        out = tmp_path / "merged.csv"
        with pytest.raises(ValueError) as excinfo:
            concat_well_shards_to_file([a], out, required_columns=["well_id"])
        assert "well_id" in str(excinfo.value)

    def test_required_column_missing_from_one_shard_fails_loud(self, tmp_path):
        # The real bug: pd.concat UNIONS columns, so a column present in only SOME shards survives
        # (NaN-filled) in the merged frame. A post-concat-only check would pass. The per-shard
        # check must catch shard B and name it.
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        pd.DataFrame([{"well_id": A01, "v": 1}]).to_csv(a, index=False)
        pd.DataFrame([{"v": 2}]).to_csv(b, index=False)  # missing well_id
        with pytest.raises(ValueError) as excinfo:
            concat_well_shards_to_file([a, b], tmp_path / "merged.csv", required_columns=["well_id"])
        message = str(excinfo.value)
        assert str(b) in message
        assert "well_id" in message

    def test_sort_columns_orders_output_and_ignores_absent(self, tmp_path):
        a = tmp_path / "a.csv"
        pd.DataFrame([{"well_id": B01, "t": 2}, {"well_id": A01, "t": 1}]).to_csv(a, index=False)
        out = tmp_path / "merged.csv"
        # 't' present -> sorts by it; 'absent' silently ignored.
        concat_well_shards_to_file([a], out, sort_columns=["t", "absent"])
        merged = pd.read_csv(out)
        assert list(merged["t"]) == [1, 2]

    def test_empty_input_fails_loud(self, tmp_path):
        with pytest.raises(ValueError):
            concat_well_shards_to_file([], tmp_path / "merged.csv")

    def test_roundtrips_via_collect_well_shard_paths(self, tmp_path):
        # The intended call shape: collect paths off disk, then concat them.
        _write_shard(tmp_path, A01, rows=[{"well_id": A01, "frame": 0}], validated=True)
        _write_shard(tmp_path, B01, rows=[{"well_id": B01, "frame": 0}], validated=True)
        shard_paths = collect_well_shard_paths(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP)
        out = artifact_path(tmp_path, SHARD_STEP, SHARD_ARTIFACT, EXP, path_mode="merged")
        concat_well_shards_to_file(shard_paths, out, required_columns=["well_id"])
        merged = pd.read_csv(out)
        assert sorted(merged["well_id"]) == [A01, B01]
