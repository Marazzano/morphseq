from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pandas as pd
import pytest
import yaml

from src.core.data.manifest_types import SourceArtifact
from src.core.run.provenance import (
    RunProvenanceBundle,
    read_selected_identity,
    write_run_provenance,
)
from tests.core.fixtures.manifest_v2 import synthetic_manifest_v2


ADAPTER_REVISION = "a" * 40


def _bundle(tmp_path):
    manifest = synthetic_manifest_v2(asset_root=tmp_path / "assets")
    source_path = tmp_path / "declared_source.csv"
    source_path.write_text("snip_id\nsnip::alpha\n", encoding="utf-8")
    source = SourceArtifact(
        experiment_id="experiment::one",
        source_name="snip_inventory",
        path=source_path,
        schema_version="fixture-v1",
        size_bytes=source_path.stat().st_size,
        mtime_ns=source_path.stat().st_mtime_ns,
        row_count=1,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    manifest = type(manifest)(
        observation_table=manifest.observation_table,
        asset_table=manifest.asset_table,
        resolved_sample_table=manifest.resolved_sample_table,
        source_inventory=(source,),
        schema_report=manifest.schema_report,
        cohort_report=manifest.cohort_report,
        split_assignments=manifest.split_assignments,
        policy=manifest.policy,
    )
    return RunProvenanceBundle.from_manifest_result(
        manifest_result=manifest,
        resolved_config={"model": {"input_dim": [1, 288, 128]}, "wandb": {"enabled": False}},
        adapter_git_revision=ADAPTER_REVISION,
    ), manifest


def test_writes_complete_local_bundle_and_round_trips_selected_keys(tmp_path):
    bundle, manifest = _bundle(tmp_path)
    output = tmp_path / "run" / "provenance"

    written = write_run_provenance(run_artifacts_dir=output, bundle=bundle)

    expected = {
        "adapter_git_revision.txt",
        "cohort_report.json",
        "mapping_policy.json",
        "observation_ids.csv",
        "policies.json",
        "provenance_manifest.json",
        "resolved_config.yaml",
        "selected_asset_keys.csv",
        "sources.json",
        "split_assignments.csv",
    }
    assert {path.name for path in written.files} == expected
    observations, asset_keys = read_selected_identity(output)
    expected_observations = tuple(manifest.resolved_sample_table["snip_id"])
    expected_keys = tuple(
        (row.snip_id, row.snip_product_key, None)
        for row in manifest.resolved_sample_table.itertuples(index=False)
    )
    assert observations == expected_observations
    assert asset_keys == expected_keys

    config = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert config["model"]["input_dim"] == [1, 288, 128]
    expected_policy = json.loads(json.dumps(asdict(manifest.policy)))
    assert json.loads(
        (output / "policies.json").read_text(encoding="utf-8")
    ) == expected_policy
    sources = json.loads((output / "sources.json").read_text(encoding="utf-8"))
    assert sources[0]["source_name"] == "snip_inventory"
    assert sources[0]["sha256"] == bundle.sources[0].sha256
    assert (output / "adapter_git_revision.txt").read_text(encoding="utf-8").strip() == ADAPTER_REVISION


def test_rejects_relative_or_existing_run_artifacts_dir(tmp_path):
    bundle, _ = _bundle(tmp_path)
    with pytest.raises(ValueError, match="absolute configured path"):
        write_run_provenance(run_artifacts_dir="relative/provenance", bundle=bundle)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        write_run_provenance(run_artifacts_dir=existing, bundle=bundle)


def test_rejects_asset_order_that_disagrees_with_observation_order(tmp_path):
    bundle, _ = _bundle(tmp_path)
    bad = RunProvenanceBundle(
        resolved_config=bundle.resolved_config,
        observation_ids=bundle.observation_ids,
        selected_asset_keys=tuple(reversed(bundle.selected_asset_keys)),
        split_assignments=bundle.split_assignments,
        policies=bundle.policies,
        sources=bundle.sources,
        cohort_report=bundle.cohort_report,
        mapping_policy=bundle.mapping_policy,
        adapter_git_revision=bundle.adapter_git_revision,
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        write_run_provenance(run_artifacts_dir=tmp_path / "bad-order", bundle=bad)


def test_wandb_receives_the_exact_local_bundle_directory(tmp_path):
    bundle, _ = _bundle(tmp_path)
    output = tmp_path / "wandb-provenance"
    captured = {}

    class FakeArtifact:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

        def add_dir(self, path):
            captured["directory"] = path

    class FakeRun:
        def log_artifact(self, artifact):
            captured["artifact"] = artifact

    write_run_provenance(
        run_artifacts_dir=output,
        bundle=bundle,
        wandb_run=FakeRun(),
        wandb_artifact_factory=FakeArtifact,
    )

    assert captured["directory"] == str(output)
    assert captured["kwargs"]["type"] == "run-provenance"
    assert isinstance(captured["artifact"], FakeArtifact)


def test_invalid_source_hash_and_duplicate_split_identity_fail_by_source_or_id(tmp_path):
    bundle, _ = _bundle(tmp_path)
    bad_source = dict(asdict(bundle.sources[0]), sha256="short")
    bad_hash_bundle = RunProvenanceBundle(
        **{**bundle.__dict__, "sources": (bad_source,)}
    )
    with pytest.raises(ValueError, match="snip_inventory.*invalid full SHA-256"):
        write_run_provenance(run_artifacts_dir=tmp_path / "bad-hash", bundle=bad_hash_bundle)

    duplicate_splits = pd.concat(
        [bundle.split_assignments, bundle.split_assignments.iloc[[0]]], ignore_index=True
    )
    duplicate_bundle = RunProvenanceBundle(
        **{**bundle.__dict__, "split_assignments": duplicate_splits}
    )
    with pytest.raises(ValueError, match="duplicate split assignment.*animal::alpha"):
        write_run_provenance(run_artifacts_dir=tmp_path / "bad-split", bundle=duplicate_bundle)
