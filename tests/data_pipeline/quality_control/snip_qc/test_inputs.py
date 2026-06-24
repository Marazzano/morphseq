"""snip_qc inputs tests — registry-resolved flag assembly, fail-loud on missing source/column."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    artifact_path,
)
from data_pipeline.quality_control.snip_qc.inputs import load_snip_qc_flag_inputs

EXP = "20250912"
WELL = "20250912_B01"
_SOURCES = [
    ("death_detection_qc", "death_detection_qc"),
    ("surface_area_qc", "surface_area_qc"),
    ("mask_quality_qc", "mask_quality_qc"),
]
_FLAG_COLS = [
    "viability_dead_flag",
    "persistence_dead_flag",
    "sa_outlier_flag",
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
]


def _write_source(root, step, artifact, snip_ids, flag_cols):
    path = artifact_path(root, step, artifact, EXP, path_mode=PATH_MODE_PER_WELL, well_id=WELL)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"snip_id": snip_ids})
    for col in flag_cols:
        df[col] = pd.array([False] * len(snip_ids), dtype=bool)
    df.to_csv(path, index=False)
    return path


def _seed_all(root, snip_ids):
    _write_source(root, "death_detection_qc", "death_detection_qc", snip_ids,
                  ["viability_dead_flag", "persistence_dead_flag"])
    _write_source(root, "surface_area_qc", "surface_area_qc", snip_ids, ["sa_outlier_flag"])
    _write_source(root, "mask_quality_qc", "mask_quality_qc", snip_ids,
                  ["edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"])


def test_assembles_all_flag_columns(tmp_path):
    snip_ids = ["s1", "s2"]
    _seed_all(tmp_path, snip_ids)
    out = load_snip_qc_flag_inputs(
        output_root=tmp_path, experiment_id=EXP, well_id=WELL, sources=_SOURCES, flag_columns=_FLAG_COLS
    )
    assert list(out.columns) == ["snip_id", *_FLAG_COLS]
    assert sorted(out["snip_id"]) == snip_ids
    assert all(out[c].dtype == bool for c in _FLAG_COLS)


def test_missing_source_artifact_fails_loud(tmp_path):
    # only seed two of three sources
    _write_source(tmp_path, "death_detection_qc", "death_detection_qc", ["s1"],
                  ["viability_dead_flag", "persistence_dead_flag"])
    _write_source(tmp_path, "surface_area_qc", "surface_area_qc", ["s1"], ["sa_outlier_flag"])
    with pytest.raises(FileNotFoundError, match="mask_quality_qc"):
        load_snip_qc_flag_inputs(
            output_root=tmp_path, experiment_id=EXP, well_id=WELL, sources=_SOURCES, flag_columns=_FLAG_COLS
        )


def test_missing_flag_column_fails_loud(tmp_path):
    snip_ids = ["s1"]
    # death source missing persistence_dead_flag
    _write_source(tmp_path, "death_detection_qc", "death_detection_qc", snip_ids, ["viability_dead_flag"])
    _write_source(tmp_path, "surface_area_qc", "surface_area_qc", snip_ids, ["sa_outlier_flag"])
    _write_source(tmp_path, "mask_quality_qc", "mask_quality_qc", snip_ids,
                  ["edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"])
    with pytest.raises(ValueError, match="persistence_dead_flag"):
        load_snip_qc_flag_inputs(
            output_root=tmp_path, experiment_id=EXP, well_id=WELL, sources=_SOURCES, flag_columns=_FLAG_COLS
        )


def test_mismatched_snip_universe_across_sources_fails_loud(tmp_path):
    _write_source(tmp_path, "death_detection_qc", "death_detection_qc", ["s1", "s2"],
                  ["viability_dead_flag", "persistence_dead_flag"])
    _write_source(tmp_path, "surface_area_qc", "surface_area_qc", ["s1"], ["sa_outlier_flag"])  # missing s2
    _write_source(tmp_path, "mask_quality_qc", "mask_quality_qc", ["s1", "s2"],
                  ["edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"])
    with pytest.raises(ValueError, match="snip_id set differs"):
        load_snip_qc_flag_inputs(
            output_root=tmp_path, experiment_id=EXP, well_id=WELL, sources=_SOURCES, flag_columns=_FLAG_COLS
        )
