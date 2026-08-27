from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.object_extraction.snip_processing.contract import (
    REQUIRED_COLUMNS_SNIP_MANIFEST,
)
from data_pipeline.object_extraction.snip_processing.pipelines.validate_snip_manifest import (
    validate_snip_manifest_df,
)


def _valid_manifest_row(tmp_path: Path) -> dict:
    """One schema-complete snip_manifest row, with the on-disk files it points at.

    BUILT FROM THE CONTRACT, NOT HAND-LISTED. The previous fixture spelled out a column set frozen
    in March and drifted 14 columns behind REQUIRED_COLUMNS_SNIP_MANIFEST, so the "happy path" test
    had been asserting nothing but its own staleness. Starting from the contract means a newly
    required column makes this test fail with "no fixture value for <col>" — a prompt to decide what
    a valid value is — rather than the fixture quietly describing a schema that no longer exists.
    """
    for name in ("img.tif", "mask.png", "snip.jpg", "raw.tif"):
        (tmp_path / name).write_bytes(b"x")

    values = {
        "snip_id": "emb1_BF_f0003",
        "mask_type": "embryo",
        "experiment_id": "E",
        "well_id": "E_A01",
        "well_index": "A01",
        "image_id": "E_A01_BF_t0003",
        "embryo_id": "emb1",
        "time_index": 3,
        "image_path": str(tmp_path / "img.tif"),
        "exported_mask_path": str(tmp_path / "mask.png"),
        "yolk_mask_path": None,  # nullable
        "processed_snip_path": str(tmp_path / "snip.jpg"),
        "raw_crop_path": str(tmp_path / "raw.tif"),
        "image_micrometers_per_pixel": 0.65,
        "target_pixel_size_um": 1.0,
        "output_height_px": 512,
        "output_width_px": 512,
        "blend_radius_um": 30.0,
        "background_mean": 128.0,
        "background_std": 30.0,
        "background_definition": "annulus",
        "rotation_angle_rad": 0.1,
        "rotation_angle_deg": 5.7,
        "rotation_source": "embryo_only",
        "pipeline_version": "abc123",
        "snip_processing_config_hash": "cfg123",
        "processing_timestamp_utc": "2026-03-01T00:00:00Z",
        "processed_file_size_bytes": 1,
        "raw_file_size_bytes": 1,
        "is_valid": True,
        "error_message": None,  # nullable
    }

    missing = [column for column in REQUIRED_COLUMNS_SNIP_MANIFEST if column not in values]
    assert not missing, (
        f"no fixture value for {missing} — REQUIRED_COLUMNS_SNIP_MANIFEST gained columns this "
        "fixture does not supply. Add a realistic value rather than deleting the assertion."
    )
    return {column: values[column] for column in REQUIRED_COLUMNS_SNIP_MANIFEST}


def test_validate_snip_manifest_happy_path(tmp_path: Path) -> None:
    validate_snip_manifest_df(pd.DataFrame([_valid_manifest_row(tmp_path)]))


def test_a_valid_snip_must_carry_its_output_path(tmp_path: Path) -> None:
    """`is_valid=True` with a null processed_snip_path is a contradiction and must fail.

    RENAMED FROM test_validate_snip_manifest_missing_file_fails, which tested nothing.
    validate_snip_manifest_df is a SCHEMA validator — it never touches the filesystem, so a row
    pointing at a nonexistent path passes it. That test only went green because its inline schema
    had drifted 14 columns behind the contract and raised for missing COLUMNS instead; the name
    described a check that does not live at this seam.

    What this function does enforce is conditional nullability: a row claiming a successful snip
    must carry the output it claims to have produced. Existence-on-disk is the path-taking
    validate_snip_manifest's job, one layer up.
    """
    row = _valid_manifest_row(tmp_path)
    row["processed_snip_path"] = None

    with pytest.raises(ValueError, match="non-null processed_snip_path"):
        validate_snip_manifest_df(pd.DataFrame([row]))

