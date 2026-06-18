"""Tests for image_materialization/materialized_image_paths.py — pure path math, no disk I/O."""

from pathlib import Path

import pytest

from data_pipeline.image_materialization.materialized_image_paths import (
    ALLOWED_IMAGE_PRODUCT_TYPES,
    materialized_image_path,
    projection_frame_path,
)

ROOT = Path("/data/built_image_data")
EXP = "20250912"
WELL = "20250912_B01"
CHANNEL = "BF"
T = 3


class TestProjectionFramePath:
    def test_candidate_contains_candidate_subdir(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T, candidate=True)
        parts = p.parts
        assert "candidate" in parts
        assert "materialized_images" in parts

    def test_live_omits_candidate_subdir(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T, candidate=False)
        assert "candidate" not in p.parts

    def test_filename_is_canonical_image_id(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T)
        # image_id = {well_id}_{channel_id}_t{time_index:04d}
        assert p.name == f"{WELL}_{CHANNEL}_t{T:04d}.png"

    def test_channel_dir_in_path(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T)
        assert CHANNEL in p.parts

    def test_projection_subdir_in_path(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T)
        assert "projection" in p.parts

    def test_experiment_id_in_path(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T)
        assert EXP in p.parts

    def test_custom_ext(self):
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=T, ext="tif")
        assert p.suffix == ".tif"


class TestMaterializedImagePathValidation:
    def test_projection_with_z_index_raises(self):
        with pytest.raises(ValueError, match="z_index=None"):
            materialized_image_path(ROOT, experiment_id=EXP, well_id=WELL,
                                    channel_id=CHANNEL, time_index=T,
                                    image_product_type="projection", z_index=1)

    def test_z_stack_without_z_index_raises(self):
        with pytest.raises(ValueError, match="z_index"):
            materialized_image_path(ROOT, experiment_id=EXP, well_id=WELL,
                                    channel_id=CHANNEL, time_index=T,
                                    image_product_type="z_stack", z_index=None)

    def test_unknown_product_type_raises_with_known_list(self):
        with pytest.raises(ValueError, match="Known types"):
            materialized_image_path(ROOT, experiment_id=EXP, well_id=WELL,
                                    channel_id=CHANNEL, time_index=T,
                                    image_product_type="volume")

    def test_bare_well_index_raises(self):
        # "B01" is a local label, not a global well_id — validate_well_id must reject it.
        with pytest.raises(ValueError):
            projection_frame_path(ROOT, experiment_id=EXP, well_id="B01",
                                  channel_id=CHANNEL, time_index=T)

    def test_disallowed_extension_raises(self):
        with pytest.raises(ValueError, match="extension"):
            materialized_image_path(ROOT, experiment_id=EXP, well_id=WELL,
                                    channel_id=CHANNEL, time_index=T,
                                    image_product_type="projection", ext="bmp")

    def test_z_stack_path_contains_z_index(self):
        p = materialized_image_path(ROOT, experiment_id=EXP, well_id=WELL,
                                    channel_id=CHANNEL, time_index=T,
                                    image_product_type="z_stack", z_index=5)
        assert "z_stack" in p.parts
        assert "z0005" in p.name
        assert f"t{T:04d}" in p.name

    def test_no_disk_io(self):
        # All calls above must work without touching the filesystem.
        # This is trivially true (pure Path math), but we confirm ROOT doesn't need to exist.
        assert not ROOT.exists()
        p = projection_frame_path(ROOT, experiment_id=EXP, well_id=WELL,
                                  channel_id=CHANNEL, time_index=0)
        assert isinstance(p, Path)

    def test_allowed_product_types_constant(self):
        assert "projection" in ALLOWED_IMAGE_PRODUCT_TYPES
        assert "z_stack" in ALLOWED_IMAGE_PRODUCT_TYPES
