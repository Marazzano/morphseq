"""Tests for image_materialization/materialized_image_paths.py — pure path math, no disk I/O."""

from pathlib import Path

import pytest

from data_pipeline.acquisition.image_materialization.materialized_image_paths import (
    ALLOWED_IMAGE_PRODUCT_TYPES,
    ALLOWED_PROVENANCE_SUFFIXES,
    focus_index_map_path,
    materialized_image_path,
    projection_frame_path,
    z_stack_frame_path,
)

ROOT = Path("/data/built_image_data")
EXP = "20250912"
WELL = "20250912_B01"
CHANNEL = "BF"
T = 3


def _index_of(parts, value):
    """Index of the FIRST occurrence of value in parts (well_id also contains the channel-ish bits)."""
    return list(parts).index(value)


class TestChannelFirstGrammar:
    """The locked channel-first grammar: {well_id}/{channel_id}/{product}/... (channel BEFORE product)."""

    def test_projection_is_channel_then_projection_then_method(self):
        p = materialized_image_path(
            ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
            image_product_type="projection", projection_method="focus_stack",
        )
        parts = p.parts
        # well_id, then channel, then projection, then method — in that order.
        i_well = _index_of(parts, WELL)
        i_chan = parts.index(CHANNEL, i_well + 1)
        i_proj = parts.index("projection", i_chan + 1)
        i_meth = parts.index("focus_stack", i_proj + 1)
        assert i_well < i_chan < i_proj < i_meth
        assert p.name == f"{WELL}_{CHANNEL}_t{T:04d}.png"

    def test_z_stack_is_channel_then_z_stack(self):
        p = materialized_image_path(
            ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
            image_product_type="z_stack", z_index=5,
        )
        parts = p.parts
        i_well = _index_of(parts, WELL)
        i_chan = parts.index(CHANNEL, i_well + 1)
        i_zs = parts.index("z_stack", i_chan + 1)
        assert i_well < i_chan < i_zs

    def test_never_emits_old_product_first_order(self):
        # Negative guard: the helper must NEVER place the product segment before the channel.
        proj = materialized_image_path(
            ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
            image_product_type="projection", projection_method="focus_stack",
        )
        zs = materialized_image_path(
            ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
            image_product_type="z_stack", z_index=2,
        )
        for p in (proj, zs):
            parts = p.parts
            i_well = _index_of(parts, WELL)
            i_chan = parts.index(CHANNEL, i_well + 1)
            # old order was {well_id}/projection/{channel} and {well_id}/z_stack/{channel}:
            # i.e. a product segment sitting BETWEEN well_id and the channel. Forbid that.
            between = parts[i_well + 1 : i_chan]
            assert "projection" not in between
            assert "z_stack" not in between

    def test_generic_projection_requires_projection_method(self):
        with pytest.raises(ValueError, match="projection_method"):
            materialized_image_path(
                ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
                image_product_type="projection", projection_method=None,
            )

    def test_z_stack_rejects_projection_method(self):
        with pytest.raises(ValueError, match="projection_method"):
            materialized_image_path(
                ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
                image_product_type="z_stack", z_index=1, projection_method="focus_stack",
            )

    def test_projection_method_is_in_path(self):
        p = materialized_image_path(
            ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
            image_product_type="projection", projection_method="max_projection",
        )
        assert "max_projection" in p.parts


class TestFocusIndexMapPath:
    def test_lands_under_focus_index_map_subdir(self):
        p = focus_index_map_path(
            ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
        )
        parts = p.parts
        i_chan = _index_of(parts, CHANNEL)
        assert parts.index("projection", i_chan) < parts.index("focus_stack")
        assert "focus_index_map" in parts
        assert p.name == f"{WELL}_{CHANNEL}_t{T:04d}.npz"

    def test_npz_is_a_provenance_suffix_not_an_image_suffix(self):
        # .npz must be a provenance suffix and must NOT be accepted as a primary image suffix.
        from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
            ALLOWED_IMAGE_SUFFIXES,
        )
        assert ".npz" in ALLOWED_PROVENANCE_SUFFIXES
        assert ".npz" not in ALLOWED_IMAGE_SUFFIXES
        with pytest.raises(ValueError, match="image extension"):
            materialized_image_path(
                ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
                image_product_type="z_stack", z_index=0, ext="npz",
            )

    def test_rejects_non_npz_ext(self):
        with pytest.raises(ValueError, match="provenance extension"):
            focus_index_map_path(
                ROOT, experiment_id=EXP, well_id=WELL, channel_id=CHANNEL, time_index=T,
                ext="png",
            )


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


class TestZStackFramePath:
    def test_z_stack_wrapper_path_contains_product_channel_and_z_index(self):
        p = z_stack_frame_path(
            ROOT,
            experiment_id=EXP,
            well_id=WELL,
            channel_id=CHANNEL,
            time_index=T,
            z_index=5,
        )
        assert "z_stack" in p.parts
        assert CHANNEL in p.parts
        assert p.name == f"{WELL}_{CHANNEL}_z0005_t{T:04d}.png"

    def test_z_stack_wrapper_requires_z_index(self):
        with pytest.raises(TypeError):
            z_stack_frame_path(
                ROOT,
                experiment_id=EXP,
                well_id=WELL,
                channel_id=CHANNEL,
                time_index=T,
            )


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
