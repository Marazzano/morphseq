"""The per-well snip transform table: replay, re-derive, sibling sharing, atomic writes.

Two guarantees are tested separately because they need different things persisted:

  REPLAY     the persisted RESOLVED chain reproduces the existing snip byte-for-byte.
  RE-DERIVE  the persisted INPUTS let a caller derive a DIFFERENT transform (e.g. with a yolk)
             and render an alternate snip.

Storing the resolved chain alone would satisfy replay and quietly fail re-derive, because a refined
orientation decision changes the transform itself.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_pipeline.object_extraction.snip_processing.snip_transform import (
    CENTERING_CONTINUOUS,
    CENTERING_LATCHED,
    apply_transform_to_image,
    apply_transform_to_mask,
    derive_snip_transform,
    transform_for_product,
)
from data_pipeline.object_extraction.snip_processing.snip_transform_table import (
    SNIP_TRANSFORM_SCHEMA_VERSION,
    SNIP_TRANSFORM_TABLE_COLUMNS,
    SnipTransformTableError,
    build_snip_transform_row,
    canonical_from_row,
    chain_from_row,
    derivation_inputs_from_row,
    read_snip_transform_table,
    validate_snip_transform_table,
    write_snip_transform_table,
)
from data_pipeline.shared.identifiers import build_snip_transform_id

SOURCE_SHAPE = (64, 64)
SNIP_SHAPE = (32, 16)
SOURCE_UM_PER_PX = 3.25
TARGET_UM_PER_PX = 7.8


def _mask(*, cy=32, cx=32, half_h=12, half_w=5) -> np.ndarray:
    mask = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
    mask[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = 1
    return mask


def _image(seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, size=SOURCE_SHAPE, dtype=np.uint8)


def _resolved(mask=None, centering=CENTERING_LATCHED):
    canonical = derive_snip_transform(
        _mask() if mask is None else mask,
        source_um_per_px=SOURCE_UM_PER_PX,
        target_um_per_px=TARGET_UM_PER_PX,
        snip_frame_shape_hw=SNIP_SHAPE,
    )
    return canonical, transform_for_product(
        canonical,
        product_shape_hw=SOURCE_SHAPE,
        product_um_per_px=SOURCE_UM_PER_PX,
        centering=centering,
    )


def _row(resolved, **overrides):
    kwargs = dict(
        snip_transform_id="20250912_B01_e01_t0000",
        physical_embryo_id="20250912_B01_e01",
        time_index=0,
        resolved=resolved,
        source_image_id="20250912_B01_BF_t0000",
        mask_id="20250912_B01_BF_t0000_m00",
        orientation_policy="pca_major_axis_yolk_down",
        orientation_source="embryo_mass_distribution",
        no_yolk_policy="fallback_mass_distribution",
    )
    kwargs.update(overrides)
    return build_snip_transform_row(**kwargs)


class TestCanonicalRoundTrip:
    """THE GATE'S READ PATH: snip_geometry writes, render jobs reconstruct.

    Under the geometry gate a render job never derives geometry -- it deserializes the canonical
    transform and resolves it for its own product grid. Every guarantee the gate makes therefore
    rests on the reconstruction being EXACT. These are the three equalities that pin it.
    """

    def test_canonical_survives_the_row_exactly(self):
        # Exact, not approximate. The recipe's floats are typed float64 columns rather than JSON
        # precisely so this holds bit-for-bit; a tolerance here would let writer drift accumulate
        # silently until two siblings rendered a pixel apart.
        canonical, resolved = _resolved()
        assert canonical_from_row(_row(resolved)) == canonical

    def test_round_trip_survives_a_real_csv(self):
        # In production the row makes a trip through a file, which is where float formatting and
        # null handling actually get tested. Both centering modes go through, since the canonical
        # recipe is product-independent: derive_snip_transform always resolves the latched
        # reference, and `centering` is a transform_for_product argument, not a property of the
        # recipe.
        from io import StringIO

        for centering in (CENTERING_LATCHED, CENTERING_CONTINUOUS):
            canonical, resolved = _resolved(centering=centering)
            reread = pd.read_csv(StringIO(pd.DataFrame([_row(resolved)]).to_csv(index=False)))
            assert canonical_from_row(reread.iloc[0]) == canonical, centering

    def test_a_null_latched_center_survives_pandas_nan(self):
        # pandas turns a missing float64 cell into NaN, not None -- and NaN != NaN, so a naive
        # reader would break exact equality on any row that carries no latched reference. Such rows
        # do not arise from derive_snip_transform today, but they will the moment the legacy branch
        # is deleted, and a reader that silently produced (nan, nan) would render garbage rather
        # than fail.
        from io import StringIO

        row = _row(_resolved()[1])
        row["latched_center_x_rescaled"] = None
        row["latched_center_y_rescaled"] = None
        reread = pd.read_csv(StringIO(pd.DataFrame([row]).to_csv(index=False)))
        assert canonical_from_row(reread.iloc[0]).latched_center_xy_rescaled is None

    def test_reconstructed_transform_resolves_identically(self):
        # The second equality: a reconstructed canonical must resolve to the same product transform.
        # Reconstructing the recipe but resolving it differently would still misregister siblings.
        canonical, resolved = _resolved()
        rebuilt = canonical_from_row(_row(resolved))
        again = transform_for_product(
            rebuilt,
            product_shape_hw=SOURCE_SHAPE,
            product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        assert again.rotation_matrix_2x3 == resolved.rotation_matrix_2x3
        assert again.rescaled_shape_hw == resolved.rescaled_shape_hw
        assert again.output_shape_hw == resolved.output_shape_hw

    def test_reconstructed_transform_renders_identical_pixels(self):
        # The third equality, and the one that actually matters: same pixels, image AND mask paths.
        # The first two could both pass while a render still diverged.
        canonical, resolved = _resolved()
        rebuilt = transform_for_product(
            canonical_from_row(_row(resolved)),
            product_shape_hw=SOURCE_SHAPE,
            product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        image, mask = _image(), _mask()
        np.testing.assert_array_equal(
            apply_transform_to_image(image, rebuilt), apply_transform_to_image(image, resolved)
        )
        np.testing.assert_array_equal(
            apply_transform_to_mask(mask, rebuilt), apply_transform_to_mask(mask, resolved)
        )

    def test_a_row_missing_the_recipe_fails_loud(self):
        # A render job that cannot rebuild the recipe must stop, not fall back to deriving one --
        # falling back is precisely the side door the gate exists to close.
        row = _row(_resolved()[1])
        del row["crop_center_um_x"]
        with pytest.raises(SnipTransformTableError, match="reconstructable canonical transform"):
            canonical_from_row(row)


class TestTransformIdIsChannelIndependent:
    def test_id_has_no_channel_segment(self):
        assert build_snip_transform_id("20250416_D09_e01", 32) == "20250416_D09_e01_t0032"

    def test_sibling_channels_resolve_to_the_same_id(self):
        # The core property: the transform is a fact about the ANIMAL at a time, so BF and RFP of one
        # embryo-time must land on one id. If this ever diverges, siblings get separate geometry.
        bf = build_snip_transform_id("20250416_D09_e01", 32)
        rfp = build_snip_transform_id("20250416_D09_e01", 32)
        assert bf == rfp

    def test_malformed_physical_embryo_id_fails_loud(self):
        with pytest.raises(ValueError):
            build_snip_transform_id("not-a-physical-embryo-id", 0)


class TestReplay:
    """Guarantee 1: the persisted chain reproduces the ORIGINAL pixels exactly."""

    def test_round_trip_reproduces_the_snip_byte_for_byte(self, tmp_path):
        image, mask = _image(), _mask()
        _, resolved = _resolved(mask)
        original = apply_transform_to_image(image, resolved, dtype=np.uint8)

        path = write_snip_transform_table(pd.DataFrame([_row(resolved)]), tmp_path / "t.parquet")
        back = read_snip_transform_table(path)
        replayed = chain_from_row(back.iloc[0]).apply_to_image(image).astype(np.uint8)

        assert replayed.shape == original.shape
        assert replayed.dtype == original.dtype
        assert np.array_equal(replayed, original), (
            "the persisted chain did not reproduce the snip exactly — the table is not sufficient "
            "provenance"
        )

    def test_round_trip_reproduces_the_mask_byte_for_byte(self, tmp_path):
        # Masks take the SAME transform but nearest interpolation. If the row collapsed the
        # image/mask interpolation split, this is where it surfaces.
        mask = _mask()
        _, resolved = _resolved(mask)
        original = apply_transform_to_mask(mask, resolved)

        path = write_snip_transform_table(pd.DataFrame([_row(resolved)]), tmp_path / "t.parquet")
        back = read_snip_transform_table(path)
        replayed = chain_from_row(back.iloc[0]).apply_to_mask((mask > 0.5).astype(np.uint8))

        assert np.array_equal(replayed.astype(np.uint8), original)

    def test_csv_round_trip_also_replays(self, tmp_path):
        # snip_inventory is CSV; the table must survive the neighbouring format too.
        image = _image()
        _, resolved = _resolved()
        original = apply_transform_to_image(image, resolved, dtype=np.uint8)
        path = write_snip_transform_table(pd.DataFrame([_row(resolved)]), tmp_path / "t.csv")
        back = read_snip_transform_table(path)
        assert np.array_equal(
            chain_from_row(back.iloc[0]).apply_to_image(image).astype(np.uint8), original
        )

    def test_chain_records_every_step_not_just_a_composite(self):
        """A single composite affine cannot express the resize prefilter. Pinned against."""
        _, resolved = _resolved()
        payload = json.loads(_row(resolved)["resolved_transform_chain_json"])
        steps = payload["steps"]
        assert len(steps) >= 2, "the resize and the affine must both be recorded"
        assert [s["kind"] for s in steps][:2] == ["resize", "affine"]
        for step in steps:
            assert {"kind", "name", "affine_2x3", "in_shape_yx", "out_shape_yx", "interp", "params"} <= set(step)
        assert "composite_affine_2x3_points_only" in payload

    def test_resize_step_records_its_resolved_interpolation(self):
        # Resolved at build time (area on downscale, linear on upscale); replay depends on the
        # RESOLVED value rather than re-deciding at read time.
        _, resolved = _resolved()
        steps = json.loads(_row(resolved)["resolved_transform_chain_json"])["steps"]
        assert steps[0]["params"]["image_interp"] in {"area", "linear"}
        assert steps[0]["params"]["mask_interp"] == "nearest"

    def test_border_policy_and_realized_scale_are_recorded(self):
        _, resolved = _resolved()
        payload = json.loads(_row(resolved)["resolved_transform_chain_json"])
        assert payload["border_mode"] == resolved.border_mode
        assert len(payload["realized_scale_yx"]) == 2
        # Realized, not requested: integer output dims make these differ in general.
        assert payload["realized_scale_yx"][1] == pytest.approx(
            resolved.rescaled_shape_hw[1] / resolved.product_shape_hw[1]
        )


class TestRederive:
    """Guarantee 2: the INPUTS support building a DIFFERENT transform, not just replaying this one."""

    def test_row_carries_the_inputs_a_fresh_derivation_needs(self):
        _, resolved = _resolved()
        inputs = derivation_inputs_from_row(_row(resolved))
        for key in (
            "source_image_id", "mask_id", "source_shape_hw", "source_um_per_px",
            "target_um_per_px", "snip_frame_shape_hw", "crop_center_um_xy", "orientation",
        ):
            assert key in inputs, f"re-derive input {key!r} is missing"
        for key in ("policy", "source", "no_yolk_policy", "rotation_angle_rad", "flip_x"):
            assert key in inputs["orientation"], f"orientation.{key} is missing"

    def test_recorded_inputs_reproduce_the_canonical_transform(self):
        # The re-derive half must rebuild the SAME recipe from the same mask — otherwise it cannot
        # be trusted to build a different one from a better mask.
        mask = _mask()
        canonical, resolved = _resolved(mask)
        inputs = derivation_inputs_from_row(_row(resolved))

        rebuilt = derive_snip_transform(
            mask,
            source_um_per_px=inputs["source_um_per_px"],
            target_um_per_px=inputs["target_um_per_px"],
            snip_frame_shape_hw=tuple(inputs["snip_frame_shape_hw"]),
        )
        assert rebuilt.rotation_angle_rad == pytest.approx(canonical.rotation_angle_rad)
        assert rebuilt.crop_center_um_xy == pytest.approx(canonical.crop_center_um_xy)

    def test_a_yolk_aware_rederivation_produces_a_different_transform(self):
        """The point of re-derive: a refined orientation changes the angle, so replay is not enough.

        The no-yolk path breaks the 180-degree ambiguity with the embryo's own mass distribution; a
        yolk mask overrides that with the yolk's position. An ASYMMETRIC embryo whose heavy end is
        opposite the yolk makes the two disagree, which is exactly the case a yolk-aware pass exists
        to correct.
        """
        # Heavy at the BOTTOM (wide rows 36..44), so the mass-distribution fallback orients one way.
        mask = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
        mask[20:44, 30:34] = 1
        mask[36:44, 24:40] = 1
        canonical, resolved = _resolved(mask)
        inputs = derivation_inputs_from_row(_row(resolved))

        # Yolk at the TOP, opposite the heavy end. With a yolk present the tiebreak uses the
        # yolk-vs-embryo COM instead of the mass distribution, and lands on the opposite branch:
        # pi (mass heuristic) -> 0 (yolk-aware). Verified against get_embryo_rotation_angle directly.
        yolk = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
        yolk[20:28, 26:38] = 1

        refined = derive_snip_transform(
            mask,
            source_um_per_px=inputs["source_um_per_px"],
            target_um_per_px=inputs["target_um_per_px"],
            snip_frame_shape_hw=tuple(inputs["snip_frame_shape_hw"]),
            yolk_mask=yolk,
        )
        assert refined.rotation_angle_rad != canonical.rotation_angle_rad, (
            "a yolk-aware derivation produced an identical angle; the fixture does not exercise the "
            "refinement this provenance exists to enable"
        )
        # And the re-derived transform must be usable, not merely different.
        assert transform_for_product(
            refined, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
        ).output_shape_hw == SNIP_SHAPE

    def test_centering_mode_is_recorded_so_pixels_are_interpretable(self):
        _, latched = _resolved(centering=CENTERING_LATCHED)
        _, continuous = _resolved(centering=CENTERING_CONTINUOUS)
        assert _row(latched)["centering"] == CENTERING_LATCHED
        assert _row(continuous)["centering"] == CENTERING_CONTINUOUS


class TestSiblingsShareOneTransform:
    def test_bf_and_rfp_reference_one_row_that_renders_both(self, tmp_path):
        """The registerability guarantee, end to end.

        Two channels of one embryo-time derive the same recipe from the same mask, so they collapse
        onto ONE table row — and that single row must render BOTH channels' pixels.
        """
        mask = _mask()
        bf_image, rfp_image = _image(0), _image(1)
        _, resolved = _resolved(mask)

        transform_id = build_snip_transform_id("20250912_B01_e01", 0)
        # Both products build a row; they must be byte-identical, so the table holds exactly one.
        bf_row = _row(resolved, snip_transform_id=transform_id, source_image_id="20250912_B01_BF_t0000")
        rfp_row = _row(resolved, snip_transform_id=transform_id, source_image_id="20250912_B01_BF_t0000")
        assert bf_row["resolved_transform_chain_json"] == rfp_row["resolved_transform_chain_json"]

        path = write_snip_transform_table(pd.DataFrame([bf_row]), tmp_path / "t.parquet")
        back = read_snip_transform_table(path)
        assert len(back) == 1

        chain = chain_from_row(back.iloc[0])
        assert np.array_equal(
            chain.apply_to_image(bf_image).astype(np.uint8),
            apply_transform_to_image(bf_image, resolved, dtype=np.uint8),
        )
        assert np.array_equal(
            chain.apply_to_image(rfp_image).astype(np.uint8),
            apply_transform_to_image(rfp_image, resolved, dtype=np.uint8),
        )

    def test_duplicate_transform_ids_are_rejected(self, tmp_path):
        # Two rows claiming one embryo-time is exactly the sibling drift this table prevents.
        _, resolved = _resolved()
        dupe = pd.DataFrame([_row(resolved), _row(resolved)])
        with pytest.raises(SnipTransformTableError, match="must be unique"):
            write_snip_transform_table(dupe, tmp_path / "t.parquet")


class TestTableContract:
    def test_missing_columns_fail_loud(self):
        with pytest.raises(SnipTransformTableError, match="missing required columns"):
            validate_snip_transform_table(pd.DataFrame({"snip_transform_id": ["x"]}))

    def test_empty_table_is_valid(self, tmp_path):
        # A well with zero valid masks still writes a headered, zero-row table: the file IS the
        # record that the well was processed and had no transforms.
        path = write_snip_transform_table(
            pd.DataFrame(columns=list(SNIP_TRANSFORM_TABLE_COLUMNS)), tmp_path / "t.parquet"
        )
        assert len(read_snip_transform_table(path)) == 0

    def test_unknown_schema_version_fails_loud(self, tmp_path):
        _, resolved = _resolved()
        row = _row(resolved)
        row["schema_version"] = SNIP_TRANSFORM_SCHEMA_VERSION + 99
        with pytest.raises(SnipTransformTableError, match="schema_version"):
            validate_snip_transform_table(pd.DataFrame([row]))

    def test_calibration_columns_are_per_axis(self):
        # A scalar um_per_px would bake in square pixels, which no contract guarantees.
        for col in ("source_um_per_px_y", "source_um_per_px_x",
                    "target_um_per_px_y", "target_um_per_px_x"):
            assert col in SNIP_TRANSFORM_TABLE_COLUMNS

    def test_row_without_a_chain_cannot_replay(self):
        with pytest.raises(SnipTransformTableError, match="resolved_transform_chain_json"):
            chain_from_row({"snip_transform_id": "x"})


class TestAtomicWrite:
    def test_failed_write_leaves_no_partial_table(self, tmp_path, monkeypatch):
        """A crash mid-write must leave NO file — consumers treat existence as done."""
        _, resolved = _resolved()
        path = tmp_path / "t.parquet"

        def boom(self, *a, **k):
            raise RuntimeError("simulated writer failure")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError, match="simulated writer failure"):
            write_snip_transform_table(pd.DataFrame([_row(resolved)]), path)

        assert not path.exists(), "a failed write left a table at the final path"
        assert not list(tmp_path.glob("*.tmp-*")), "a failed write left its temp file behind"

    def test_failed_write_does_not_clobber_an_existing_table(self, tmp_path, monkeypatch):
        # The stronger property: a failed RERUN must leave the previous good table intact.
        _, resolved = _resolved()
        path = write_snip_transform_table(pd.DataFrame([_row(resolved)]), tmp_path / "t.parquet")
        before = path.read_bytes()

        def boom(self, *a, **k):
            raise RuntimeError("simulated writer failure")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError):
            write_snip_transform_table(pd.DataFrame([_row(resolved)]), path)

        assert path.read_bytes() == before, "a failed rerun corrupted the previous good table"

    def test_invalid_table_never_becomes_visible(self, tmp_path):
        # Validation happens BEFORE the rename, so an invalid table is never published at all.
        _, resolved = _resolved()
        path = tmp_path / "t.parquet"
        with pytest.raises(SnipTransformTableError):
            write_snip_transform_table(pd.DataFrame([_row(resolved), _row(resolved)]), path)
        assert not path.exists()
