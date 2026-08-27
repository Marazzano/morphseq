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
    build_resolved_chain_payload,
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


def _product_row(resolved, *, snip_transform_id="20250912_B01_e01_t0000", **overrides):
    """A PRODUCT inventory row: the FK plus this product's compiled chain.

    Separate from _row because the two grains carry different things. The canonical row is
    channel-independent; the chain belongs here, where it describes the raster ONE product actually
    rendered.
    """
    row = {
        "snip_transform_id": snip_transform_id,
        "resolved_transform_chain_json": build_resolved_chain_payload(resolved),
    }
    row.update(overrides)
    return row


def _row(canonical, **overrides):
    kwargs = dict(
        snip_transform_id="20250912_B01_e01_t0000",
        physical_embryo_id="20250912_B01_e01",
        time_index=0,
        canonical=canonical,
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
        assert canonical_from_row(_row(canonical)) == canonical

    def test_round_trip_survives_a_real_csv(self):
        # In production the row makes a trip through a file, which is where float formatting and
        # null handling actually get tested. Both centering modes go through, since the canonical
        # recipe is product-independent: derive_snip_transform always resolves the latched
        # reference, and `centering` is a transform_for_product argument, not a property of the
        # recipe.
        from io import StringIO

        for centering in (CENTERING_LATCHED, CENTERING_CONTINUOUS):
            canonical, resolved = _resolved(centering=centering)
            reread = pd.read_csv(StringIO(pd.DataFrame([_row(canonical)]).to_csv(index=False)))
            assert canonical_from_row(reread.iloc[0]) == canonical, centering

    def test_a_null_latched_center_survives_pandas_nan(self):
        # pandas turns a missing float64 cell into NaN, not None -- and NaN != NaN, so a naive
        # reader would break exact equality on any row that carries no latched reference. Such rows
        # do not arise from derive_snip_transform today, but they will the moment the legacy branch
        # is deleted, and a reader that silently produced (nan, nan) would render garbage rather
        # than fail.
        from io import StringIO

        row = _row(_resolved()[0])
        row["legacy_center_on_target_rescaled_rotated_grid_x"] = None
        row["legacy_center_on_target_rescaled_rotated_grid_y"] = None
        reread = pd.read_csv(StringIO(pd.DataFrame([row]).to_csv(index=False)))
        rebuilt = canonical_from_row(reread.iloc[0])
        assert rebuilt.legacy_center_on_target_rescaled_rotated_grid_xy is None

    def test_reconstructed_transform_resolves_identically(self):
        # The second equality: a reconstructed canonical must resolve to the same product transform.
        # Reconstructing the recipe but resolving it differently would still misregister siblings.
        canonical, resolved = _resolved()
        rebuilt = canonical_from_row(_row(canonical))
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
            canonical_from_row(_row(canonical)),
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
        row = _row(_resolved()[0])
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


class TestTwoGrainsTwoGuarantees:
    """The lifecycle, tested at its two REAL boundaries rather than as one fused round trip.

    The old tests conflated them: they wrote a canonical row and replayed pixels from it, which only
    worked because the shared row wrongly carried one product's compiled chain. Separating them is
    the point --

        canonical row  -> RE-DERIVE: recompile geometry for ANY product
        product row    -> REPLAY:    reproduce THAT product's exact raster
    """

    def test_the_canonical_row_recompiles_for_any_product(self):
        # Guarantee 1. The canonical row is product-independent, so a reconstructed recipe must
        # resolve identically for whichever grid asks.
        canonical, resolved = _resolved()
        rebuilt = canonical_from_row(_row(canonical))
        again = transform_for_product(
            rebuilt,
            product_shape_hw=SOURCE_SHAPE,
            product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        assert again.rotation_matrix_2x3 == resolved.rotation_matrix_2x3
        assert again.rescaled_shape_hw == resolved.rescaled_shape_hw

    def test_the_product_row_replays_its_own_pixels(self):
        # Guarantee 2, through the FULL lifecycle: derive -> persist canonical -> reconstruct ->
        # resolve for a product -> persist that product's chain -> replay.
        image, mask = _image(), _mask()
        canonical, _ = _resolved(mask)

        rebuilt = canonical_from_row(_row(canonical))
        resolved = transform_for_product(
            rebuilt,
            product_shape_hw=SOURCE_SHAPE,
            product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        original = apply_transform_to_image(image, resolved, dtype=np.uint8)

        product_row = _product_row(resolved)
        replayed = chain_from_row(product_row).apply_to_image(image).astype(np.uint8)
        assert np.array_equal(replayed, original), (
            "the product row's chain did not reproduce its own raster -- the inventory is not "
            "sufficient provenance"
        )

    def test_the_product_row_replays_the_mask_path_too(self):
        # Image and mask take different interpolation, so replaying one proves nothing about the
        # other.
        mask = _mask()
        canonical, resolved = _resolved(mask)
        original = apply_transform_to_mask(mask, resolved)
        replayed = chain_from_row(_product_row(resolved)).apply_to_mask(
            (mask > 0.5).astype(np.uint8)
        )
        assert np.array_equal(replayed, original)

    def test_the_chain_records_every_step_not_a_composite(self):
        # A composite matrix is COORDINATE truth only: it cannot express the anti-aliased resize
        # prefilter, so a snip replayed from one would alias with no error.
        _canonical, resolved = _resolved()
        payload = json.loads(_product_row(resolved)["resolved_transform_chain_json"])
        kinds = [step["kind"] for step in payload["steps"]]
        assert "resize" in kinds and "affine" in kinds, kinds
        assert payload["realized_scale_yx"], "the REALIZED scale must be recorded, not the requested"

    def test_the_canonical_row_carries_no_chain(self):
        # THE OWNERSHIP RULE. A channel-independent row has no single resolved chain -- BF native,
        # RFP native and a 4x z_stack compile the same recipe differently. Storing one here would
        # make whichever product wrote it authoritative for all of them.
        assert "resolved_transform_chain_json" not in _row(_resolved()[0])
        assert "resolved_transform_chain_json" not in SNIP_TRANSFORM_TABLE_COLUMNS


class TestFlipXIsReserved:
    def test_flip_x_true_is_rejected(self):
        # The column exists for schema continuity, but no render path applies a reflection. A row
        # claiming a flip its pixels never received is worse than no column at all.
        row = _row(_resolved()[0])
        row["flip_x"] = True
        with pytest.raises(SnipTransformTableError, match="flip_x=True is not supported"):
            validate_snip_transform_table(pd.DataFrame([row]))

    def test_flip_x_false_passes(self):
        validate_snip_transform_table(pd.DataFrame([_row(_resolved()[0])]))


class TestRederive:
    """Guarantee 2: the INPUTS support building a DIFFERENT transform, not just replaying this one."""

    def test_row_carries_the_inputs_a_fresh_derivation_needs(self):
        canonical, resolved = _resolved()
        inputs = derivation_inputs_from_row(_row(canonical))
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
        inputs = derivation_inputs_from_row(_row(canonical))

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
        inputs = derivation_inputs_from_row(_row(canonical))

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

    def test_centering_mode_is_recorded_on_the_PRODUCT_row(self):
        # Centering is a RESOLUTION choice, not a property of the canonical recipe -- it is an
        # argument to transform_for_product, and two products of one embryo-time could in principle
        # resolve differently. So the mode that produced a given raster belongs with that raster,
        # which is the single most important field for interpreting a snip across the migration.
        _, latched = _resolved(centering=CENTERING_LATCHED)
        _, continuous = _resolved(centering=CENTERING_CONTINUOUS)
        assert json.loads(_product_row(latched)["resolved_transform_chain_json"])["centering"] == (
            CENTERING_LATCHED
        )
        assert json.loads(
            _product_row(continuous)["resolved_transform_chain_json"]
        )["centering"] == CENTERING_CONTINUOUS


class TestSiblingsShareOneTransform:
    def test_bf_and_rfp_reference_one_row_and_compile_it_separately(self, tmp_path):
        """The registerability guarantee, with ownership correct.

        Two channels of one embryo-time derive the SAME recipe from the same mask, so they collapse
        onto ONE canonical row -- that collapse is what makes sibling geometry drift
        unrepresentable. Each then compiles that shared recipe into its OWN chain, because a chain
        is expressed in one product's coordinates and cannot be shared.
        """
        mask = _mask()
        canonical, _ = _resolved(mask)
        transform_id = build_snip_transform_id("20250912_B01_e01", 0)

        # ONE canonical row. Both products reference it; neither owns a copy.
        row = _row(canonical, snip_transform_id=transform_id)
        path = write_snip_transform_table(pd.DataFrame([row]), tmp_path / "t.parquet")
        shared = canonical_from_row(read_snip_transform_table(path).iloc[0])

        # Each product compiles it for its own grid. Same recipe in, different chains out.
        bf = transform_for_product(
            shared, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        rfp = transform_for_product(
            shared, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )

        bf_image, rfp_image = _image(0), _image(1)
        bf_snip = chain_from_row(_product_row(bf, snip_transform_id=transform_id)).apply_to_image(
            bf_image
        )
        rfp_snip = chain_from_row(_product_row(rfp, snip_transform_id=transform_id)).apply_to_image(
            rfp_image
        )

        # Same geometry: different pixels in, identical placement out.
        assert bf_snip.shape == rfp_snip.shape
        assert not np.array_equal(bf_snip, rfp_snip), "the fixture must feed different pixels"
        np.testing.assert_array_equal(
            chain_from_row(_product_row(bf, snip_transform_id=transform_id)).apply_to_image(
                bf_image
            ),
            bf_snip,
        )

    def test_duplicate_transform_ids_are_rejected(self, tmp_path):
        # Two rows claiming one embryo-time is exactly the sibling drift this table prevents.
        canonical, resolved = _resolved()
        dupe = pd.DataFrame([_row(canonical), _row(canonical)])
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
        canonical, resolved = _resolved()
        row = _row(canonical)
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
        canonical, resolved = _resolved()
        path = tmp_path / "t.parquet"

        def boom(self, *a, **k):
            raise RuntimeError("simulated writer failure")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError, match="simulated writer failure"):
            write_snip_transform_table(pd.DataFrame([_row(canonical)]), path)

        assert not path.exists(), "a failed write left a table at the final path"
        assert not list(tmp_path.glob("*.tmp-*")), "a failed write left its temp file behind"

    def test_failed_write_does_not_clobber_an_existing_table(self, tmp_path, monkeypatch):
        # The stronger property: a failed RERUN must leave the previous good table intact.
        canonical, resolved = _resolved()
        path = write_snip_transform_table(pd.DataFrame([_row(canonical)]), tmp_path / "t.parquet")
        before = path.read_bytes()

        def boom(self, *a, **k):
            raise RuntimeError("simulated writer failure")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError):
            write_snip_transform_table(pd.DataFrame([_row(canonical)]), path)

        assert path.read_bytes() == before, "a failed rerun corrupted the previous good table"

    def test_invalid_table_never_becomes_visible(self, tmp_path):
        # Validation happens BEFORE the rename, so an invalid table is never published at all.
        canonical, resolved = _resolved()
        path = tmp_path / "t.parquet"
        with pytest.raises(SnipTransformTableError):
            write_snip_transform_table(pd.DataFrame([_row(canonical), _row(canonical)]), path)
        assert not path.exists()
