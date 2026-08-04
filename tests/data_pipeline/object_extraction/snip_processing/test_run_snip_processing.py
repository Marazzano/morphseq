"""End-to-end test for run_snip_processing.

Uses FakePredictor to produce real frame_masks output, writes synthetic
grayscale images to a temp directory, and runs run_snip_processing end-to-end.
No GPU or real SAM2 is invoked.
"""

from __future__ import annotations

import json

from data_pipeline.object_extraction.segmentation.masks.mask_rle import (
    decode_binary_mask_rle,
    encode_binary_mask_rle,
)

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.backends.sam2_video.fake_predictor import (
    FakePredictor,
    segment_one_well_fake,
)
from data_pipeline.object_extraction.segmentation.sam2_video.model_loader import Sam2VideoModelConfig
from data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video import (
    Sam2WellInput,
    run_sam2_video_for_wells,
)
from data_pipeline.acquisition.metadata_ingest.collection_provenance import (
    build_collection_provenance,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
    build_physical_embryo_registry,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_INVENTORY_COLUMNS,
    validate_snip_inventory_contract,
)
from data_pipeline.shared.identifiers import build_image_id, build_well_id
from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_processing import run_snip_processing


WELL_ID = build_well_id("20250912", "B01")
N_FRAMES = 3
N_OBJECTS = 2
IMG_W, IMG_H = 64, 64


def _make_frame_inventory(tmp_images: Path) -> pd.DataFrame:
    # Fixture images must be byte-stable across runs, or a pixel regression test pins the RNG
    # rather than the pipeline. Use a LOCAL Generator, never np.random.seed(): the code under test
    # calls the global np.random.seed() itself (309 in _estimate_background, 42 in
    # generate_background_noise), so a globally-seeded fixture would have its state clobbered
    # mid-test and its determinism would silently depend on call ordering.
    rng = np.random.default_rng(20250912)
    rows = []
    for t in range(N_FRAMES):
        image_id = build_image_id(WELL_ID, "BF", t)
        img_path = tmp_images / f"{image_id}.png"
        img = rng.integers(50, 200, (IMG_H, IMG_W), dtype=np.uint8)
        skio.imsave(str(img_path), img, check_contrast=False)
        rows.append({
            "experiment_id": "20250912",
            "well_id": WELL_ID,
            "image_id": image_id,
            "time_index": t,
            "z_index": pd.NA,
            "channel_id": "BF",
            # Required by frame_inventory_contract and part of frame identity: the effective key is
            # (image_id, product_key), so the product atoms must be present for a product-aware
            # source lookup to resolve.
            "image_product_type": "projection",
            "projection_method": "focus_stack",
            "image_path": str(img_path),
            "image_width_px": IMG_W,
            "image_height_px": IMG_H,
            "image_micrometers_per_pixel": 2.17,
        })
    return pd.DataFrame(rows)


def _make_frame_masks(frame_inventory: pd.DataFrame) -> pd.DataFrame:
    well = Sam2WellInput(
        well_id=WELL_ID,
        model_frame_view=frame_inventory,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=N_OBJECTS)
    config = Sam2VideoModelConfig(
        models_root=Path("/fake/models"),
        config_path=Path("/fake/config.yaml"),
        checkpoint_path=Path("/fake/checkpoint.pt"),
        device="cpu",
        model_id="fake_predictor:v1",
    )
    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well], model_config=config, segment_one_well=segment_one_well_fake,
        )
    return results[0].frame_masks


def _write_inputs(tmp_path, *, include_registry=True):
    """Write frame_inventory, frame_masks, and (optionally) the registry shard to tmp_path.

    Returns (frame_masks, frame_masks_csv, frame_inventory_csv, registry_csv). The registry is
    built from the SAME frame_masks via the real Stage-2 builder — exactly how the pipeline does
    it — so the join reproduces the identity the old mint chain produced.
    """
    tmp_images = tmp_path / "images"
    tmp_images.mkdir()

    frame_inventory = _make_frame_inventory(tmp_images)
    frame_masks = _make_frame_masks(frame_inventory)

    frame_masks_csv = tmp_path / "frame_masks.csv"
    frame_inventory_csv = tmp_path / "frame_inventory.csv"
    registry_csv = tmp_path / "physical_embryo_registry.csv"
    frame_masks.to_csv(frame_masks_csv, index=False)
    frame_inventory.to_csv(frame_inventory_csv, index=False)
    if include_registry:
        # Provenance comes from the real CREATE site, not a hand-rolled dict: "20250912" is not a
        # `_coll_` id, so build_collection_provenance returns the inert single-source payload
        # (n_sources == 1 -> NORMAL merge policy), which is the legacy behavior this fixture wants.
        # It only joins raw_root into a string and never reads disk, so tmp_path need not exist.
        provenance = build_collection_provenance("20250912", tmp_path / "raw")
        build_physical_embryo_registry(frame_masks, provenance).to_csv(registry_csv, index=False)

    return frame_masks, frame_masks_csv, frame_inventory_csv, registry_csv


def _run_geometry(tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv, **kw):
    """Run THE GATE, then hand back the table every render job must read.

    Every render test goes through this because that is now the only legal path: a render job that
    could derive its own geometry would reopen exactly the drift the gate closes, so
    run_snip_processing refuses rather than falling back.
    """
    from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_geometry import (
        run_snip_geometry,
    )

    table = tmp_path / "snip_transforms.csv"
    run_snip_geometry(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        output_path=table,
        target_pixel_size_um=kw.get("target_pixel_size_um", 2.17),
        output_height_px=kw.get("output_height_px", 64),
        output_width_px=kw.get("output_width_px", 64),
    )
    return table


def test_run_snip_processing_produces_inventory(tmp_path):
    frame_masks, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    output_csv = tmp_path / "snip_inventory.csv"
    snips_dir = tmp_path / "snips"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=output_csv,
        snips_dir=snips_dir,
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    assert output_csv.exists(), "snip_inventory.csv was not written"
    df = pd.read_csv(output_csv)

    # One row per valid mask row.
    n_valid = int(frame_masks["is_valid_mask"].astype(bool).sum())
    assert len(df) == n_valid, f"expected {n_valid} rows, got {len(df)}"

    # Required columns present.
    required = [
        "snip_id", "embryo_id", "physical_embryo_id", "experiment_id", "well_id",
        "image_id", "time_index", "channel_id", "mask_id", "track_id",
        "image_path", "processed_snip_path", "is_valid_snip", "error_message",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"missing columns: {missing}"

    # All snips should be valid.
    failures = df[~df["is_valid_snip"].astype(bool)]
    assert failures.empty, f"some snips failed:\n{failures[['snip_id', 'error_message']]}"

    # snip_id is unique.
    assert not df["snip_id"].duplicated().any(), "duplicate snip_ids"

    # Pixel files exist on disk.
    for _, row in df.iterrows():
        png = tmp_path / str(row["processed_snip_path"])
        assert png.exists(), f"pixel file missing: {png}"


def test_run_snip_processing_joins_registry_physical_embryo_id(tmp_path):
    """The joined physical_embryo_id equals what the registry resolves for each (well, track)."""
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    output_csv = tmp_path / "snip_inventory.csv"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=output_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    snips = pd.read_csv(output_csv)
    registry = pd.read_csv(registry_csv)
    expected = {
        (str(r["well_id"]), str(r["track_id"])): str(r["physical_embryo_id"])
        for _, r in registry.iterrows()
    }
    for _, row in snips.iterrows():
        key = (str(row["well_id"]), str(row["track_id"]))
        assert str(row["physical_embryo_id"]) == expected[key], (
            f"snip physical_embryo_id for {key} did not match the registry"
        )


def test_run_snip_processing_writes_valid_headered_inventory_for_empty_well(tmp_path):
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    frame_masks = pd.read_csv(frame_masks_csv).iloc[0:0]
    frame_masks.to_csv(frame_masks_csv, index=False)
    output_csv = tmp_path / "snip_inventory.csv"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=output_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    result = pd.read_csv(output_csv)
    assert result.empty
    assert tuple(result.columns) == SNIP_INVENTORY_COLUMNS
    validate_snip_inventory_contract(result)


def test_run_snip_processing_snip_pixels_are_stable(tmp_path):
    """Pin the produced snip PIXELS, so a refactor that silently changes them fails here.

    This is the regression gate for the snip-product refactor: the geometry/render split, the
    dtype threading through crop_to_embryo_bounds, and the recipe seam must leave the default
    BF path producing exactly the bytes it produces today.

    Two assertions at different strengths, deliberately:

    * **Decoded-array equality is the load-bearing one.** It is what "the pixels did not change"
      actually means, and it is immune to PNG encoder/metadata churn.
    * **File byte-hash is a secondary signal.** It can shift on an encoder or metadata change with
      the pixels untouched; if it breaks ALONE, investigate before assuming a defect.

    Both depend on the two GLOBAL np.random seeds the pipeline sets for itself — 309 in
    ``_estimate_background`` (sampled over ``valid_masks.index``) and 42 in
    ``generate_background_noise``. Because the background estimate is drawn from the *row set and
    ordering* of ``valid_masks``, ANY change to how that frame is built or iterated changes every
    output byte in the well. That coupling is the reason this test exists.
    """
    import hashlib

    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    output_csv = tmp_path / "snip_inventory.csv"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=output_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    df = pd.read_csv(output_csv).sort_values("snip_id")
    assert df["is_valid_snip"].astype(bool).all(), "fixture must produce only valid snips"

    # sha256 of the written PNG bytes, keyed by snip_id. Captured from this fixture with the
    # seeded generator above; verified identical across independent runs in separate tmpdirs.
    #
    # RE-PINNED DELIBERATELY when the entrypoint moved onto the snip transform seam. That commit
    # varies TWO things at once — the resample kernel (skimage `rescale`/`resize` -> cv2, which
    # applies INTER_AREA on downscale) and the render path (the affine now writes straight onto the
    # snip canvas, eliminating the bounds-expanded rotation intermediate and its separate crop).
    #
    # Centering is deliberately held at `legacy_latched`, which reproduces BOTH legacy's centering
    # REFERENCE (the center measured on the rescaled, ROTATED mask) and its int() QUANTIZER — so
    # placement is not an intended contributor here and these bytes reflect kernel + render path.
    # The continuous-centering repair moves the reference to the source, pre-rotation mask and
    # re-pins these values again separately.
    expected_png_sha256 = {
        "20250912_B01_e01_BF_t0000": "711abd84c13c6beb",
        "20250912_B01_e01_BF_t0001": "a30b945f0a4a8e34",
        "20250912_B01_e01_BF_t0002": "a54aae17275a3cdc",
        "20250912_B01_e02_BF_t0000": "9ce06e208d9f962d",
        "20250912_B01_e02_BF_t0001": "bb3c224b6c51ec9d",
        "20250912_B01_e02_BF_t0002": "a3f70b9cfb3a71b7",
    }
    assert set(df["snip_id"]) == set(expected_png_sha256), (
        "fixture produced a different snip set than the pinned baseline"
    )

    mismatched_bytes = []
    for _, row in df.iterrows():
        snip_id = str(row["snip_id"])
        png = tmp_path / str(row["processed_snip_path"])

        # Load-bearing: the decoded array must be exactly what it was. dtype and shape are
        # asserted explicitly because a silent dtype change (the H1 truncation hazard) would
        # otherwise only surface as an opaque hash mismatch.
        decoded = skio.imread(str(png))
        assert decoded.dtype == np.uint8, f"{snip_id}: expected uint8, got {decoded.dtype}"
        assert decoded.shape == (64, 64), f"{snip_id}: unexpected shape {decoded.shape}"

        byte_digest = hashlib.sha256(png.read_bytes()).hexdigest()[:16]
        if byte_digest != expected_png_sha256[snip_id]:
            mismatched_bytes.append((snip_id, expected_png_sha256[snip_id], byte_digest))

    assert not mismatched_bytes, (
        "snip PNG bytes changed from the pinned baseline "
        "(snip_id, expected, actual):\n  "
        + "\n  ".join(f"{s}: {e} -> {a}" for s, e, a in mismatched_bytes)
        + "\n\nIf ONLY this assertion fails and the decoded arrays are unchanged, suspect a PNG "
          "encoder/metadata change rather than a pipeline defect. If the background statistics "
          "also moved, suspect a change to the valid_masks row set or iteration order (see the "
          "global-seed coupling in the docstring)."
    )


def test_run_snip_processing_background_estimate_is_stable(tmp_path):
    """Pin ``_estimate_background``, the upstream input to every rendered byte.

    Separated from the pixel test so a failure says WHICH layer moved: if this fails too, the
    background statistics changed (row set / iteration order / seed); if only the pixel test
    fails, the render path changed while its inputs held.
    """
    from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_processing import (
        _estimate_background,
    )

    _, frame_masks_csv, frame_inventory_csv, _ = _write_inputs(tmp_path, include_registry=False)
    frame_masks = pd.read_csv(frame_masks_csv)
    valid_masks = frame_masks[frame_masks["is_valid_mask"].astype(bool)].copy()
    inventory_index = pd.read_csv(frame_inventory_csv).set_index("image_id")

    bg_mean, bg_std = _estimate_background(valid_masks, inventory_index)

    assert bg_mean == pytest.approx(124.44167564655173, rel=0, abs=1e-9)
    assert bg_std == pytest.approx(43.340736738744425, rel=0, abs=1e-9)


def test_run_snip_processing_fails_loud_on_missing_registry_match(tmp_path):
    """A valid mask whose track is absent from the registry raises (contract violation)."""
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)

    # Drop every registry row -> no track resolves -> the first valid mask must fail loud.
    pd.read_csv(registry_csv).iloc[0:0].to_csv(registry_csv, index=False)

    with pytest.raises(ValueError, match="No physical_embryo_registry entry"):
        run_snip_processing(
            frame_masks_csv=frame_masks_csv,
            frame_inventory_csv=frame_inventory_csv,
            physical_embryo_registry_csv=registry_csv,
            output_csv=tmp_path / "snip_inventory.csv",
            snips_dir=tmp_path / "snips",
            output_root=tmp_path,
            target_pixel_size_um=2.17,
            output_height_px=64,
            output_width_px=64,
        )


def test_uint16_source_is_refused_not_silently_rescaled(tmp_path):
    """HAZARD ZERO. A non-uint8 frame must fail, never be autoscaled into a display frame.

    The removed code used DTYPE AS A PROXY FOR RECIPE::

        if image.dtype != np.uint8:
            image = rescale_intensity(image, in_range="image", out_range=(0, 255))

    ``in_range="image"`` is per-frame min/max autoscaling, so each frame got its own affine map
    keyed to its own extremes -- a 2-copy and a 0-copy embryo in different frames land on the same
    0-255 span. That is exact erasure of relative intensity, and it is silent. The guard meant it
    never fired on 8-bit BF and would ALWAYS fire on uint16 fluorescence: dormant precisely until
    it would destroy the measurement this pipeline exists to make.

    The failure is per ROW (the entrypoint catches per mask and marks is_valid_snip=False), so one
    bad frame invalidates its own snip instead of taking down a well full of good BF rows.
    """
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)

    # Rewrite every source frame as uint16 with values far above the 8-bit range, so a rescale
    # would be both obvious and destructive.
    inventory = pd.read_csv(frame_inventory_csv)
    for path in inventory["image_path"]:
        rng = np.random.default_rng(7)
        skio.imsave(
            str(path),
            rng.integers(3000, 12000, (IMG_H, IMG_W), dtype=np.uint16),
            check_contrast=False,
        )

    out_csv = tmp_path / "snip_inventory.csv"
    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=out_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    df = pd.read_csv(out_csv)
    assert not df["is_valid_snip"].astype(bool).any(), (
        "a uint16 source produced a 'valid' snip -- it was silently rescaled into a display frame"
    )
    reasons = " ".join(str(r) for r in df["error_message"].fillna(""))
    assert "uint8" in reasons, f"failure was not attributed to the dtype barrier: {reasons!r}"

    # And no pixels were written for them: a rescaled uint16 snip on disk is the artifact that
    # would later be measured as if it carried real intensity.
    assert not list((tmp_path / "snips").rglob("*.png"))


def test_a_second_product_renders_from_its_own_source(tmp_path):
    """A non-BF product renders from ITS OWN frames, into ITS OWN subtree, preserving uint16.

    The end-to-end claim of P1+P2 together: the geometry comes from the BF mask, the pixels come
    from the requested source product, and nothing in between autoscales them. Before the product
    key existed there was no way to ASK for this; before the recipe registry there was no way to
    render it without CLAHE.
    """
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)

    # Add an RFP sibling for every BF frame: same well, same timepoints, uint16 at a scale that
    # would be destroyed by any 8-bit coercion.
    inventory = pd.read_csv(frame_inventory_csv)
    rfp_rows = []
    for _, row in inventory.iterrows():
        rfp_id = build_image_id(WELL_ID, "RFP", int(row["time_index"]))
        rfp_path = tmp_path / "images" / f"{rfp_id}.png"
        rng = np.random.default_rng(11 + int(row["time_index"]))
        skio.imsave(
            str(rfp_path),
            rng.integers(3000, 12000, (IMG_H, IMG_W), dtype=np.uint16),
            check_contrast=False,
        )
        rfp = dict(row)
        rfp.update({
            "image_id": rfp_id,
            "channel_id": "RFP",
            "projection_method": "max",
            "image_path": str(rfp_path),
        })
        rfp_rows.append(rfp)
    pd.concat([inventory, pd.DataFrame(rfp_rows)], ignore_index=True).to_csv(
        frame_inventory_csv, index=False
    )

    out_csv = tmp_path / "rfp_snip_inventory.csv"
    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=out_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
        snip_product_key="RFP__projection__max__no_change",
    )

    df = pd.read_csv(out_csv)
    assert df["is_valid_snip"].astype(bool).all(), (
        f"RFP snips failed: {df['error_message'].dropna().tolist()[:3]}"
    )
    assert (df["snip_product_key"] == "RFP__projection__max__no_change").all()

    # Its own subtree, beneath the embryo -- not mixed in with the BF product.
    written = sorted((tmp_path / "snips").rglob("RFP__projection__max__no_change/*.png"))
    assert written, "no RFP pixels were written under the product directory"

    # THE POINT: uint16 survived. A snip that had been through the removed rescale, or through
    # clahe_blend, would come back 8-bit.
    snip = skio.imread(str([p for p in written if not p.name.endswith("_embryo.png")][0]))
    assert snip.dtype == np.uint16, f"no_change produced {snip.dtype}; dosage information is gone"
    assert snip.max() > 255, "values were compressed into the 8-bit range"


def test_a_render_job_cannot_derive_its_own_geometry(tmp_path):
    """THE GATE. Without the transform table, rendering must FAIL rather than fall back.

    A silent fallback to derivation is the whole failure mode the gate closes: two product jobs
    deriving independently can straddle a frame_masks regeneration and produce siblings that no
    longer register, with no error anywhere.
    """
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    out_csv = tmp_path / "snip_inventory.csv"
    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        output_csv=out_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )
    df = pd.read_csv(out_csv)
    assert not df["is_valid_snip"].astype(bool).any()
    reasons = " ".join(str(r) for r in df["error_message"].fillna(""))
    assert "snip_geometry must run" in reasons, reasons


def test_a_changed_mask_fails_the_job_not_the_row(tmp_path):
    """MASK-HASH FREEZE, enforced as a PREFLIGHT CONTRACT rather than a row condition.

    A hash mismatch is not a bad embryo. It says the renderer is consuming a different REVISION of
    frame_masks than the geometry gate certified, so placement and mask no longer describe the same
    segmentation event. Every resulting artifact is individually plausible -- which is what makes it
    dangerous -- so marking rows invalid would prevent their use while NORMALIZING a condition that
    must never occur, and hundreds of invalid rows would bury the real diagnosis.

    The job therefore fails before writing anything, reports ALL mismatches at once, and says how to
    recover.
    """
    from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_processing import (
        MaskProvenanceError,
    )

    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    table = _run_geometry(tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv)

    # Mutate one mask's PIXELS (not merely its encoding) after geometry was derived.
    masks = pd.read_csv(frame_masks_csv)
    decoded = decode_binary_mask_rle(json.loads(str(masks.loc[0, "mask_rle"])))
    decoded[decoded.shape[0] // 2, :] = True
    masks.loc[0, "mask_rle"] = json.dumps(encode_binary_mask_rle(decoded))
    masks.to_csv(frame_masks_csv, index=False)

    out_csv = tmp_path / "snip_inventory.csv"
    snips_dir = tmp_path / "snips"
    with pytest.raises(MaskProvenanceError) as excinfo:
        run_snip_processing(
            frame_masks_csv=frame_masks_csv,
            frame_inventory_csv=frame_inventory_csv,
            physical_embryo_registry_csv=registry_csv,
            snip_transform_table_csv=table,
            output_csv=out_csv,
            snips_dir=snips_dir,
            output_root=tmp_path,
            target_pixel_size_um=2.17,
            output_height_px=64,
            output_width_px=64,
        )

    message = str(excinfo.value)
    assert "Hash mismatches: 1" in message, message
    assert "Regenerate snip_geometry" in message, "the error must say how to recover"
    # NOTHING was written -- not a partial inventory, not one pixel.
    assert not out_csv.exists()
    assert not list(snips_dir.rglob("*.png"))


def test_a_reencoded_mask_is_not_a_mismatch(tmp_path):
    """The complement: re-serializing an UNCHANGED mask must not trip the freeze.

    Two RLE strings can encode one binary mask, so hashing the string rather than the decoded pixels
    would fail on a no-op rewrite -- catching nothing real while blocking legitimate reruns.
    """
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    table = _run_geometry(tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv)

    masks = pd.read_csv(frame_masks_csv)
    for i in masks.index:
        decoded = decode_binary_mask_rle(json.loads(str(masks.loc[i, "mask_rle"])))
        masks.loc[i, "mask_rle"] = json.dumps(encode_binary_mask_rle(decoded))
    masks.to_csv(frame_masks_csv, index=False)

    out_csv = tmp_path / "snip_inventory.csv"
    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=table,
        output_csv=out_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )
    assert pd.read_csv(out_csv)["is_valid_snip"].astype(bool).all()


def test_siblings_share_one_transform_row(tmp_path):
    """One embryo-time, one canonical transform -- referenced by every product, copied by none."""
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    table = pd.read_csv(_run_geometry(tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv))

    assert not table["snip_transform_id"].duplicated().any()
    # Channel-independent by construction: the id names an animal at a time, not a channel.
    assert not table["snip_transform_id"].str.contains("_BF_|_RFP_").any()


def test_the_product_row_records_how_this_raster_was_made(tmp_path):
    """Each product row must be self-describing about ITS OWN rendering.

    The shared transform row holds the product-independent recipe; these columns hold what that
    recipe compiled into for one product. Without them a consumer holding a snip cannot tell which
    source product it came from, whether it is pixel-comparable to a sibling, or what dtype the
    pixels actually landed as.
    """
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    out_csv = tmp_path / "snip_inventory.csv"
    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        snip_transform_table_csv=_run_geometry(
            tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv
        ),
        output_csv=out_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )
    row = pd.read_csv(out_csv).iloc[0]

    assert row["source_image_product_key"] == "BF__projection__focus_stack"
    assert row["snip_transform_id"], "the FK into the shared geometry row must be populated"
    # PIXEL-registerability, distinct from physical: same transform id AND same grid id.
    assert row["output_grid_id"] == "yx64x64_um2.17x2.17"
    # Read back FROM THE WRITTEN FILE. A uint16 source silently written as uint8 would look correct
    # in every other column.
    assert row["pixel_dtype"] == "uint8"

    chain = json.loads(row["resolved_transform_chain_json"])
    assert [s["kind"] for s in chain["steps"]] == ["resize", "affine"]


def test_siblings_share_a_transform_but_own_their_chains(tmp_path):
    """The registerability guarantee at the artifact layer.

    Two products of one embryo-time reference ONE geometry row -- that shared reference is what
    makes drift unrepresentable -- while each records its own compiled chain and dtype, because a
    chain is expressed in one product's coordinates and cannot be shared.
    """
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    inventory = pd.read_csv(frame_inventory_csv)
    rfp_rows = []
    for _, r in inventory.iterrows():
        rfp_id = build_image_id(WELL_ID, "RFP", int(r["time_index"]))
        rfp_path = tmp_path / "images" / f"{rfp_id}.png"
        rng = np.random.default_rng(31 + int(r["time_index"]))
        skio.imsave(
            str(rfp_path),
            rng.integers(3000, 12000, (IMG_H, IMG_W), dtype=np.uint16),
            check_contrast=False,
        )
        row = dict(r)
        row.update({
            "image_id": rfp_id, "channel_id": "RFP",
            "projection_method": "max", "image_path": str(rfp_path),
        })
        rfp_rows.append(row)
    pd.concat([inventory, pd.DataFrame(rfp_rows)], ignore_index=True).to_csv(
        frame_inventory_csv, index=False
    )

    table = _run_geometry(tmp_path, frame_masks_csv, frame_inventory_csv, registry_csv)
    written = {}
    for key in ("BF__projection__focus_stack__clahe_blend", "RFP__projection__max__no_change"):
        out_csv = tmp_path / f"{key}.csv"
        run_snip_processing(
            frame_masks_csv=frame_masks_csv,
            frame_inventory_csv=frame_inventory_csv,
            physical_embryo_registry_csv=registry_csv,
            snip_transform_table_csv=table,
            output_csv=out_csv,
            snips_dir=tmp_path / "snips",
            output_root=tmp_path,
            target_pixel_size_um=2.17,
            output_height_px=64,
            output_width_px=64,
            snip_product_key=key,
        )
        written[key] = pd.read_csv(out_csv)

    bf, rfp = written.values()
    for name, df in written.items():
        assert df["is_valid_snip"].astype(bool).all(), (
            f"{name}: {df['error_message'].dropna().tolist()[:2]}"
        )

    # ONE shared geometry row per embryo-time.
    assert set(bf["snip_transform_id"]) == set(rfp["snip_transform_id"])
    # Same output grid -> PIXEL-registerable, not merely physically related.
    assert set(bf["output_grid_id"]) == set(rfp["output_grid_id"])
    # But each owns its dtype: the whole reason no_change exists.
    assert bf["pixel_dtype"].iloc[0] == "uint8"
    assert rfp["pixel_dtype"].iloc[0] == "uint16"

    # EXACTLY ONE PRODUCT OWNS THE LEGACY ALIAS. The flat layout predates product keys and has no
    # product dimension, so only the product legacy consumers meant can claim it. This was found by
    # the collision guard rather than by design: with every product linking, the RFP job failed
    # instead of silently repointing BF's alias at RFP pixels.
    assert bf["legacy_flat_snip_path"].notna().all()
    assert rfp["legacy_flat_snip_path"].isna().all(), (
        "a non-default product claimed the legacy flat path; it is reachable only through its "
        "product path and snip_product_key"
    )
