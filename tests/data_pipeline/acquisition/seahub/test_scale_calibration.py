from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.acquisition.seahub.scale_calibration import (
    CALIBRATION_OUTPUT_COLUMNS,
    SeaHubScaleCalibrationConfig,
    calibrate_reconciled_source_fovs,
    calibrate_source_fov_scales,
    load_packaged_scale_reference,
)


def _reference_row(stage_hpf: float) -> pd.Series:
    reference = load_packaged_scale_reference()
    return reference.loc[reference["stage_hpf"].eq(stage_hpf)].iloc[0]


def _mask_rows(
    *,
    source_fov_id: str = "fov-001",
    stage_hpf: float = 24.0,
    areas: list[float],
    valid: list[bool] | None = None,
) -> pd.DataFrame:
    valid = valid if valid is not None else [True] * len(areas)
    return pd.DataFrame(
        {
            "source_fov_id": [source_fov_id] * len(areas),
            "stage_hpf": [stage_hpf] * len(areas),
            "mask_area_px": areas,
            "is_valid_mask": valid,
            "source_embryo_id": [f"{source_fov_id}_p{i:02d}" for i in range(1, len(areas) + 1)],
        }
    )


def test_packaged_reference_is_the_agreed_nine_stage_table() -> None:
    reference = load_packaged_scale_reference()

    assert reference["stage_hpf"].tolist() == [12.0, 14.0, 15.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0]
    assert reference["reference_version"].unique().tolist() == ["v1"]
    assert reference.loc[reference["stage_hpf"].eq(14.0), "stage_prior_um_per_px"].item() == pytest.approx(
        4.250471607060187
    )
    assert reference.loc[reference["stage_hpf"].eq(96.0), "strict_wt_p50_area_um2"].item() == pytest.approx(
        1_068_921.6363
    )


def test_one_regularized_estimate_is_emitted_per_fov() -> None:
    reference = _reference_row(24.0)
    raw_scale = 6.0
    area = float(reference["strict_wt_p50_area_um2"]) / raw_scale**2
    masks = _mask_rows(areas=[area] * 8)

    result = calibrate_source_fov_scales(masks)

    assert len(result) == 1
    assert tuple(result.columns) == CALIBRATION_OUTPUT_COLUMNS
    row = result.iloc[0]
    prior = float(reference["stage_prior_um_per_px"])
    assert row["source_fov_id"] == "fov-001"
    assert row["n_mask_rows"] == 8
    assert row["n_valid_masks"] == 8
    assert row["median_valid_mask_area_px"] == pytest.approx(area)
    assert row["raw_inferred_um_per_px"] == pytest.approx(raw_scale)
    assert row["regularization_raw_weight"] == pytest.approx(0.5)
    assert row["image_micrometers_per_pixel"] == pytest.approx(prior + 0.5 * (raw_scale - prior))
    assert row["calibration_status"] == "placeholder"
    assert row["scale_estimation_status"] == "mask_area_regularized"
    assert not row["relative_bound_applied"]
    assert not row["absolute_bound_applied"]


def test_all_embryos_from_a_fov_receive_the_same_estimate_on_many_to_one_join() -> None:
    masks = pd.concat(
        [
            _mask_rows(source_fov_id="fov-a", areas=[20_000.0] * 8),
            _mask_rows(source_fov_id="fov-b", areas=[30_000.0] * 8),
        ],
        ignore_index=True,
    )
    calibration = calibrate_source_fov_scales(masks)
    annotated = masks.merge(
        calibration[["source_fov_id", "image_micrometers_per_pixel"]],
        on="source_fov_id",
        how="left",
        validate="many_to_one",
    )

    assert len(calibration) == 2
    assert annotated.groupby("source_fov_id")["image_micrometers_per_pixel"].nunique().eq(1).all()


def test_fewer_than_four_valid_positive_masks_uses_prior_only() -> None:
    masks = _mask_rows(
        stage_hpf=48.0,
        areas=[20_000.0, 21_000.0, 22_000.0, 23_000.0, 0.0, np.nan, 24_000.0, 25_000.0],
        valid=[True, True, True, False, True, True, False, False],
    )
    result = calibrate_source_fov_scales(masks)
    row = result.iloc[0]
    prior = float(_reference_row(48.0)["stage_prior_um_per_px"])

    assert row["n_mask_rows"] == 8
    assert row["n_valid_masks"] == 3
    assert row["calibration_status"] == "placeholder"
    assert row["scale_estimation_status"] == "stage_prior_fallback_insufficient_masks"
    assert row["calibration_method"] == "stage_prior_only"
    assert row["calibration_issue"] == "insufficient_valid_masks:3<4"
    assert np.isnan(row["raw_inferred_um_per_px"])
    assert row["regularization_raw_weight"] == 0.0
    assert row["image_micrometers_per_pixel"] == pytest.approx(prior)


@pytest.mark.parametrize(
    ("stage_hpf", "raw_scale", "expected", "relative_clipped", "absolute_clipped"),
    [
        (14.0, 10.0, 4.250471607060187 * 1.20, True, False),
        (14.0, 0.1, 3.5, True, True),
        (96.0, 20.0, 9.0, True, True),
    ],
)
def test_relative_and_absolute_guardrails_are_hard_and_auditable(
    stage_hpf: float,
    raw_scale: float,
    expected: float,
    relative_clipped: bool,
    absolute_clipped: bool,
) -> None:
    reference = _reference_row(stage_hpf)
    area = float(reference["strict_wt_p50_area_um2"]) / raw_scale**2
    result = calibrate_source_fov_scales(_mask_rows(stage_hpf=stage_hpf, areas=[area] * 8))
    row = result.iloc[0]

    assert row["raw_inferred_um_per_px"] == pytest.approx(raw_scale)
    assert row["image_micrometers_per_pixel"] == pytest.approx(expected)
    assert row["relative_bound_applied"] == relative_clipped
    assert row["absolute_bound_applied"] == absolute_clipped
    assert 3.5 <= row["image_micrometers_per_pixel"] <= 9.0


def test_conflicting_stages_within_one_source_fov_fail_loudly() -> None:
    masks = _mask_rows(areas=[20_000.0] * 8)
    masks.loc[7, "stage_hpf"] = 36.0

    with pytest.raises(ValueError, match="conflicting stages"):
        calibrate_source_fov_scales(masks)


def test_source_census_preserves_missing_masks_and_unresolved_stages() -> None:
    source_fovs = pd.DataFrame(
        {
            "source_fov_id": ["with-masks", "without-masks", "unresolved", "unreferenced"],
            "stage_hpf": [24.0, 48.0, pd.NA, 30.0],
        }
    )
    mask_rows = _mask_rows(source_fov_id="with-masks", stage_hpf=24.0, areas=[20_000.0] * 8)
    result = calibrate_source_fov_scales(source_fovs, mask_rows)
    by_fov = result.set_index("source_fov_id")

    assert result["source_fov_id"].tolist() == source_fovs["source_fov_id"].tolist()
    assert len(result) == 4
    assert by_fov.loc["without-masks", "n_valid_masks"] == 0
    assert by_fov.loc["without-masks", "scale_estimation_status"] == (
        "stage_prior_fallback_insufficient_masks"
    )
    assert by_fov.loc["without-masks", "image_micrometers_per_pixel"] == pytest.approx(
        _reference_row(48.0)["stage_prior_um_per_px"]
    )
    assert by_fov.loc["unresolved", "scale_estimation_status"] == (
        "global_fallback_unresolved_stage"
    )
    assert by_fov.loc["unresolved", "calibration_issue"] == "unresolved_stage"
    assert by_fov.loc["unresolved", "image_micrometers_per_pixel"] == 7.8
    assert by_fov.loc["unreferenced", "scale_estimation_status"] == (
        "global_fallback_unreferenced_stage"
    )
    assert by_fov.loc["unreferenced", "calibration_issue"] == "stage_not_in_reference:30"
    assert by_fov.loc["unreferenced", "image_micrometers_per_pixel"] == 7.8
    assert set(result["calibration_status"]) == {"placeholder"}


def test_production_wrapper_joins_image_id_masks_to_reconciled_fov_stage() -> None:
    reconciled = pd.DataFrame(
        {
            "source_fov_id": ["with-masks", "without-masks", "unresolved"],
            "stage_hpf": [24.0, 48.0, pd.NA],
        }
    )
    manifest = pd.DataFrame(
        {
            "image_id": ["with-masks"] * 10,
            "embryo_position": list(range(1, 11)),
            "mask_score": [0.95] * 10,
            "mask_area_px": [20_000.0] * 8 + [0.0, np.nan],
        }
    )

    result = calibrate_reconciled_source_fovs(reconciled, manifest)
    by_fov = result.set_index("source_fov_id")

    assert result["source_fov_id"].tolist() == reconciled["source_fov_id"].tolist()
    assert by_fov.loc["with-masks", "n_mask_rows"] == 10
    assert by_fov.loc["with-masks", "n_valid_masks"] == 8
    assert by_fov.loc["with-masks", "scale_estimation_status"] == "mask_area_regularized"
    assert by_fov.loc["without-masks", "scale_estimation_status"] == (
        "stage_prior_fallback_insufficient_masks"
    )
    assert by_fov.loc["unresolved", "image_micrometers_per_pixel"] == 7.8


def test_production_wrapper_honors_optional_mask_score_threshold() -> None:
    reconciled = pd.DataFrame({"image_id": ["fov-a"], "stage_hpf": [24.0]})
    manifest = pd.DataFrame(
        {
            "image_id": ["fov-a"] * 8,
            "mask_area_px": [20_000.0] * 8,
            "mask_score": [0.95, 0.95, 0.95, 0.95, 0.5, 0.5, 0.5, 0.5],
        }
    )

    result = calibrate_reconciled_source_fovs(
        reconciled,
        manifest,
        min_mask_score=0.9,
    )

    assert result.iloc[0]["n_valid_masks"] == 4
    assert result.iloc[0]["scale_estimation_status"] == "mask_area_regularized"


def test_configuration_keeps_agreed_production_guardrails() -> None:
    config = SeaHubScaleCalibrationConfig()

    assert config.min_valid_masks == 4
    assert config.raw_scale_weight == 0.5
    assert config.relative_bound_fraction == 0.20
    assert config.absolute_min_um_per_px == 3.5
    assert config.absolute_max_um_per_px == 9.0
    assert config.global_fallback_um_per_px == 7.8
