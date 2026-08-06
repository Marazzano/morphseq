"""Conservative source-FOV pixel-scale calibration for SeaHub images.

SeaHub source images contain eight embryos in one field of view (FOV).  Scale is therefore
estimated once for the FOV and must be broadcast unchanged to all embryos cropped from it.  The
estimate uses the median area of valid embryo masks and a versioned, stage-specific strict-WT
surface-area reference::

    raw_um_per_px = sqrt(strict_wt_p50_area_um2 / median_valid_mask_area_px)

The raw value is deliberately only half-weighted relative to the stage prior, then constrained to
within 20% of that prior and to the absolute interval [3.5, 9.0] um/px.  FOVs with fewer than four
valid masks use the stage prior without attempting a mask-derived estimate.  An unresolved or
unreferenced stage retains the explicit 7.8 um/px global fallback so reconciliation's intentional
stage-failure pass-through remains processable.

This module only computes a reviewable FOV-level table.  It does not modify frame inventories,
snips, QC verdicts, or pipeline configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REFERENCE_VERSION = "v1"
ABSOLUTE_MIN_UM_PER_PX = 3.5
ABSOLUTE_MAX_UM_PER_PX = 9.0
RELATIVE_BOUND_FRACTION = 0.20
MIN_VALID_MASKS = 4
RAW_SCALE_WEIGHT = 0.50
GLOBAL_FALLBACK_UM_PER_PX = 7.8

_REFERENCES_DIR = Path(__file__).resolve().parent / "references"
_REFERENCE_COLUMNS = (
    "reference_version",
    "stage_hpf",
    "stage_prior_um_per_px",
    "strict_wt_reference_n",
    "strict_wt_p50_area_um2",
    "prior_n_fovs",
    "prior_n_embryos",
    "reference_source",
)

CALIBRATION_OUTPUT_COLUMNS = (
    "source_fov_id",
    "source_stage_value",
    "stage_hpf",
    "image_micrometers_per_pixel",
    "calibration_status",
    "scale_estimation_status",
    "calibration_method",
    "calibration_issue",
    "calibration_reference_version",
    "calibration_reference_source",
    "strict_wt_reference_n",
    "strict_wt_p50_area_um2",
    "stage_prior_um_per_px",
    "prior_n_fovs",
    "prior_n_embryos",
    "n_mask_rows",
    "n_valid_masks",
    "median_valid_mask_area_px",
    "raw_inferred_um_per_px",
    "regularization_raw_weight",
    "regularized_um_per_px_before_bounds",
    "relative_lower_bound_um_per_px",
    "relative_upper_bound_um_per_px",
    "absolute_lower_bound_um_per_px",
    "absolute_upper_bound_um_per_px",
    "relative_bound_applied",
    "absolute_bound_applied",
    "min_valid_masks_required",
)


@dataclass(frozen=True)
class SeaHubScaleCalibrationConfig:
    """Small, explicit set of calibration guardrails."""

    reference_version: str = DEFAULT_REFERENCE_VERSION
    min_valid_masks: int = MIN_VALID_MASKS
    raw_scale_weight: float = RAW_SCALE_WEIGHT
    relative_bound_fraction: float = RELATIVE_BOUND_FRACTION
    absolute_min_um_per_px: float = ABSOLUTE_MIN_UM_PER_PX
    absolute_max_um_per_px: float = ABSOLUTE_MAX_UM_PER_PX
    global_fallback_um_per_px: float = GLOBAL_FALLBACK_UM_PER_PX

    def __post_init__(self) -> None:
        if not str(self.reference_version).strip():
            raise ValueError("reference_version must be non-empty.")
        if int(self.min_valid_masks) < 1:
            raise ValueError("min_valid_masks must be >= 1.")
        if not 0.0 <= float(self.raw_scale_weight) <= 1.0:
            raise ValueError("raw_scale_weight must be in [0, 1].")
        if not 0.0 <= float(self.relative_bound_fraction) < 1.0:
            raise ValueError("relative_bound_fraction must be in [0, 1).")
        if float(self.absolute_min_um_per_px) <= 0:
            raise ValueError("absolute_min_um_per_px must be > 0.")
        if float(self.absolute_max_um_per_px) <= float(self.absolute_min_um_per_px):
            raise ValueError("absolute_max_um_per_px must exceed absolute_min_um_per_px.")
        if not (
            float(self.absolute_min_um_per_px)
            <= float(self.global_fallback_um_per_px)
            <= float(self.absolute_max_um_per_px)
        ):
            raise ValueError("global_fallback_um_per_px must lie within the absolute bounds.")


def packaged_scale_reference_path(version: str = DEFAULT_REFERENCE_VERSION) -> Path:
    """Return the repository-packaged SeaHub scale-reference path."""
    return _REFERENCES_DIR / f"seahub_scale_reference_{version}.csv"


def _validated_reference(reference: pd.DataFrame, *, expected_version: str) -> pd.DataFrame:
    missing = [column for column in _REFERENCE_COLUMNS if column not in reference.columns]
    if missing:
        raise ValueError(f"SeaHub scale reference is missing columns: {missing}.")

    out = reference.loc[:, _REFERENCE_COLUMNS].copy()
    versions = out["reference_version"].dropna().astype(str).unique().tolist()
    if versions != [expected_version]:
        raise ValueError(
            "SeaHub scale reference version mismatch: "
            f"expected {expected_version!r}, found {versions}."
        )

    numeric_columns = (
        "stage_hpf",
        "stage_prior_um_per_px",
        "strict_wt_reference_n",
        "strict_wt_p50_area_um2",
        "prior_n_fovs",
        "prior_n_embryos",
    )
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any() or not np.isfinite(out[column].to_numpy(dtype=float)).all():
            raise ValueError(f"SeaHub scale reference has non-finite {column!r} values.")

    if out["stage_hpf"].duplicated().any():
        raise ValueError("SeaHub scale reference contains duplicate stage_hpf rows.")
    if (out["stage_prior_um_per_px"] <= 0).any():
        raise ValueError("SeaHub scale reference stage priors must be > 0.")
    if (out["strict_wt_reference_n"] < 1).any():
        raise ValueError("SeaHub scale reference strict_wt_reference_n must be >= 1.")
    if (out["strict_wt_p50_area_um2"] <= 0).any():
        raise ValueError("SeaHub scale reference strict-WT areas must be > 0.")
    if out["reference_source"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("SeaHub scale reference requires non-empty reference_source values.")
    return out.sort_values("stage_hpf").reset_index(drop=True)


def load_packaged_scale_reference(
    version: str = DEFAULT_REFERENCE_VERSION,
) -> pd.DataFrame:
    """Load and validate the versioned nine-stage SeaHub calibration reference."""
    path = packaged_scale_reference_path(version)
    if not path.is_file():
        available = sorted(item.name for item in _REFERENCES_DIR.glob("seahub_scale_reference_*.csv"))
        raise FileNotFoundError(
            f"SeaHub scale reference {path.name!r} was not found. Available: {available}."
        )
    return _validated_reference(pd.read_csv(path), expected_version=version)


def _coerce_valid_mask_flags(values: pd.Series) -> pd.Series:
    """Return strict booleans while accepting ordinary CSV bool/0/1 spellings."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.map(
        lambda value: str(value).strip().casefold() if not pd.isna(value) else "false"
    )
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"is_valid_mask contains unrecognized boolean values: {unknown}.")
    return normalized.map(mapping).astype(bool)


def _reference_for_stage(reference: pd.DataFrame, stage_hpf: float) -> pd.Series | None:
    matches = np.isclose(
        reference["stage_hpf"].to_numpy(dtype=float),
        float(stage_hpf),
        rtol=0.0,
        atol=1e-6,
    )
    if int(matches.sum()) == 0:
        return None
    if int(matches.sum()) > 1:  # defensive; reference validation already rejects duplicates
        raise ValueError(f"Multiple SeaHub scale references matched stage_hpf={stage_hpf:g}.")
    return reference.loc[matches].iloc[0]


def _source_fov_ids(df: pd.DataFrame, *, table_label: str) -> pd.Series:
    """Resolve the current source_fov_id contract or the production manifest's image_id alias."""
    if "source_fov_id" in df.columns:
        values = df["source_fov_id"]
    elif "image_id" in df.columns:
        values = df["image_id"]
    else:
        raise ValueError(f"{table_label} must contain source_fov_id or image_id.")
    identifiers = values.astype("string")
    if identifiers.isna().any() or identifiers.str.strip().eq("").any():
        raise ValueError(f"{table_label} contains null/empty source FOV identities.")
    return identifiers.astype(str)


def calibrate_reconciled_source_fovs(
    reconciled_fovs: pd.DataFrame,
    mask_manifest: pd.DataFrame,
    *,
    config: SeaHubScaleCalibrationConfig | None = None,
    reference_df: pd.DataFrame | None = None,
    stage_col: str = "stage_hpf",
    mask_area_col: str = "mask_area_px",
    mask_score_col: str = "mask_score",
    min_mask_score: float | None = None,
) -> pd.DataFrame:
    """Adapt production reconciliation + SAM2 tables into the all-FOV estimator.

    Reconciliation supplies the authoritative FOV census and stage.  The current SAM2 manifest
    identifies the same FOV as ``image_id`` and has no ``is_valid_mask`` column, so finite positive
    ``mask_area_px`` values are usable by default.  If a future manifest supplies
    ``is_valid_mask``, that verdict is also honored.  ``min_mask_score`` is opt-in; no arbitrary
    score threshold is silently imposed by calibration.
    """
    if stage_col not in reconciled_fovs.columns:
        raise ValueError(f"reconciled_fovs is missing stage column {stage_col!r}.")
    source = pd.DataFrame(
        {
            "source_fov_id": _source_fov_ids(
                reconciled_fovs, table_label="reconciled_fovs"
            ),
            "stage_hpf": reconciled_fovs[stage_col],
        }
    )

    if mask_manifest.empty:
        masks = pd.DataFrame(
            columns=["source_fov_id", "mask_area_px", "is_valid_mask"]
        )
    else:
        if mask_area_col not in mask_manifest.columns:
            raise ValueError(f"mask_manifest is missing area column {mask_area_col!r}.")
        areas = pd.to_numeric(mask_manifest[mask_area_col], errors="coerce")
        usable = pd.Series(
            np.isfinite(areas.to_numpy(dtype=float)) & areas.gt(0).to_numpy(),
            index=mask_manifest.index,
            dtype=bool,
        )
        if "is_valid_mask" in mask_manifest.columns:
            usable &= _coerce_valid_mask_flags(mask_manifest["is_valid_mask"])
        if min_mask_score is not None:
            if mask_score_col not in mask_manifest.columns:
                raise ValueError(
                    f"min_mask_score was supplied but mask_manifest lacks {mask_score_col!r}."
                )
            scores = pd.to_numeric(mask_manifest[mask_score_col], errors="coerce")
            usable &= np.isfinite(scores.to_numpy(dtype=float)) & scores.ge(
                float(min_mask_score)
            )
        masks = pd.DataFrame(
            {
                "source_fov_id": _source_fov_ids(
                    mask_manifest, table_label="mask_manifest"
                ),
                "mask_area_px": areas,
                "is_valid_mask": usable,
            }
        )

    return calibrate_source_fov_scales(
        source,
        masks,
        config=config,
        reference_df=reference_df,
    )


def calibrate_source_fov_scales(
    source_fovs: pd.DataFrame,
    mask_rows: pd.DataFrame | None = None,
    *,
    config: SeaHubScaleCalibrationConfig | None = None,
    reference_df: pd.DataFrame | None = None,
    source_fov_col: str = "source_fov_id",
    stage_col: str = "stage_hpf",
    mask_area_col: str = "mask_area_px",
    valid_mask_col: str = "is_valid_mask",
) -> pd.DataFrame:
    """Return exactly one guarded pixel-scale estimate per source FOV.

    ``source_fovs`` is the authoritative census and may include an unresolved stage. ``mask_rows``
    may be empty or omit a source FOV entirely; only rows marked valid with a finite, positive mask
    area contribute to the median.  For convenience and backwards compatibility, callers may pass
    one combined embryo-mask table as ``source_fovs`` and omit ``mask_rows``; its FOV/stage columns
    are collapsed to the source census after checking stage consistency.
    """
    config = config or SeaHubScaleCalibrationConfig()
    source_required = (source_fov_col, stage_col)
    missing = [column for column in source_required if column not in source_fovs.columns]
    if missing:
        raise ValueError(f"SeaHub source FOV rows are missing columns: {missing}.")
    if source_fovs.empty:
        raise ValueError("SeaHub source FOV rows are empty; no source FOVs can be calibrated.")

    reference = (
        load_packaged_scale_reference(config.reference_version)
        if reference_df is None
        else _validated_reference(reference_df, expected_version=config.reference_version)
    )
    source = source_fovs.loc[:, source_required].copy()
    fov_ids = source[source_fov_col].astype("string")
    if fov_ids.isna().any() or fov_ids.str.strip().eq("").any():
        raise ValueError("SeaHub source FOV rows contain null/empty identities.")
    source[source_fov_col] = fov_ids.astype(str)

    # A combined mask table legitimately repeats each FOV.  Collapse it only after confirming
    # that all non-null source stages agree.  An explicit source census must itself be unique.
    combined_input = mask_rows is None and {mask_area_col, valid_mask_col}.issubset(source_fovs.columns)
    if combined_input:
        for source_fov_id, group in source.groupby(source_fov_col, sort=False):
            numeric_stages = pd.to_numeric(group[stage_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(numeric_stages) and not np.allclose(
                numeric_stages, numeric_stages[0], rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"SeaHub source FOV {source_fov_id!r} has conflicting stages: "
                    f"{sorted(set(numeric_stages.tolist()))}."
                )
        source = source.drop_duplicates(source_fov_col, keep="first").reset_index(drop=True)
        masks_input = source_fovs
    else:
        if source[source_fov_col].duplicated().any():
            duplicates = sorted(
                source.loc[source[source_fov_col].duplicated(keep=False), source_fov_col].unique()
            )
            raise ValueError(f"SeaHub source FOV census has duplicate identities: {duplicates}.")
        masks_input = mask_rows

    mask_required = (source_fov_col, mask_area_col, valid_mask_col)
    if masks_input is None:
        masks = pd.DataFrame(columns=mask_required)
    else:
        mask_missing = [column for column in mask_required if column not in masks_input.columns]
        if mask_missing:
            raise ValueError(f"SeaHub mask rows are missing columns: {mask_missing}.")
        masks = masks_input.loc[:, mask_required].copy()
    if not masks.empty:
        mask_fov_ids = masks[source_fov_col].astype("string")
        if mask_fov_ids.isna().any() or mask_fov_ids.str.strip().eq("").any():
            raise ValueError("SeaHub mask rows contain null/empty source FOV identities.")
        masks[source_fov_col] = mask_fov_ids.astype(str)
        unknown_fovs = sorted(set(masks[source_fov_col]) - set(source[source_fov_col]))
        if unknown_fovs:
            raise ValueError(f"SeaHub mask rows contain FOVs absent from the source census: {unknown_fovs}.")
        masks[mask_area_col] = pd.to_numeric(masks[mask_area_col], errors="coerce")
        masks["_valid_mask"] = _coerce_valid_mask_flags(masks[valid_mask_col])
        masks["_usable_mask"] = (
            masks["_valid_mask"]
            & np.isfinite(masks[mask_area_col].to_numpy(dtype=float))
            & masks[mask_area_col].gt(0)
        )
    else:
        masks[mask_area_col] = pd.Series(dtype=float)
        masks["_usable_mask"] = pd.Series(dtype=bool)

    mask_groups = {
        str(source_fov_id): group
        for source_fov_id, group in masks.groupby(source_fov_col, sort=False)
    }

    output_rows: list[dict[str, object]] = []
    for _, source_row in source.iterrows():
        source_fov_id = str(source_row[source_fov_col])
        source_stage_value = source_row[stage_col]
        numeric_stage = pd.to_numeric(pd.Series([source_stage_value]), errors="coerce").iloc[0]
        stage_hpf = float(numeric_stage) if pd.notna(numeric_stage) and np.isfinite(numeric_stage) else np.nan
        reference_row = _reference_for_stage(reference, stage_hpf) if np.isfinite(stage_hpf) else None
        group = mask_groups.get(source_fov_id, masks.iloc[0:0])
        usable_areas = group.loc[group["_usable_mask"], mask_area_col].to_numpy(dtype=float)
        n_valid_masks = int(len(usable_areas))
        median_area = float(np.median(usable_areas)) if n_valid_masks else np.nan
        if reference_row is None:
            prior = np.nan
            strict_wt_p50 = np.nan
            raw_scale = np.nan
            applied_raw_weight = 0.0
            unbounded = float(config.global_fallback_um_per_px)
            status = (
                "global_fallback_unresolved_stage"
                if not np.isfinite(stage_hpf)
                else "global_fallback_unreferenced_stage"
            )
            method = "global_fallback"
            issue = (
                "unresolved_stage"
                if not np.isfinite(stage_hpf)
                else f"stage_not_in_reference:{stage_hpf:g}"
            )
            reference_source = "global_placeholder_7.8_um_per_px"
            strict_wt_reference_n = np.nan
            prior_n_fovs = np.nan
            prior_n_embryos = np.nan
            relative_lower = np.nan
            relative_upper = np.nan
            relative_bounded = unbounded
            relative_bound_applied = False
        elif n_valid_masks >= int(config.min_valid_masks):
            prior = float(reference_row["stage_prior_um_per_px"])
            strict_wt_p50 = float(reference_row["strict_wt_p50_area_um2"])
            raw_scale = float(np.sqrt(strict_wt_p50 / median_area))
            applied_raw_weight = float(config.raw_scale_weight)
            unbounded = prior + applied_raw_weight * (raw_scale - prior)
            status = "mask_area_regularized"
            method = "strict_wt_p50_over_fov_median_mask_area"
            issue = ""
            reference_source = str(reference_row["reference_source"])
            strict_wt_reference_n = int(reference_row["strict_wt_reference_n"])
            prior_n_fovs = int(reference_row["prior_n_fovs"])
            prior_n_embryos = int(reference_row["prior_n_embryos"])
            relative_lower = prior * (1.0 - float(config.relative_bound_fraction))
            relative_upper = prior * (1.0 + float(config.relative_bound_fraction))
            relative_bounded = float(np.clip(unbounded, relative_lower, relative_upper))
            relative_bound_applied = not np.isclose(unbounded, relative_bounded)
        else:
            prior = float(reference_row["stage_prior_um_per_px"])
            strict_wt_p50 = float(reference_row["strict_wt_p50_area_um2"])
            raw_scale = np.nan
            applied_raw_weight = 0.0
            unbounded = prior
            status = "stage_prior_fallback_insufficient_masks"
            method = "stage_prior_only"
            issue = (
                f"insufficient_valid_masks:{n_valid_masks}<"
                f"{int(config.min_valid_masks)}"
            )
            reference_source = str(reference_row["reference_source"])
            strict_wt_reference_n = int(reference_row["strict_wt_reference_n"])
            prior_n_fovs = int(reference_row["prior_n_fovs"])
            prior_n_embryos = int(reference_row["prior_n_embryos"])
            relative_lower = prior * (1.0 - float(config.relative_bound_fraction))
            relative_upper = prior * (1.0 + float(config.relative_bound_fraction))
            relative_bounded = unbounded
            relative_bound_applied = False

        final_scale = float(
            np.clip(
                relative_bounded,
                float(config.absolute_min_um_per_px),
                float(config.absolute_max_um_per_px),
            )
        )
        output_rows.append(
            {
                "source_fov_id": str(source_fov_id),
                "source_stage_value": source_stage_value,
                "stage_hpf": stage_hpf,
                "image_micrometers_per_pixel": final_scale,
                # The inferred scale is useful operationally but is not physical
                # metrology. Keep the shared modality contract honest and carry
                # the inference state separately in scale_estimation_status.
                "calibration_status": "placeholder",
                "scale_estimation_status": status,
                "calibration_method": method,
                "calibration_issue": issue,
                "calibration_reference_version": str(config.reference_version),
                "calibration_reference_source": reference_source,
                "strict_wt_reference_n": strict_wt_reference_n,
                "strict_wt_p50_area_um2": strict_wt_p50,
                "stage_prior_um_per_px": prior,
                "prior_n_fovs": prior_n_fovs,
                "prior_n_embryos": prior_n_embryos,
                "n_mask_rows": int(len(group)),
                "n_valid_masks": n_valid_masks,
                "median_valid_mask_area_px": median_area,
                "raw_inferred_um_per_px": raw_scale,
                "regularization_raw_weight": applied_raw_weight,
                "regularized_um_per_px_before_bounds": float(unbounded),
                "relative_lower_bound_um_per_px": float(relative_lower),
                "relative_upper_bound_um_per_px": float(relative_upper),
                "absolute_lower_bound_um_per_px": float(config.absolute_min_um_per_px),
                "absolute_upper_bound_um_per_px": float(config.absolute_max_um_per_px),
                "relative_bound_applied": relative_bound_applied,
                "absolute_bound_applied": not np.isclose(relative_bounded, final_scale),
                "min_valid_masks_required": int(config.min_valid_masks),
            }
        )

    return pd.DataFrame(output_rows, columns=CALIBRATION_OUTPUT_COLUMNS)


__all__ = [
    "ABSOLUTE_MAX_UM_PER_PX",
    "ABSOLUTE_MIN_UM_PER_PX",
    "CALIBRATION_OUTPUT_COLUMNS",
    "DEFAULT_REFERENCE_VERSION",
    "GLOBAL_FALLBACK_UM_PER_PX",
    "MIN_VALID_MASKS",
    "RAW_SCALE_WEIGHT",
    "RELATIVE_BOUND_FRACTION",
    "SeaHubScaleCalibrationConfig",
    "calibrate_reconciled_source_fovs",
    "calibrate_source_fov_scales",
    "load_packaged_scale_reference",
    "packaged_scale_reference_path",
]
