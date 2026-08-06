"""surface_area_qc compute — stage-binned two-sided area outlier flag, one row per snip.

Pure logic, explicit inputs, no global config, fail loud. The flag is::

    sa_outlier_flag = area_um2 > k_upper * p95(stage)  OR  area_um2 < k_lower * p5(stage)

where ``(p5, p95)`` are interpolated per snip at ``predicted_stage_hpf`` from the validated
reference curve. A snip whose upstream stage is intentionally unresolved emits
``sa_outlier_flag=False`` with ``surface_area_qc_applicability=not_applicable``.
The output carries the FULL snip spine taken from the universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
)

from .config import SurfaceAreaQCConfig
from .reference import interpolate_reference_band
from data_pipeline.quality_control.applicability import (
    QC_APPLICABILITY_EXCLUSION,
    QC_APPLICABILITY_NOT_APPLICABLE,
)


def compute_surface_area_flag(
    area_um2: float,
    stage_hpf: float,
    surface_area_reference_df: pd.DataFrame,
    *,
    k_upper: float,
    k_lower: float,
) -> bool:
    """Return True if ``area_um2`` falls outside the stage-interpolated tolerance band."""
    p5, p95 = interpolate_reference_band(stage_hpf, surface_area_reference_df)
    too_large = area_um2 > k_upper * p95
    too_small = area_um2 < k_lower * p5
    return bool(too_large or too_small)


def compute_surface_area_qc_flags(
    mask_geometry_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    snip_universe_df: pd.DataFrame,
    surface_area_reference_df: pd.DataFrame,
    *,
    config: SurfaceAreaQCConfig,
) -> pd.DataFrame:
    """Return one surface_area_qc row per snip in the universe (full spine + sa_outlier_flag)."""
    area_col = config.area_column
    stage_col = config.stage_column

    universe = snip_universe_df.copy()
    _require_unique_snip_id(universe, "surface_area_qc universe")

    area_lookup = _one_to_one_lookup(mask_geometry_df, area_col, "mask_geometry", config.missing_area_policy)
    stage_lookup = _one_to_one_lookup(stage_df, stage_col, "stage_predictions", config.missing_stage_policy)

    out = universe[list(SNIP_ID_SPINE_COLUMNS)].copy()
    flags: list[bool] = []
    applicability: list[str] = []
    for snip_id in out["snip_id"].astype(str):
        if snip_id not in area_lookup.index:
            raise ValueError(
                f"surface_area_qc: snip_id {snip_id!r} in the universe has no mask_geometry row "
                f"({area_col!r}). Every snip needs an area to be judged."
            )
        if snip_id not in stage_lookup.index:
            raise ValueError(
                f"surface_area_qc: snip_id {snip_id!r} in the universe has no stage_predictions row "
                f"({stage_col!r}). surface_area_qc is stage-binned; there is no stage-free band in MVP."
            )
        area = float(area_lookup.loc[snip_id])
        stage_value = stage_lookup.loc[snip_id]
        if pd.isna(stage_value):
            if config.missing_stage_policy != "not_applicable":
                raise ValueError(
                    f"surface_area_qc: snip_id {snip_id!r} has no resolved stage "
                    f"(missing_stage_policy={config.missing_stage_policy!r})."
                )
            flags.append(False)
            applicability.append(QC_APPLICABILITY_NOT_APPLICABLE)
            continue
        stage = float(stage_value)
        flags.append(
            compute_surface_area_flag(
                area, stage, surface_area_reference_df, k_upper=config.k_upper, k_lower=config.k_lower
            )
        )
        applicability.append(QC_APPLICABILITY_EXCLUSION)

    out["sa_outlier_flag"] = pd.array(flags, dtype=bool)
    out["surface_area_qc_applicability"] = applicability
    return out


def _require_unique_snip_id(df: pd.DataFrame, label: str) -> None:
    if "snip_id" not in df.columns:
        raise ValueError(f"{label}: missing snip_id column.")
    dupes = df["snip_id"][df["snip_id"].duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"{label}: duplicate snip_id(s) {dupes[:5]}; expected one row per snip.")


def _one_to_one_lookup(df: pd.DataFrame, value_col: str, label: str, missing_policy: str) -> pd.Series:
    """Return a snip_id -> value Series, failing loud on dup snip_id, missing/non-finite values."""
    if "snip_id" not in df.columns:
        raise ValueError(f"surface_area_qc: {label} input missing snip_id column.")
    if value_col not in df.columns:
        raise ValueError(
            f"surface_area_qc: {label} input missing required column {value_col!r}. "
            f"Available: {sorted(df.columns)}."
        )
    _require_unique_snip_id(df, f"surface_area_qc {label}")

    keyed = df.set_index(df["snip_id"].astype(str))[value_col]
    values = pd.to_numeric(keyed, errors="coerce")
    if missing_policy == "fail":
        bad = keyed.index[values.isna() | ~np.isfinite(values.to_numpy(dtype=float))].tolist()
        if bad:
            raise ValueError(
                f"surface_area_qc: {label} column {value_col!r} has null/non-finite value(s) for "
                f"snip_id(s) {bad[:5]}. (missing_*_policy=fail)"
            )
    return values
