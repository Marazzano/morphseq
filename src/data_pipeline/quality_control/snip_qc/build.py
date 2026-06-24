"""snip_qc build — pure verdict logic. No path/IO imports, no product artifact names.

Consumes ``snip_universe_df`` (the full snip spine) and an already-assembled ``qc_flags_df``
(snip_id + the flag columns named by the exclusion map). For each snip, ``qc_fail_reasons`` is the
pipe-delimited list of reasons whose flag is true (in the map's declared order), and
``use_snip = (qc_fail_reasons == "")``. Returns exactly SNIP_QC_REQUIRED_COLUMNS.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
)

from .contract import SNIP_QC_REQUIRED_COLUMNS


def build_snip_qc_verdict(
    snip_universe_df: pd.DataFrame,
    qc_flags_df: pd.DataFrame,
    *,
    exclusion_reasons: dict[str, str],
) -> pd.DataFrame:
    """Return the per-snip verdict table (full spine + use_snip + qc_fail_reasons)."""
    # Start from the FULL spine — never from [["snip_id"]] (the result must satisfy the contract).
    missing_spine = [c for c in SNIP_ID_SPINE_COLUMNS if c not in snip_universe_df.columns]
    if missing_spine:
        raise ValueError(f"snip_qc build: universe missing spine column(s) {missing_spine}.")
    out = snip_universe_df[list(SNIP_ID_SPINE_COLUMNS)].copy()

    _require_unique(out, "snip_id", "snip_qc universe")
    _require_unique(qc_flags_df, "snip_id", "snip_qc flags")
    if set(out["snip_id"].astype(str)) != set(qc_flags_df["snip_id"].astype(str)):
        raise ValueError("snip_qc build: qc_flags_df snip_id set must match the universe exactly.")

    for col in exclusion_reasons.values():
        if col not in qc_flags_df.columns:
            raise ValueError(
                f"snip_qc build: flag column {col!r} (named by an exclusion reason) is not in "
                f"qc_flags_df. Columns present: {sorted(qc_flags_df.columns)}."
            )
        if qc_flags_df[col].isna().any():
            raise ValueError(f"snip_qc build: flag column {col!r} has null value(s).")
        if qc_flags_df[col].dtype != bool:
            raise ValueError(f"snip_qc build: flag column {col!r} must be boolean dtype.")

    flags_by_snip = qc_flags_df.set_index(qc_flags_df["snip_id"].astype(str))
    reasons_out: list[str] = []
    for snip_id in out["snip_id"].astype(str):
        row = flags_by_snip.loc[snip_id]
        fired = [reason for reason, col in exclusion_reasons.items() if bool(row[col])]
        reasons_out.append("|".join(fired))

    out["qc_fail_reasons"] = reasons_out
    out["use_snip"] = pd.array([r == "" for r in reasons_out], dtype=bool)
    return out[SNIP_QC_REQUIRED_COLUMNS]


def _require_unique(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df.columns:
        raise ValueError(f"{label}: missing key column {key!r}.")
    dupes = df[key][df[key].duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"{label}: duplicate {key} value(s) {dupes[:5]}; expected one row per {key}.")
