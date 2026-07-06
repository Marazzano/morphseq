"""fraction_alive compute — viability fraction per snip from snip-resolution masks.

The embryo mask is the cropped frame_masks RLE that snip_processing already saved beside each snip
(``embryo_mask`` in snip_inventory; legacy alias ``embryo_mask_snip_path``) — same crop transform as
the snip image, so it is pixel-aligned, with no model and no re-prediction. The ``via`` (dead-tissue) mask comes from the
per-snip ``snip_auxiliary_masks`` product. Both live in the one snip coordinate space governed by
``snip_frame_shape``, so the overlap is well defined. (``compute_fraction_alive`` still aligns
shapes defensively as a cheap safety net.) Empty embryo mask -> null (documented); missing VIA mask
-> policy (fail loud by default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io

from data_pipeline.feature_extraction.fraction_alive._legacy_compute import compute_fraction_alive
from data_pipeline.feature_extraction.shared.feature_table_utils import SNIP_FEATURE_TABLE_SPINE_COLUMNS

from .contract import FRACTION_ALIVE_TABLE_COLUMNS

MISSING_VIA_FAIL = "fail"
MISSING_VIA_NULL = "null"

VIA_MASK_TYPE = "via"


def _via_path_by_snip(snip_auxiliary_masks_df: pd.DataFrame) -> dict[str, str]:
    """Return ``{snip_id -> via_mask_path}`` for valid via auxiliary masks."""
    via = snip_auxiliary_masks_df[
        (snip_auxiliary_masks_df["auxiliary_mask_type"] == VIA_MASK_TYPE)
        & snip_auxiliary_masks_df["is_valid_auxiliary_mask"].astype(bool)
    ]
    return {str(s): str(p) for s, p in zip(via["snip_id"], via["auxiliary_mask_path"])}


def compute_fraction_alive_features(
    snip_inventory_df: pd.DataFrame,
    snip_auxiliary_masks_df: pd.DataFrame,
    *,
    snip_frame_shape,
    output_root=None,
    missing_via_policy: str = MISSING_VIA_FAIL,
) -> pd.DataFrame:
    """Return one fraction_alive row per snip from the saved embryo mask + the snip via mask."""
    via_by_snip = _via_path_by_snip(snip_auxiliary_masks_df)

    def _resolve(path: str) -> str:
        """Resolve a stored snip-mask path: relative paths root at ``output_root``.

        ``embryo_mask`` / ``embryo_mask_snip_path`` / auxiliary-mask paths are stored relative to
        ``output_root`` by snip_processing; an already-absolute path is used as-is. No hidden global
        root (the retired ``path_contracts`` helper that used to live here raised on call).
        """
        p = Path(str(path))
        if output_root is not None and not p.is_absolute():
            p = Path(output_root) / p
        return str(p)

    rows: list[dict] = []
    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])

        embryo_path = snip.get("embryo_mask")
        if embryo_path is None or (isinstance(embryo_path, float) and pd.isna(embryo_path)):
            embryo_path = snip.get("embryo_mask_snip_path")
        if embryo_path is None or (isinstance(embryo_path, float) and pd.isna(embryo_path)):
            raise ValueError(
                f"fraction_alive: snip {snip_id!r} has no embryo_mask. snip_processing "
                "must save the cropped embryo mask for every valid snip."
            )
        embryo_mask = io.imread(_resolve(str(embryo_path)))

        via_path = via_by_snip.get(snip_id)
        if via_path is None:
            if missing_via_policy == MISSING_VIA_NULL:
                fraction = np.nan
            else:
                raise ValueError(
                    f"fraction_alive: no valid {VIA_MASK_TYPE!r} mask for snip {snip_id!r} in "
                    "snip_auxiliary_masks. Run snip_auxiliary_masks with a via checkpoint, or set "
                    "missing_via_policy='null'."
                )
        else:
            via_mask = io.imread(via_path)
            fraction = compute_fraction_alive(
                embryo_mask, via_mask, snip_frame_shape=snip_frame_shape
            )

        row = {col: snip[col] for col in SNIP_FEATURE_TABLE_SPINE_COLUMNS}
        row["fraction_alive"] = fraction
        rows.append(row)

    return pd.DataFrame(rows, columns=FRACTION_ALIVE_TABLE_COLUMNS)
