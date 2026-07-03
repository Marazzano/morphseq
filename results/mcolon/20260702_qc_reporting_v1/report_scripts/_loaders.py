"""Review-time input loaders — the ONLY thing that changes when these reports are wired into the DAG.

Today each report is driven ad-hoc: paths are hand-globbed from the real 20250912 per-well output.
When a report is promoted into src/data_pipeline/<product>/report.py, these `load_*` calls are the
single seam that gets swapped for `artifact_path(step=..., path_mode="merged")`. The report bodies
(the renderer calls, grain logic, derived values) do not change. Merge contracts are NOT touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = _REPO_ROOT / "data_pipeline_output"
EXPERIMENT_ID = "20250912"


def merged(stage: str, product: str, suffix: str, *, ext: str = "csv") -> pd.DataFrame:
    """Concat every per-well shard for one product into the merged experiment view (stands in for
    the merged registry artifact). ``suffix`` is the filename stem after the well_id."""
    root = DATA_ROOT / stage / EXPERIMENT_ID / product / "per_well"
    paths = sorted(root.glob(f"*/*_{suffix}.{ext}"))
    if not paths:
        raise FileNotFoundError(f"No per-well {suffix}.{ext} under {root}")
    read = pd.read_parquet if ext == "parquet" else pd.read_csv
    return pd.concat((read(p) for p in paths), ignore_index=True)


def snip_image_paths() -> pd.DataFrame:
    """snip_id -> absolute processed-snip image path, for galleries."""
    inv = merged("object_extraction", "snips", "snip_inventory")
    inv["resolved_image_path"] = inv["processed_snip_path"].apply(lambda rel: str(DATA_ROOT / rel))
    return inv[["snip_id", "resolved_image_path"]]


def snip_mask_paths() -> pd.DataFrame:
    """snip_id -> absolute per-snip embryo-mask PNG (same-size crop as the processed snip). For
    reports that recompute a mask-derived geometry (curvature centerline) at render time."""
    inv = merged("object_extraction", "snips", "snip_inventory")
    inv["resolved_mask_path"] = inv["embryo_mask_snip_path"].apply(lambda rel: str(DATA_ROOT / rel))
    return inv[["snip_id", "image_id", "resolved_mask_path"]]


def pixel_size_by_image_id() -> pd.DataFrame:
    """image_id -> source_micrometers_per_pixel from frame_inventory (the micron calibration a
    micron-aware feature like centerline_length_um needs; scale is crop-invariant)."""
    inv = merged("acquisition", "frame_inventory", "frame_inventory")
    col = "source_micrometers_per_pixel"
    if col not in inv.columns:
        raise KeyError(f"frame_inventory has no {col!r}; columns: {list(inv.columns)}")
    return inv[["image_id", col]].drop_duplicates("image_id").rename(
        columns={col: "pixel_size_um"}
    )
