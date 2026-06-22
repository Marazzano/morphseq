"""Snip image source — read model inputs from the per-well snip_inventory manifest.

This is the §9.1 image-source seam (specs/model_input_handoff_contract.md): the
embeddings step locates its input snips by reading the **manifest**, NOT by globbing
the legacy ``bf_embryo_snips/<exp>/*.jpg`` tree and NOT via the broken
``src.core.data.dataset_configs`` imports in ``services/gen_embeddings.py``.

Contract of the snip_inventory CSV this reads (the live `object_extraction` product):
- ``snip_id`` — stable per-snip identity (the join key; carries all identity, so no
  metadata join is needed to encode).
- ``processed_snip_path`` — the model-input PNG, stored **relative to the manifest's
  own directory** (resolved here against that directory).
- ``is_valid_snip`` — usability gate. We default to keeping only valid snips; biology
  gating ("encode all, gate downstream") happens later in analysis_ready.

Path-pure + pandas only: no torch, no model, no orchestration import. Runnable from
the 3.10 main env or the 3.9 encode env alike — it just resolves file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Union

import pandas as pd

PathLike = Union[str, Path]

SNIP_ID_COL = "snip_id"
PROCESSED_SNIP_PATH_COL = "processed_snip_path"
IS_VALID_SNIP_COL = "is_valid_snip"


class SnipInput(NamedTuple):
    """One model-input snip: its stable id and the resolved absolute image path."""

    snip_id: str
    image_path: Path


def collect_snip_inputs(
    snip_inventory_csv: PathLike,
    *,
    valid_only: bool = True,
) -> List[SnipInput]:
    """Read a snip_inventory CSV and return its model-input snips (id + resolved path).

    ``processed_snip_path`` is stored relative to the manifest's directory; it is
    resolved here so callers get absolute, existence-checkable paths.

    Fail-loud at the contract boundary: a missing required column names the column
    AND the file; a resolved image path that does not exist names the snip AND the
    path. (A planning-time caller wants to learn the inventory is stale here, not
    deep inside an encode loop.)

    Args:
        snip_inventory_csv: per-well or merged snip_inventory CSV.
        valid_only: keep only rows with ``is_valid_snip == True`` (default). The
            biological ``use_snip`` gate is downstream (analysis_ready), not here.

    Returns:
        ``SnipInput`` rows in manifest order (``shuffle=False`` is the encode contract).

    Raises:
        FileNotFoundError: if the CSV, or any resolved image path, is missing.
        KeyError: if a required column is absent.
    """
    csv_path = Path(snip_inventory_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"snip_inventory CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in (SNIP_ID_COL, PROCESSED_SNIP_PATH_COL):
        if col not in df.columns:
            raise KeyError(
                f"snip_inventory CSV {csv_path} is missing required column "
                f"'{col}'. Got columns: {list(df.columns)}."
            )

    if valid_only and IS_VALID_SNIP_COL in df.columns:
        df = df[df[IS_VALID_SNIP_COL].astype(bool)]

    manifest_dir = csv_path.parent
    inputs: List[SnipInput] = []
    for snip_id, rel_path in zip(df[SNIP_ID_COL], df[PROCESSED_SNIP_PATH_COL]):
        rel = Path(rel_path)
        image_path = rel if rel.is_absolute() else (manifest_dir / rel)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Processed snip image for {snip_id} not found: {image_path} "
                f"(from processed_snip_path='{rel_path}' relative to {manifest_dir}). "
                f"The snip_inventory is stale or the snips were not materialized."
            )
        inputs.append(SnipInput(snip_id=str(snip_id), image_path=image_path))

    return inputs
