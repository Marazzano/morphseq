"""Split the dataset-level drop-in frame_inventory manifest into per-well shards.

The submission surface is ONE ``dropin_frame_inventory.csv`` (all wells); the operational spine is
per-well ``{well_id}_frame_inventory.csv`` shards. This module fans the big ingress file into those
shards — the big file is ingress-only (read by discovery + this split), never by a well-local compute
stage. After this split the shards ARE registry artifacts and go through the SAME strict gate the
native materializer's shards do.

Identity is derived from the atoms (``experiment_id`` + ``well_index``) via the grammar — never
f-stringed inline. A single experiment is enforced (matching ``discover_wells_from_handoff``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.shared.identifiers import build_well_id


def split_dropin_inventory_by_well(
    manifest_csv: Path, output_dir: Path
) -> dict[str, Path]:
    """Split ``manifest_csv`` into per-well ``{well_id}_frame_inventory.csv`` shards under ``output_dir``.

    Returns a mapping ``well_id -> shard_path``. Each shard carries only that well's rows; the
    derived ``well_id`` column is added so the shard is self-describing (the gate recomputes it from
    the atoms and fails loud on any disagreement — it is never trusted as authored).
    """
    df = pd.read_csv(manifest_csv)
    for col in ("experiment_id", "well_index"):
        if col not in df.columns:
            raise ValueError(
                f"[split_dropin_inventory] {manifest_csv} is missing required column {col!r}."
            )

    experiments = sorted(df["experiment_id"].dropna().astype(str).str.strip().unique())
    if len(experiments) != 1:
        raise ValueError(
            f"[split_dropin_inventory] a drop-in submission must contain exactly one experiment_id; "
            f"found {experiments}."
        )
    experiment_id = experiments[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    well_ids = df.apply(
        lambda r: build_well_id(experiment_id, str(r["well_index"]).strip()), axis=1
    )

    shards: dict[str, Path] = {}
    for well_id, group in df.groupby(well_ids, sort=False):
        shard_path = output_dir / f"{well_id}_frame_inventory.csv"
        group.to_csv(shard_path, index=False)
        shards[str(well_id)] = shard_path
    return shards
