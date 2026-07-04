"""analysis_ready assemble — the merged-level fan-in join.

One row per ``snip_id``. Left-joins the merged per-snip feature tables + the snip_qc verdict onto
the identity spine (the snip_qc table owns the authoritative snip_id universe), then broadcasts the
per-well plate_metadata across snips by ``well_id``. Every join is a LEFT join onto the base — no
feature source may add or drop a snip; the row count and snip_id uniqueness are asserted after each
join (belt check; each source's own contract already forbids dup snip_ids).

Grain notes:
  - feature tables (curvature/stage/mask_geometry/pose/fraction_alive) + snip_qc: 1 row / snip_id.
  - latents: 1 row / snip_id, but a WIDE dynamic z_mu_*/z_sigma_* block (matched by prefix).
  - plate_metadata: 1 row / well_id → broadcast onto snips (the stage_predictions precedent).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contract import (
    ANALYSIS_READY_REQUIRED_PLATE_COLUMNS,
    broadcast_plate_columns,
    is_latent_column,
)

_SPINE_JOIN_KEY = "snip_id"
_PLATE_JOIN_KEY = "well_id"


def _left_join_on_snip(base: pd.DataFrame, add: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Left-join ``add`` onto ``base`` by snip_id, dropping spine columns ``add`` re-carries (only
    the payload is new), and asserting the base row count / snip_id uniqueness are preserved."""
    if _SPINE_JOIN_KEY not in add.columns:
        raise ValueError(f"analysis_ready: {source} table has no '{_SPINE_JOIN_KEY}' column.")
    if add[_SPINE_JOIN_KEY].duplicated().any():
        dups = add.loc[add[_SPINE_JOIN_KEY].duplicated(), _SPINE_JOIN_KEY].head(5).tolist()
        raise ValueError(f"analysis_ready: {source} has duplicate snip_id values: {dups}")

    # Keep only snip_id + this source's NEW columns (anything already on base is spine/provenance).
    new_cols = [c for c in add.columns if c == _SPINE_JOIN_KEY or c not in base.columns]
    n_before = len(base)
    merged = base.merge(add[new_cols], on=_SPINE_JOIN_KEY, how="left", validate="one_to_one")

    if len(merged) != n_before:
        raise ValueError(
            f"analysis_ready: joining {source} changed the row count "
            f"({n_before} -> {len(merged)}); a LEFT join must not add or drop snips."
        )
    return merged


def assemble_analysis_ready(
    *,
    snip_qc: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    latents: pd.DataFrame,
    plate_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Fan-in join. ``snip_qc`` is the base (it carries the full spine + the authoritative snip_id
    set). ``feature_tables`` maps source-name -> per-snip table. ``latents`` is the wide latent
    table. ``plate_metadata`` is per-well and gets broadcast by well_id.
    """
    if _SPINE_JOIN_KEY not in snip_qc.columns:
        raise ValueError("analysis_ready: snip_qc base table has no 'snip_id' column.")
    if snip_qc[_SPINE_JOIN_KEY].duplicated().any():
        raise ValueError("analysis_ready: snip_qc base has duplicate snip_id values.")
    if _PLATE_JOIN_KEY not in snip_qc.columns:
        raise ValueError(
            "analysis_ready: snip_qc base has no 'well_id' — the spine must carry it for the "
            "plate_metadata broadcast."
        )

    out = snip_qc.copy()

    # 1. Each per-snip feature table (deterministic order for stable column layout).
    for source in sorted(feature_tables):
        out = _left_join_on_snip(out, feature_tables[source], source=source)

    # 2. Latents — same snip_id join, but the new columns are the dynamic z_mu_*/z_sigma_* block.
    out = _left_join_on_snip(out, latents, source="latent_embeddings")

    # 3. Plate metadata — broadcast per-well row onto every snip in that well.
    out = _broadcast_plate(out, plate_metadata)

    return out


def _broadcast_plate(base: pd.DataFrame, plate: pd.DataFrame) -> pd.DataFrame:
    """Broadcast per-well plate_metadata onto snips by well_id (one plate row -> all its snips)."""
    if _PLATE_JOIN_KEY not in plate.columns:
        raise ValueError("analysis_ready: plate_metadata has no 'well_id' column.")
    if plate[_PLATE_JOIN_KEY].duplicated().any():
        dups = plate.loc[plate[_PLATE_JOIN_KEY].duplicated(), _PLATE_JOIN_KEY].head(5).tolist()
        raise ValueError(f"analysis_ready: plate_metadata has duplicate well_id rows: {dups}")

    missing_required = [c for c in ANALYSIS_READY_REQUIRED_PLATE_COLUMNS if c not in plate.columns]
    if missing_required:
        raise ValueError(
            f"analysis_ready: plate_metadata is missing required field(s) {missing_required}."
        )

    # Every well present in the spine MUST have a plate row (a missing well is a data error, not a
    # silent NaN — same hard-fail stage_predictions enforces).
    spine_wells = set(base[_PLATE_JOIN_KEY].unique())
    plate_wells = set(plate[_PLATE_JOIN_KEY].unique())
    orphan_wells = sorted(spine_wells - plate_wells)
    if orphan_wells:
        raise ValueError(
            f"analysis_ready: {len(orphan_wells)} well(s) in the spine have no plate_metadata row: "
            f"{orphan_wells[:5]}. Every snip's well must have a plate row."
        )

    carry_cols = [_PLATE_JOIN_KEY] + broadcast_plate_columns(plate.columns)
    n_before = len(base)
    out = base.merge(plate[carry_cols], on=_PLATE_JOIN_KEY, how="left", validate="many_to_one")
    if len(out) != n_before:
        raise ValueError(
            f"analysis_ready: plate broadcast changed the row count ({n_before} -> {len(out)})."
        )
    return out


def read_and_assemble(
    *,
    snip_qc_path: Path,
    feature_paths: dict[str, Path],
    latents_path: Path,
    plate_metadata_csv: Path,
) -> pd.DataFrame:
    """Filesystem adapter: read each merged input (parquet or csv by suffix), assemble, return."""

    def _read(p: Path) -> pd.DataFrame:
        p = Path(p)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    return assemble_analysis_ready(
        snip_qc=_read(snip_qc_path),
        feature_tables={name: _read(p) for name, p in feature_paths.items()},
        latents=_read(latents_path),
        plate_metadata=pd.read_csv(plate_metadata_csv),
    )
