"""snip_qc entrypoint — the thin filesystem adapter that builds the final verdict.

Loads the feature universe (validated snip_inventory), assembles the MVP exclusion flag columns
from their registered per-well source shards (death_detection_qc, surface_area_qc, mask_quality_qc),
builds the verdict, validates it (registry as verifier), then writes the per-well shard. The source
(step, artifact) pairs are passed explicitly here; inputs.py resolves their paths via the registry.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build import build_snip_qc_verdict
from .contract import SNIP_QC_EXCLUSION_REASONS, validate_snip_qc
from .inputs import load_snip_qc_flag_inputs

# MVP exclusion flag sources: (step, artifact). Each step here has a single registered artifact, so
# the artifact key is explicit for clarity (inputs.py would otherwise infer it).
_MVP_FLAG_SOURCES: list[tuple[str, str]] = [
    ("death_detection_qc", "death_detection_qc"),
    ("surface_area_qc", "surface_area_qc"),
    ("mask_quality_qc", "mask_quality_qc"),
]


def run_snip_qc(
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    snip_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
) -> None:
    snip_universe = pd.read_csv(snip_inventory_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    flag_columns = list(SNIP_QC_EXCLUSION_REASONS.values())
    qc_flags = load_snip_qc_flag_inputs(
        output_root=Path(output_root),
        experiment_id=experiment_id,
        well_id=well_id,
        sources=_MVP_FLAG_SOURCES,
        flag_columns=flag_columns,
    )

    verdict = build_snip_qc_verdict(
        snip_universe, qc_flags, exclusion_reasons=SNIP_QC_EXCLUSION_REASONS
    )

    validate_snip_qc(verdict, physical_embryo_registry_df=registry, check_sources=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The snip_qc verdict is the final operational table — persisted as parquet (registry artifact
    # extension); write by suffix so a future CSV override still works.
    if output_path.suffix == ".parquet":
        verdict.to_parquet(output_path, index=False)
    else:
        verdict.to_csv(output_path, index=False)
