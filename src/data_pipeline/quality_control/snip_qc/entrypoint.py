"""snip_qc entrypoint — thin filesystem adapter that builds the final per-snip verdict.

Receives the pre-resolved source plan (resolved_sources + exclusion_reasons) deserialized
from the tracked resolved_sources JSON artifact. Does not resolve paths itself — that was
done at DAG planning time by flag_input_resolver.py and persisted to JSON.

Flow: load snip_universe + registry → load + verify flag inputs (inputs.py) →
      build verdict (build.py) → validate (contract.py) → write.

See: docs/refactors/streamline-snakemake/target/specs/quality_control/snip_qc_verdict_and_flag_resolver.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build import build_snip_qc_verdict
from .contract import validate_snip_qc
from .flag_input_resolver import ResolvedFlagSource
from .inputs import load_snip_qc_flag_inputs


def run_snip_qc(
    *,
    snip_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    resolved_sources: tuple[ResolvedFlagSource, ...],
    exclusion_reasons: dict[str, str],
) -> None:
    snip_universe = pd.read_csv(snip_inventory_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    qc_flags = load_snip_qc_flag_inputs(resolved_sources)

    verdict = build_snip_qc_verdict(
        snip_universe, qc_flags, exclusion_reasons=exclusion_reasons
    )

    validate_snip_qc(verdict, physical_embryo_registry_df=registry, check_sources=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        verdict.to_parquet(output_path, index=False)
    else:
        verdict.to_csv(output_path, index=False)
