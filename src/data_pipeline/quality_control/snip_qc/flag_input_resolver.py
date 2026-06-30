"""snip_qc flag-input resolver — adapter between payload contracts and paths.py.

THIS IS ADAPTER/WIRING LOGIC. It is the only module in snip_qc permitted to import
pipeline_orchestrator.orchestration.paths. No build, inputs, contract, or entrypoint module
should import paths.py — a test enforces this boundary.

Joins two existing registries:
  1. _SOURCE_PAYLOADS: eligible source steps + their payload columns (imported from source contracts)
  2. PIPELINE_STEPS / artifact_path: step/artifact -> filesystem path

Given requested flag columns (SNIP_QC_EXCLUSION_FLAGS or a config override), returns ResolvedFlagSource objects
with concrete per-well artifact paths. Used at DAG planning time to declare Snakemake inputs
and serialized as JSON so the runtime task receives the exact same plan.

See: docs/refactors/streamline-snakemake/target/specs/quality_control/snip_qc_verdict_and_flag_resolver.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    PIPELINE_STEPS,
    artifact_path,
)
from data_pipeline.quality_control.death_detection.contract import DEATH_DETECTION_QC_PAYLOAD_COLUMNS
from data_pipeline.quality_control.mask_quality_qc.contract import MASK_QUALITY_QC_PAYLOAD_COLUMNS
from data_pipeline.quality_control.surface_area_qc.contract import SURFACE_AREA_QC_PAYLOAD_COLUMNS

# Explicit eligible source universe — snip_qc's decision about which upstream QC products
# may contribute exclusion flags. Adding a new source requires:
#   1. Importing its *_PAYLOAD_COLUMNS constant above
#   2. Adding it here
#   3. Adding the flag column to SNIP_QC_EXCLUSION_FLAGS (or config)
# The resolver enforces all three are consistent.
_SOURCE_PAYLOADS: dict[str, tuple[str, ...]] = {
    "death_detection_qc": DEATH_DETECTION_QC_PAYLOAD_COLUMNS,
    "surface_area_qc":    SURFACE_AREA_QC_PAYLOAD_COLUMNS,
    "mask_quality_qc":    MASK_QUALITY_QC_PAYLOAD_COLUMNS,
}


@dataclass(frozen=True)
class ResolvedFlagSource:
    """One resolved source: a step, its per-well CSV artifact key, the flag columns to read, and the path."""

    step: str
    artifact_key: str
    flag_columns: tuple[str, ...]
    path: Path

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "artifact_key": self.artifact_key,
            "flag_columns": list(self.flag_columns),
            "path": str(self.path),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResolvedFlagSource:
        return cls(
            step=d["step"],
            artifact_key=d["artifact_key"],
            flag_columns=tuple(d["flag_columns"]),
            path=Path(d["path"]),
        )


def resolve_snip_qc_flag_sources(
    exclusion_flags: tuple[str, ...],
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
) -> tuple[ResolvedFlagSource, ...]:
    """Return one ResolvedFlagSource per needed source step with concrete per-well artifact path.

    Pure: no disk reads, no side effects. Safe to call at DAG planning time.
    Raises ValueError if any requested flag cannot be resolved unambiguously.
    """
    flag_to_step = _build_flag_column_index(exclusion_flags)

    step_to_flags: dict[str, list[str]] = {}
    for flag, step in flag_to_step.items():
        step_to_flags.setdefault(step, []).append(flag)

    resolved = []
    for step, flags in sorted(step_to_flags.items()):
        artifact_key = _find_per_well_csv_artifact_key(step)
        path = Path(str(artifact_path(
            output_root, step, artifact_key, experiment_id,
            path_mode=PATH_MODE_PER_WELL, well_id=well_id,
        )))
        resolved.append(ResolvedFlagSource(
            step=step,
            artifact_key=artifact_key,
            flag_columns=tuple(sorted(flags)),
            path=path,
        ))
    return tuple(resolved)


def _build_flag_column_index(exclusion_flags: tuple[str, ...]) -> dict[str, str]:
    """Return flag_column -> source_step for all requested flags.

    Optimistic: scans _SOURCE_PAYLOADS and collects all candidate steps per flag.
    Reports all resolution failures together so the caller sees the full picture at once:
      - zero candidates -> flag not in any eligible source payload
      - multiple candidates -> ambiguous; lists all claiming steps
    """
    requested = set(exclusion_flags)

    candidates: dict[str, list[str]] = {col: [] for col in requested}
    for step, payload_cols in _SOURCE_PAYLOADS.items():
        for col in payload_cols:
            if col in candidates:
                candidates[col].append(step)

    errors: list[str] = []
    index: dict[str, str] = {}
    for col, steps in sorted(candidates.items()):
        if len(steps) == 0:
            errors.append(
                f"  {col!r}: not found in any eligible source payload "
                f"(eligible steps: {sorted(_SOURCE_PAYLOADS)})"
            )
        elif len(steps) > 1:
            errors.append(
                f"  {col!r}: ambiguous — claimed by multiple sources {steps}; "
                "payloads must be disjoint for auto-resolution"
            )
        else:
            index[col] = steps[0]

    if errors:
        raise ValueError(
            "snip_qc resolver: could not resolve all requested flag columns:\n"
            + "\n".join(errors)
            + "\nFix _SOURCE_PAYLOADS or the exclusion_flags config."
        )
    return index


def _find_per_well_csv_artifact_key(step: str) -> str:
    """Return the sole per-well CSV artifact key for step.

    Fails loud if the step is not in PIPELINE_STEPS, or if zero or multiple
    per-well CSV artifacts match (rather than silently using the wrong one).
    """
    if step not in PIPELINE_STEPS:
        raise ValueError(
            f"snip_qc resolver: source step {step!r} is not registered in PIPELINE_STEPS. "
            f"Known steps: {sorted(PIPELINE_STEPS)}. "
            "Register the step in paths.py before adding it to _SOURCE_PAYLOADS."
        )
    step_entry = PIPELINE_STEPS[step]
    csv_keys = [
        key
        for key, patterns in step_entry["artifacts"].items()
        if PATH_MODE_PER_WELL in patterns and patterns[PATH_MODE_PER_WELL].endswith(".csv")
    ]
    if len(csv_keys) != 1:
        raise ValueError(
            f"snip_qc resolver: step {step!r} has {len(csv_keys)} per-well CSV artifact(s) "
            f"{csv_keys}; expected exactly one. "
            "If the step has multiple CSV artifacts, specify artifact_key explicitly."
        )
    return csv_keys[0]
