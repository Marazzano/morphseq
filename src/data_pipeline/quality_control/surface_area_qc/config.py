"""surface_area_qc config — defaults, run-override resolution, and the self-documenting band.

Canonical thresholds are ``k_upper=1.4`` / ``k_lower=0.9`` (what actually runs). The legacy
1.2/0.9 signature defaults were stale and are NOT carried forward. On resolution the product
prints a plain-language statement of the active band so its meaning is never reverse-engineered
from code (see feature_world.md surface_area_qc; a test asserts the statement).

``k_lower`` was raised 0.7 -> 0.8 -> 0.9 on 2026-07-02: area alone cannot separate bad (yolk-only)
SAM2 masks from real, small/thin embryos, so this is a deliberate, imperfect tradeoff, not a full
fix. Chosen to clear the round-blob population out of the passing bands (verified empirically),
at the explicit cost of also losing real thin/dorsal-pose embryos (accepted: less information
content than a bad mask, per product call). See
docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# Product-local defaults. Run-level overrides live under quality_control.surface_area_qc.
SURFACE_AREA_QC_DEFAULTS: dict = {
    "reference_version": "v1",
    "area_column": "area_um2",
    "stage_column": "predicted_stage_hpf",
    "k_upper": 1.4,  # flag when area_um2 > k_upper * p95 (too large)
    "k_lower": 0.9,  # flag when area_um2 < k_lower * p5  (too small)
    "missing_reference_policy": "fail",  # fail | (future) documented fallback
    "missing_area_policy": "fail",       # fail | (future) documented flag behavior
    # Reconciliation may intentionally pass an unresolved stage through. The
    # surface-area band cannot judge it, so preserve the row without excluding it.
    "missing_stage_policy": "not_applicable",
}


@dataclass(frozen=True)
class SurfaceAreaQCConfig:
    reference_version: str
    area_column: str
    stage_column: str
    k_upper: float
    k_lower: float
    missing_reference_policy: str
    missing_area_policy: str
    missing_stage_policy: str


def resolve_config(overrides: dict | None = None) -> SurfaceAreaQCConfig:
    """Merge run-level overrides onto the product defaults and return a frozen config."""
    merged = dict(SURFACE_AREA_QC_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(SURFACE_AREA_QC_DEFAULTS)
        if unknown:
            raise ValueError(
                f"surface_area_qc: unknown config key(s) {sorted(unknown)}. "
                f"Known keys: {sorted(SURFACE_AREA_QC_DEFAULTS)}."
            )
        merged.update(overrides)
    return SurfaceAreaQCConfig(
        reference_version=str(merged["reference_version"]),
        area_column=str(merged["area_column"]),
        stage_column=str(merged["stage_column"]),
        k_upper=float(merged["k_upper"]),
        k_lower=float(merged["k_lower"]),
        missing_reference_policy=str(merged["missing_reference_policy"]),
        missing_area_policy=str(merged["missing_area_policy"]),
        missing_stage_policy=str(merged["missing_stage_policy"]),
    )


def band_statement(config: SurfaceAreaQCConfig) -> str:
    """Return the plain-language statement of the active band (required form, values filled)."""
    return (
        f"surface_area_qc band: flag {config.area_column} OUTSIDE "
        f"[ k_lower({config.k_lower:.2f}) x p5 , k_upper({config.k_upper:.2f}) x p95 ]\n"
        f"  percentiles interpolated per snip at {config.stage_column} "
        "(wildtype reference, fixed for MVP);\n"
        "  k = tolerance multiplier beyond the reference curve.\n"
        f'  -> "too small" if area < {config.k_lower:.2f} x p5;  '
        f'"too large" if area > {config.k_upper:.2f} x p95.'
    )
