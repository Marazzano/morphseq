"""surface_area_qc config — defaults, run-override resolution, and the self-documenting band.

Canonical thresholds are ``k_upper=1.4`` / ``k_lower=0.70`` (what actually runs). The legacy
1.2/0.9 signature defaults were stale and are NOT carried forward. On resolution the product
prints a plain-language statement of the active band so its meaning is never reverse-engineered
from code (see feature_world.md surface_area_qc; a test asserts the statement).

``k_lower`` history -- the band's lower edge has moved twice, and the history is kept because each
move encodes a different failure mode:

* 0.7 -> 0.8 -> 0.9 on 2026-07-02: area alone cannot separate bad (yolk-only) SAM2 masks from
  real, small/thin embryos, so this was a deliberate, imperfect tradeoff, not a full fix. Chosen
  to clear the round-blob population out of the passing bands (verified empirically), at the
  explicit cost of also losing real thin/dorsal-pose embryos (accepted: less information content
  than a bad mask, per product call).
* 0.9 -> 0.70 on 2026-09-02: that accepted cost turned out to land on real short mutants, which
  for a pbx4/pbx1b screen are precisely the animals of interest -- "small" IS the phenotype, and
  the packaged reference is a WILDTYPE percentile curve, so a wildtype-calibrated p5 floor
  systematically rejects the screen's targets. Measured on the pbx pilot (208 snips, 48 hpf,
  both plates), the two populations separate with a 0.30-wide empty gap in ``area_um2 / p5``:
      real short embryos   0.756 - 0.832  (total_length_um 1766-2486, width_um 649-671 = normal)
      ---- empty 0.457 .. 0.756 ----
      broken masks         0.457 (762 um "embryo"), 0.311 (30 um "embryo")
  0.70 sits inside that gap. The yolk-only round blobs the 0.9 raise was aimed at still fail,
  because they land below 0.457 -- so this recovers real animals without readmitting the
  population the previous raise was protecting against.

``k_upper=1.4`` was deliberately NOT touched: it is what catches whole-frame mask blow-outs
(observed up to 12.7x p95, with "lengths" of 5000-8000 um). Only the lower edge moved.

Per-run overrides live under ``quality_control.surface_area_qc`` in the merged config and reach
``resolve_config`` via ``tasks.py cmd_surface_area_qc``. See
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
    "k_lower": 0.70,  # flag when area_um2 < k_lower * p5  (too small); see the k_lower history above
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
