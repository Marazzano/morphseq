"""focus_qc config — defaults + run-override resolution.

Thresholds from the 2026-06-30 fine-tuning review (see
docs/refactors/streamline-snakemake/target/specs/quality_control/z_stack_focus_motion_blur_qc_and_slice_selection.md):
``local_context`` normalization, Sobel ``grad > 0.02``, ``interior_strong_edge_fraction < 0.50``.
This is a deliberate compromise — a low-information exclusion as much as a strict focus exclusion —
not a claim that every excluded embryo is technically out-of-focus.
"""

from __future__ import annotations

from dataclasses import dataclass

FOCUS_QC_DEFAULTS: dict = {
    "projection_product_key": "BF__projection__focus_stack",
    "interior_erosion_pixels": 12,            # primary erosion; smaller fallback below min_interior_pixels
    "interior_erosion_fallback_pixels": 6,
    "normalization_mode": "local_context",    # mask + dilated local context band, robust 1st-99th pct
    "strong_edge_sobel_threshold": 0.02,      # Sobel gradient cutoff for a "strong edge" pixel
    "interior_strong_edge_fraction_threshold": 0.50,  # focus_flag = fraction < this
    "min_interior_pixels": 200,               # below this, fall back to smaller erosion / full mask
    "missing_projection_policy": "fail",      # fail | (future) documented flag behavior
    "missing_mask_policy": "fail",
}


@dataclass(frozen=True)
class FocusQCConfig:
    projection_product_key: str
    interior_erosion_pixels: int
    interior_erosion_fallback_pixels: int
    normalization_mode: str
    strong_edge_sobel_threshold: float
    interior_strong_edge_fraction_threshold: float
    min_interior_pixels: int
    missing_projection_policy: str
    missing_mask_policy: str


def resolve_config(overrides: dict | None = None) -> FocusQCConfig:
    """Merge run-level overrides onto the product defaults and return a frozen config."""
    merged = dict(FOCUS_QC_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(FOCUS_QC_DEFAULTS)
        if unknown:
            raise ValueError(
                f"focus_qc: unknown config key(s) {sorted(unknown)}. "
                f"Known keys: {sorted(FOCUS_QC_DEFAULTS)}."
            )
        merged.update(overrides)
    return FocusQCConfig(
        projection_product_key=str(merged["projection_product_key"]),
        interior_erosion_pixels=int(merged["interior_erosion_pixels"]),
        interior_erosion_fallback_pixels=int(merged["interior_erosion_fallback_pixels"]),
        normalization_mode=str(merged["normalization_mode"]),
        strong_edge_sobel_threshold=float(merged["strong_edge_sobel_threshold"]),
        interior_strong_edge_fraction_threshold=float(
            merged["interior_strong_edge_fraction_threshold"]
        ),
        min_interior_pixels=int(merged["min_interior_pixels"]),
        missing_projection_policy=str(merged["missing_projection_policy"]),
        missing_mask_policy=str(merged["missing_mask_policy"]),
    )
