"""mask_quality_qc config — defaults + run-override resolution.

Thresholds migrated from the legacy `segmentation_qc` config block, with one correction: the
overlap IoU cutoff is the spec's 0.10 (the legacy `max_mask_overlap_fraction=0.3` was a
different, looser knob and is not carried forward).
"""

from __future__ import annotations

from dataclasses import dataclass

MASK_QUALITY_QC_DEFAULTS: dict = {
    "margin_pixels": 2,            # edge-contact margin (px)
    "min_component_fraction": 0.05,  # significant-component size vs largest
    "iou_threshold": 0.10,        # pairwise IoU cutoff for overlap (spec canon, not legacy 0.3)
    "missing_mask_policy": "fail",  # fail | (future) documented flag behavior
}


@dataclass(frozen=True)
class MaskQualityQCConfig:
    margin_pixels: int
    min_component_fraction: float
    iou_threshold: float
    missing_mask_policy: str


def resolve_config(overrides: dict | None = None) -> MaskQualityQCConfig:
    """Merge run-level overrides onto the product defaults and return a frozen config."""
    merged = dict(MASK_QUALITY_QC_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(MASK_QUALITY_QC_DEFAULTS)
        if unknown:
            raise ValueError(
                f"mask_quality_qc: unknown config key(s) {sorted(unknown)}. "
                f"Known keys: {sorted(MASK_QUALITY_QC_DEFAULTS)}."
            )
        merged.update(overrides)
    return MaskQualityQCConfig(
        margin_pixels=int(merged["margin_pixels"]),
        min_component_fraction=float(merged["min_component_fraction"]),
        iou_threshold=float(merged["iou_threshold"]),
        missing_mask_policy=str(merged["missing_mask_policy"]),
    )
