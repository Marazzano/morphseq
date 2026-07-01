"""motion_blur_qc config — defaults + run-override resolution.

MVP policy, fixed from the 2026-06-30 review:
adjacent z-plane mask-pixel NCC below 0.90 is a bad pair, and a snip is flagged when more than
10% of valid adjacent pairs are bad. Missing masks/stacks and undefined NCC support fail loud.
"""

from __future__ import annotations

from dataclasses import dataclass

MOTION_BLUR_QC_DEFAULTS: dict = {
    "z_stack_product_key": "BF__z_stack",
    "bad_z_pair_ncc_threshold": 0.90,
    "bad_pair_frac_threshold": 0.10,
    "flat_pair_variance_epsilon": 0.0,
    "missing_z_stack_policy": "fail",
    "missing_mask_policy": "fail",
}


@dataclass(frozen=True)
class MotionBlurQCConfig:
    z_stack_product_key: str
    bad_z_pair_ncc_threshold: float
    bad_pair_frac_threshold: float
    flat_pair_variance_epsilon: float
    missing_z_stack_policy: str
    missing_mask_policy: str


def resolve_config(overrides: dict | None = None) -> MotionBlurQCConfig:
    """Merge run-level overrides onto the product defaults and return a frozen config."""
    merged = dict(MOTION_BLUR_QC_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(MOTION_BLUR_QC_DEFAULTS)
        if unknown:
            raise ValueError(
                f"motion_blur_qc: unknown config key(s) {sorted(unknown)}. "
                f"Known keys: {sorted(MOTION_BLUR_QC_DEFAULTS)}."
            )
        merged.update(overrides)
    return MotionBlurQCConfig(
        z_stack_product_key=str(merged["z_stack_product_key"]),
        bad_z_pair_ncc_threshold=float(merged["bad_z_pair_ncc_threshold"]),
        bad_pair_frac_threshold=float(merged["bad_pair_frac_threshold"]),
        flat_pair_variance_epsilon=float(merged["flat_pair_variance_epsilon"]),
        missing_z_stack_policy=str(merged["missing_z_stack_policy"]),
        missing_mask_policy=str(merged["missing_mask_policy"]),
    )
