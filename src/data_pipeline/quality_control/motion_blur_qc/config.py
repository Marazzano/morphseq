"""motion_blur_qc config — defaults + run-override resolution.

MVP policy, fixed from the 2026-06-30 review:
adjacent z-plane mask-pixel NCC below 0.90 is a bad pair, and a snip is flagged when more than
0% of valid adjacent pairs are bad. Missing masks/stacks and undefined NCC support fail loud.
"""

from __future__ import annotations

from dataclasses import dataclass

MOTION_BLUR_QC_DEFAULTS: dict = {
    "z_stack_product_key": "BF__z_stack",
    "bad_z_pair_ncc_threshold": 0.90,
    "bad_pair_frac_threshold": 0.0,
    "flat_pair_variance_epsilon": 0.0,
    "missing_z_stack_policy": "fail",
    "missing_mask_policy": "fail",
    # Physical resolution the NCC is computed at, INDEPENDENT of how z-slices happen to be stored.
    # Planes are resampled to this before the NCC; None disables resampling and computes on whatever
    # is on disk. Never upsamples (see compute._resample_planes_to_target).
    #
    # Deliberately 15.09 and NOT the 6.5 µm/px storage target: bad_z_pair_ncc_threshold=0.90 was
    # tuned against Keyence z-slices as they were then materialized (native 3.7744 µm/px under the
    # old blind downsample_factor=4 => 15.0889). Pinning QC here keeps that tuned operating point
    # intact while the write policy moves z-slices to 6.5, which is the entire reason this knob
    # exists — storage resolution is a storage decision, QC resolution is a QC decision. It also
    # makes 0.90 scope-invariant: previously the same threshold silently meant one thing for Keyence
    # (15.09) and another for YX1 (12.92). Change this only alongside a re-tune of the threshold.
    "qc_micrometers_per_pixel": 15.0889,
}


@dataclass(frozen=True)
class MotionBlurQCConfig:
    z_stack_product_key: str
    bad_z_pair_ncc_threshold: float
    bad_pair_frac_threshold: float
    flat_pair_variance_epsilon: float
    missing_z_stack_policy: str
    missing_mask_policy: str
    qc_micrometers_per_pixel: float | None = None


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
        qc_micrometers_per_pixel=(
            None
            if merged["qc_micrometers_per_pixel"] is None
            else float(merged["qc_micrometers_per_pixel"])
        ),
    )
