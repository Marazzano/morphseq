"""death_detection config — defaults + run-override resolution.

Thresholds migrated from the legacy `death_detection` config block. The per-frame axis is
`time_index` (the spine column); `lead_time_hr` is in HOURS and is converted to elapsed time via
the frame-timing input (never subtracted from a raw frame index).
"""

from __future__ import annotations

from dataclasses import dataclass

DEATH_DETECTION_DEFAULTS: dict = {
    "persistence_threshold": 0.80,    # required post-inflection dead fraction
    "lead_time_hr": 4.0,              # HOURS subtracted from the inflection's elapsed time
    "decline_rate_threshold": 0.05,   # min decline rate for a candidate inflection
    "dead_fraction_threshold": 0.90,  # per-frame fraction_alive cutoff (viability + persistence evidence)
    "min_timepoints": 3,              # min observations per animal before death detection is attempted
    "smoothing_window": 5,            # savgol window for noisy traces (odd; clamped to series length)
}


@dataclass(frozen=True)
class DeathDetectionConfig:
    persistence_threshold: float
    lead_time_hr: float
    decline_rate_threshold: float
    dead_fraction_threshold: float
    min_timepoints: int
    smoothing_window: int


def resolve_config(overrides: dict | None = None) -> DeathDetectionConfig:
    """Merge run-level overrides onto the product defaults and return a frozen config."""
    merged = dict(DEATH_DETECTION_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(DEATH_DETECTION_DEFAULTS)
        if unknown:
            raise ValueError(
                f"death_detection: unknown config key(s) {sorted(unknown)}. "
                f"Known keys: {sorted(DEATH_DETECTION_DEFAULTS)}."
            )
        merged.update(overrides)
    return DeathDetectionConfig(
        persistence_threshold=float(merged["persistence_threshold"]),
        lead_time_hr=float(merged["lead_time_hr"]),
        decline_rate_threshold=float(merged["decline_rate_threshold"]),
        dead_fraction_threshold=float(merged["dead_fraction_threshold"]),
        min_timepoints=int(merged["min_timepoints"]),
        smoothing_window=int(merged["smoothing_window"]),
    )
