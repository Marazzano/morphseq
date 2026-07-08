"""The one `PeakAcceptancePolicy`: promote-to-peak criteria, in one place.

Extracted from the inline accept/reject block that used to live inside
`peak_counting._detect_kde_peak_basins_sample_support` (the
`basin_fraction < min_sample_fraction` / `prominence < min_prominence_ratio`
checks, gated by `if len(peak_locations_tuple) > 1:`). Relocating it here does
not change behavior -- the `> 1` guard is preserved exactly -- it just gives
the promotion criteria one home so bootstrap draws and the final full-data
resolution can share one config and one implementation (COMPOSE_single_path_plan
Part 1 Sec 1.3).

`filter_detection_candidates` covers candidate-local criteria (sample-fraction
and prominence, evaluated per candidate against its own basin). It is the
Stage-1 slice of the policy: the full two-method split (adding
`validate_resolved_basins` as a distinct post-assignment empirical-mass floor)
is fleshed out incrementally as later stages need it; for now
`validate_resolved_basins` is available but only checks the same
min_component_mass_frac floor already read elsewhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PeakAcceptancePolicy:
    """One config, one implementation, applied identically everywhere a
    candidate peak is promoted to a final peak."""

    min_sample_fraction: float = 0.05
    min_prominence_ratio: float = 0.10
    min_component_mass_fraction: float = 0.10

    def filter_detection_candidates(
        self,
        *,
        n_candidates: int,
        basin_fractions: Sequence[float],
        prominences: Sequence[float | None],
    ) -> tuple[list[bool], list[list[str]]]:
        """Candidate-local accept/reject, matching the legacy inline block.

        Preserves the `n_candidates > 1` guard exactly: a single-candidate
        distribution is always accepted (nothing to disambiguate against), and
        only the multi-candidate case is gated by the sample-fraction and
        prominence thresholds.
        """
        accepted_mask: list[bool] = []
        reject_reasons: list[list[str]] = []
        for idx in range(n_candidates):
            basin_fraction = float(basin_fractions[idx]) if idx < len(basin_fractions) else 0.0
            prominence = prominences[idx] if idx < len(prominences) else None
            accepted = True
            reasons: list[str] = []
            if n_candidates > 1:
                if basin_fraction < float(self.min_sample_fraction):
                    accepted = False
                    reasons.append(f"sample_fraction_below_{self.min_sample_fraction:.3f}")
                if prominence is not None and prominence < float(self.min_prominence_ratio):
                    accepted = False
                    reasons.append(f"prominence_below_{self.min_prominence_ratio:.3f}")
            accepted_mask.append(accepted)
            reject_reasons.append(reasons)
        return accepted_mask, reject_reasons

    def validate_resolved_basins(
        self, *, basin_component_mass_fractions: Sequence[float]
    ) -> list[bool]:
        """The empirical-mass floor applied to FINAL (post-assignment) basins:
        a basin holding less than `min_component_mass_fraction` of total
        empirical mass is discarded, never resurrected to force a target
        count (COMPOSE_single_path_plan Sec 1.3)."""
        return [
            float(mass) >= float(self.min_component_mass_fraction)
            for mass in basin_component_mass_fractions
        ]


__all__ = ["PeakAcceptancePolicy"]
