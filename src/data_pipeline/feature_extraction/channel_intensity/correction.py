"""Background-corrected intensity: raw evidence joined to a versioned null.

RAW STAYS RAW. Every column here is an explicitly named ADDITION. Nothing overwrites an extracted
value, so a later change of estimator cannot retroactively alter what was measured -- and the two can
always be compared.

NOT CLIPPED AT ZERO. A negative corrected value means the null over-subtracted for that embryo, which
is exactly the diagnostic you want when judging whether the pooled null is right. Clipping would hide
it and make an over-subtracting estimator look fine.

MEAN AND INTEGRATED ARE BOTH EMITTED, DELIBERATELY. Integrated intensity scales with embryo volume,
so comparing it across timepoints conflates dosage with growth -- an older embryo is bigger and
therefore brighter in total regardless of copy number. At matched stage the mean is the safer default;
integrated is right when embryo size is controlled. Choosing one here would silently make that call
for every downstream analysis.
"""

from __future__ import annotations

from typing import Mapping

# Above this, the dosage call is not trustworthy: saturation compresses a 2-copy embryo toward
# 1-copy, which destroys the comparison in the one direction that looks like clean data. Emitted as
# a flag rather than a filter -- dropping the row would hide the acquisition problem that caused it.
SATURATION_CONCERN_FRACTION = 0.001


def correct_row(row: Mapping[str, object], null: Mapping[str, object]) -> dict[str, object]:
    """Join one raw evidence row to its well's null and derive the corrected columns.

    The null's identity travels ONTO the row (``null_estimator``, ``well_null_mode_dn``) so any row
    is self-explaining without a join back to the null table. A corrected number whose provenance
    requires a second lookup tends to get quoted without it.
    """
    mode = float(null["null_mode_dn"])
    embryo_px = int(row["embryo_px"])
    embryo_sum = float(row["embryo_sum_dn"])

    embryo_mean = (embryo_sum / embryo_px) if embryo_px else float("nan")
    clipped = int(row.get("embryo_clipped_px", 0))
    saturated_fraction = (clipped / embryo_px) if embryo_px else 0.0

    return {
        # Provenance of the correction, carried on the row itself.
        "null_estimator": null["null_estimator"],
        "well_null_mode_dn": mode,
        "well_null_valid": bool(null["null_valid"]),
        "well_null_drift_dn": float(null.get("null_drift_dn", 0.0)),
        # Raw, restated for convenience -- identical to the extracted values, never recomputed
        # differently.
        "embryo_mean_dn": embryo_mean,
        "embryo_integrated_dn": embryo_sum,
        # Corrected. Explicit additions; not clipped.
        "embryo_mean_bgsub_dn": embryo_mean - mode,
        "embryo_integrated_bgsub_dn": embryo_sum - mode * embryo_px,
        # Signal relative to background spread. NaN rather than inf when the null has no spread,
        # since a zero-width background means the estimate is degenerate, not that the signal is
        # infinite.
        "embryo_bgsub_z": (
            (embryo_mean - mode) / float(null["null_robust_sigma_dn"])
            if float(null.get("null_robust_sigma_dn", 0.0)) > 0
            else float("nan")
        ),
        # The dosage gate. A downstream copy-number call must consult this.
        "embryo_saturated_fraction": saturated_fraction,
        "embryo_saturation_concern": saturated_fraction > SATURATION_CONCERN_FRACTION,
        # Whether this row supports a dosage comparison at all. Both conditions are about the
        # MEASUREMENT, never about the biology -- nothing here looks at genotype, and it must not:
        # a measurement that depended on its own hypothesis would be worthless.
        "intensity_dosage_usable": bool(
            null["null_valid"]
            and bool(row.get("annulus_area_fraction", 0.0) > 0.5)
            and saturated_fraction <= SATURATION_CONCERN_FRACTION
        ),
    }
