"""QC gate for whether an embryo's intensity measurement is usable, and for WHICH question.

WHY NOT A BRIGHTNESS RATIO. "More than Nx dimmer than the others, therefore unusable" is the wrong
rule: A09 is 4-7x dimmer than G09 and carries essentially the same information after normalization
(entropy gap +0.03 to +0.83 bits). Exposure, copy number, expression level and optics all move
absolute brightness while leaving a rich spatial signal intact.

WHY NOT A RAW LEVEL COUNT EITHER. An earlier draft of this proposed ">100 distinct occupied levels"
as the whole gate. That is fragile: read noise, hot pixels, background offset and large masks all
add distinct values without adding signal. MEASURED HERE -- D06 at t2 occupies 144 levels (passing
a >100 rule) but separates from background by only 3.0 sigma and has just 42 EFFECTIVE states. The
raw count over-reports it by 3.4x.

So the gate is three independent checks, because they fail independently:

    effective_states  = 2**entropy   how many intensity states are MEANINGFULLY used, which
                                     downweights levels held by one noisy pixel
    separation        = (mean - background_mode) / background_sigma
                                     is the embryo distinguishable from the well it sits in
    saturated_frac                   is the bright end intact, or clipped into the ceiling

AND THE ANSWER DEPENDS ON THE QUESTION. Per-embryo normalization removes absolute scale -- which is
exactly the dosage signal. So passing this gate licenses PATTERN work (texture, morphology,
spatial structure), not dosage. Dosage must be read on native intensities with exposure carried,
where a 6x brightness difference is the measurement rather than a nuisance.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from data_pipeline.feature_extraction.channel_intensity.pooling import (  # noqa: E402
    HIST_BIN_WIDTH_DN,
    estimate_well_null,
)

EXP = "20260624_2x_td_bf_pbx_coll_plate01"
CENTERS = (np.arange(2048) + 0.5) * HIST_BIN_WIDTH_DN

# CALIBRATED ON THE FULL PLATE, not chosen a priori. 64 was a guess (2**6) and it is too
# permissive: at 64 the normalized far-pair difference sits at 3.44x the matched-brightness floor
# (t1), because the plate reaches embryos near the resolution limit. Sweeping the threshold against
# that floor:
#
#     min_eff   t1 kept   t1 norm/floor    t2 kept   t2 norm/floor
#        64       135         3.44           108         1.68
#       128       116         1.80            99         1.25
#       192        91         2.28            91         1.29
#       256        66         1.75            74         1.55
#
# 128 halves the t1 ratio while keeping 116/135 embryos. 256 buys nothing and costs half the
# sample. The non-monotonicity above 128 is sampling noise in the floor estimate, not structure.
MIN_EFFECTIVE_STATES = 128.0
# The other two remain a priori, NOT calibrated -- stated so the difference is visible.
MIN_SEPARATION_SIGMA = 5.0    # embryo mean this many background sigmas above the background mode
MAX_SATURATED_FRAC = 0.01     # >1% of pixels at the ceiling compresses the bright tail


def shannon_bits(counts) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def evaluate(row, null) -> dict:
    """Three independent usability checks for one embryo-time."""
    counts = np.asarray(row["embryo_hist_counts"], dtype=float)
    mode = float(null["null_mode_dn"])
    sigma = float(null["null_robust_sigma_dn"])
    n_px = int(row["embryo_px"])

    entropy = shannon_bits(counts)
    # EFFECTIVE states, not unique values: 2**H asks how many states are actually carrying the
    # distribution, so a level occupied by a single noisy pixel contributes almost nothing.
    effective = float(2 ** entropy) if np.isfinite(entropy) else float("nan")
    occupied = int((counts > 0).sum())

    mean_dn = float((counts * CENTERS[: len(counts)]).sum() / n_px) if n_px else float("nan")
    separation = (mean_dn - mode) / sigma if sigma else float("nan")
    saturated = int(row["embryo_clipped_px"]) / n_px if n_px else float("nan")

    checks = {
        "resolution_ok": bool(effective >= MIN_EFFECTIVE_STATES),
        "separation_ok": bool(separation >= MIN_SEPARATION_SIGMA),
        "unsaturated_ok": bool(saturated <= MAX_SATURATED_FRAC),
    }
    return {
        "occupied_levels": occupied,
        "effective_states": effective,
        "entropy_bits": entropy,
        "separation_sigma": separation,
        "saturated_frac": saturated,
        **checks,
        # PASSING LICENSES PATTERN WORK ONLY. Normalization removes the absolute scale that dosage
        # is carried in, so this flag must never be read as "safe to pool for dosage".
        "usable_for_pattern": all(checks.values()),
    }


def main() -> None:
    shards = sorted(glob.glob(str(
        MORPHSEQ_ROOT / ".pbx_smoke/out/object_extraction" / EXP
        / "channel_intensity/per_well/*/RFP__projection__max/channel_intensity.csv"
    )))
    frames = [f for f in (pd.read_csv(p) for p in shards) if len(f)]
    if not frames:
        raise SystemExit("no non-empty shards on disk")
    d = pd.concat(frames, ignore_index=True)
    for column in ("annulus_hist_counts", "embryo_hist_counts"):
        d[column] = d[column].map(json.loads)

    nulls = {k: estimate_well_null(g.to_dict("records"))
             for k, g in d.groupby(["well_id", "time_index"])}

    rows = []
    for _, r in d.iterrows():
        verdict = evaluate(r, nulls[(r["well_id"], int(r["time_index"]))])
        rows.append({"well": r["well_id"][-3:], "time_index": int(r["time_index"]),
                     "area_px": int(r["embryo_px"]), **verdict})
    q = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    print(f"=== INTENSITY QC, {len(q)} embryo-times ===")
    print(f"thresholds: effective_states >= {MIN_EFFECTIVE_STATES:.0f}, "
          f"separation >= {MIN_SEPARATION_SIGMA} sigma, saturated <= {MAX_SATURATED_FRAC:.0%}\n")
    for name in ("resolution_ok", "separation_ok", "unsaturated_ok", "usable_for_pattern"):
        print(f"  {name:20s} {q[name].sum():3d}/{len(q)}")

    print("\n=== WHY A RAW LEVEL COUNT IS NOT ENOUGH ===")
    misleading = q[(q.occupied_levels > 100) & ~q.separation_ok]
    print(f"pass '>100 occupied levels' but FAIL separation: {len(misleading)}")
    if len(misleading):
        print(misleading.sort_values("separation_sigma")[
            ["well", "time_index", "area_px", "occupied_levels", "effective_states",
             "separation_sigma"]].head(10).round(1).to_string(index=False))
        print("\n  occupied over-reports these: the raw count includes levels held by a pixel or")
        print("  two of noise, which 2**H discounts.")

    out = Path(__file__).resolve().parents[1] / "output" / "intensity_qc.csv"
    q.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
