"""Dosage classes, clustered WITHIN each timepoint, then checked for consistency across them.

WHY PER TIMEPOINT. Fluorescence intensity is not stable over developmental time -- a pan-nuclear
marker tracks cell number, exposure can differ between sessions, and stage/focus/z all move a max
projection. Clustering pooled across timepoints therefore partitions TIME as much as genotype, and
does it while producing three clean-looking clusters. Each timepoint is its own comparison: same
file, same exposure, same session.

WHAT MAKES IT A REAL TEST RATHER THAN A DECORATION. Copy number is FIXED for a physical embryo. So
the classes are discovered independently at each timepoint, and then the same embryo's class
assignment is compared ACROSS timepoints. If the three independent clusterings agree on which
embryos are bright, that agreement is evidence the axis is dosage. If they disagree, the clusters
are reading something time-varying and no amount of normalization fixes it.

The consistency check is the result. The clusters themselves are not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Anchor to the repo root: this script lives four levels down under results/, so relative paths
# would resolve against whatever directory it happens to be invoked from.
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from data_pipeline.feature_extraction.channel_intensity.pooling import (  # noqa: E402
    HIST_BIN_WIDTH_DN,
    estimate_well_null,
)

OUT = MORPHSEQ_ROOT / ".pbx_smoke/out"
EXP = "20260624_2x_td_bf_pbx_coll_plate01"
PRODUCT = "RFP__projection__max"
MERGED = OUT / "object_extraction" / EXP / "channel_intensity" / f"{EXP}_channel_intensity.csv"

raw = pd.read_csv(MERGED)
for column in ("annulus_hist_counts", "embryo_hist_counts"):
    raw[column] = raw[column].map(json.loads)

pd.set_option("display.width", 220)


def hist_mean(counts) -> float:
    counts = np.asarray(counts, dtype=float)
    centers = (np.arange(len(counts)) + 0.5) * HIST_BIN_WIDTH_DN
    return float((counts * centers).sum() / counts.sum()) if counts.sum() else float("nan")


# --- background, per well ---------------------------------------------------------------------
# The null stays per WELL (pooled over that well's timepoints) because it estimates the optical
# background of a physical well, which is a property of the well and its rim -- not of a timepoint.
# NULL PER (WELL, TIMEPOINT). Pooling a well's annuli across timepoints assumes the background is
# stationary, which is false here: the three timepoints are three separate ND2s from three days at
# two different exposures. MEASURED on A01 -- the pooled null gives 400 DN while t0's own background
# is 656 DN, a 256 DN under-subtraction on a ~1100 DN signal (23% error), biased toward exactly the
# timepoint whose exposure differs.
nulls = {
    key: estimate_well_null(g.to_dict("records"))
    for key, g in raw.groupby(["well_id", "time_index"])
}

rows = []
for _, row in raw.iterrows():
    null = nulls[(row["well_id"], int(row["time_index"]))]
    mean_dn = hist_mean(row["embryo_hist_counts"])
    exposure = float(row.get("exposure_ms", float("nan")))
    bgsub = mean_dn - null["null_mode_dn"]
    rows.append({
        "well_id": row["well_id"],
        "time_index": int(row["time_index"]),
        # The physical embryo is the identity that must persist across timepoints -- the whole point
        # of the consistency check. Fall back to well_id when the column is absent (one embryo/well).
        "embryo": str(row.get("physical_embryo_id") or row["well_id"]),
        "exposure_ms": exposure,
        "bgsub_dn": bgsub,
        # Per-ms only when the row KNOWS its exposure. Never a guessed default: dividing by a
        # fabricated exposure is the silent error the whole exposure chain exists to prevent.
        "bgsub_per_ms": bgsub / exposure if np.isfinite(exposure) and exposure > 0 else np.nan,
        "snr": bgsub / null["null_robust_sigma_dn"] if null["null_robust_sigma_dn"] else np.nan,
        "clipped_px": int(row["embryo_clipped_px"]),
    })
d = pd.DataFrame(rows)

have_exposure = d["bgsub_per_ms"].notna().any()
value_col = "bgsub_per_ms" if have_exposure else "bgsub_dn"
print(f"=== {len(d)} embryo-times | clustering on {value_col} "
      f"({'exposure-normalized' if have_exposure else 'NO exposure on rows — raw DN'}) ===\n")


def classify_within(values: np.ndarray, k: int = 3) -> np.ndarray:
    """Assign 1-D values to k ordered classes by 1-D k-means, class 0 = dimmest.

    1-D k-means rather than a fixed cut: a fixed threshold would impose the answer, and with n this
    small a full mixture model would fit noise. Initialised at quantiles so the result is
    deterministic -- a random init would make the class labels themselves unreproducible.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < k:
        return np.full(len(values), -1)
    centers = np.quantile(finite, np.linspace(0.15, 0.85, k))
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new = np.array([
            values[labels == i].mean() if (labels == i).any() else centers[i] for i in range(k)
        ])
        if np.allclose(new, centers):
            break
        centers = np.sort(new)
    return np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)


# --- cluster INDEPENDENTLY within each timepoint ----------------------------------------------
d["dosage_class"] = -1
for t, group in d.groupby("time_index"):
    labels = classify_within(group[value_col].to_numpy())
    d.loc[group.index, "dosage_class"] = labels
    print(f"--- timepoint {t} (n={len(group)}) ---")
    view = group.assign(dosage_class=labels).sort_values(value_col, ascending=False)
    print(view[["embryo", value_col, "snr", "clipped_px", "dosage_class"]].to_string(index=False))
    print()

# --- THE ACTUAL TEST: does one embryo keep its class across timepoints? ------------------------
print("=== CONSISTENCY: copy number is FIXED, so a physical embryo must keep its class ===")
pivot = d.pivot_table(index="embryo", columns="time_index", values="dosage_class", aggfunc="first")
print(pivot.to_string())

consistent = pivot.apply(lambda r: r.dropna().nunique() == 1, axis=1)
print(f"\n  embryos with a STABLE class across all timepoints: {int(consistent.sum())}/{len(pivot)}")
if len(pivot):
    print(f"  fraction stable: {consistent.mean():.0%}")
print(
    "\n  A high fraction means three INDEPENDENT clusterings agreed about which embryos are bright,\n"
    "  which is what a fixed, heritable property looks like. A low fraction means the clusters are\n"
    "  tracking something time-varying (stage, focus, z) and are not reading copy number."
)

# --- the intensity ladder, for whether classes sit at plausible ratios -------------------------
print("\n=== CLASS AMPLITUDE RATIOS (a copy-number ladder should read ~0 : 1 : 2) ===")
for t, group in d.groupby("time_index"):
    means = group.groupby("dosage_class")[value_col].mean().sort_index()
    if len(means) >= 2 and means.iloc[0] > 0:
        ratios = " : ".join(f"{v / means.iloc[0]:.2f}" for v in means)
    else:
        ratios = "n/a (dimmest class at or below zero)"
    print(f"  t{t}: {ratios}   (means: {', '.join(f'{v:.2f}' for v in means)})")

d.to_csv(str(Path(__file__).resolve().parents[1] / "output" / "dosage_per_timepoint.csv"), index=False)
print("\nwrote dosage_per_timepoint.csv")
