"""THE QUESTION: after background correction and normalization, do the dosage classes carry the
same amount of information?

The worry this answers, stated precisely. The transgene is PAN-NUCLEAR, so fluorescence intensity is
a proxy for CELL ABUNDANCE. A 1-copy and a 2-copy embryo are measuring the same underlying quantity
at different scale. If dosage were a pure multiplicative factor, normalizing would recover identical
information and the classes could be analysed together. If instead the dim class sits nearer the
noise floor, it loses information the bright class keeps -- and joint analysis silently favours the
brighter embryos.

That is a question about INFORMATION CONTENT, not about whether the means separate. Two
distributions can separate cleanly and still carry different amounts of information about cell
abundance.

WHY THE ENTROPY OF THE RAW HISTOGRAM IS NOT THE ANSWER. Bins are fixed-width in DN, so a 2x brighter
distribution spreads over ~2x as many bins for free. Its entropy rises by ~1 bit purely from being
brighter -- an artifact of the bin grid, not information about the embryo. The comparison must be
made after putting every embryo on a common scale, which is exactly what "does it survive
normalization" means.

MEASURED ON THE RFP RASTER (RFP__projection__max, native uint16). BF contributes only the
segmentation mask -- detection is BF-only -- never the intensities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from data_pipeline.feature_extraction.channel_intensity.pooling import (  # noqa: E402
    HIST_BIN_WIDTH_DN,
    estimate_well_null,
)

EXP = "20260624_2x_td_bf_pbx_coll_plate01"
MERGED = Path(".pbx_smoke/out/object_extraction") / EXP / "channel_intensity" / f"{EXP}_channel_intensity.csv"

raw = pd.read_csv(MERGED)
for column in ("annulus_hist_counts", "embryo_hist_counts"):
    raw[column] = raw[column].map(json.loads)
assert (raw["source_image_product_key"] == "RFP__projection__max").all(), "must be RFP pixels"

CENTERS = (np.arange(2048) + 0.5) * HIST_BIN_WIDTH_DN
nulls = {w: estimate_well_null(g.to_dict("records")) for w, g in raw.groupby("well_id")}
classes = pd.read_csv("dosage_per_timepoint.csv") if Path("dosage_per_timepoint.csv").exists() else None


def entropy_bits(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def pixels_from_hist(counts) -> np.ndarray:
    """Reconstruct the pixel population from its histogram (bin centers, repeated by count)."""
    counts = np.asarray(counts, dtype=int)
    return np.repeat(CENTERS[: len(counts)], counts)


rows = []
for _, r in raw.iterrows():
    null = nulls[r["well_id"]]
    px = pixels_from_hist(r["embryo_hist_counts"])
    bg = px - null["null_mode_dn"]          # background-corrected, NOT clipped at zero
    exposure = float(r.get("exposure_ms", np.nan))
    per_ms = bg / exposure if np.isfinite(exposure) and exposure > 0 else bg * np.nan

    mean_bg = float(bg.mean())
    sigma = float(null["null_robust_sigma_dn"])

    # FOUR NORMALIZATIONS, because the answer may depend on which is used and quoting only the
    # flattering one would be cherry-picking.
    #   /mean   pure scale removal -- the literal "is it just a scale factor?" test
    #   /median robust to the bright tail
    #   z       location AND scale removed
    #   log1p   right for a multiplicative rather than additive process
    variants = {
        "raw_dn": bg,
        "per_ms": per_ms,
        "div_mean": bg / mean_bg if mean_bg != 0 else bg * np.nan,
        "div_median": bg / np.median(bg) if np.median(bg) != 0 else bg * np.nan,
        "zscore": (bg - bg.mean()) / bg.std() if bg.std() > 0 else bg * np.nan,
        "log1p_z": None,
    }
    shifted = bg - bg.min() + 1.0
    lg = np.log1p(shifted)
    variants["log1p_z"] = (lg - lg.mean()) / lg.std() if lg.std() > 0 else lg * np.nan

    out = {
        "well_id": r["well_id"],
        "time_index": int(r["time_index"]),
        "embryo_px": int(r["embryo_px"]),
        "exposure_ms": exposure,
        "mean_bgsub_dn": mean_bg,
        "snr": mean_bg / sigma if sigma else np.nan,
        "clipped_px": int(r["embryo_clipped_px"]),
        # Effective dynamic range above the noise floor, in units of the background sigma. This is
        # the "usable range" a normalization cannot manufacture: it is set at acquisition.
        "dynamic_range_sigmas": float((np.percentile(bg, 99) - np.percentile(bg, 1)) / sigma) if sigma else np.nan,
        "cv": float(bg.std() / mean_bg) if mean_bg else np.nan,
    }
    # Entropy on a COMMON 256-bin grid per variant, so bin count never differs between embryos --
    # any entropy difference is then about distribution shape, not about the grid.
    for name, values in variants.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[f"H_{name}"] = np.nan
            continue
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[f"H_{name}"] = np.nan
            continue
        counts, _ = np.histogram(finite, bins=256, range=(lo, hi))
        out[f"H_{name}"] = entropy_bits(counts)
    rows.append(out)

info = pd.DataFrame(rows)
if classes is not None:
    key = classes[["well_id", "time_index", "dosage_class"]]
    info = info.merge(key, on=["well_id", "time_index"], how="left")

pd.set_option("display.width", 240)
print("=== PER EMBRYO-TIME (RFP, background-corrected) ===")
show = ["well_id", "time_index", "dosage_class", "mean_bgsub_dn", "snr", "dynamic_range_sigmas",
        "cv", "clipped_px", "H_raw_dn", "H_div_mean", "H_zscore", "H_log1p_z"]
print(info[[c for c in show if c in info]].round(3).to_string(index=False))

print("\n=== BY DOSAGE CLASS: does normalization equalize information? ===")
agg = info.groupby("dosage_class").agg(
    n=("mean_bgsub_dn", "size"),
    mean_dn=("mean_bgsub_dn", "mean"),
    snr=("snr", "mean"),
    dyn_range_sigmas=("dynamic_range_sigmas", "mean"),
    cv=("cv", "mean"),
    clipped=("clipped_px", "sum"),
    H_raw=("H_raw_dn", "mean"),
    H_div_mean=("H_div_mean", "mean"),
    H_z=("H_zscore", "mean"),
    H_log1p=("H_log1p_z", "mean"),
).round(3)
print(agg.to_string())

print("\n=== THE VERDICT TEST ===")
print("If dosage were a PURE SCALE FACTOR, entropy after /mean would be equal across classes and")
print("the raw-entropy gap would vanish. Spread across classes, in bits:\n")
for column, label in (("H_raw_dn", "raw DN"), ("H_div_mean", "divided by mean"),
                      ("H_zscore", "z-scored"), ("H_log1p_z", "log1p + z")):
    if column in info:
        by_class = info.groupby("dosage_class")[column].mean().dropna()
        if len(by_class) >= 2:
            print(f"  {label:18s} range = {by_class.max() - by_class.min():.3f} bits "
                  f"({', '.join(f'{v:.2f}' for v in by_class)})")

print("\n  A spread that COLLAPSES after normalization => the raw gap was a brightness artifact and")
print("  the classes carry equal information: they can be analysed together.")
print("  A spread that SURVIVES => the dim class really carries less, and joint analysis is unsafe.")

print("\n=== THE FLOOR, which no normalization can lift ===")
print("Dynamic range above background, in sigmas -- set at ACQUISITION, not recoverable later:")
for cls, group in info.groupby("dosage_class"):
    print(f"  class {int(cls)}: {group['dynamic_range_sigmas'].mean():7.2f} sigmas   "
          f"(SNR {group['snr'].mean():6.2f}, CV {group['cv'].mean():.2f})")

info.to_csv("information_content.csv", index=False)
print("\nwrote information_content.csv")
