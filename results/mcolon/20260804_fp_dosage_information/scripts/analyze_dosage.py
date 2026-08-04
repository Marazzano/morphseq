"""Dosage analysis: does copy-number information survive background correction and normalization?

Reads the raw channel_intensity shards, pools the well-level background null, corrects, and then
asks the actual question -- whether the information content is equal across dosage classes after
normalization, or whether the dim class sits closer to the noise floor and loses what the bright
class keeps.
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
ROOT = OUT / "object_extraction" / EXP / "channel_intensity" / "per_well"

frames = []
for well_dir in sorted(ROOT.iterdir()):
    csv = well_dir / PRODUCT / "channel_intensity.csv"
    if csv.exists():
        frames.append(pd.read_csv(csv))
raw = pd.concat(frames, ignore_index=True)
# The entrypoint now skips invalid masks as targets, but shards written before that fix still carry
# them. Drop by the same rule so a re-read of old shards cannot resurrect them.
if "embryo_px" in raw:
    full_frame = raw["embryo_px"] > 1_000_000
    if full_frame.any():
        print(f"DROPPING {int(full_frame.sum())} full-frame (failed-segmentation) target rows")
        raw = raw[~full_frame].reset_index(drop=True)
for column in ("annulus_hist_counts", "embryo_hist_counts"):
    raw[column] = raw[column].map(json.loads)

pd.set_option("display.width", 250)
print(f"=== RAW EVIDENCE: {len(raw)} embryo-times, {raw.well_id.nunique()} wells ===")
show = [
    c
    for c in (
        "well_id",
        "time_index",
        "embryo_px",
        "embryo_clipped_px",
        "annulus_px",
        "annulus_excluded_px",
        "annulus_neighbor_count",
    )
    if c in raw
]
print(raw[show].to_string(index=False))


def hist_mean(counts: list[int]) -> float:
    counts = np.asarray(counts, dtype=float)
    centers = (np.arange(len(counts)) + 0.5) * HIST_BIN_WIDTH_DN
    return float((counts * centers).sum() / counts.sum()) if counts.sum() else float("nan")


# --- pooled null, per well -------------------------------------------------------------------
print("\n=== POOLED BACKGROUND NULL (per well) ===")
nulls = {}
for well, group in raw.groupby("well_id"):
    null = estimate_well_null(group.to_dict("records"))
    nulls[well] = null
    print(
        f"{well}  mode={null['null_mode_dn']:8.1f}  sigma={null['null_robust_sigma_dn']:7.1f}  "
        f"px={null['null_pooled_px']:>10,}  drift={null['null_drift_dn']:6.1f}  "
        f"by_time={ {k: round(v, 1) for k, v in null['null_mode_dn_by_time'].items()} }"
    )

# --- correction ------------------------------------------------------------------------------
rows = []
for _, row in raw.iterrows():
    null = nulls[row["well_id"]]
    embryo_mean = hist_mean(row["embryo_hist_counts"])
    rows.append(
        {
            "well_id": row["well_id"],
            "time_index": int(row["time_index"]),
            "mask_id": row["mask_id"],
            "embryo_px": int(row["embryo_px"]),
            "embryo_mean_dn": embryo_mean,
            "null_mode_dn": null["null_mode_dn"],
            "null_sigma_dn": null["null_robust_sigma_dn"],
            "embryo_mean_bgsub_dn": embryo_mean - null["null_mode_dn"],
            "embryo_snr": (embryo_mean - null["null_mode_dn"]) / null["null_robust_sigma_dn"]
            if null["null_robust_sigma_dn"]
            else float("nan"),
            "clipped_px": int(row["embryo_clipped_px"]),
            "hist": np.asarray(row["embryo_hist_counts"], dtype=float),
        }
    )
corrected = pd.DataFrame(rows)

print("\n=== BACKGROUND-CORRECTED INTENSITY ===")
print(
    corrected.drop(columns=["hist"]).sort_values("embryo_mean_bgsub_dn", ascending=False).to_string(index=False)
)


# --- information content ---------------------------------------------------------------------
def shannon_bits(counts: np.ndarray) -> float:
    """Entropy of the intensity distribution, in bits, over occupied bins only."""
    p = counts[counts > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def occupied_bins(counts: np.ndarray) -> int:
    return int((counts > 0).sum())


print("\n=== INFORMATION CONTENT PER EMBRYO-TIME ===")
info = []
for _, row in corrected.iterrows():
    counts = row["hist"]
    centers = (np.arange(len(counts)) + 0.5) * HIST_BIN_WIDTH_DN
    total = counts.sum()
    cdf = np.cumsum(counts) / total
    p01 = centers[np.searchsorted(cdf, 0.01)]
    p99 = centers[np.searchsorted(cdf, 0.99)]
    info.append(
        {
            "well_id": row["well_id"],
            "time_index": row["time_index"],
            "bgsub_mean": row["embryo_mean_bgsub_dn"],
            "snr": row["embryo_snr"],
            "entropy_bits": shannon_bits(counts),
            "occupied_bins": occupied_bins(counts),
            "p01_dn": p01,
            "p99_dn": p99,
            "range_dn": p99 - p01,
            # THE KEY COLUMN. Entropy of the raw histogram is inflated purely by being brighter: a
            # 2x brighter distribution spreads over 2x as many fixed-width bins for free. Rescaling
            # each embryo to a common span before binning removes that trivial gain and asks whether
            # any information SURVIVES normalization.
            "entropy_norm_bits": shannon_bits(
                np.histogram(
                    np.repeat(centers, counts.astype(int)) / max(row["embryo_mean_bgsub_dn"], 1e-9),
                    bins=256,
                    range=(0, 4),
                )[0].astype(float)
            ),
        }
    )
info_df = pd.DataFrame(info)
print(info_df.to_string(index=False))

# --- IS THE VARIATION DOSAGE, OR IS IT TIME? -------------------------------------------------
# Copy number is FIXED per embryo. If intensity for one embryo swings across its own timelapse by
# as much as it differs between embryos, then between-embryo brightness is not reading dosage and
# no clustering on it can recover copy number.
# EXPOSURE NORMALIZATION. Reads exposure_ms off the row when frame_inventory carries it (see
# nd2_illumination.py); falls back to the MEASURED pbx values otherwise, stated explicitly rather
# than silently, so a run against artifacts that predate the column is still interpretable and is
# never mistaken for a run that had real per-frame exposure.
_MEASURED_PBX_EXPOSURE_MS = {0: 600.0, 1: 300.0, 2: 300.0}
if "exposure_ms" in raw.columns and raw["exposure_ms"].notna().any():
    exposure_by_time = raw.groupby("time_index")["exposure_ms"].first().to_dict()
    print("\n=== EXPOSURE: read from frame_inventory ===")
else:
    exposure_by_time = _MEASURED_PBX_EXPOSURE_MS
    print("\n=== EXPOSURE: frame_inventory lacks exposure_ms; using MEASURED pbx values ===")
    print("    (t0=600ms, t1=t2=300ms, read from the ND2 text metadata -- see"
          " DOSAGE_INFORMATION_ANALYSIS.md)")
print("   ", {k: float(v) for k, v in exposure_by_time.items()})

corrected["exposure_ms"] = corrected["time_index"].map(exposure_by_time)
corrected["bgsub_per_ms"] = corrected["embryo_mean_bgsub_dn"] / corrected["exposure_ms"]

print("\n=== WITHIN-EMBRYO VARIATION ACROSS TIME (dosage is fixed; this should be small) ===")
by_embryo = corrected.assign(track=corrected["mask_id"].str.replace(r"_t\d+_", "_", regex=True))
for track, group in by_embryo.groupby("track"):
    ordered = group.sort_values("time_index")
    vals = ordered["embryo_mean_bgsub_dn"].to_numpy()
    norm = ordered["bgsub_per_ms"].to_numpy()
    fold = float(vals.max()) / max(float(vals.min()), 1e-9)
    fold_norm = float(np.nanmax(norm)) / max(float(np.nanmin(norm)), 1e-9)
    print(
        f"  {track[-24:]:24s} {np.array2string(vals, precision=0):>24s} fold={fold:5.1f}x"
        f"  | /ms {np.array2string(norm, precision=2):>22s} fold={fold_norm:5.1f}x"
    )

within = by_embryo.groupby("track")["embryo_mean_bgsub_dn"].agg(lambda v: v.max() / max(v.min(), 1e-9))
within_norm = by_embryo.groupby("track")["bgsub_per_ms"].agg(
    lambda v: np.nanmax(v) / max(np.nanmin(v), 1e-9)
)
between = corrected["embryo_mean_bgsub_dn"].max() / max(corrected["embryo_mean_bgsub_dn"].min(), 1e-9)
print(f"\n  median WITHIN-embryo fold range : {within.median():.1f}x")
print(f"  median WITHIN-embryo, EXPOSURE-NORMALIZED : {within_norm.median():.1f}x")
print(f"  BETWEEN-embryo fold range       : {between:.1f}x")
print(
    "  If within is comparable to between, brightness is dominated by something that changes over\n"
    "  the timelapse -- stage, focus, bleaching -- not by copy number."
)

print("\n=== THE QUESTION: is entropy explained by brightness? ===")
for column in ("entropy_bits", "entropy_norm_bits"):
    r = np.corrcoef(info_df["bgsub_mean"], info_df[column])[0, 1]
    print(f"  corr(bgsub_mean, {column:18s}) = {r:+.3f}")
print(
    "\n  A strong POSITIVE correlation for entropy_bits with a collapse toward zero for\n"
    "  entropy_norm_bits means the raw entropy gap was a binning artefact of brightness and the\n"
    "  information is scale-free -- i.e. dim and bright embryos carry the SAME information and can\n"
    "  be analysed together after normalization.\n"
    "  A correlation that SURVIVES normalization means the dim class really does carry less, and\n"
    "  joint analysis would favour the bright class."
)
