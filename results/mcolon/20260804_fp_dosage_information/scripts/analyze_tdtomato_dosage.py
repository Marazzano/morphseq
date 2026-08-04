"""Dosage structure WITHIN the genetically-equivalent tdtomato wells.

WHY THIS POPULATION AND NOT THE WHOLE PLATE. The genotype sheet has three groups:

    column 1      ab                             8 wells   NON-transgenic
    columns 2-7   tdtomato                      48 wells
    columns 8-12  pbx4_pbx1b_crispant_tdtomato  40 wells

The transgene is the SAME in both fluorescent groups; what differs is the crispant perturbation.
So copy number is not a designed variable anywhere on this plate, and clustering across groups
measures the perturbation, not dosage -- which is exactly the error an earlier pass made by fitting
a "3-class dosage ladder" to four wells that turned out to be a control plus one transgenic group.

The 48 tdtomato wells ARE the clean population: genetically equivalent by design, so any brightness
structure among them is het/homo segregation plus noise. If a transgene segregates, a clutch of
carriers should split roughly 1:2:1 (homo:het:non), and the het/homo split should sit at ~2x.

WHAT WOULD COUNT AS A RESULT, decided before looking:
  - a BIMODAL split among fluorescent tdtomato embryos at a ratio near 2x  -> dosage is readable
  - a unimodal spread, or a split at a ratio far from 2x                   -> it is not, and the
                                                                             spread is expression
                                                                             level, not copy number
Both outcomes are informative; only the first supports analysing het and homo together.

Everything is compared WITHIN one timepoint (same ND2, same exposure, same session) because
fluorescence is not stable across developmental time.
"""

from __future__ import annotations

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
MERGED = (
    MORPHSEQ_ROOT / ".pbx_smoke/out/object_extraction" / EXP
    / "channel_intensity" / f"{EXP}_channel_intensity.csv"
)
PLATE_XLSX = MORPHSEQ_ROOT / ".pbx_smoke/in/plate_metadata" / f"{EXP}_well_metadata.xlsx"
OUTDIR = Path(__file__).resolve().parents[1] / "output"


def load_genotypes() -> dict[str, str]:
    """well letter+number -> genotype, from the plate-layout sheet."""
    sheet = pd.read_excel(PLATE_XLSX, sheet_name="genotype").set_index("Unnamed: 0")
    return {
        f"{row}{int(col):02d}": str(value)
        for row, series in sheet.iterrows()
        for col, value in series.items()
        if pd.notna(value)
    }


raw = pd.read_csv(MERGED)
for column in ("annulus_hist_counts", "embryo_hist_counts"):
    raw[column] = raw[column].map(json.loads)

genotypes = load_genotypes()
raw["well"] = raw["well_id"].str.rsplit("_", n=1).str[-1]
raw["genotype"] = raw["well"].map(genotypes)

print(f"=== {len(raw)} embryo-times, {raw.well_id.nunique()} wells ===")
print(raw.groupby("genotype")["well_id"].nunique().to_string())

CENTERS = (np.arange(2048) + 0.5) * HIST_BIN_WIDTH_DN
nulls = {w: estimate_well_null(g.to_dict("records")) for w, g in raw.groupby("well_id")}

rows = []
for _, r in raw.iterrows():
    null = nulls[r["well_id"]]
    counts = np.asarray(r["embryo_hist_counts"], dtype=float)
    mean_dn = float((counts * CENTERS[: len(counts)]).sum() / counts.sum()) if counts.sum() else np.nan
    bgsub = mean_dn - null["null_mode_dn"]
    exposure = float(r.get("exposure_ms", np.nan))
    rows.append({
        "well_id": r["well_id"],
        "well": r["well"],
        "genotype": r["genotype"],
        "time_index": int(r["time_index"]),
        "bgsub_dn": bgsub,
        "bgsub_per_ms": bgsub / exposure if np.isfinite(exposure) and exposure > 0 else np.nan,
        "snr": bgsub / null["null_robust_sigma_dn"] if null["null_robust_sigma_dn"] else np.nan,
        "embryo_px": int(r["embryo_px"]),
        "clipped_px": int(r["embryo_clipped_px"]),
    })
d = pd.DataFrame(rows)
value = "bgsub_per_ms" if d["bgsub_per_ms"].notna().any() else "bgsub_dn"

# --- the non-transgenic controls set the "no transgene" floor --------------------------------
print(f"\n=== THE FLOOR: non-transgenic (ab) controls, {value} ===")
ab = d[d.genotype == "ab"]
if len(ab):
    for t, g in ab.groupby("time_index"):
        print(f"  t{t}: n={len(g):3d}  median={g[value].median():8.3f}  p95={g[value].quantile(0.95):8.3f}  max={g[value].max():8.3f}")
    print("\n  An embryo above the ab p95 is fluorescent above what a non-carrier produces.")
else:
    print("  (no ab wells in this run)")

# --- structure within tdtomato ----------------------------------------------------------------
print(f"\n=== THE QUESTION: structure within tdtomato (genetically equivalent), {value} ===")
td = d[d.genotype == "tdtomato"]
for t, g in td.groupby("time_index"):
    vals = g[value].dropna().sort_values()
    if len(vals) < 4:
        print(f"\n--- t{t}: n={len(vals)}, too few to assess ---")
        continue
    floor = ab[ab.time_index == t][value].quantile(0.95) if len(ab) else np.nan
    carriers = vals[vals > floor] if np.isfinite(floor) else vals

    print(f"\n--- t{t}: n={len(vals)} tdtomato embryos, {len(carriers)} above the ab floor ---")
    qs = [0, 10, 25, 50, 75, 90, 100]
    print("   percentiles: " + "  ".join(f"p{q}={np.percentile(vals, q):.2f}" for q in qs))

    if len(carriers) >= 6:
        # THE 2x TEST. Split the carriers at the largest gap in log space -- a copy-number split is
        # multiplicative, so a gap in log space is what a het/homo boundary looks like. Then ask
        # whether the two groups sit ~2x apart, which is what copy number predicts.
        logs = np.log2(carriers.to_numpy())
        gaps = np.diff(logs)
        cut = int(np.argmax(gaps))
        lo, hi = carriers.to_numpy()[: cut + 1], carriers.to_numpy()[cut + 1 :]
        if len(lo) >= 2 and len(hi) >= 2:
            ratio = hi.mean() / lo.mean()
            print(f"   largest log-gap split: {len(lo)} dim / {len(hi)} bright, ratio = {ratio:.2f}x")
            print(f"     dim  mean {lo.mean():8.3f}   bright mean {hi.mean():8.3f}")
            verdict = ("CONSISTENT with a 1-vs-2 copy split" if 1.6 <= ratio <= 2.6
                       else "NOT a copy-number ratio (copy number predicts ~2.0x)")
            print(f"     -> {verdict}")
            # A gap is only meaningful if it stands out from the others.
            print(f"     largest gap {gaps[cut]:.3f} vs median gap {np.median(gaps):.3f} "
                  f"(log2 units; a real mode boundary should dominate)")
        else:
            print("   split leaves a group of <2; no ratio computed")

    # Carrier fraction: a segregating transgene in a carrier incross gives ~75% fluorescent.
    if np.isfinite(floor) and len(vals):
        print(f"   fluorescent fraction: {len(carriers)}/{len(vals)} = {len(carriers)/len(vals):.0%} "
              f"(a het incross predicts ~75%)")

OUTDIR.mkdir(parents=True, exist_ok=True)
d.to_csv(OUTDIR / "tdtomato_dosage.csv", index=False)
print(f"\nwrote {OUTDIR / 'tdtomato_dosage.csv'}")
