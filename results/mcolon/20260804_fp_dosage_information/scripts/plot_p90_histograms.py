"""Per-timepoint histogram of each embryo's 90th-percentile RFP intensity.

WHY p90 AND NOT THE MEAN. The mean is diluted by however much of the embryo is dim -- yolk, dark
tissue, whatever fraction of the mask is not expressing -- so two embryos with the same copy number
but different body composition get different means. The 90th percentile asks "how bright are this
embryo's BRIGHTEST pixels", which is much closer to a per-nucleus expression level and far less
sensitive to what else is inside the mask. For a copy-number question that is the better statistic.

It is computed EXACTLY from the stored histogram (2048 bins x 32 DN), not by re-reading pixels --
which is the payoff for having persisted histograms rather than summary statistics.

ONE PANEL PER TIMEPOINT, never pooled: the three timepoints are separate ND2s from different days,
t0 at 600 ms and t1/t2 at 300 ms, so pooling would smear an exposure difference into the
distribution. Values are per-ms so the panels are comparable, but they stay separate panels.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from data_pipeline.feature_extraction.channel_intensity.pooling import (  # noqa: E402
    HIST_BIN_WIDTH_DN,
    estimate_well_null,
)

EXP = "20260624_2x_td_bf_pbx_coll_plate01"
OUTDIR = Path(__file__).resolve().parents[1] / "output"
CENTERS = (np.arange(2048) + 0.5) * HIST_BIN_WIDTH_DN

# Colour-blind safe, and deliberately NOT the genotype palette: these are acquisition groups, not
# the wildtype/het/homo axis that palette encodes.
GROUP_COLOR = {
    "ab": "#808080",
    "tdtomato": "#2166AC",
    "pbx4_pbx1b_crispant_tdtomato": "#B2182B",
}
GROUP_LABEL = {
    "ab": "ab (non-transgenic)",
    "tdtomato": "tdtomato",
    "pbx4_pbx1b_crispant_tdtomato": "pbx4/pbx1b crispant",
}


def percentile_from_hist(counts: np.ndarray, q: float) -> float:
    """Exact q-th percentile of the pixel population, read off the histogram."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return float(CENTERS[: len(counts)][np.searchsorted(np.cumsum(counts) / total, q)])


def load_genotypes() -> dict[str, str]:
    sheet = pd.read_excel(
        MORPHSEQ_ROOT / ".pbx_smoke/in/plate_metadata" / f"{EXP}_well_metadata.xlsx",
        sheet_name="genotype",
    ).set_index("Unnamed: 0")
    return {
        f"{row}{int(col):02d}": str(v)
        for row, series in sheet.iterrows()
        for col, v in series.items()
        if pd.notna(v)
    }


shards = sorted(glob.glob(str(
    MORPHSEQ_ROOT / ".pbx_smoke/out/object_extraction" / EXP
    / "channel_intensity/per_well/*/RFP__projection__max/channel_intensity.csv"
)))
frames = [f for f in (pd.read_csv(p) for p in shards) if len(f)]
if not frames:
    raise SystemExit("no non-empty shards on disk")
raw = pd.concat(frames, ignore_index=True)
raw["embryo_hist_counts"] = raw["embryo_hist_counts"].map(json.loads)
raw["annulus_hist_counts"] = raw["annulus_hist_counts"].map(json.loads)
raw["well"] = raw["well_id"].str.rsplit("_", n=1).str[-1]
raw["genotype"] = raw["well"].map(load_genotypes())

# Null per (well, TIMEPOINT): the background is contemporaneous with the signal it corrects, and
# t0's 600 ms frames have a genuinely higher background than t1/t2's 300 ms ones.
nulls = {k: estimate_well_null(g.to_dict("records")) for k, g in raw.groupby(["well_id", "time_index"])}

rows = []
for _, r in raw.iterrows():
    null = nulls[(r["well_id"], int(r["time_index"]))]
    exposure = float(r.get("exposure_ms", np.nan))
    p90 = percentile_from_hist(r["embryo_hist_counts"], 0.90) - null["null_mode_dn"]
    rows.append({
        "well": r["well"],
        "genotype": r["genotype"],
        "time_index": int(r["time_index"]),
        "exposure_ms": exposure,
        "p90_bgsub_dn": p90,
        "p90_per_ms": p90 / exposure if np.isfinite(exposure) and exposure > 0 else np.nan,
    })
d = pd.DataFrame(rows)
value = "p90_per_ms" if d["p90_per_ms"].notna().any() else "p90_bgsub_dn"

times = sorted(d["time_index"].unique())
fig, axes = plt.subplots(len(times), 1, figsize=(9, 3.0 * len(times)), sharex=True)
axes = np.atleast_1d(axes)

# One shared bin grid across panels so the eye compares shapes, not binnings.
finite = d[value].replace([np.inf, -np.inf], np.nan).dropna()
bins = np.linspace(0, float(np.ceil(finite.max() * 1.05)), 36)

for ax, t in zip(axes, times):
    at_t = d[d.time_index == t]
    present = [g for g in GROUP_COLOR if (at_t.genotype == g).any()]
    ax.hist(
        [at_t[at_t.genotype == g][value].dropna() for g in present],
        bins=bins, stacked=True,
        color=[GROUP_COLOR[g] for g in present],
        label=[GROUP_LABEL[g] for g in present],
        edgecolor="white", linewidth=0.5,
    )
    # The ab p95 is the empirical "no transgene" line: right of it is fluorescence a non-carrier
    # does not produce.
    ab = at_t[at_t.genotype == "ab"][value].dropna()
    if len(ab):
        ax.axvline(ab.quantile(0.95), color="#404040", ls="--", lw=1.2,
                   label="ab p95 (carrier floor)" if t == times[0] else None)
    exposure = at_t["exposure_ms"].dropna()
    stamp = f"{exposure.iloc[0]:.0f} ms" if len(exposure) else "exposure unknown"
    ax.set_title(f"t{t}  —  n={len(at_t)} embryos, RFP exposure {stamp}", fontsize=10, loc="left")
    ax.set_ylabel("embryos")
    ax.legend(fontsize=8, frameon=False)

axes[-1].set_xlabel("embryo p90 RFP intensity, background-subtracted (DN/ms)")
fig.suptitle(
    "Per-embryo 90th-percentile RFP intensity, by timepoint\n"
    f"{EXP} — {d.well.nunique()} wells on disk",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))

OUTDIR.mkdir(parents=True, exist_ok=True)
png = OUTDIR / "p90_intensity_histograms.png"
fig.savefig(png, dpi=150)
d.to_csv(OUTDIR / "p90_intensity.csv", index=False)

print(f"wrote {png}")
for t in times:
    at_t = d[d.time_index == t]
    ab = at_t[at_t.genotype == "ab"][value].dropna()
    floor = ab.quantile(0.95) if len(ab) else np.nan
    print(f"\nt{t}: n={len(at_t)}  floor(ab p95)={floor:.2f}")
    for g in [g for g in GROUP_COLOR if (at_t.genotype == g).any()]:
        v = at_t[at_t.genotype == g][value].dropna()
        above = (v > floor).sum() if np.isfinite(floor) else 0
        print(f"   {GROUP_LABEL[g]:24s} n={len(v):3d}  median={v.median():7.2f}  above floor={above}")
