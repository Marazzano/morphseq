# Design: `channel_intensity_qc` — features to extract, and how to gate on them

Status: **proposal**, written from the full-plate analysis (96 wells, 334 embryo-times). Nothing
here is wired into the DAG yet.

---

## The thing this fixes

The analysis kept saying "normalization cannot recover an embryo that was never digitized above
noise". That is stated as an *analysis caveat* in the write-ups, which is the wrong place for it:

```
dim image containing only:  100, 101, 102, 103
stretched to:               0, 85, 170, 255
still only 4 distinct measurements
```

An embryo like that should never reach the analysis. It is a **QC exclusion**, decided once from
extracted features, not a footnote every downstream reader has to remember. The corollary is that
the features the gate needs must be *extracted and persisted*, not recomputed ad hoc — which is
what the exploratory `scripts/intensity_qc_gate.py` currently does.

---

## What is already extracted (nothing new needed at object grain)

`object_extraction/channel_intensity` already persists, per embryo-time x source product:

| column | why it is sufficient |
|---|---|
| `embryo_hist_counts` | 2048 x 32-DN histogram — entropy, effective states, percentiles all derive from it exactly |
| `embryo_sum_dn`, `embryo_sumsq_dn`, `embryo_px` | exact moments |
| `embryo_clipped_px` | saturation at the ceiling |
| `annulus_hist_counts`, `annulus_px` | the background this embryo is judged against |
| `exposure_ms`, `illumination_power` | acquisition scale, so DN/ms is available |

**No pixel re-read is required for any feature below.** That is the payoff for having persisted
histograms rather than summary statistics — the QC module is arithmetic over an existing table.

---

## Features to extract (new module: `feature_extraction/channel_intensity_qc`)

Grain: **one row per (embryo-time, source_image_product_key)**, matching `channel_intensity`.

### Resolution — is the signal digitized finely enough to survive normalization?

| feature | definition | why not the obvious alternative |
|---|---|---|
| `entropy_bits` | Shannon H of the embryo histogram | |
| `effective_states` | `2 ** entropy_bits` | NOT `occupied_levels`. Measured: H05 shows 102 occupied levels but 16 effective — read noise inflates the raw count 6x. A level held by one pixel barely moves entropy. |
| `occupied_levels` | count of non-empty bins | keep for diagnosis, do not gate on it |

### Separation — is the embryo distinguishable from the well it sits in?

| feature | definition |
|---|---|
| `separation_sigma` | `(embryo_mean − annulus_mode) / annulus_robust_sigma` |
| `well_null_mode_dn`, `well_null_sigma_dn` | copied onto the row so it is self-explaining without a join |

The null must be estimated **per (well, timepoint)**, never pooled across timepoints — measured on
this plate, t0 backgrounds run 560–688 DN against 336–400 at t1/t2 because t0 was a 600 ms exposure.
Pooling under-subtracts t0 by ~256 DN on a ~1100 DN signal.

### Saturation — is the bright end intact?

| feature | definition |
|---|---|
| `saturated_frac` | `embryo_clipped_px / embryo_px` |

Saturation compresses bright embryos toward dim ones while looking like clean data, so it must be
visible rather than inferred.

### Geometry — is this an embryo at all?

| feature | definition | rationale |
|---|---|---|
| `area_px` | mask area | 5 t2 masks were 425k–4.6M px against a real embryo's ~85k: whole-well blobs |
| `aspect_ratio` | bbox long/short | orientation proxy; ranged 1.01–5.31 on this plate |
| `bbox_fill_fraction` | `area_px / bbox_area` | 0.15 means a fragmented mask, not a fish |

Geometry is included because it is the confound the intensity features cannot see: a mask that
caught more of the fish integrates more signal at the same expression level.

---

## How the gate applies

### Flags annotate; they do not drop rows

This follows the existing convention (`focus_flag`, `motion_blur_flag`) — QC writes a flag column
and a report, and **nothing in the pipeline removes a row on its own**. The exclusion decision is
the analyst's, made explicitly downstream, because a row silently vanishing is how a filtered
population gets mistaken for a measured one.

### Three independent checks, because they fail independently

```
resolution_ok   effective_states  >= 128
separation_ok   separation_sigma  >= 5
unsaturated_ok  saturated_frac    <= 0.01
```

### The verdict is per-QUESTION, not a single boolean

This is the part most easily got wrong, and the earlier write-ups did get it wrong:

| flag | means |
|---|---|
| `usable_for_pattern` | all three checks pass — texture/morphology/spatial work may normalize and pool |
| `usable_for_dosage` | all three **and** `exposure_ms` present **and** the analysis must NOT normalize |

Per-embryo normalization removes the absolute scale, which **is** the dosage signal. A single
`usable` boolean would license erasing the quantity a dosage analysis is trying to read.

### Threshold provenance must be recorded

| threshold | status |
|---|---|
| `effective_states >= 128` | **CALIBRATED** on the full plate against the matched-brightness floor (see below) |
| `separation_sigma >= 5` | a priori, uncalibrated |
| `saturated_frac <= 0.01` | a priori, uncalibrated |

Marking which is which matters: 64 was a guess (2**6) and proved too permissive. Swept against the
floor:

| min effective | t1 kept | t1 residual/floor | t2 kept | t2 residual/floor |
|---|---|---|---|---|
| 64 | 135 | 3.44 | 108 | 1.68 |
| **128** | **116** | **1.80** | **99** | **1.25** |
| 192 | 91 | 2.28 | 91 | 1.29 |
| 256 | 66 | 1.75 | 74 | 1.55 |

128 halves the t1 ratio while keeping 116/135. The non-monotonicity above it is sampling noise in
the floor estimate, not structure.

**The other two thresholds should be calibrated the same way before being trusted.**

---

## Why a brightness ratio is NOT among the checks

The obvious rule — "more than Nx dimmer than its peers, therefore unusable" — is wrong, and the
plate shows it. A09 is 4–7x dimmer than G09 and carries the same information after normalization
(residual 0.287 bits against a 0.17–0.23 floor). Exposure, expression level, copy number and optics
all move absolute brightness while leaving spatial signal intact.

The operative constraint is a **resolution floor**: a 5x-dimmer embryo with 300 effective states is
fine; a 2x-dimmer one with 40 is not.

---

## Wiring (mirrors `feature_extraction/mask_geometry`)

```
feature_extraction/channel_intensity_qc/
    compute.py      pure: channel_intensity rows -> QC rows
    config.py       thresholds as a dataclass, with provenance in comments
    contract.py     column contract for the merge
    entrypoint.py   I/O half
```

- registry step `channel_intensity_qc`, stage `feature_extraction`, `PER_WELL_THEN_MERGE`
- rule reads the per-well `channel_intensity` shard; **no pixel access**, so it is CPU-cheap and can
  run anywhere in the DAG after intensity
- merge is a dumb concat with `required_columns`, like every other merge
- a report under `quality_control/reporting`, following `focus_qc` / `motion_blur_qc`

---

## Open questions

1. **Should `separation_sigma` and `saturated_frac` thresholds be calibrated, and against what?**
   The resolution threshold had a natural target (the matched-brightness entropy floor). Separation
   and saturation need their own, and I do not have one yet.
2. **Does the gate belong at embryo-time grain or track grain?** An embryo that fails at one
   timepoint and passes at two others is currently three independent verdicts. A track-level rollup
   may be what analyses actually want.
3. ~~**`effective_states` depends on pixel count.**~~ **RESOLVED — it does not.** Measured on the
   plate (plausible masks only): `corr(effective_states, area_px) = +0.018`, and small vs large
   masks give median effective states of 263 vs 269. Restricted to well-separated embryos the
   correlation is `-0.035` against `+0.751` for separation. So `effective_states` tracks SIGNAL
   QUALITY, not mask size, and the threshold does not penalise small embryos. This was the check
   most likely to invalidate the whole gate, so it is worth having done first.

---

# CORRECTION: `effective_states` was bit-depth dependent

The gate as first written binned entropy on the **fixed 32-DN grid**, so the number depended on the
image encoding, not just the data. Demonstrated directly — one distribution, two encodings, same
32-DN bins:

| encoding | effective_states |
|---|---|
| 8-bit | **5.7** |
| 16-bit | **1479.8** |
| ratio | **259x for identical data** |

A gate at `>= 128` would have rejected **every 8-bit image on principle**, penalising acquisition
format rather than data quality.

## The fix: bin as a fraction of range, not in absolute DN

Bin each embryo's own 0.5–99.5 percentile span into a **fixed number** of bins (256). Fixing the
count rather than the width is what makes it encoding-free:

| encoding | range-relative effective_states |
|---|---|
| 8-bit | 184.4 |
| 16-bit | 151.1 |
| ratio | **0.82x** |

`entropy_fixed_grid_bits` is retained as a diagnostic — its scale-dependence is exactly the subject
of the entropy-scale analysis, so it must stay available, just not be gated on.

## Recalibrated, and the sweep is now monotonic

| min effective | t1 kept | t1 ratio | t2 kept | t2 ratio |
|---|---|---|---|---|
| 0 | 165 | 6.89 | 160 | 7.98 |
| 64 | 138 | 3.72 | 109 | 1.39 |
| 96 | 123 | 1.86 | 105 | 1.25 |
| **128** | **102** | **1.42** | **99** | **1.18** |
| 160 | 38 | 0.59 | 66 | 0.72 |

128 brings both timepoints to ~1.2–1.4x the floor while keeping ~100 embryos each. 160 goes below
the floor but keeps only 38 at t1 — over-pruning to chase a number.

**The sweep is monotonic here**, unlike the earlier one on the absolute-DN measure, which is further
evidence the encoding-free version measures the intended thing.

## Also considered and rejected: `span_in_sigmas`

`(p99 − p01) / background_sigma` is bit-depth-free by construction and was the obvious simpler
candidate. Head-to-head at matched retention it is consistently the weaker discriminator:

| retention | t1: effective_states | t1: span_in_sigmas | t2: effective_states | t2: span_in_sigmas |
|---|---|---|---|---|
| 85% | 3.38 | 4.23 | 5.09 | 6.87 |
| 70% | **1.01** | 3.31 | **1.17** | 1.88 |
| 55% | 1.13 | 1.62 | 0.96 | 1.01 |

They correlate at ~0.8, so they measure related things, but entropy weights *how* the pixels are
distributed across the range rather than just how wide it is. Kept as a reported feature; not the
gate.
