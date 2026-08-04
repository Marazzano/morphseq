# Does dosage survive normalization? — analysis plan and decisions

Working notes, written while running unattended. Every decision below is recorded so it can be
audited later rather than reconstructed from code.

## The question

The transgene is a **pan-nuclear** marker, so fluorescence intensity is a proxy for **cell
abundance**. A 1-copy and a 2-copy embryo differ by a **scale factor** on the same underlying
quantity — they are not measuring different things.

So the worry is specific: *can embryos of different copy number be analyzed together?* If dosage is
a pure multiplicative scale, normalizing recovers identical information and the answer is yes. If
1-copy sits closer to the noise floor, it loses information the 2-copy retains, and joint analysis
silently favours the brighter class.

That is an empirical question about **information content after normalization**, not about whether
the means separate. Two distributions can separate cleanly and still carry different amounts of
information about cell abundance.

## Why this is not just "compare the means"

Three failure modes a mean-difference test cannot see:

1. **Floor compression.** If 1-copy signal approaches the background, its dynamic range above noise
   shrinks. The mean shifts fine; the *usable* range does not scale.
2. **Saturation compression.** The opposite end — a 2-copy embryo is the most likely to clip, and
   clipping compresses 2-copy toward 1-copy while looking like clean data. Max-over-Z projection
   makes this worse (it takes the brightest plane per pixel).
3. **Variance structure.** Poisson-like noise scales with sqrt(signal), so a 2x brighter sample has
   ~1.4x the noise but 2x the signal — SNR is not scale-invariant. Normalizing by the mean does not
   normalize the information.

## Method

### Step 1 — measurement (already built, needs running)

Per embryo-time, on the NATIVE uint16 raster (never the rendered snip — INTER_AREA mixes photons
across pixel boundaries and biases both saturation counts and distribution tails):

- embryo-interior histogram + exact sum/sumsq/pixel count
- annulus histogram, with neighbour ("close fish") exclusion
- saturation counts at both ends

### Step 2 — background correction

Pooled well-level annulus **mode** (not mean: the well rim autofluoresces at ~1019 DN vs ~575
mid-well, and embryo halo bleeds inward — both right-tailed, both drag a mean). Subtract per embryo,
never clipped at zero.

### Step 3 — cluster into 3 dosage classes

Cluster on background-corrected **mean** intensity (not integrated — integrated scales with embryo
volume, which confounds dosage with developmental stage). Expect the classes to sit at roughly
0 : 1 : 2 relative amplitude if the marker is linear in copy number.

DECISION TO REVISIT: cluster per timepoint or pooled across timepoints? Pooling assumes expression
is stable across 33-77hpf, which for a pan-nuclear marker tracking cell number it is NOT — total
cells grow. Plan: cluster **within** timepoint, then check class membership is consistent for the
same physical embryo across timepoints. Inconsistency is itself a finding.

### Step 4 — the actual question: information content

For each dosage class, after normalization, compute and compare:

| measure | what it answers |
|---|---|
| Shannon entropy of the intensity distribution | raw information content, in bits |
| entropy after per-class scale normalization | does normalizing equalize it? |
| effective dynamic range above background (p99 - mode) / robust sigma | usable SNR |
| coefficient of variation within class | is relative precision preserved? |
| saturated fraction | is 2-copy losing its top end? |

Normalizations to compare, because the answer may depend on which is used:
- divide by class mean (pure scale removal)
- divide by class median
- z-score within class
- log1p then z-score (if the distribution is multiplicative rather than additive)

### Step 5 — the verdict

The claim "the same information is there" is supported only if, after normalization:
- entropies are equal within noise across classes
- effective dynamic range is comparable
- neither floor nor ceiling compression is class-specific

If 1-copy shows reduced entropy or compressed range, joint analysis is NOT safe without an explicit
per-class correction — and the honest answer is to say which classes can be pooled and which cannot.

## Confounds that must be reported, not hidden

- **Exposure/gain are not carried in frame_inventory.** Constant exposure across the plate cannot be
  verified from the artifacts. Within one timelapse well it is a reasonable but UNVERIFIED
  assumption; cross-experiment comparison is unsupported until exposure is carried.
- **Genotype is not known to the pipeline.** It arrives as a per-well string at analysis_ready, and
  0/1/2 copy number exists nowhere in the pipeline. So the three classes here are DISCOVERED from
  intensity, not labelled — which means "did clustering recover the truth?" is a separate question
  from "is the information equal", and needs the genotype table to answer.
- **Only 4 wells in the smoke run.** Sufficient to prove the measurement path; NOT sufficient for a
  dosage conclusion. The full plate (96 wells x 3 timepoints) is needed before any claim about
  information content.

---

# RESULT — first run on real RFP pixels (2026-08-04, 4 wells)

The measurement path works end to end: 12 valid embryo-times measured on the native uint16 RFP
raster, background-corrected against a pooled per-well annulus mode. Zero saturated pixels anywhere,
so no ceiling compression. **The dosage question is not answerable from this data, for a reason that
is itself the finding.**

## The blocking finding: intensity is not stable within one embryo

Copy number is FIXED for an embryo. So within-embryo variation across its own timelapse is a pure
noise floor for any dosage measurement — whatever it is, dosage differences must exceed it to be
readable.

| embryo | bg-corrected mean by timepoint (DN) | fold range |
|---|---|---|
| A01 | 1102 → 473 → 309 | **3.6x** |
| A02 | 1747 → 149 → 154 | **11.8x** |
| B01 | 356 → 38 → 71 | **9.3x** |
| B02 | no background estimate (see below) | — |

- median WITHIN-embryo fold range: **9.3x**
- BETWEEN-embryo fold range: **45.8x**

These are the same order of magnitude. **A 1-vs-2-copy difference is a 2x effect, and it is buried
under a ~9x within-embryo swing.** Clustering brightness into three classes here would partition
timepoints and acquisition conditions, not genotypes — and it would do so while producing three
clean-looking clusters, which is the dangerous part.

Every embryo is brightest at t0 and drops sharply afterwards. That monotone-decreasing shape across
all four wells is the signature of **photobleaching or a per-timepoint exposure change**, not
biology — a pan-nuclear marker tracking cell number should INCREASE as cells divide. Whatever it is,
it is confounded with the exact axis dosage would be read on.

## What must be resolved before dosage can be asked again

1. **Exposure and gain are still not in frame_inventory.** This was flagged as a confound before the
   run; it is now the leading candidate for the t0 cliff and can no longer be deferred. Until
   exposure is carried per frame, a brightness drop is uninterpretable — it cannot be distinguished
   from bleaching or from a real change.
2. **Compare at matched timepoint only.** Within-timepoint between-embryo spread is the only
   comparison this data supports. With 4 wells that is 4 embryos per timepoint — far too few, which
   is why the full plate is not optional.
3. **Then re-ask the information question.** The entropy machinery is built and runs; it was not
   reported here because with the between-embryo signal confounded by time, an entropy comparison
   across "classes" would be comparing timepoints wearing genotype labels.

## Secondary finding: a failed segmentation destroys its neighbour's background

B02 carries a full-frame 2304x2304 mask (area 4,987,142 px, ~100x a real embryo, confidence 0.0,
is_valid_mask False). It is correctly RETAINED as a neighbour — an invalid mask is still a fish
emitting photons — but because it covers the whole frame, dilating the neighbour union erased 100%
of the real embryo's annulus: `annulus_px` 0, `annulus_excluded_px` 132,627, pooled null NaN.

So B02 has no background estimate at all, caused entirely by a different mask's failure. Options
recorded in the commit; capping exclusion area is most likely right but needs a threshold chosen
against more than one well.

## What IS established

- native uint16 survives to measurement (no per-frame rescale, no 8-bit conversion)
- zero saturation, so the bright end is not compressed
- pooled annulus mode gives a stable background (~368-400 DN across wells)
- background drift across time is 192-352 DN, which is NOT negligible against the dimmest
  embryos (38-154 DN bg-corrected) — the per-timepoint null already computed should be preferred
  over the pooled one for those rows
- SNR spans 0.24 to 13.7, so the dim end genuinely sits AT the noise floor. That is the floor
  compression this document predicted, and it is real: those embryos cannot carry the same
  information as the bright ones no matter how they are normalized.

---

# ROOT CAUSE FOUND — the exposure changed between timepoints

The "timepoints" in this experiment are not a timelapse. They are **three separate ND2 files
acquired on three different days**:

```
20260624_..._t33hpf.nd2   fluorescence Exposure: 600 ms
20260625_..._t52hpf.nd2   fluorescence Exposure: 300 ms
20260626_..._t77hpf.nd2   fluorescence Exposure: 300 ms
```

**t33hpf was acquired at 2x the exposure of the other two.** Laser power (Celesta line at 20.0) and
DIA iris (7.5 / 18.1) are byte-identical across all three files, so exposure is the only acquisition
parameter that moved — and t0 is exactly the timepoint that was mysteriously brightest in every
single well.

Exposure IS present in the ND2 text metadata (`Camera Settings: Exposure: N ms`). It is simply not
carried through to `frame_inventory`, which is why the confound was invisible to the analysis and had
to be recovered by reading the raw files.

## Normalizing by exposure removes half the problem

| embryo | raw fold range | per-ms normalized (DN/ms) | fold range |
|---|---|---|---|
| A01 | 3.6x | 1.84, 1.58, 1.03 | **1.8x** |
| A02 | 11.8x | 2.91, 0.50, 0.52 | **5.9x** |
| B01 | 9.3x | 0.59, 0.13, 0.24 | **4.7x** |

median within-embryo fold range: **9.3x -> 4.7x**

So exposure explains roughly half the within-embryo swing and the rest is still unexplained —
plausibly real biology (stage), focus, or z-position, all of which move a max-projection.

A 4.7x residual noise floor is still far too large to read a 2x dosage difference against. But this
is now a tractable, identified problem rather than an unexplained one.

## THE FIX, in priority order

1. **Carry exposure (and laser power) per frame into `frame_inventory`.** It is in the ND2 text
   metadata already; nothing needs to be re-acquired. This is the single highest-value change and it
   unblocks everything else. Without it, ANY cross-timepoint fluorescence comparison in this project
   is silently wrong by whatever factor the exposure differed — this experiment happened to differ by
   exactly 2x, which is the same size as the dosage effect being hunted.
2. **Compare only within a timepoint** until (1) lands. Same file, same exposure, same session.
3. **Then re-ask dosage** on the full plate at a single timepoint, where 96 wells give enough
   embryos for three classes to be distinguishable from three lumps.

## Revised verdict on the original question

"Is the same information there after normalization?" cannot be answered yet, but the run establishes
its precondition is currently NOT met: **the dim end of the observed range sits at SNR 0.24-1.2, at
the noise floor.** Those embryos cannot carry information equal to the SNR-13.7 bright end under any
normalization — normalization rescales a distribution, it does not recover dynamic range that was
never digitized. If the dim class turns out to be 1-copy rather than an exposure artefact, joint
analysis with 2-copy embryos is NOT safe.

Establishing whether that dim end is 1-copy or is simply the 300 ms sessions requires fix (1).

---

# FIX IN PROGRESS — exposure is now read from the ND2

Priority (1) above is underway.

**Landed:**
- `scope/yx1/nd2_illumination.py` — parses `exposure_ms`, `illumination_power`, `dia_iris_intensity`
  per channel out of the ND2 free-text dump, with 8 alignment tests. Verified on all three pbx files:
  BF 11 ms constant, tdtomato 600/300/300.
- `exposure_ms` / `illumination_power` / `dia_iris_intensity` on the YX1 acquisition inventory,
  populated from the same single ND2 open. NaN when unparsed — never a default, because a fabricated
  exposure is indistinguishable from a measured one downstream.

**Remaining:** thread the three columns through `image_materialization` onto `frame_inventory`, which
is where `channel_intensity` reads. That crosses the emitted-column contract and both scope
materializers, so it is a separate change.

Once it lands, `channel_intensity` rows can carry exposure and the analysis can normalize per-ms
instead of guessing — turning the 9.3x within-embryo swing into the 4.7x measured above, and making
the residual (stage/focus/z) the next thing to characterize rather than an unknown mixed in with an
instrument setting.

## Note for the frame_inventory step: exposure is NOT a per-time value

`materialize_well_yx1.py` builds a `time_lookup` keyed on `time_index` alone, valid because
`elapsed_time_s` / `acquisition_time_s` are constant across z and channel within a timepoint.

**Exposure is not.** It differs BY CHANNEL — 11 ms for BF and 600 ms for tdtomato in the same file.
Reusing the `time_index`-only lookup shape would assign whichever channel happened to be first to
every row, which for these files means stamping BF's 11 ms onto the RFP rows: an exposure column
that is present, plausible, and wrong by 55x on exactly the channel it exists to describe.

The lookup must be keyed `(time_index, channel_id)`.
