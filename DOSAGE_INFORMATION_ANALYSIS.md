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
