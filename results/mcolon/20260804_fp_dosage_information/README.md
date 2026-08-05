# FP dosage: is the same information there after normalization?

**Question.** The transgene is a **pan-nuclear** marker, so RFP intensity is a proxy for **cell
abundance**. A 1-copy and a 2-copy embryo therefore measure the *same* underlying quantity at
different scale. Can they be analysed together? If dosage is a pure multiplicative factor,
normalizing recovers identical information and the answer is yes.

**Answer: no.** The information does not survive normalization, and the reason generalises beyond
this dataset.

---

## The result

Shannon entropy of the background-corrected RFP intensity distribution, spread across the three
brightness classes:

| normalization | spread across classes (bits) |
|---|---|
| raw DN | **4.462** |
| divided by mean (pure scale removal) | **4.462** |
| z-scored (location + scale removed) | **4.462** |
| log1p + z (multiplicative model) | 4.531 |

**Four normalizations, the same gap.**

### Why — the mechanism

Every normalization worth having is **affine**. The information loss here is **digitization**, which
is not. Counting the distinct intensity levels each embryo actually occupies:

| class | occupied levels |
|---|---|
| 0 (dim) | 16 – 31 |
| 1 | 17 – 293 |
| 2 (bright) | 449 – 1004 |

Pixel counts are comparable — B01 t1 has **more** pixels (104,402) than B02 t0 (47,259) — but ~30x
fewer levels to spread them over. Rescaling moves levels; it cannot create them. An embryo digitized
into 16 levels does not become a 1000-level measurement by dividing by its mean.

---

## Class 0 vs class 1: how uncomparable, exactly?

The three-class spread is dominated by class 2 (30x brighter than anything else). The pair that
matters in practice:

| | class 0 | class 1 |
|---|---|---|
| mean signal | 153.9 DN | 907.8 DN |
| SNR (mean / background sigma) | **1.06** | **7.09** |
| dynamic range | 2.18 sigma | 22.88 sigma |
| entropy | 2.96 bits | 5.46 bits |
| CV within embryo | 0.65 | 0.72 |

- entropy gap: **2.50 bits**
- within-class SD: 0.55 (class 0), 1.00 (class 1) — so the gap is ~2.5x the within-class spread.
  Separated, but **not cleanly**: the classes overlap in information content more than their mean
  brightness suggests.
- **SNR ratio 6.71x** → matching class 1's relative precision on class 0 needs **45x more pixels**,
  or ~6.7x longer exposure.

**Class 0 sits at SNR ~1.06**: its signal is about equal to the background fluctuation it was
corrected against. The correction is as large as the answer.

---

## What would fix it

**Acquisition, not analysis.** Longer exposure or higher gain for the dim population so it is
digitized across a comparable number of levels. This is a plate-design decision and it is the
actionable finding here.

---

## Caveats, stated plainly

- **4 embryos.** With 3 classes the assignment is nearly forced; the clustering has almost no power.
- **The classes are DISCOVERED from intensity, not known genotypes.** The class amplitude ratio is
  30–60x, far wider than copy number alone explains (a 0/1/2 series should read ~0:1:2), so these
  are probably not copy-number classes. Confirming that needs the genotype table.
- **The information argument does not depend on that.** It says any population spread across this
  brightness range carries unequal information, whatever produced the spread.
- A separate confound was found and fixed upstream: fluorescence exposure was **600 ms at t33hpf vs
  300 ms at t52/t77hpf**, a 2x artifact the same size as a dosage effect. Exposure is now carried
  per-frame from the ND2 through to the intensity rows, and the analysis normalizes per-ms.

---

## Contents

```
scripts/analyze_information_content.py   THE ANSWER: entropy under 4 normalizations
scripts/analyze_dosage_per_timepoint.py  cluster WITHIN each timepoint, test class stability
scripts/analyze_dosage.py                first pass: background correction + exposure confound
scripts/run_intensity_smoke.py           drives channel_intensity outside Snakemake
output/information_content.csv           per embryo-time entropy / SNR / dynamic range
output/dosage_per_timepoint.csv          per-timepoint class assignments
DOSAGE_INFORMATION_ANALYSIS.md           working notes, decisions recorded as they were made
```

Data source: `.pbx_smoke/out/.../channel_intensity/` for
`20260624_2x_td_bf_pbx_coll_plate01`, product `RFP__projection__max` (native uint16).
BF contributes only the segmentation mask — detection is BF-only — never the intensities.

---

# CORRECTION (same day): the class labels were wrong, and the ladder is not copy number

**Class 0 is the non-transgenic control**, not a dim measurement. No transgene, no fluorescence,
SNR ~1.06 because there is nothing there. Its low information content is the negative control
behaving correctly — not evidence about dosage. The earlier "class 0 vs class 1" section below
answers a question about a control, which is not the question worth asking.

**The comparison that matters is class 1 (het, 1 copy) vs class 2 (homo, 2 copies).**

| | class 1 (het) | class 2 (homo) |
|---|---|---|
| mean signal | 907.8 DN | 6776.1 DN |
| SNR | 7.09 | 16.94 |
| dynamic range | 22.88 sigma | 38.13 sigma |
| entropy | 5.46 bits | 7.42 bits |
| CV within embryo | 0.72 | 0.51 |

- entropy gap **1.96 bits**, pooled within-class SD 0.72 → **effect size 2.74**
- SNR ratio **2.39x**

## But these are almost certainly NOT het and homo

**Brightness ratio homo/het = 7.46x. Copy number predicts 2.0x.**

Off by nearly 4x. A 2-copy embryo cannot be 7x brighter than a 1-copy one from dosage alone.

The per-timepoint ladder (exposure-normalized DN/ms) shows why:

```
t0:  B02=16.57   A02= 2.91   A01= 1.84   B01=0.59     ratios 5.69x, 1.59x, 3.09x
t1:  B02=12.43   A01= 1.58   A02= 0.50   B01=0.13     ratios 7.89x, 3.18x, 3.89x
t2:  B02=22.19   A01= 1.03   A02= 0.52   B01=0.24     ratios 21.55x, 2.00x, 2.16x
```

There is **one bright outlier (B02, 6–22x above everything) and three dim embryos** clustered within
~3x of each other, whose ordering even swaps between timepoints (A02 above A01 at t0, below at
t1/t2). That is not a 0/1/2 ladder. The k-means was forced to emit three classes and split the dim
group arbitrarily.

**So the class labels are an artifact of asking for k=3 on 4 embryos**, and no conclusion about
het-vs-homo comparability can be drawn from this data. What the data does support:

1. the measurement path works end to end on real RFP pixels with exposure carried
2. a population spread across a 30x brightness range carries unequal information, and no affine
   normalization recovers it — that argument is independent of what produced the spread
3. B02 is genuinely different from the rest by a wide margin, stably, at every timepoint

Answering the het-vs-homo question needs the **genotype table** (so classes are known rather than
discovered) and the **full plate** (so there are enough embryos per class to see a 2x step against
the within-class spread, which here is comparable to the step itself).

---

# MATCHED-PAIR RESULT: A09 vs G09 — 4-7x within one genotype

The user identified A09 and G09 as embryos lying in the same orientation. They are the best-
controlled comparison available:

| | area ratio | aspect | MEAN ratio | INTEGRATED ratio |
|---|---|---|---|---|
| t0 | 1.12x | 2.49 vs 2.62 | **6.79x** | 6.08x |
| t1 | 1.14x | 4.23 vs 3.49 | **6.19x** | 5.43x |
| t2 | 1.08x | 3.41 vs 3.03 | **4.30x** | 4.66x |

Both elongated laterals, within 8-14% on area at every timepoint.

## Every designed variable is held constant

From the plate sheet, A09 and G09 are the **same genotype**
(`pbx4_pbx1b_crispant_tdtomato`) and the **same treatment** (PTU). Same clutch, same well
conditions, matched size, matched orientation.

So the 4-7x difference has no experimental explanation. What remains is the transgene itself:
expression level, mosaicism, or different insertion sites — **not copy number**, which could only
produce 2x.

## Two things this settles

**1. With area matched, mean and integrated AGREE.** 6.79 vs 6.08, 6.19 vs 5.43, 4.30 vs 4.66.
Their disagreement in the unmatched comparison was area doing the work. The earlier claim that
"integrated is the worse statistic" was really "area varies" — a weaker and different claim. When
geometry is controlled the two statistics carry the same information, which is what they should do.

**2. Brightness does not read copy number in this line.** A 4-7x spread between two embryos that
are identical in every controlled respect is far too large for a 1-vs-2 copy step, and it is
*within* a genotype group rather than between groups.

## Correction to the earlier B02/G09 numbers

An earlier pass used B02 and G09 and reported ~2x (1.80/2.27/1.89), reading it as possible dosage.
That pair was **not** matched: B02 is compact (aspect 1.01-1.33) against G09's elongated
(2.62-3.49), and `corr(mean, aspect)` is +0.35 population-wide. The ~2x was contaminated by
presentation. **A09/G09 supersedes it.**

---

# THE ANSWER, on the matched pair: YES, normalization preserves the information

A09 vs G09 is the right test for this question — same genotype, same treatment, matched size and
orientation, differing only in brightness (4-7x).

| t | mean ratio | occupied levels | raw entropy gap | **normalized entropy gap** |
|---|---|---|---|---|
| 0 | 6.79x | 304 vs 1381 | +2.69 bits | **+0.45 bits** |
| 1 | 6.19x | 209 vs 1053 | +2.56 bits | **+0.83 bits** |
| 2 | 4.30x | 343 vs 1562 | +2.29 bits | **+0.03 bits** |

**~85-99% of the apparent information gap disappears once the two are put on a common scale.** The
raw gap was brightness spreading across more fixed-width bins, not the dim embryo knowing less.

## Why it works here, and where it stops working

Normalization rescales the intensity levels an embryo occupies; it cannot create new ones. So the
question is whether the dim embryo has *enough* levels left to describe its distribution:

| distinct levels | max representable entropy | |
|---|---|---|
| 16 | 4.00 bits | the earlier "class 0" embryos |
| 32 | 5.00 bits | |
| 209 | 7.71 bits | **A09** |
| 1053 | 10.04 bits | **G09** |

A09 sits at 209-343 levels and carries 6.42-7.26 bits — **not clipped**, with headroom. That is why
normalizing recovers it.

The failure threshold is not "6x dimmer". It is **when the digitizer runs out of levels**. The
class-0 embryos at 16-31 levels have a 4.0-4.95 bit ceiling, so no rescaling can make them
comparable to a 10-bit embryo — but those were non-transgenic controls with no signal, which is a
different situation from a genuinely dim carrier.

## This corrects the earlier three-class conclusion

The earlier "4.46 bits, unchanged under four normalizations" compared classes spanning
non-transgenic controls to the brightest embryo, i.e. an embryo with ~nothing to measure against one
with full dynamic range. That comparison is dominated by the control and answers a question about
the noise floor, not about whether two real measurements can be pooled.

**Practical answer:** two carriers differing ~6x in brightness DO carry the same information after
normalization, and can be analysed together. The caution is a floor check, not a ratio check —
verify an embryo occupies enough distinct levels (say >100) rather than that it is within some
brightness factor of the others.

---

# QC GATE — and the distinction that matters more than the gate

## Normalization is licensed for PATTERN, not for DOSAGE

Per-embryo normalization removes the absolute scale. For dosage, that scale **is** the signal — a
6x brightness difference is the measurement, not a nuisance. So:

```
pattern / texture / morphology :  normalize AFTER QC
dosage / absolute expression   :  keep native intensities + exposure, do NOT normalize
```

The earlier conclusion ("two carriers 6x apart carry the same information and can be analysed
together") is correct **for pattern work only**. Stated without that qualifier it was wrong, because
it would license erasing the very quantity a dosage analysis is trying to read.

## A brightness ratio is the wrong gate

A09 is 4-7x dimmer than G09 and carries essentially the same information after normalization
(entropy gap +0.03 to +0.83 bits). Exposure, copy number, expression level and optics all move
absolute brightness while leaving spatial signal intact.

## But so is a raw level count

An earlier draft proposed ">100 distinct occupied levels" as the whole gate. Measured on this data,
**7 embryo-times pass that rule and fail on separation**:

| well | t | area | occupied | **effective (2^H)** | separation |
|---|---|---|---|---|---|
| D06 | 2 | 20k | 144 | **42** | 3.0σ |
| E11 | 2 | 85k | 473 | **251** | 3.4σ |
| H05 | 1 | 4473k | 102 | **16** | 4.2σ |
| E01 | 1 | 4563k | 265 | **28** | 4.8σ |

H05 over-reports by 6x — 102 occupied levels, 16 effective. It is a whole-well segmentation blob
whose "levels" are read noise. That is the numerical-taxidermy case, and a raw count waves it
through.

Note one prediction that did NOT hold here: `corr(occupied, area) = -0.27`, i.e. area is not
inflating the count on this plate, because the largest masks are mostly-uniform background. Area
inflation is a real hazard in general but is not the operative one in this dataset.

## The gate (`scripts/intensity_qc_gate.py`)

Three checks, because they fail independently:

```
effective_states = 2**entropy   >= 64      how many states are MEANINGFULLY used
separation       = (mean - bg_mode)/bg_sigma >= 5   distinguishable from its own well
saturated_frac                  <= 1%      bright tail intact
```

`2**H` rather than unique-value count: a level held by one noisy pixel contributes almost nothing
to entropy, so the effective count discounts exactly what inflates the raw one.

Result on 165 embryo-times: resolution 125, separation 141, unsaturated 164, **all three: 122**.

Thresholds are starting points to calibrate against the full plate, not established cutoffs.

---

# AT 52 WELLS: the carriers are ONE CONTINUOUS MODE, not two

With ~2x the data, the picture resolves — and it overturns the earlier "~2x split".

## The split my method reported was an artifact

Largest-log-gap splitting still returns a number (2.49x at t1, 2.62x at t2, QC-gated), but looking
at the sorted values shows why that is meaningless:

```
t2:  12.6 12.6 14.2 14.2 14.8 14.8 15.6 15.6 18.5 19.5 21.3 21.3 22.1 24.3
     26.5 28.7 28.7 31.7 33.1 36.7 37.7 43.4 44.3 60.0 60.0 65.8 65.8
```

That is a continuum. A gap-finder must return *some* boundary, so it cuts an arbitrary point and
reports a ratio.

## Tested against a unimodal null

Comparing the observed largest log-gap to 2000 log-normal samples of the same size and spread:

| t | n | largest gap | null median | null p95 | p | verdict |
|---|---|---|---|---|---|---|
| 1 | 27 | 0.582 | 0.446 | 0.943 | **0.281** | one continuous mode |
| 2 | 27 | 0.439 | 0.529 | 1.096 | **0.672** | one continuous mode |

No evidence of bimodality. The carrier distribution is **4.4-5.2x wide** — consistent with the
A09/G09 matched pair (4-7x) and far too wide for a 2x copy-number step.

## Converging lines of evidence

1. **A09 vs G09** — same genotype, same treatment, matched size and orientation: 4-7x apart
2. **Population shape** — 27 QC-passing carriers per timepoint form one continuous log-normal-ish
   distribution spanning 4-5x, with no significant gap
3. **The earlier 2x** — came from a 4-embryo forced k=3, and from an unmatched pair (B02 compact vs
   G09 elongated)

**Brightness in this line reflects continuous expression-level variation, not discrete copy
number.** Any 3-class dosage assignment from intensity would be imposing structure that the data
does not contain.

## Two open items

- **Carrier fraction is 91-94%**, up from 77-85% at 26 wells, against ~75% for a het incross.
  Either the ab floor is too permissive or this is not a simple het incross. Worth resolving.
- **The ~3.0 DN/ms dim group persists** (6 embryos at t2, was 3 at 26 wells) and sits well below
  the continuum. It is a distinct population, not the low tail — still unexplained.

---

# DOES NORMALIZATION PRESERVE INFORMATION? Answered against a matched-brightness null

The right control, as the user framed it: compare normalization's effect against a null of
**similarly fluorescent, unnormalized** embryos. Their entropy difference is the floor — the
biological + measurement scatter you get even when brightness is already matched, so it is not
attributable to scale.

QC-passing embryos only, within a single timepoint.

| | t1 | t2 |
|---|---|---|
| **NULL** — pairs <1.2x apart in brightness, raw | **0.150 bits** (382 pairs) | **0.195 bits** (205 pairs) |
| pairs >2x apart, raw | 1.475 bits (631 pairs) | 1.538 bits (468 pairs) |
| pairs >2x apart, **normalized** | **0.290 bits** | **0.149 bits** |
| normalized vs the null floor | 1.94x | **0.77x** |

**At t2, normalized brightness-mismatched pairs are indistinguishable from already-matched pairs** —
0.149 vs a 0.195 floor. Normalization fully closed a 1.538-bit gap. At t1 it closes ~5x of a
1.475-bit gap but leaves a residual at ~2x the floor.

## Within the normal range, raw entropy is almost purely a brightness readout

Restricting to the IQR of the carrier distribution — no controls, no dim outliers, only
1.76-2.16x of brightness spread:

| | t1 | t2 |
|---|---|---|
| corr(brightness, H_raw) | **+0.923** | **+0.817** |
| corr(brightness, H_norm) | **-0.187** | **-0.044** |

Raw entropy tracks brightness at r>0.8 even across a <2x range. After normalization the dependence
is gone. The raw-entropy differences reported earlier in this document were therefore measuring the
intensity scale, not the amount of biological structure.

## What this establishes

For **pattern/texture/morphology** questions, embryos spanning the carrier brightness range can be
normalized and pooled: the residual information difference is at or below what two
already-matched embryos show anyway.

This does **not** license pooling for dosage — normalization removes the absolute scale, which is
where a copy-number signal would live. See the pattern/dosage split above.

---

# THE FORMAL ARGUMENT (`scripts/plot_entropy_scale_proof.py` → `output/entropy_scale_proof.png`)

## The claim, stated so it can be falsified

Let X be an embryo's pixel-intensity distribution and s a scale factor. With **fixed** bin width w
(here 32 DN), a value v lands in bin ⌊v/w⌋, so scaling by s widens the occupied index range by s.
A scale change is a change of measure with constant Jacobian — it adds log₂(s) to entropy and
nothing to shape:

```
    H_fixed(sX) = H_fixed(X) + log₂(s)          (1)
    H_norm (sX) = H_norm (X)                    (2)
```

where H_norm bins on a grid rescaled to each sample's own percentile span.

Together these give a sharp prediction for **any** two embryos i, j:

```
    H_raw(j) − H_raw(i)   ==  log₂( mean_j / mean_i )   if the shapes are the same
    H_norm(j) − H_norm(i) ==  0                          if the shapes are the same
```

**Any departure is real distributional difference.** That is the argument: it converts "is the
information the same?" into a residual against a known law, so the null has teeth.

## A. The identity, verified on synthetic data

One log-normal distribution, rescaled 1x → 8x:

| | max deviation |
|---|---|
| fixed-grid ΔH vs log₂(s) | **0.0057 bits** |
| normalized ΔH across 8x | **0.0000 bits** |

## B. Two real embryos, 5.2x apart

G06 (12.6 DN/ms) vs F05 (65.8 DN/ms):

| | value |
|---|---|
| brightness ratio | 5.20x |
| predicted raw ΔH = log₂(5.20) | **+2.38 bits** |
| observed raw ΔH | **+2.01 bits** |
| observed normalized ΔH | **+0.28 bits** |

Panel B2 is the visual core: divided by their own means, the two histograms nearly superimpose. The
raw 2.01-bit gap was the scale term; 0.28 bits of genuine shape difference survives.

## C. Every pair, against the prediction

| t | pairs | corr(ΔH_raw, log₂ ratio) | slope (theory 1.000) | residual |
|---|---|---|---|---|
| 1 | 1891 | **+0.986** | 0.968 | 0.178 bits |
| 2 | 1081 | **+0.973** | 0.900 | 0.296 bits |

Raw ΔH follows the log₂ law along the diagonal; normalized ΔH collapses onto zero.

**Nearly all raw entropy variation between embryos is the scale term the theory predicts.** The
slopes sit slightly below 1.0 (0.90–0.97), which is the expected finite-bin departure — the identity
is exact only in the fine-bin limit, and the dimmest embryos occupy few enough bins to feel it.

## Why this is stronger than the earlier comparisons

Earlier passes reported entropy gaps and asked whether they "looked small". This makes a
quantitative prediction from first principles and measures the residual against it. The residual —
0.18–0.30 bits — is the honest estimate of real biological difference between embryos, and it is
comparable to the 0.15–0.20 bit floor measured between already-matched-brightness pairs.

## Confirmed at 3x the sample

Re-run at 83 wells / 289 embryo-times (from 165), 10,069 pairs (from ~3,000):

| t | embryos | pairs | r | slope (theory 1.000) | residual | median &#124;ΔH_norm&#124; |
|---|---|---|---|---|---|---|
| 1 | 113 | 6328 | **+0.985** | **1.041** | 0.276 bits | 0.309 |
| 2 | 87 | 3741 | **+0.979** | 0.942 | 0.364 bits | 0.238 |

The t1 slope moved from 0.968 → **1.041**, i.e. *toward* theory and slightly past it, which is how a
real law behaves as it is measured more precisely — not how a fitted artifact behaves. r stays
above 0.979 with 3x the pairs.

QC at this sample: 289 embryo-times, resolution 226, separation 252, unsaturated 284, **all three
219**.

---

# FULL PLATE (96 wells, 334 embryo-times) — final

Run 23443692 completed 374/374 steps. 91 wells with rows, 4 with no measurable embryo, 1 shard
absent.

## The law holds decisively

| t | embryos | pairs | r | slope (theory 1.000) | residual |
|---|---|---|---|---|---|
| 1 | 135 | **9,045** | **+0.9838** | 1.039 | 0.282 bits |
| 2 | 108 | **5,778** | **+0.9740** | 0.978 | 0.403 bits |

14,823 pairs. Slopes 0.978 and 1.039 straddle the theoretical 1.000. The prediction
`ΔH_raw = log₂(brightness ratio)` is confirmed.

## But the null test got WORSE at scale — and that is the useful finding

| normalized far-pairs vs the matched-brightness floor | 26 wells | full plate |
|---|---|---|
| t1 | 1.94x | **3.44x** |
| t2 | 0.77x | **1.68x** |

The earlier "t2 fully closes the gap" was a small-sample result and does not survive. **Reported as
a correction, not buried.**

### Why: the full plate reaches the resolution limit

Brightness now spans 54x (t1) and 381x (t2), pulling in embryos that pass the gate but sit near the
digitization floor. Splitting far-pairs by how well-resolved the *dimmer* member is:

| dimmer member | t1 &#124;ΔH_norm&#124; | t2 &#124;ΔH_norm&#124; |
|---|---|---|
| well resolved | **0.287** | **0.288** bits |
| poorly resolved | 0.874 | 0.612 bits |

**When the dim embryo is well resolved, normalization works — 0.287 bits against a 0.17-0.23 floor.
When it is not, it does not.** This is the "normalization cannot create levels" argument confirmed
directly on data rather than asserted.

## Threshold recalibrated: 64 → 128 effective states

64 was a guess (2**6). Swept against the null floor:

| min_eff | t1 kept | t1 norm/floor | t2 kept | t2 norm/floor |
|---|---|---|---|---|
| 64 | 135 | 3.44 | 108 | 1.68 |
| **128** | **116** | **1.80** | **99** | **1.25** |
| 192 | 91 | 2.28 | 91 | 1.29 |
| 256 | 66 | 1.75 | 74 | 1.55 |

128 halves the t1 ratio while keeping 116/135. Above that the ratio moves non-monotonically, which
is sampling noise in the floor estimate rather than structure — so 128, not more.

Final gate on 334 embryo-times: resolution 248, separation 296, unsaturated 329, **all three 242**.

## The answer, final form

**Yes — normalization preserves the information, PROVIDED the dim embryo is adequately digitized.**

- adequately resolved (≥128 effective states): residual 0.287 bits vs a 0.17-0.23 bit floor
- poorly resolved: residual 0.61-0.87 bits, and no rescaling recovers it

So the operative constraint is not a brightness ratio but a **resolution floor**. A 5x-dimmer embryo
with 300 effective states is fine; a 2x-dimmer one with 40 is not.

For **pattern/morphology** work: pool freely above the gate.
For **dosage**: do not normalize at all — the scale is the signal.
