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
