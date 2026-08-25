# TODO / Spec: Phenotype discreteness — within-group modality along a morphology axis

**Status:** idea captured, not started.
**Origin:** `results/mcolon/20260617_morph_axis_investigation/morph_axis_scatter.py`

> NOTE: The `morph_axis_investigation` scatter work to date is on **completely
> simulated / fake data** — a sandbox to understand the method. Nothing here is a
> biological claim yet.

## The question (first principles)

This is a **within-group** question, not a between-class one:

> Conditional on a group of embryos defined by its biological label, is *that
> group's own* distribution along a morphology axis **continuous** (one lump) or
> **discrete** (bimodal / splits)?

Things this is explicitly **not**:
- Not clustering / unsupervised discovery — the grouping is **given** by biology.
  We start from the biological labels.
- Not separability between two phenotype classes (that's what the current
  `morph_axis_scatter.py` measures — set it aside for this question).
- Not distance from wildtype. Wildtype is not a contrast group we subtract.

## Role of wildtype: the continuum calibration

Wildtype is the reference for **what "continuous" looks like**. It calibrates the
shape of a unimodal / continuum null. Run the modality test on wildtype siblings
→ it should read **continuous**. That is the proof the statistic correctly
recognizes a continuum. (WT siblings within each gene's experiments are the right
pool; two independent WT pools also bound the natural run-to-run spread of the
statistic.)

## Method

1. **Axis (label-free).** Project each embryo onto a single scalar morphology
   axis defined *without* the phenotype labels — WT growth axis, or a morphospace
   PC of `total_length_um` × `mean_curvature_per_um`. Label-free matters: the
   modality we find must be a property of the group, not something a supervised
   two-class axis imposed.

2. **Per-group modality statistic.** For each (group × 4-hpf time-bin), take that
   group's projected values and run a **single-sample modality test**:
   Hartigan's dip (p-value), or 1-vs-2-component GMM (BIC gap / likelihood
   ratio). One number: one lump or two.

3. **Read the groups.**
   - wildtype → expect continuous (calibration passes).
   - each mutant phenotype group → recovers the known biology from distribution
     shape alone (e.g. b9d2 continuous, cep290 discrete — whichever the biology
     is). We know they differ going in; the deliverable is a **method that
     recovers that difference.**

4. **Validate on simulated data first** — dial a controlled distribution from
   fully-continuous to clearly-bimodal and confirm the statistic tracks it, THEN
   apply to real references.

## Extension (after the method works)

Once the within-group modality test is trustworthy on the hand-defined 2-D
morphology axis:

- **Generalize the axis to the embedding.** Replace the hand morphometrics with
  the learned space — a PC of `z_mu_b` ("MPC space"), or eventually the modality
  test run directly in the full embedding — so the morphology axis is *discovered
  from the embedding* rather than hand-picked.
- Gate on the distribution check: only trust an embedding-derived axis once it
  passes the same continuous-null calibration on wildtype that the hand axis did.
- The morphology assets we ultimately define are then grounded in whatever axis
  passes that check.

## Connectedness, not mode-counting; density prior from wildtype

Reframe the test from "modality" (which bakes in a peak count) to
**connectedness**: is the group's support one connected, gap-free region along
the axis, or does it fracture? Continuous = one connected piece; discrete =
broken support. This handles n≥2 uniformly (a group that fractures into 2 or 5
both read "discrete"; counting how many is the discovery/clustering layer, kept
separate).

A "gap" is only meaningful against a **density prior** — how densely the axis
would be filled if the phenotype were continuous. Without it, sparse sampling
manufactures fake gaps. Wildtype supplies the prior: it is the known continuum,
so its density-along-the-axis is the shape of "a filled continuum." The prior
must be **stage-matched** — WT spread/position move with development — so the
object is `p(axis position | wildtype, time-bin)`. A mutant gap is real only if
it's a density deficit relative to the stage-matched WT profile.

Mechanically this points at density-aware connectedness (super-level sets /
cluster-tree, or DBSCAN with neighborhood scale set from WT) rather than a dip
test.

## Distinguishing variance from modes (the core problem)

Variance and modes are genuinely confusable and this is the hard part. A
high-variance mutant can be a **stretched continuum** (connected, just wider) —
that is NOT discreteness. Discreteness is **broken support** (can't cross the
middle without hitting a should-be-filled gap). Variance is width; modality is
gaps. Three bounds keep them separate:

1. **Normalize each distribution to a SHAPE (kills variance as a confound).**
   Rescale every group — wildtype and each mutant — to **unit mass and unit
   scale** per time-bin, so you compare density *profiles*, not absolute counts
   or widths. Two distinct normalizations, don't conflate them:
   - **Global mass-and-scale normalization (required, non-circular):** account
     for the phenotype's own total spread/count. A wide-variance continuum is
     uniformly *thinner*; without this it looks patchy just for being spread
     thin. Normalizing the mutant to itself fixes the "wide variance but still a
     continuum" case. This is normalizing for phenotype density itself — and it
     is legitimate.
   - **Local-shape normalization (forbidden, circular):** dividing by the density
     *valley itself* erases the very gap you're testing for. Never let a group's
     own valley set its own tolerance.

   **Wildtype's normalized shape defines the tolerated patchiness.** If accrued
   WT is itself patchy at some relative level, that patchiness *is* the continuum
   standard — a mutant is allowed to be at least that patchy. The test quantity
   is **scale-and-mass-invariant relative valley depth**: mutant interior dip vs.
   its own flanks, compared to how deep an interior dip WT tolerates relative to
   *its* flanks. Non-circular because the tolerance (bar) comes from WT's shape
   while the thing tested is the mutant's shape — each normalized to itself, but
   the mutant's valley never sets its own bar.

2. **Gap-depth + gap-width thresholds, calibrated on stage-matched WT.** How deep
   (valley density vs flanks) and how wide (fraction of axis spanned) must an
   interior trough be before it's a mode and not thin sampling. Both calibrated
   on WT so they aren't pulled from thin air.

3. **Resampling null at matched n.** Resample the stage-matched WT continuum at
   the mutant's n and ask how often *that* fakes a gap this deep by chance. If a
   WT-sized draw reproduces it 30% of the time, there's nothing. This guards
   against underpowering masquerading as a real gap — and it's the sample-size
   control we agreed on.

## The real output: the variance→mode transition over time

None of this is static. A phenotype can start as a pure variance increase (one
connected, widening blob) and only later fracture into modes as fates separate.
So the deliverable isn't "is this group discrete" — it's **when along
developmental time does connected-but-wide become broken.** That transition point
is the signal (it dates the fate decision), and it's far more informative than a
single discrete/continuous label.

## Testing / tuning plan

- **Anchors for tuning thresholds:** cep290 and b9d2 homozygous groups are the
  labeled known cases (one continuum-like, one discrete — from biology). Tune the
  gap-depth / gap-width / null thresholds on these two until the method recovers
  the known answer, then treat those thresholds as fixed.
- **2-D is fine for testing.** The length × curvature space already used for the
  cep290 / b9d2 scatters is an acceptable test bed; connectedness generalizes to
  2-D support directly (WT prior becomes a 2-D density).
- **Simulated data first** (controllable continuous↔bimodal), then the real
  anchors.

## Caveat: a 1-D axis can hide a mode

Two genuinely separated blobs can collapse into one lump if they differ along a
direction the axis discarded. So any "continuous" call on the hand axis is
provisional until the embedding-space version (stage 2) agrees — a second reason
for the extension beyond generality.

## Deliverable

Per (group × time-bin): the group's projected-value distribution (histogram /
violin) with its connectedness statistic annotated, wildtype shown as the
stage-matched continuous-null. Per group, the connectedness trajectory over
developmental time, marking the variance→mode transition where it occurs.



# Theroretica addednum to be refined: 
I actually think you’ve hit on a much cleaner decomposition than “continuous vs. discrete.” What you’re really building is a decision tree for phenotype geometry. The important shift is that each statistic has one job, rather than trying to summarize everything with a single number.

I’d organize it like this.

For each phenotype axis:

1. Is this axis phenotypically different from wildtype?

        ↓ No
    Stop.
    "Wildtype-like."

        ↓ Yes

2. Is the mutant still a single connected continuum
   relative to wildtype?

        ↓ Yes

3. Describe the continuum.

        ↓ No

4. Describe the discrete structure.

Notice that “describe the continuum” and “describe the discrete structure” are different scientific questions. They shouldn’t use the same statistics.

⸻

Stage 1: Is the axis different from wildtype?

This is essentially a distribution shift problem.

Possible statistics:
	•	KL divergence
	•	Jensen-Shannon divergence (often nicer because it’s symmetric and bounded)
	•	Wasserstein distance
	•	Earth Mover’s Distance

Purpose:

Is this phenotype axis biologically changing at all?

If not, don’t spend effort describing its geometry.

This aligns beautifully with your earlier observation that classifier directions outperform the full embedding. Most embedding directions simply won’t differ from wildtype and can be ignored.

⸻

Stage 2: Is it connected?

This is no longer a density question.

It is a support geometry question.

You’re asking:

Does the observed phenotype occupy one connected region, or is the support broken?

Candidate statistics:
	•	MST longest-edge distribution
	•	H₀ persistence (equivalent information, richer summary)
	•	Conductance
	•	Fiedler value

Each is probing connectivity from a different angle.

⸻

MST

Answers:

Are there unusually large unsupported jumps?

Good at finding:
	•	gaps
	•	bridges
	•	unsupported regions

Weakness:

Can mistake long sparse tails for gaps.

⸻

Conductance

Answers:

Is there a bottleneck?

Good at distinguishing

******
*****
****
***
**
*

from

*****
     *****

because the taper still has many local connections.

⸻

Fiedler value

Answers:

How globally connected is this graph?

It is essentially a smooth version of conductance.

Probably more mathematically stable.

⸻

One thing I like is that these methods should agree on clear examples.

If they disagree,

that’s informative.

Maybe the phenotype has
	•	one weak bridge,
	•	a severe tail,
	•	or too little data.

⸻

Stage 3: Describe a connected continuum

This is where I think your framework becomes genuinely novel.

Suppose we’ve already decided

this phenotype is one continuum.

Now we ask:

What kind of continuum?

Examples:

Compact

******
******
******

Low variance.

⸻

Broad

***************
************
********

Large variance.

⸻

Skewed

******
*****
***
**
*

Long tail.

⸻

Heavy-tailed

A few very severe embryos.

⸻

Heteroskedastic

Variance changes along the axis.

⸻

Multi-density

Dense core with diffuse outskirts.

⸻

Notice none of these are clustering questions.

They’re describing the geometry of one connected support.

This is where classical distribution descriptors become useful:
	•	variance
	•	skewness
	•	kurtosis
	•	quantiles
	•	entropy
	•	tail index
	•	robust spread (MAD, IQR)

⸻

Stage 4: If not connected

Only then do you ask

How many stable components exist?

That’s a different problem.

Possible methods:
	•	HDBSCAN
	•	persistence
	•	hierarchical clustering
	•	spectral clustering

Importantly,

this is not your first question.

It’s downstream.

⸻

Confidence should live outside the tree

I really think confidence should be orthogonal.

Every decision gets attached to

confidence = 0.91

or

Low confidence:
N = 5
Wildtype bootstrap overlap = large

Confidence shouldn’t determine whether something is continuous.

It determines how much we trust that conclusion.

⸻

Relative to wildtype

This is the part I think is the strongest idea.

Nothing should be absolute.

Instead

Mutant conductance
───────────────
Wildtype conductance distribution

or

Mutant MST gap
────────────
Bootstrap WT gaps

Everything becomes

relative support geometry.

That automatically adjusts for developmental stage, natural variability, and sampling density.

⸻

One refinement I’d make

There’s one place I’d tweak your plan slightly.

You said:

“If it’s continuous, then ask what the spread looks like.”

I’d broaden “spread” to shape.

Spread is only one aspect.

Instead think:

Connected continuum

↓

Shape descriptor

    spread
    skew
    tails
    density decay
    anisotropy

Because biologically,

cep290 isn’t just “high variance.”

It’s

a broad, approximately uniform continuum.

Another mutant might be

a narrow core with a rare severe tail.

Same variance.

Very different biology.

⸻

The philosophy I think is emerging

I actually think the unifying principle is this:

Every phenotype axis has two independent properties:
	1.	Where is the support? (support geometry)
	2.	How is probability distributed on that support? (density geometry)

That’s a distinction statisticians don’t always make explicitly, but it’s exactly what you’re circling around.

Support geometry asks questions like:
	•	Is there one connected region?
	•	Are there gaps?
	•	How many disconnected components?

Density geometry asks:
	•	Is the mass concentrated or diffuse?
	•	Is there a long tail?
	•	Is it symmetric?
	•	Does density decay smoothly?

Those are orthogonal concepts. Two distributions can have identical support but very different densities, or identical densities on completely different supports.

I think that’s the conceptual leap your framework is making. Before describing how embryos are distributed, first determine the shape of the space they actually occupy. Once you’ve established that the support is a single connected continuum, then describing the density on that continuum becomes meaningful. If the support is disconnected, you’ve entered a different biological regime where clustering and describing each component separately is the appropriate next step.

To me, that’s a very coherent organizing principle, and it’s considerably cleaner than trying to invent a single “continuity score” that has to do everything.

use KL to determine if different from wildtype
Yes, I agree with the idea of KL divergence, but I’d slightly modify the tool.

For 1D phenotype axes, I’d prefer:

Primary: Jensen-Shannon divergence or Wasserstein distance
Secondary: KL divergence if density estimation is stable

KL is fragile because it blows up when one distribution has near-zero density where the other has mass. That can be biologically meaningful, but with small N it can also be pure gremlin dust 🧪. Jensen-Shannon is safer because it is symmetric and bounded. Wasserstein is also nice because it measures “how far the mutant distribution moved” along the axis.

The clean version:

Use divergence/distance from WT to decide whether an axis is worth describing.
Use graph/connectivity descriptors to decide whether that changed axis is continuous or discrete.
Use shape descriptors to describe the continuum.
Use WT bootstrap to set all thresholds.
Use matched-N confidence to say how much to trust the call.

1. How to determine thresholds

Do not set thresholds absolutely.

The threshold should come from wildtype itself.

For each phenotype axis and developmental time window:

1. Sample wildtype embryos with matched N.
2. Compute the descriptor.
3. Repeat many times.
4. Get a null distribution.
5. Compare mutant descriptor to that null.

Example:

descriptor = largest MST edge z-score

WT bootstrap:
0.8, 1.1, 1.4, 1.0, 1.3, 0.9, ...

Mutant:
3.7

Conclusion:
mutant has an unusually large gap relative to WT.

So thresholds become empirical:

WT 95th percentile = suspicious
WT 99th percentile = strong

For “different from WT”:

JS divergence(mutant, WT) > WT-vs-WT bootstrap 95th percentile

For “discrete”:

MST gap statistic > WT matched-N 95th percentile
conductance < WT matched-N 5th percentile
Fiedler value < WT matched-N 5th percentile

That gives every threshold the same meaning:

This pattern is unlikely under normal wildtype variability at this N.

That is the whole spine of the method.

2. What to do with small sample size

Small N should not mean “impossible.”

It should mean:

we can compute descriptors, but confidence is limited.

For N=5, the MST has 4 edges. That is enough to notice a dramatic gap, but not enough to estimate a full density well.

So I’d split methods by sample-size tolerance:

Works better at low N:
- MST max edge
- MST edge ratio
- range / IQR
- Wasserstein distance
- rank-based tests

Needs more N:
- KL divergence
- KDE-based density
- skew/kurtosis
- multimodality tests
- HDBSCAN-style clustering

I’d make the output explicit:

axis: curvature
changed_from_WT: yes
continuity_call: continuous
confidence: moderate
reason: N=8, MST and conductance within WT range, WT bootstrap overlap low

Or:

axis: tail curvature
changed_from_WT: yes
continuity_call: possible discrete split
confidence: low
reason: N=5, one large MST edge, but matched-N WT null is wide

The key is to report confidence as uncertainty, not as a magic yes/no gate.

3. A practical confidence score

I’d use three ingredients:

confidence =
    sample size adequacy
  + bootstrap separation from WT
  + stability under resampling

Meaning:

Sample size adequacy:
Do we have enough embryos for this descriptor?

Bootstrap separation:
How extreme is mutant relative to matched-N WT?

Stability:
If we remove one embryo or resample embryos, does the call stay the same?

For example:

high confidence:
- N ≥ 12
- mutant descriptor beyond WT 99th percentile
- call stable under leave-one-out

moderate confidence:
- N 8–12
- beyond WT 95th percentile
- mostly stable

low confidence:
- N 5–7
- weakly beyond WT
- call changes when one embryo is removed

insufficient:
- N < 5 for distribution shape claims

This is especially important because at low N, one weird embryo can create a fake “gap.”

4. The refined decision tree

For each axis:

1. Test difference from WT
   Use JS divergence / Wasserstein / KL.
   Threshold from WT-vs-WT matched-N bootstrap.

2. If not different:
   label = wildtype-like
   stop.

3. If different:
   test support continuity.
   Use MST gap, conductance, Fiedler value.
   Threshold each against matched-N WT bootstrap.

4. If continuous:
   describe continuum shape:
   spread, skew, tail, quantiles, density decay.

5. If discrete:
   defer to component discovery:
   how many pieces, how stable, what each piece looks like.

6. Attach confidence:
   N, WT bootstrap percentile, leave-one-out stability.

The mantra is:

No raw thresholds.
No absolute discreteness.
Everything calibrated to matched wildtype.
Every call gets confidence.

That makes the framework much harder to fool.