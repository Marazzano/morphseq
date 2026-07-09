## Motivation

We have two biological samples: a reference population and a target population.

Our goal is to characterize how their phenotype distributions are organized.

The current framing of `discrete` versus `continuous` is too coarse. A target can be more modal, more concentrated, or more separated than reference even when its modes remain connected by intermediate phenotypes. That is not a failure case. It is a richer phenotype description.

Rather than asking whether the target is simply discrete or continuous, we ask:

> How is the target distribution organized relative to the reference distribution?

This benchmark should produce a quantitative profile of distributional organization, not a single classifier.

The benchmark treats phenotype distributions as density landscapes. Peaks are regions where phenotypes accumulate; valleys are lower-density regions that connect or separate peaks. A target distribution can differ from reference by changing peak number, peak concentration, peak shape, valley depth, bridge density, or robustness of these structures. These properties should be measured separately before any downstream interpretation such as `more discrete` or `more continuous`.

---

## Biological and Statistical Framing

Biologically, we are interested in common phenotypic trends or outcomes within a population, and in how those outcomes relate to one another.

Statistically, these trends correspond to modes of the distribution. Intuitively, we refer to these modes as peaks: regions where many similar observations accumulate.

Equally important is the relationship between peaks. Some peaks are linked by many intermediate phenotypes, while others are separated by relatively few. To describe this continuity, we consider the valleys between peaks. Statistically, valleys are regions of weaker support separating neighboring high-support regions. Some valleys may contain continuous intermediate observations, while others may be nearly empty. The depth of valleys and the density of bridges between peaks describe how strongly peaks are connected.

There are two fundamental objects in the distributional landscape:

- Peaks: regions of relatively high density where observations accumulate.
- Valleys: lower-density regions that connect or separate neighboring peaks.

These objects exist on different spatial scales. Peaks describe local accumulation of similar phenotypes. Valleys describe weaker support, bridges, and bottlenecks between neighboring peaks.

Together, peaks and valleys provide a simple language for describing phenotype organization:

- Peaks describe the major phenotypic trends or outcomes.
- Valleys describe how continuous or separated those trends are from one another.

Downstream interpretations such as `more continuous` or `more discrete` can be derived from peak-valley organization:

- More continuous: peaks are connected by substantial intermediate density.
- More discrete: peaks are separated by deeper valleys or weaker bridges.

---

## Descriptive Ontology

To characterize a phenotype distribution, we describe three complementary aspects.

### 1. Peak Properties

Each peak is characterized individually. These properties describe the phenotypic trends themselves.

Core questions:

- How many peaks are there?
- How much mass does each peak contain?
- How tall is each peak?
- How wide or compact is each peak?
- What shape is each peak?
- Are peaks internally smooth, skewed, elongated, diffuse, or heterogeneous?

Examples of peak properties:

- peak count
- peak mass
- peak height
- peak width
- peak compactness
- peak anisotropy
- internal density heterogeneity
- internal distance distribution
- peak persistence across scale

### 2. Valley Properties

Valleys describe the relationships between peaks. They quantify how strongly peaks are connected by intermediate phenotypes.

Core questions:

- How deep are the valleys between peaks?
- How much bridge density connects peaks?
- How far apart are peaks relative to their widths?
- Do peaks merge gradually or only at very low density?
- Are the modes weakly separated, strongly separated, or effectively disconnected?

Examples of valley properties:

- valley depth
- saddle or merge density
- bridge density
- bridge mass
- inter-peak distance
- inter-peak distance relative to peak width
- graph bottleneck strength
- component persistence across density or radius scale

### 3. Relative Organization of Peaks and Valleys

Beyond individual peaks and valleys, we characterize the overall organization of the distribution.

Core questions:

- Are peak masses balanced, or is one peak dominant?
- Are peak heights similar, or is there a hierarchy?
- Are valleys uniformly deep, or are some peaks more connected than others?
- Are peaks arranged as satellites, chains, branches, or balanced groups?
- Is the target more organized into peaks than the reference?

Examples of relative organization:

- peak mass balance
- peak height balance
- peak prominence distribution
- valley depth distribution
- bridge-density distribution
- inter-peak distance distribution
- peak-valley persistence across scale

A distribution is not summarized by the number of peaks alone. The phenotype is described by the joint organization of peak properties and valley properties.

---

## Support-Aware Mode Estimation

The benchmark is not primarily about KDE. It is about recovering modal organization. KDE is one possible estimator, and its bandwidth matters because it can determine whether nearby peaks remain distinct or become merged. But the conceptual target is broader:

```text
support geometry
  -> estimation regime selection
  -> observed modal organization
  -> robustness as evidence
  -> biological interpretation
```

This makes support-aware mode estimation an estimation procedure, not part of the ontology. The ontology defines what exists: peaks, valleys, and their organization. The estimation procedure describes how those objects can be recovered from finite observations.

### 1. Support Geometry

Before choosing an estimator, characterize the geometry of the observed support.

The purpose is to estimate the characteristic spatial scales of peaks and valleys:

- What is the characteristic local support width inside dense regions?
- Are there stable bottlenecks or bridges between neighboring dense regions?
- Are candidate valleys broad, narrow, or bridge-like?
- Is the support approximately isotropic, strongly anisotropic, or highly heterogeneous?
- Does the support suggest that one global smoothing scale is appropriate?

Possible diagnostics include:

- local neighbor distances
- local covariance or anisotropy
- MST edge-length structure
- candidate bridge edges
- graph bottlenecks
- bridge-to-local-width ratios

These diagnostics should be interpreted as evidence about support geometry, not as final mode calls.

Support geometry may also indicate that candidate valleys or bridges are below the resolution of the observed sample. In that case, the appropriate output is a resolution warning, not a forced mode call.

### 2. Estimation Regime Selection

Use the support geometry to choose an appropriate mode-estimation regime.

Examples:

- diffuse approximately isotropic support: global KDE may be sufficient
- dense peaks separated by short but stable valleys: a narrower smoothing scale may be needed to preserve valleys
- large differences in local density: adaptive estimation may be needed
- strong anisotropy: anisotropic or locally scaled estimation may be needed
- graph-like bridges or bottlenecks: graph-based or topology-aware estimators may be useful

The important point is that support geometry informs the estimation strategy. Bandwidth, graph scale, neighborhood size, or any other smoothing parameter should not be treated as an arbitrary preprocessing choice.

### 3. Robustness and Biological Interpretation

Biologists are not interested in every local maximum of a density estimate. They are interested in reproducible phenotypic trends.

Finite samples inevitably contain structures that arise from both biology and sampling variability. The goal of the benchmark is therefore not simply to detect peaks and valleys, but to determine which observed structures are sufficiently robust to support biological interpretation.

A point cloud with several dense islands could arise from:

- true biological modes
- uneven sampling of one continuous distribution
- density-weighted dropout in low-density regions
- underresolution of a smooth distribution
- outlier or satellite contamination

Therefore, peak and valley estimation should first describe observed structure, then ask which parts of that structure persist under reasonable perturbations of the data and estimator.

Operationally:

- A peak is an observed locally high-density region identified by the chosen peak-finding procedure.
- A valley is an observed low-density or low-support region between peaks.

These are observed structures, not automatic claims about hidden biology.

```text
observed peak    != guaranteed biological state
observed valley  != guaranteed true absence of support
observed islands != guaranteed discreteness
```

Robustness is the epistemic criterion that turns an observed structure into evidence. The estimator proposes a modal organization; robustness determines which parts of that proposal deserve to be treated as biological signal.

Support geometry asks what spatial scales are present. Robustness asks which of those scales are reproducible. A bottleneck may exist in the observed support, but it provides stronger biological evidence if it survives subsampling, reasonable changes in smoothing scale, or changes in estimation regime.

### 4. Robustness Assessment

After modal organization is estimated, evaluate whether inferred peaks and valleys remain robust.

A peak or valley is considered robust if it persists under reasonable perturbations of the observations and estimation procedure.

The objective is not to recover every possible peak. The objective is to recover the set of peaks and valleys whose existence is consistently supported by the available evidence.

Perturbations include:

- downsampling
- bootstrap resampling
- estimator family
- smoothing scale
- graph scale
- outlier removal

Robust structures may be supported by:

- recurrence under subsampling or downsampling
- persistence across smoothing or graph scales
- persistence across estimator families
- stability of bridge density
- stability of valley depth
- stability of peak membership
- stability of peak mass or mass fraction
- stability of peak concentration or compactness

Only after this final validation step should observed peaks be interpreted as candidate biological modes.

---

## Reference-vs-Target Comparison

After each distribution is described, we compare the target against the reference.

The goal is not only to determine whether the target differs from WT. The goal is to determine how its organization differs.

A target may be:

- more concentrated
- more strongly organized into peaks
- more compact within peaks
- more balanced across peaks
- more separated by deep valleys
- more connected by intermediate bridge density
- more anisotropic, skewed, or diffuse
- more sensitive to sampling or resolution

Example interpretation:

> The target has three mass-significant observed peaks. Compared with reference, these peaks are more compact and more height-balanced. The intervening valleys are deeper and bridge density is weaker, but support continuity remains.

This is more informative than saying only:

> target is discrete

or:

> target is continuous

---

## Generative Benchmark Logic

To test statistics for modal organization, the synthetic benchmark should generate distributions by independently controlling the properties we want to recover.

The generator should not rely on vague scenario names like `patchy`, `compact`, or `spiral`. Instead, each synthetic distribution should be specified by interpretable configuration fields.

Conceptually:

```text
distribution =
    peak_organization
  + peak_geometry
  + within_peak_density_profile
  + valley_structure
  + observation_process
```

These fields should be sufficient to loosely generate the intended distribution and to explain what each statistic is supposed to recover.

### 1. Peak Organization

Describes how many peaks exist and how their mass is arranged.

Examples:

```text
one_peak
two_peaks
three_peaks
one_large_plus_satellite
chain_of_peaks
branching_peaks
many_weak_peaks
```

Key properties:

- peak count
- peak mass balance
- peak spacing
- peak arrangement

### 2. Peak Geometry

Describes the shape of each peak.

Examples:

```text
round
anisotropic
elongated
ridge
arc
spiral
branch
core_tail
```

Key point:

> Shape does not define peak count.

Importantly, a spiral, ridge, arc, or tail can be one peak/mode if density does not rise again into another mass-significant peak.

### 3. Within-Peak Density Profile

Describes how probability mass is distributed inside each peak.

Examples:

```text
gaussian
uniform
radial_decay
ridge_decay
flat_core
core_tail_monotone
beaded
```

Important contrast:

```text
spiral + ridge_decay = one continuous spiral trend
spiral + beaded      = multiple peaks arranged along a spiral
```

Same geometry, different modal organization.

### 4. Valley Structure

Describes how peaks are connected or separated.

Examples:

```text
no_bridge
low_bridge
moderate_bridge
high_bridge
monotone_tail
satellite_bump
```

Interpretation:

- high bridge: peaks exist but are substantially connected
- low bridge: peaks are more strongly separated
- no bridge: support is effectively disconnected
- monotone tail: one peak extends without forming another peak
- satellite bump: a tail rises again into a weak secondary peak

High-bridge cases are important. They should be reported as:

```text
more modal / more peaked than reference
weakly or moderately separated
support connected
```

not as strict disconnected support.

### 5. Observation Process

Describes how finite sampling transforms the underlying distribution into the observed point cloud.

Examples:

```text
iid
underresolved
uneven_sampling
missing_segment
density_weighted_dropout
outlier_contaminated
```

This separates true distributional organization from sampling artifacts.

For example, observed islands may reflect true peaks, but they may also reflect underresolution or uneven sampling. The benchmark should make these cases explicit.

Concrete contrast:

```text
one_peak_spiral_ridge_decay + iid
one_peak_spiral_ridge_decay + missing_segment
one_peak_spiral_ridge_decay + underresolved
spiral_beaded + iid
```

The first three share the same underlying density but differ in observation process. The last has a different underlying density and true modal organization.

---

## Controlled Synthetic Contrasts

The synthetic suite should include controlled contrasts where one property changes while others are held fixed.

### Same Peak Count, Different Concentration

```text
one_peak_compact
one_peak_diffuse
```

Purpose:

Test whether concentration statistics respond to compactness without confusing it for peak count.

### Same Peak Count, Different Shape

```text
one_peak_round
one_peak_elongated
one_peak_arc
one_peak_spiral
```

Purpose:

Test whether shape, curvature, or anisotropy creates false modal structure.

### Same Peak Count, Different Valley Strength

```text
two_peaks_high_bridge
two_peaks_low_bridge
two_peaks_no_bridge
```

Purpose:

Test whether valley statistics distinguish weak, strong, and disconnected separation.

### Same Geometry, Different Modal Organization

```text
spiral_ridge_decay
spiral_beaded
```

Purpose:

Test whether the method distinguishes one continuous trend from multiple peaks arranged along the same geometry.

### Same True Distribution, Different Observation Process

```text
one_peak_well_sampled
one_peak_underresolved
one_peak_uneven_sampling
one_peak_missing_segment
```

Purpose:

Test whether apparent fragmentation is reported as a sampling or resolution issue rather than overinterpreted as biological discreteness.

---

## Metric Families

The benchmark should evaluate statistics according to the descriptive property they are meant to recover.

### Peak Metrics

Questions:

- How many peaks are present?
- How much mass does each peak contain?
- How tall, wide, compact, or anisotropic is each peak?
- How stable are peaks across scale or resampling?

Examples:

- HDR component count
- component persistence across HDR thresholds
- peak mass distribution
- peak height distribution
- peak width / area
- peak anisotropy
- peak compactness
- peak persistence across scale

### Valley Metrics

Questions:

- How separated are peaks?
- How much intermediate density connects them?
- Do peaks merge at high density or only at low density?

Examples:

- valley depth
- merge density
- saddle density
- bridge mass
- bridge density
- inter-peak distance
- relative separation
- graph bottleneck strength

### Concentration Metrics

Questions:

- Does the target pack mass into smaller high-density regions than reference?
- Is the target sharper or more compact, even if peak count is unchanged?

Examples:

- HDR mass-area curve
- HDR area AUC
- effective area at fixed HDR mass

Important distinction:

> Concentration is not the same as peak count.

A one-peak target can be more concentrated than reference without being more modal.

### Organization Metrics

Questions:

- Are peaks balanced or hierarchical?
- Are there satellites?
- Are peaks arranged as a chain, branch, or symmetric group?
- Are valleys uniformly deep or heterogeneous?

Examples:

- peak mass entropy
- peak height entropy
- largest peak mass fraction
- peak prominence distribution
- valley depth distribution
- bridge-density distribution
- inter-peak distance distribution

### Support Geometry Diagnostics

Questions:

- What is the characteristic local support width inside dense regions?
- Are valleys or bridge regions broad, narrow, or bottleneck-like?
- Is the observed support isotropic, anisotropic, or locally heterogeneous?
- Does the support geometry suggest a single characteristic scale or multiple local scales?

Examples:

- kNN radius distribution
- local neighbor distance distribution
- local covariance or anisotropy
- MST edge-length structure
- candidate bridge edges
- graph bottleneck strength
- bridge-to-local-width ratios

These diagnostics describe intrinsic support geometry. They do not declare biological modes on their own.

### Estimation Regime Metrics

Questions:

- Is one global smoothing scale adequate for recovering the observed organization?
- Would a narrower smoothing scale preserve valleys without fragmenting peaks?
- Do local density differences suggest adaptive smoothing?
- Does anisotropy suggest anisotropic or locally scaled estimation?
- Would a graph-based or topology-aware estimator better preserve bridges and bottlenecks?

Examples:

- recommended estimation regime
- smoothing scale relative to local support width
- evidence for adaptive estimation
- evidence for anisotropic or locally scaled estimation
- evidence for graph-based or topology-aware estimation
- sensitivity of peak and valley calls to smoothing scale

Estimator and scale selection should be evaluated as support-aware modeling decisions, not as arbitrary hyperparameter searches.

### Mode Robustness Metrics

A peak or valley is robust when its existence is consistently supported under reasonable perturbations of the observations and estimation procedure.

Questions:

- Which inferred peaks persist across smoothing scales or estimator families?
- Which peaks recur under subsampling or downsampling?
- Which valleys remain stable across the modal estimation procedure?
- How much mass is consistently assigned to each peak?
- Is each peak's concentration or compactness stable across resampling?
- Are bridge density, valley depth, peak membership, and peak mass stable enough to support biological interpretation?

Perturbations:

- downsampling
- bootstrap resampling
- estimator family
- smoothing scale
- graph scale
- outlier removal

Evidence of robustness:

- peak persistence across smoothing scales
- valley persistence across smoothing scales
- estimator-family stability
- subsampling recurrence
- bridge-density stability
- valley-depth stability
- peak-membership stability
- peak-mass stability
- peak-mass-fraction stability
- peak-concentration stability

Robustness is the evidence filter between observed modal structure and candidate biological organization.

### Resolution / Confidence Diagnostics

Questions:

- Is the sample dense enough to trust the estimated peak and valley structure?
- Are conclusions sensitive to sampling, filtering, or finite-sample uncertainty?

Examples:

- local sampling density
- bootstrap variability
- downsampling variability
- outlier sensitivity
- support overlap
- filtering sensitivity

Confidence diagnostics qualify interpretation. They do not replace the peak-valley description or define the objects being measured.

---

## Desired Benchmark Output

The final output should be a profile, not a binary label.

For the first implementation, the top-level pairwise profile should stay compact:

```text
target_id
reference_id
concentration_effect
peak_organization_effect
separation_effect
bridge_density_effect
support_geometry_flag
recommended_estimation_regime
robustness_flag
resolution_confidence_flag
interpretation
```

Detailed peak and valley tables can then carry:

```text
peak_count_effect
peak_mass_effect
peak_height_effect
peak_width_effect
peak_shape_effect
valley_depth_effect
relative_separation_effect
peak_balance_effect
organization_effect
mode_robustness_effect
```

Effect directions should be signed consistently for all pairwise comparisons:

```text
positive concentration_effect:
  target is more concentrated than reference

positive peak_organization_effect:
  target has more persistent / mass-significant peaks than reference

positive separation_effect:
  target peaks are more strongly separated than reference

positive bridge_density_effect:
  target has more intermediate bridge density / continuity than reference
```

This convention is important because stronger separation and stronger bridge density point in opposite biological directions.

Example profile:

```text
Compared with reference:
  support geometry: locally heterogeneous
  recommended estimation regime: adaptive or locally scaled estimator
  concentration effect: positive
  peak organization effect: positive
  separation strength: weak/moderate
  bridge density: substantial
  support disconnected: no
  mode robustness: adequate
  resolution/confidence: adequate
```

Plain-language interpretation:

> The target support is locally heterogeneous, so peak and valley estimates should be interpreted under an adaptive or locally scaled estimation regime. Under that regime, the target distribution is more organized into peaks than the reference, and those peaks are more compact. However, the peaks remain connected by appreciable bridge density, so this supports a more-modal organization interpretation rather than strict support discreteness.

---

## Working Principle

The goal is not to find one estimator that never fails.

The goal is to build a support-aware benchmark and reporting framework that describes recoverable structure:

```text
characteristic support width
support anisotropy or heterogeneity
stable bridges and bottlenecks
appropriate mode-estimation regime
more concentrated
more peaked
different peak shapes
different peak masses
different peak heights
different valley depths
more or less bridge density
more or less separated
more or less balanced
more or less robust across smoothing and resampling
more or less well resolved
```

This lets us describe biologically realistic phenotypes without forcing every case into `discrete` or `continuous`.

The benchmark should make quantitative claims like:

> The target is more compact, more peak-organized, and more valley-separated than reference.

rather than:

> The target is discrete.

Discreteness can be a downstream interpretation, but it should not be the primary raw measurement.


some thought : Lmao then this is **way easier** than most interpretability problems.

For L2 logistic regression, the classifier is just:

[
\text{logit}_c(x)=w_c^\top x+b_c
]

So there is no hidden reasoning to decode. The prediction is literally a weighted sum of your 80 raw features.

## What explains one embryo’s prediction

For embryo (i), class (c), and feature (j):

[
\text{contribution}*{icj}=x*{ij}w_{cj}
]

These contributions sum to the class logit, apart from the intercept:

[
\sum_j x_{ij}w_{cj}+b_c=\text{logit}_c(x_i)
]

For a comparison between two classes, which is usually more meaningful:

[
\text{contribution}_{i,c\text{ vs }k,j}
=======================================

x_{ij}(w_{cj}-w_{kj})
]

So if you want to know why an embryo is pushed toward `pbx1b_pbx4_crispant` rather than control, inspect:

```python
x * (coef_double - coef_control)
```

Positive values push toward double crispant. Negative values push toward control.

## But your real question is slightly different

You do not only want:

> Why did it classify embryo (i) as genotype (c)?

You want:

> Which classifier-weighted raw directions cause margin space to gain same-genotype neighbors and lose cross-genotype neighbors?

That requires connecting coefficients to **distances in margin space**.

Suppose your margin representation is:

[
m(x)=Wx+b
]

For two embryos (x_i) and (x_j):

[
m(x_i)-m(x_j)=W(x_i-x_j)
]

Notice the intercept vanishes.

Their squared distance in margin space is:

[
\lVert W(x_i-x_j)\rVert^2
]

So margin geometry is entirely determined by:

```text
raw embryo difference
→ multiplied by classifier coefficient matrix
→ distances measured after that projection
```

This means the classifier pulls embryos together when their raw differences lie mostly in directions that (W) suppresses, and pushes them apart when their differences align with heavily weighted classifier directions.

That is exactly why margin can discard cross-genotype raw ties and replace them with genotype-aligned neighbors. Your uploaded analysis is consistent with a linear supervised projection reorganizing fine local geometry while retaining more coarse structure. 

## The cleanest attribution for your problem

For an embryo pair (i,j), let:

```python
delta_x = x_i - x_j
```

Each margin dimension (c) receives:

```python
delta_margin_c = coef_[c] @ delta_x
```

Then the pair’s squared margin distance is:

```python
margin_dist2 = np.sum((coef_ @ delta_x) ** 2)
```

To attribute this distance back to raw features, you have two choices.

### Simple signed contribution per margin

```python
feature_margin_contribution = delta_x * coef_[c]
```

This tells you which raw features create separation along margin (c).

### Exact quadratic decomposition of total margin distance

Because:

[
|W\Delta x|^2
=============

\Delta x^\top W^\top W\Delta x
]

the effective metric in raw space is:

```python
M = coef_.T @ coef_
```

This matrix is the classifier-induced geometry.

That is a beautiful object for your analysis:

* large diagonal (M_{jj}): raw feature (j) strongly affects margin distance
* large off-diagonal (M_{jk}): features (j) and (k) jointly affect margin distance
* eigenvectors of (M): raw-space directions most preserved/amplified by the classifier
* null space of (M): raw directions discarded by margin space

In other words, your logistic regression defines a **Mahalanobis-like supervised distance**:

[
d_{\text{margin}}^2(x_i,x_j)
============================

(x_i-x_j)^\top M(x_i-x_j)
]

where:

[
M=W^\top W
]

That is probably the deepest, cleanest explanation of what margin space is doing.

## What L2 changes

L2 regularization means coefficients are shrunk toward zero:

```text
large weights become smaller
correlated predictors may share weight
weak directions are suppressed
```

It does **not** make coefficients uninterpretable. But individual feature rankings may be unstable when raw latent features are correlated.

So interpret:

```text
stable coefficient directions
groups of correlated features
eigenvectors of W.T @ W
```

more strongly than “feature 37 is the magic biological feature.”

## Three analyses I would run

### 1. Coefficient and effective-metric spectrum

Compute:

```python
W = classifier.coef_
M = W.T @ W

eigenvalues, eigenvectors = np.linalg.eigh(M)
order = np.argsort(eigenvalues)[::-1]
```

Plot:

```text
eigenvalue spectrum
top raw-space classifier directions
feature loadings on each direction
```

Since you have perhaps five classes but 80 raw dimensions, the classifier geometry is extremely low rank. For multinomial logistic regression with (C) rows, rank is at most (C), and effectively often (C-1) because class logits have a redundant shared direction.

So margin space is throwing away most of the 80-dimensional geometry **by construction**.

That directly explains why coarse genotype organization may remain while local raw neighborhoods are heavily reorganized.

### 2. Explain lost and gained neighbor pairs

You already have:

```text
raw neighbors lost in margin
new margin neighbors gained
```

For each pair, calculate:

```python
delta_x = x_i - x_j
delta_margin = W @ delta_x
```

Then compare:

```text
lost pairs:
  which classifier directions were weak despite raw closeness?

gained same-genotype pairs:
  which classifier directions make them similar?
```

For gained pairs, inspect which components of `delta_margin` are especially small.

For lost cross-genotype pairs, inspect which class-contrast directions become large.

### 3. Ablate classifier directions

Rather than ablating raw features one at a time, remove one classifier direction:

```python
W_ablated = W.copy()
W_ablated[class_or_direction] = 0
margin_ablated = X @ W_ablated.T
```

Then recompute:

```text
neighbor overlap
same-genotype gained-neighbor enrichment
within-genotype preservation
experiment enrichment
```

This tells you which class contrast causes the reorganization.

For example:

```text
remove double-crispant contrast
→ does genotype consolidation disappear?
```

That is much more interpretable than generic SHAP here.

## One caution: standardization

Before interpreting coefficients, check whether logistic regression was trained after `StandardScaler`.

If yes:

```text
coefficient magnitude is comparable across standardized features
```

If no:

```text
large coefficient may merely compensate for a small feature scale
```

For contributions, use the exact representation the classifier received:

```python
X_model = scaler.transform(X_raw)
contributions = X_model * classifier.coef_[class_id]
```

Do not multiply unscaled `z_mu_b` values by coefficients trained on scaled features.

## My actual recommendation

Skip SHAP initially.

Build:

```text
1. W coefficient matrix
2. M = W.T @ W classifier-induced metric
3. eigenvectors/eigenvalues of M
4. pairwise attribution for gained and lost neighbors
5. direction-ablation effects on your existing neighborhood statistics
```

Because the model is linear, those are exact explanations. No interpretability séance required.
