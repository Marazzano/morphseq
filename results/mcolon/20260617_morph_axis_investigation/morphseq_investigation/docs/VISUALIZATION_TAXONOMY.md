# Visualization taxonomy

Three tiers. The boundary between them is doing architectural work: it keeps the
engine from turning into a report-construction octopus, and keeps `valley_visualization.py`'s
one-gene narrative out of the general verbs.

## The spine: axes vs within-cell

One distinction runs through every Tier-2 verb (inherited from
`analyze.viz.plotting.feature_over_time.plot_feature_over_time`):

```
facet_row / facet_col   →  WHERE a comparison appears (the grid layout)
within-cell grouping    →  WHAT is compared in place (the overlaid curves/fields)
```

`DistributionGrid` already captures the resolved result of this split — `row` /
`col` (`FacetCoordinate`) are the axes; the curves bucketed into a cell are the
within-cell overlay. Tier-2 verbs CONSUME this; they do not re-invent grouping
semantics (no ad-hoc `group_by` / `color_by` / `panel_by` / `overlay_by` cousins).

---

## Tier 1 — Single-axis drawing primitives

Small reusable operations that draw resolved numerical objects onto ONE axis.
No faceting, no report layout, no density fitting. Live in `engine/plotting.py`.

Two sub-kinds (they live together but are conceptually distinct):

- **Marks** (draw one thing): `strip_trace` (one KDE strip), `hdr_band_trace`
  (one 1-D HDR band).
- **Single-axis composers** (draw several resolved marks onto one axis):
  `overlay_strip_subplot` (several densities on one shared 1-D grid).

- **2-D** (NOT built — seam only): a 2-D density field, peak-basin ring,
  membership-overlap fill, raw-points scatter. These live *bespoke* inside
  `valley_visualization.py` today; extracting them is future work, gated on the
  base-vs-annotation decision below.

## Tier 2 — General faceted visualization verbs

Consume renderer-neutral resolved grids (`DistributionGrid`). Own facet layout,
within-cell visual encoding, legends, and role styling. Do NOT perform analysis
or density estimation — that is the numerical stage
(`materialize_distribution_marginals`), upstream.

- **`plot_1d_density_grid`** (built): faceted 1-D density-strip grid. One subplot
  per cell, densities overlaid within each cell; role → style (reference
  gray/dashed), role → auto gradient. Named for the mark family it draws, not
  "all 1-D distribution views":
    - ECDFs, histograms, violins would be SIBLING verbs.
    - a genuine stacked **RIDGELINE** (offset baselines within one coordinate
      system — a joyplot) is its OWN renderer that consumes the same
      `DistributionGrid`; it is not a `layout=` switch on the strip verb. (The
      overlaid/stacked/mirror ridge variants in `v0/rich_distribution_plot.py`
      are that renderer's prototype, pre-`DistributionGrid`.)
- **`plot_2d_density_grid`** (deferred): faceted 2-D density-field grid. Held off
  deliberately — see the seam below.

## Tier 3 — Custom composed figures

Own report-specific narrative, panel arrangement, and cross-panel semantics
(row order carries a story, axes may deliberately differ, one panel may show
target density while another shows overlap, legends may be hand-built, layout
proportions may encode importance). May REUSE Tier 1 and Tier 2, but are NOT
promoted into the engine.

- **`valley_visualization.py`** (entry point `render_gene`) — a 4-row, one-gene
  report (target KDE + peak rings / reference readout / density overlap / raw
  points side-by-side). Unfaceted and specific. **Keep it custom.** Do not fold
  it into the engine and do not give it an engine-verb name; when
  `plot_2d_density_grid` exists it will be the *general faceted density-field
  grid*, a different and simpler thing than this report.

---

## The deferred 2-D seam: base representation + annotation layers

When `plot_2d_density_grid` is eventually built, do NOT frame the design as
"what does the 2-D case overlay?" — that wrongly treats a density field and a
peak ring as peers. They are not: one is the substrate, the other is annotation
on it. Resolve two independent concepts instead:

```
base representation   (the substrate — exactly one)
  - density field         "where is probability mass concentrated?"
  - raw points            "what observations support the model?"
  - membership field      "where do assignments/regions intersect?"

annotation layers     (drawn on the base — zero or more)
  - peak-basin rings      "what structures did the peak model resolve?"
  - centroids
  - overlap contours
  - labels
```

A future spec stays compositional without becoming report-specific:

```python
Plot2DSpec(
    base_mark=DensityFieldMark(),
    annotation_marks=(PeakBasinRingMark(), RawPointMark(alpha=...)),
)
```

This avoids the `show_points=/show_peaks=/show_overlap=/reference_mode=...` forest
— the scent of several verbs hiding in one trench coat. Not built now; recorded
so the seam is better than "everything is an equivalent overlay."
