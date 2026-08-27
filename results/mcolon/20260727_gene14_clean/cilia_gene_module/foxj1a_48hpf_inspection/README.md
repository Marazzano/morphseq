# Foxj1a 48-hpf cilia-module inspection

Everything needed to inspect the first differential-module result lives here.

- `inspect_foxj1a_48hpf.Rmd`: readable analysis notebook.
- `global_celltype_graph.Rmd`: maps normal foxj1a expression and the
  perturbation response onto the complete v3.1.0 cell-type graph.
- `module_comparison_48hpf/`: isolated comparison of the raw differences,
  standardized effects, statistical evidence, and embryo values for all five
  approved cilia modules.
- `report.html`: rendered notebook.
- `figures/`: matching white-background PNG and PDF figures.
- `data/`: the focused embryo values and component-gene summaries shown in the
  figures.
- `qsub_inspect_foxj1a_48hpf.sh`: cluster entry point.

The figures use embryos—not individual cells—as the replicate. Per-cell data
are used only to calculate embryo summaries and inspect component genes.

The compact three-panel figure makes one argument:

1. control `foxj1a` expression tracks control motile-cilia activity,
2. foxj1a perturbation changes the motile-cilia score in selected cell types,
   and
3. the top six cell-type results are shown embryo by embryo, including the two
   FDR-significant radial-glia losses.

The companion correlation-sensitivity figure shows both Pearson and Spearman
correlations while adding cell types from highest to lowest control `foxj1a`
expression. Its lower panel shows the corresponding fall in mean foxj1a
expression. It is descriptive and is not used to select a new significance
threshold.

The global graph keeps expression and response as separate measurements.
Node color is mean control `foxj1a` expression, node size is motile-cilia score
loss, and a black outline marks BH-FDR hits. The figure reports their Pearson
and Spearman correlations for all tested graph nodes and for the top ten
foxj1a-expressing nodes. To prevent a few extreme nodes from flattening the
visible range, colors are capped at the 95th percentile and node size at the
97.5th percentile; the exported node table retains the original values.

The all-cell-type embryo grid contains every eligible gold-motile result,
ordered by BH q-value and then score change. Each facet also reports its rank
by control `foxj1a` expression so the correlation-sensitive compartment can be
inspected directly.
