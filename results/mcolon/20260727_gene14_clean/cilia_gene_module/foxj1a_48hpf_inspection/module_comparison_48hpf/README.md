# Foxj1a 48-hpf comparison across cilia modules

This subsection compares all five approved cilia modules without changing the
main foxj1a inspection analysis.

The figures answer four separate questions:

1. Where does each module increase or decrease on the global cell-type graph?
2. Which effects are large on a common standardized scale?
3. Which effects are supported after BH correction?
4. Do the strongest results look consistent across individual embryos?
5. How much does the candidate set expand from BH q-values to raw p-values?
6. Do all nominal raw-p candidates look consistent across embryos?

Raw score differences are not directly comparable between modules because the
modules contain different genes and have different score ranges. Therefore,
the raw graph gives every module its own clearly labelled 95th-percentile color
limit. The standardized graph uses one shared scale.

Files:

- `compare_modules_48hpf.Rmd`: readable executable analysis.
- `figures/`: white-background PNG and PDF figures.
- `data/`: compact tables used in the figures.

Exploratory significance outputs:

- `06_module_overlap_upset_matrix`: all 27 nominal candidate cell types across
  the five modules. Blue nodes decrease, red nodes increase, and node size
  gives the evidence tier. Angled labels keep every cell type legible.
- `07_raw_p_candidate_embryo_values.pdf`: one page per cilia module showing
  every result with raw p < 0.05. These are nominal candidates, not
  FDR-supported discoveries.
- `07_raw_p_candidates_*.png`: the same five module pages as convenient PNGs.
