# 2026-07-15 lab meeting scratch

## B9D2 phenotype-transition summary

The B9D2 version of the NWDB CEP290 phenotype-transition summary is under
`figures/b9d2_phenotype_transition/static_plots/`.

It uses the cleaned reference table from the SCI cilia QC analysis:

`results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv`

Only QC-passing homozygous embryos with penetrant labels are included. `CE`
remains `CE`, and `BA_rescue` is pooled into `HTA`; non-penetrant embryos remain
a separate gray class and are excluded from this two-curve summary. The canonical colors come from
`analyze.viz.styling.color_mapping_config`.

## B9D2 phenotype distribution by pair

`figures/b9d2_phenotype_distribution/phenotype_distribution_by_pair.png`
recreates the earlier six-pair by three-zygosity distribution figure with the same
pooled `CE`/`HTA` mapping, a retained gray non-penetrant class, and canonical package colors.

## CEP290 phenotype-transition summaries

Matching CEP290 curvature and total-length summaries are under
`figures/cep290_phenotype_transition/static_plots/`. They use the cleaned CEP290
reference table and canonical pink/teal package colors.

## B9D2 trajectories by pair

`figures/b9d2_phenotype_trajectories_by_pair/phenotype_trajectories_by_pair.png`
recreates the historical 2×6 curvature/length grid, now colored by canonical B9D2
phenotype rather than genotype.

## B9D2 curvature by phenotype and experiment

`figures/b9d2_curvature_by_phenotype_and_experiment/` contains a native
`plot_feature_over_time` figure with `CE` and `HTA` as facet rows and experiment as
the color/group variable. It reads five Build06 experiments, retains every discovered
pair (1, 2, 4, 5, 6, 7, and 8), and assigns one phenotype row per embryo by pooling
the saved B9D2 model's predictions across supported time bins.
The companion length plot uses the identical layout and labels with `total_length_um`.

## Penetrance composite (everything pooled)

`figures/penetrance_composite/` summarizes phenotype spread across WT / het / homozygous for
both genes, pooled over all experiments. Percentages are **within each zygosity** (penetrance
framing) and raw embryo counts are annotated on every view. Phenotype calls reuse
`14_plot_resolve_phenotypes_from_mess.load_gene`, so they match the trajectory figure exactly.

Three views of one table (`penetrance_table.csv`, one row per embryo):

- `A_dot_matrix.png` — zygosity x phenotype grid; dot area = n (scaled per gene), fill = %.
- `B_skinny_bars.png` — 100% stacked composition; thin segments labeled outside with leaders.
- `C_embryo_dots.png` — one dot per embryo, jittered; black bar = penetrant fraction.

Headline numbers: cep290 penetrant 8% WT / 9% het / **90% homo** (hets look like WT);
b9d2 penetrant 0% WT / **33% het** / 95% homo — the het CE+HTA fraction is the b9d2 point.

### Reproduce

```bash
/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/01_plot_b9d2_phenotype_summary.py

/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/02_plot_b9d2_phenotype_distribution_by_pair.py

/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/03_plot_cep290_phenotype_summary.py

/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/04_plot_b9d2_phenotype_trajectories_by_pair.py

/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/05_plot_b9d2_curvature_by_phenotype_and_experiment.py

/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/06_plot_b9d2_length_by_phenotype_and_experiment.py

/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260715_lab_meeting_scratch/21_plot_penetrance_composite.py
```
