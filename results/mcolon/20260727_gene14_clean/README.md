# GENE14 — clean rebuild (2026-07-27)

Replaces `20260715_gene14_hooke_dact/` (91 scripts, 290 outputs, 422 GB), where only
9 scripts were load-bearing and provenance was no longer checkable by reading.

**Structure: source-derived tables are kept separate from analyses.** The
imaging crosswalk, neutral cell-type counts, and differential-abundance choices
each have a clear home.

```
0_shared/                       cross-assay maps and shared annotations
cell_type_counts/               neutral embryo-by-cell-type measurements
control_floor/                  the MOTIVATION — read before believing any DACT
abundance_dact/                 per-timepoint DACT refits -> trajectories, heatmap
morphology/                     imaging z_mu_b classification
expression_deg/                 empty; build when needed
```

## Shared mappings and annotations

| script | writes | verified |
|---|---|---|
| `0_shared/build_cell_type_lineage.R` | `cell_type_lineage.tsv` | 379 graph cell types, 30 sulston groups (matches archive) |
| `0_shared/map_morphseq_to_sequencing.ipynb` | `morphseq_embryo_sequence_map.csv`, `preferred_morphseq_labels.csv` | Excel-backed identity map plus one preferred MorphSeq phenotype label per sequencing embryo |

The mapping notebook reads sequencing metadata, sequencing QC, and McClintock
embryo IDs directly from their source files. There is no copied
`embryo_table.tsv`.

## Cell-type counts

| script | writes | purpose |
|---|---|---|
| `cell_type_counts/build_per_embryo_celltype_counts.R` | `per_embryo_celltype_counts.tsv` | Counts every cell type in every embryo and fills absent cell types with explicit zeroes |

This table contains measurements only. It does not define controls, contrasts,
reliability thresholds, or differential-abundance results.

## Abundance DACT

| script | writes | purpose |
|---|---|---|
| `abundance_dact/contrast_plan.R` | *(no output)* | Declares every perturbation-versus-control comparison |
| `abundance_dact/cell_type_gate_policy.R` | *(no output)* | Provides reusable detection and reliability functions |
| `abundance_dact/differential_abundance.Rmd` | `output/differential_abundance.tsv` | Fits every planned contrast and adds reliability and significance columns |

## The two nuances, each in exactly one place

### Control choice — `abundance_dact/contrast_plan.R`

The contrast plan is the single source of truth:

| gene | control | timepoints |
|---|---|---|
| foxj1a / ift88 / sspo | `ctrl-inj` | 18/24/30/48 |
| cep290-mut | `cep290-negsib` | 18/24/48 |
| cep290-mut | `b9d2-negsib` | **30 — borrowed, `is_cross_rt_block=TRUE`** |
| b9d2-mut | `b9d2-negsib` | 14/18/30/48 |

**The 30hpf borrow is unavoidable, not a shortcut.** `cep290-negsib` has zero embryos at
30hpf. mcclintock did not solve this — it *dropped*
cep290@30hpf outright (`dropped_conditions.tsv`: "Missing control batch"), so the
borrowed fit is the only cep290@30hpf estimate that exists anywhere.
`is_cross_rt_block` propagates
as a column and must be rendered on every figure that shows it.

### Cell-type filtering — `abundance_dact/cell_type_gate_policy.R`

```
is_reliable =
  log_abundance_control >= -3 AND
  control_detection_fraction >= 0.5
```

Both terms concern the **control** arm — it is the denominator. Nothing looks at the
mutant, which is why the gate is identical across genes sharing a control. That
invariance is the point: the gate describes the assay, not the biology.

**`ablation_candidate` was removed.** It conflated a true ablation (cell type genuinely
gone — the strongest result possible) with a detection floor, and the heatmap masked
both. Measured cost at 48hpf: **7 real ablations hidden, 4 significant**, including
*neural progenitor cell, midbrain* independently in foxj1a (q=0.006) and sspo (q=0.011).
All 7 are retained by the new gate. `mean_cells_*` and
`perturbation_detection_fraction` ride along as
**informational** columns that drive no filtering.

The control column is named `control_detection_fraction`, rather than “sibling,”
because the crispant control is `ctrl-inj`.

## Correction to `SPLINE_VS_PER_TIMEPOINT.md`

The archived doc is titled "Spline pooling is WORSE ... for this cohort", but its table
measures **only cep290 and b9d2**. Measured on the **crispants**, the opposite holds:

| cohort | median SE ratio spline/refit | more hits |
|---|---|---|
| mutants (the doc) | 2–7× larger | refit |
| crispants | **0.45–0.88×, i.e. smaller** | **spline** (sspo@48: 41 vs 20) |

Mechanistically consistent — mutants have 7–8 siblings plus the 30hpf gap, so one smooth
curve fits badly; crispants have 11–12 per arm at all four timepoints. Effects agree
(r = 0.80–0.98). **We refit everything per-timepoint regardless**, so the pipeline is
unaffected, but the doc's claim should be scoped to the mutant cohort.

## Environment

```bash
conda run -n segmentation_grounded_sam --no-capture-output python <script>
/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript <script>
```

No local CDS copy is saved: the mcclintock CDS is already cilia-only (520 embryos across
9 arms, nothing else), so a subset would just duplicate 19 GB. Scripts that need it read
the source directly.
