# GENE14 — clean rebuild (2026-07-27)

Replaces `20260715_gene14_hooke_dact/` (91 scripts, 290 outputs, 422 GB), where only
9 scripts were load-bearing and provenance was no longer checkable by reading.

**Structure: one shared core, then independent threads.** A thread reads from
`0_shared/` and its own directory — never from another thread. That way DEGs (or
anything else) can be added later without touching what already works.

```
0_shared/                       the ONLY place that touches the big data
control_floor/                  the MOTIVATION — read before believing any DACT
abundance_dact/                 per-timepoint DACT refits -> trajectories, heatmap
morphology/                     imaging z_mu_b classification
expression_deg/                 empty; build when needed
```

## 0_shared — run in order

| script | writes | verified |
|---|---|---|
| `0_load_mcclintock.R` | `embryo_table.tsv` (520-embryo spine), `per_embryo_celltype_counts.tsv`, `provenance.tsv` | 74,898 Bl1 counts **identical** to archive |
| `1_cell_type_lineage.R` | `cell_type_lineage.tsv` | 379 graph cell types, 30 sulston groups (matches archive) |
| `2_seq_imaging_crosswalk.py` | *(imported, no output)* | exact lookup on coordinates parsed from `seq_embryo_ID` |
| `3_build_embryo_id_map.py` | `embryo_id_map.csv` | 516 unique matches; 11 honest metadata misses; 0 incomplete/colliding/time-mismatched |
| `4_attach_morphseq_labels.py` | `embryo_table_labeled.tsv`, `imaging_to_seq_crosswalk.tsv` | 169/169 resolved, 0 orphans; 101 labels **identical** to archive |
| `5_celltype_gate.py` | `celltype_gate.tsv` | 115/149/194/235 crispant cell types, **identical sets** to archive |

`0` reads mcclintock and nothing else, so the spine exists before anything attaches to
it. `2` is pure lookup logic; `3` is the only script that writes labels.

## The two nuances, each in exactly one place

### Control choice — `0_shared/5_celltype_gate.py`

`CONTROL_OF` and `BORROWED` are the single source of truth:

| gene | control | timepoints |
|---|---|---|
| foxj1a / ift88 / sspo | `ctrl-inj` | 18/24/30/48 |
| cep290-mut | `cep290-negsib` | 18/24/48 |
| cep290-mut | `b9d2-negsib` | **30 — borrowed, `xblock=TRUE`** |
| b9d2-mut | `b9d2-negsib` | 14/18/30/48 |

**The 30hpf borrow is unavoidable, not a shortcut.** `cep290-negsib` has zero embryos at
30hpf (visible in step 0's design table). mcclintock did not solve this — it *dropped*
cep290@30hpf outright (`dropped_conditions.tsv`: "Missing control batch"), so the
borrowed fit is the only cep290@30hpf estimate that exists anywhere. `xblock` propagates
as a column and must be rendered on every figure that shows it.

### Cell-type filtering — `0_shared/5_celltype_gate.py`

```
reliable = log_abund_ctrl >= -3  AND  nonzero_frac_ctrl >= 0.5
```

Both terms concern the **control** arm — it is the denominator. Nothing looks at the
mutant, which is why the gate is identical across genes sharing a control. That
invariance is the point: the gate describes the assay, not the biology.

**`ablation_candidate` was removed.** It conflated a true ablation (cell type genuinely
gone — the strongest result possible) with a detection floor, and the heatmap masked
both. Measured cost at 48hpf: **7 real ablations hidden, 4 significant**, including
*neural progenitor cell, midbrain* independently in foxj1a (q=0.006) and sspo (q=0.011).
All 7 are retained by the new gate. `mean_cells_*` and `nonzero_frac_mut` ride along as
**informational** columns that drive no filtering.

Renamed `nonzero_frac_sib` → `nonzero_frac_ctrl`: the old name lied — for crispants the
control is `ctrl-inj`, not a sibling.

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
9 arms, nothing else), so a subset would just duplicate 19 GB. `provenance.tsv` records
its path for gene-level work.
