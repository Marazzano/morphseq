# TASK E — b9d2 worked example (ACCEPTANCE TARGET, SERIAL, LAST)

**Prereq: TASK_0 + A + B + C + D merged.** This is the integration that proves the
engine composes. It is the acceptance target for the whole thing. Runs last, one
agent, not parallel.

## Source of truth
`docs/HANDOFF_NEXT_distribution_engine.md` lines 49–96 (the worked example + the
data-source warning) and `docs/PRIMITIVE_ONTOLOGY.md` "Construction walk" (Steps
0–5). Grounding (NOT a template to copy):
`results/mcolon/20260617_morph_axis_investigation/valley_visualization.py`
(`load_bins` + `render_gene`).

## The end goal, concrete
> "Generate the b9d2 phenotype distributions over time vs their controls, and watch
> the two clusters emerge."

## Scope — an end-to-end script `v0/b9d2_worked_example.py` that runs the walk:
```
for each time_bin (24 / 30 / 36 / 48 hpf):
  Step 0  Load + bin UPSTREAM. Reuse src/analyze/utils/binning.py::bin_embryos_by_time.
          The engine objects NEVER bin. Embryo grain (physical_embryo_id).
  Step 1  CARVE by label column → two Distributions per (scope × bin × role):
            reference = zygosity == "wildtype"
            target    = phenotype_clean ∈ {CE, HTA}
          Build a SHARED feature representation BEFORE the split (same feature_names
          on both). Native feature values, feature units, no frame.
  Step 2  genotype/phenotype (column) labeler on target → LabelGroup "phenotype"
          (SampleSets CE / HTA / unlabeled). [TASK_B]
  Step 3  peak_finding labeler on target AND reference, on ONE grid built from
          POOLED target+reference features → same grid_id → raster-comparable.
          [TASK_A build_grid + TASK_B peak_finding]
          Emergence = target "peak".sample_set_ids grows 1→2 across bins.
  Step 4  compare_label_groups(reference_peak_lg, target_peak_lg, policy)  [TASK_C]
Step 5  LOOP over bins, collect. NO DistributionSequenceView object — the loop +
        (scope, role) naming IS the series (reserved name only).
Plot    feature-over-time via the faceting engine [TASK_D]: CE vs HTA strips, peak
        count / centers drifting across bins.
```

## ⚠️ Data source (read the handoff warning, lines 78–96)
- **Do NOT hard-depend on the hand-cleaned CSV** (`.../gene14/tables/
  reference_b9d2_clean.csv`) as the final story. The maintained target is the
  updated Snakemake analysis-ready output (correct binning, QC gates, latents,
  `physical_embryo_id`).
- **Repo owner's call:** for THIS immediate build, an **ad-hoc column
  manipulation** to make the current columns work is FINE. Leave the worked-example
  logic above unchanged; treat "wire to updated Snakemake" as a **follow-up, not a
  blocker**. **Confirm the current analysis-ready path with the user before wiring
  it** (`⟢ CHECKPOINT: ask` below).

## What this task proves (acceptance criteria)
- The two clusters "emerge": target peak count goes 1 → 2 as hpf increases, visible
  in the output plot.
- No `if genotype … else if peak …` in the composition — the same skeleton runs.
- Objects never binned/carved themselves; the caller did (Steps 0–1 upstream).
- The plot is faceting-engine IR, not a bespoke grid.

## Out of scope
Changing any engine object/labeler/comparator/plotter. If you find a bug in A–D,
file it against that task's module and (if small) fix on a clearly-scoped commit —
don't silently reshape the ontology from the integration layer.

## Commit checkpoints
- `⟢ CHECKPOINT: ask` — before wiring the data source, confirm the analysis-ready
  path with the user (Snakemake output vs ad-hoc CSV manipulation for now).
- `⟢ COMMIT 1` — Steps 0–3 wired (load/bin/carve/label/peak on pooled grid) on real
  b9d2 data; a run produces per-bin LabelGroups. Smoke test green.
  `dist-engine(task-e): b9d2 carve→label→peak on pooled per-bin grid`
- `⟢ COMMIT 2` — Step 4 comparison + Step 5 loop + the emergence plot.
  `dist-engine(task-e): b9d2 emergence — compare + feature-over-time plot`
- Open PR. Include the output figure path in the PR body and state whether target
  peak count actually goes 1→2 (the acceptance signal).

## Definition of done
End-to-end b9d2 script runs on real data through A–D with zero labeler-type
branching; two-cluster emergence is visible in a faceting-engine plot; binning
stayed upstream; data-source decision confirmed with the user; two checkpoint
commits; PR open with the figure + emergence verdict.
