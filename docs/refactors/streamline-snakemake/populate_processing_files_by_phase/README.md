# Populate Processing Files by Phase

This directory reorganizes the module-population guides from
`populate_process_files/` into refactor phases that mirror the
Snakemake data-flow plan.  Each phase document links back to the
original component deep-dives while highlighting the specific files
and deliverables required to unlock that phase of the pipeline.

- `phase_1.md` – experiment metadata bootstrap (plate/scope manifests,
  joined metadata, image manifest).  
- Additional phase guides will be added as the refactor advances
  beyond Phase 1.

Use these summaries when planning work increments or grooming tickets:
they provide a ready-made checklist of modules, expected functions, and
downstream outputs for the current phase while keeping the detailed
per-component notes in `populate_process_files/` as the source of truth.
