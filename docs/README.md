# morphseq documentation

All documentation lives under `docs/`. Investigation artifacts are the one
deliberate exception — see **Evidence** below.

## Layout

| Path | Holds |
|---|---|
| `docs/architecture/` | Repo-level reference: system architecture, ID conventions, training/inference guides. |
| `docs/data_pipeline/` | Reference and open-work docs for `src/data_pipeline/`. Mirrors the package. |
| `docs/data_pipeline/specs/` | The design/spec corpus behind the current pipeline (formerly `docs/refactors/streamline-snakemake/`). Includes `target/specs/tech_debt/`. |
| `docs/core/` | Reference docs for `src/core`, `src/models`, `src/data`. |
| `docs/refactors/<name>/` | Multi-subsystem refactor efforts. Currently `seahub/` and `core-model/`. |
| `docs/scratch/` | Half-formed ideas and specs not yet committed to. |
| `docs/_archive/` | Superseded material. Kept for provenance; do not treat as current. |

## Where does a new doc go?

Three kinds of document, three homes:

1. **Reference** — how the system works *now*. Goes in the folder mirroring the
   code it describes (`docs/data_pipeline/`, `docs/core/`, `docs/architecture/`).
   Edited alongside the code.

2. **Open work** — what is broken, deferred, or planned. Goes in that same
   subsystem folder, in the subsystem's tracker. For the pipeline that is
   [`data_pipeline/PLANNED_REVISIONS.md`](data_pipeline/PLANNED_REVISIONS.md).
   Keep these **few**; a finding that is not in a tracker is a finding nobody
   will find.

3. **Evidence** — a dated measurement, audit, or investigation. Stays in
   `results/<user>/<YYYYMMDD>_<topic>/`, next to the notebooks, CSVs, and figures
   that produced it. **Do not move evidence into `docs/`.** Instead, record the
   open item in the relevant tracker and link back to the evidence folder.

A refactor spanning several subsystems gets `docs/refactors/<name>/` rather than
being filed under one arbitrary parent.

## Start here

- Pipeline overview — [`data_pipeline/PIPELINE_OVERVIEW.md`](data_pipeline/PIPELINE_OVERVIEW.md)
- Open pipeline work — [`data_pipeline/PLANNED_REVISIONS.md`](data_pipeline/PLANNED_REVISIONS.md)
- System architecture — [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md)
- ID grammar — [`architecture/parsing_id_conventions.md`](architecture/parsing_id_conventions.md)
- Active refactors — [`refactors/seahub/`](refactors/seahub/), [`refactors/core-model/`](refactors/core-model/)
