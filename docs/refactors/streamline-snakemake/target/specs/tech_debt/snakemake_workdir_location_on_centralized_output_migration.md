# Tech Debt: Snakemake work-directory location must be reconsidered on the centralized-output migration

**Status:** known placement decision to revisit, mdcolon 2026-07-10. Not a blocker today — the
current placement is correct for the single-user tree. The debt is that the chosen location is
**tied to the local `WORKFLOW_DIR`**, and the upcoming shared-output migration changes the
constraints that placement must satisfy.

---

## Background — why a per-experiment work directory exists

Concurrent full-pipeline runs on **different experiments** share the one `WORKFLOW_DIR`
(`src/data_pipeline/pipeline_orchestrator/`). Snakemake writes its lock + DAG bookkeeping into
`.snakemake/` **relative to the working directory (cwd), not per experiment** — so two schedulers
launched from the same `WORKFLOW_DIR` collide on a single lock, and the second refuses to start.

Fix in place (baked into `sge_job_submissions/submit_snakemake_TEMPLATE.sge`): each run passes
`snakemake --directory work_directories/{EXPERIMENT}`, giving every experiment its **own** isolated
`.snakemake/` lock + DAG tree. This is safe because the Snakefile resolves includes/config via
`workflow.basedir` and all outputs via `env.paths.output_root` (both absolute), so `--directory`
relocates **only** the `.snakemake` bookkeeping — never includes, config, or outputs.

Today that lands at:

```
src/data_pipeline/pipeline_orchestrator/work_directories/{experiment}/.snakemake/
```

(local-only; `work_directories/` is gitignored.)

## The debt — the shared centralized-output migration

There is an upcoming migration to a **centralized beta pipeline output folder shared between
mdcolon and Nicholas Lammers** (nlammers). When that lands, the Snakemake work-directory location
must be chosen deliberately, because two properties now matter that don't in the single-user tree:

1. **Preserve DAG/incremental state across experiments and across users.** The `.snakemake`
   bookkeeping is what lets a re-run skip already-complete jobs. If the work directory is pinned to
   a local `WORKFLOW_DIR` (per checkout / per user), that incremental state is **not shared** — a
   second person (or a second machine) re-running the same experiment against the shared output
   tree gets a cold DAG and may redo or, worse, contend on outputs it thinks are missing. The work
   directory should live **with the shared output tree, keyed by experiment**, so the incremental
   state travels with the data, e.g. `{shared_output_root}/work_directories/{experiment}/`.

2. **Avoid cross-user lock collisions on the SAME experiment while keeping per-experiment
   isolation.** The per-experiment split already prevents different-experiment collisions; on the
   shared tree it *also* becomes the mechanism that lets two users coordinate on the *same*
   experiment via one authoritative lock (co-located with the shared outputs) instead of two
   independent local locks that don't see each other.

## What to do at migration time

- Move `SMK_WORKDIR` from `WORKFLOW_DIR/work_directories/{experiment}` to a path under the shared
  output root (e.g. derived from `env.paths.output_root`), so `.snakemake` bookkeeping is
  **co-located with the experiment's outputs** and shared across users/machines.
- Confirm the shared filesystem's locking semantics are sound for the Snakemake lock (NFS lock
  correctness) before relying on the co-located lock for cross-user coordination.
- Keep the per-experiment keying — it is the isolation unit that makes concurrent different-
  experiment runs safe; the migration changes only *where the keyed dir lives*, not the keying.

## Why record now

The placement was chosen for the single-user local tree and is correct there. It is easy to carry
that local path forward unthinkingly into the shared migration, silently losing cross-user
incremental state and re-introducing a lock that doesn't actually coordinate the two users. Writing
it down here ties the decision to the migration that changes its constraints.
