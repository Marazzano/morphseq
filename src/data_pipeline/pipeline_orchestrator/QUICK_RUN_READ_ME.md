## Quick Run

This pipeline uses **Snakemake** so you can run the full pipeline, one experiment, multiple experiments, or individual wells from an experiment.

The key idea is that Snakemake configs can be passed **compositionally**:

```text
config.yaml                               ← tracked base science config
configs/runtime_configs/*.yaml            ← experiment / smoke overlays
sge_job_submissions/*.sge                 ← local SGE launchers
```

Two tracked defaults have distinct jobs:

- `config.yaml` controls what the workflow does (products, models, and model-server ownership).
- `profiles/default/config.yaml` controls how Snakemake schedules it (`gpu=1`, `keep-going`,
  rerun policy, and service readiness timeouts). These are Snakemake CLI settings and cannot be
  supplied by the workflow's ordinary `config.yaml`.

Runtime overlays select experiments/wells or deliberately override a base behavior. SGE launchers
pass the default profile explicitly; that is belt-and-braces rather than a requirement, since
Snakemake also auto-discovers it (below).

Later `--configfile` arguments override earlier ones.

So a normal run looks like:

> **The run policy in `profiles/default/config.yaml` applies to every run.** It supplies
> `--keep-going`, `--resources gpu=1`, `--rerun-triggers mtime`, and `--rerun-incomplete`,
> which are POLICY for this pipeline rather than per-run choices. Without
> `--keep-going` a single failing well stops Snakemake from scheduling any further jobs -- that
> ended a 96-well run at 812/1197 with ~70 wells never attempted.
>
> You do **not** have to pass a flag for this: Snakemake 7.32.4 auto-discovers `profiles/default`
> beside the Snakefile (it searches relative to the current directory, then to the Snakefile). The
> explicit `--profile ...` shown below is equivalent and harmless. Anything on the command line
> still overrides the profile, and `--workflow-profile none` skips it entirely.
>
> The Snakefile asserts the resolved values at parse time (`orchestration/run_policy.py`), so a run
> that somehow did not receive the policy fails immediately with an explanation instead of quietly
> re-running SAM2 across every well or putting several models on one card.

```bash
snakemake --profile src/data_pipeline/pipeline_orchestrator/profiles/default \
  --configfile config.yaml \
  --configfile configs/runtime_configs/runtime_config_all_20250912_and_20260410_otx_pilot.yaml \
  all --cores 1
```

This means:

```text
base defaults + runtime experiment/well selection → run target
```

The runtime config is where you list which experiments, and optionally which wells, you want to run.

---

### 1. Run one or more experiments

To run one experiment, list it under `experiments`:

```yaml
experiments:
  - 20250912
```

If no `target_wells` are provided, the pipeline uses the smart default:

```text
run all discovered wells for that experiment
```

To run multiple experiments:

```yaml
experiments:
  - 20250912
  - 20251017
```

Again, if no wells are specified, each experiment runs over all discovered wells.

---

### 2. Optionally narrow to specific wells

To run only selected wells, add `target_wells`:

```yaml
experiments:
  - 20250912

target_wells:
  20250912:
    - B01
```

This means:

```text
run experiment 20250912
only run well B01
```

You can list multiple wells:

```yaml
experiments:
  - 20250912

target_wells:
  20250912:
    - B01
    - B02
    - B03
```

Well entries can be local well slugs or global well ids:

```yaml
target_wells:
  20250912:
    - B01
    - 20250912_B02
```

`experiment_wells` is also accepted with the same structure.

The discovery step still decides what physically exists. The runtime config only narrows what the DAG tries to run.

---

### 3. Choose the Snakemake target

The config chooses **what slice of data** to run.

The Snakemake target chooses **how far through the pipeline** to run.

Common targets:

```text
front_half
  raw acquisition → materialized images → validated frame inventory shards

through_line
  raw acquisition → materialized images → segmentation/tracking/features → snip_qc

analysis_ready
  raw acquisition → full per-snip payloads → final merged analysis parquet

reports
  generate terminal report artifacts from available pipeline outputs

all
  run the full configured pipeline
```

Example: run all discovered wells for one experiment through the full pipeline:

```bash
snakemake --profile src/data_pipeline/pipeline_orchestrator/profiles/default \
  --configfile config.yaml \
  --configfile configs/runtime_configs/runtime_config_all_20250912_and_20260410_otx_pilot.yaml \
  all --cores 1
```

Example: run one well through the front half only:

```bash
snakemake --profile src/data_pipeline/pipeline_orchestrator/profiles/default \
  --configfile config.yaml \
  --configfile configs/runtime_configs/config_smoke_front_half_20250912.yaml \
  front_half --cores 1
```

Example: run one well through the biological processing path:

```bash
snakemake --profile src/data_pipeline/pipeline_orchestrator/profiles/default \
  --configfile config.yaml \
  --configfile configs/runtime_configs/config_smoke_through_line_20250912_B01.yaml \
  through_line --cores 1
```

---

### 4. Submit a cluster run with SGE

For GPU/full-pipeline reruns, submit from the orchestrator directory so SGE writes logs next to the
submission script. This example is a two-experiment `rule all` run:

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/data_pipeline/pipeline_orchestrator
qsub sge_job_submissions/submit_all_20250912_and_20260410_otx_pilot.sge
```

The runtime overlay is `configs/runtime_configs/runtime_config_all_20250912_and_20260410_otx_pilot.yaml`:

```yaml
experiments:
  - 20250912
  - 20260410_otx_pilot
```

The SGE script runs `rule all` with that overlay:

```bash
snakemake --profile src/data_pipeline/pipeline_orchestrator/profiles/default \
  --configfile config.yaml \
  --configfile configs/runtime_configs/runtime_config_all_20250912_and_20260410_otx_pilot.yaml \
  --cores 4 all
```

SGE logs are written under the repo root:

```text
/net/trapnell/vol1/home/mdcolon/proj/morphseq/logs/all_20250912_otx.<JOB_ID>.out
/net/trapnell/vol1/home/mdcolon/proj/morphseq/logs/all_20250912_otx.<JOB_ID>.err
```

Stub for a new cluster rerun:

```bash
cp configs/runtime_configs/runtime_config_all_20250912_and_20260410_otx_pilot.yaml \
  configs/runtime_configs/config_runtime_<run_name>.yaml
cp sge_job_submissions/submit_snakemake_TEMPLATE.sge \
  sge_job_submissions/submit_<run_name>.sge
# Edit experiments, optional target_wells/experiment_wells, rule target, job name, log names,
# and remove the TEMPLATE_GUARD block before submitting.
qsub sge_job_submissions/submit_<run_name>.sge
```
