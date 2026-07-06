## Quick Run

This pipeline uses **Snakemake** so you can run the full pipeline, one experiment, multiple experiments, or individual wells from an experiment.

The key idea is that Snakemake configs can be passed **compositionally**:

```text
config.yaml                  ← stable defaults
config_runtime_*.yaml        ← experiments, optional wells, smoke limits
```

Later `--configfile` arguments override earlier ones.

So a normal run looks like:

```bash
snakemake --configfile config.yaml \
  --configfile config_runtime_20250912.yaml \
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
snakemake --configfile config.yaml \
  --configfile config_runtime_20250912.yaml \
  all --cores 1
```

Example: run one well through the front half only:

```bash
snakemake --configfile config.yaml \
  --configfile config_runtime_20250912_B01.yaml \
  front_half --cores 1
```

Example: run one well through the biological processing path:

```bash
snakemake --configfile config.yaml \
  --configfile config_runtime_20250912_B01.yaml \
  through_line --cores 1
```