# Differential cell-type abundance

This analysis asks:

> Does a perturbation or MorphSeq phenotype change the abundance of a cell type
> relative to its control?

The analysis runs one comparison at one developmental time point. We call each
comparison a **contrast**.

For example:

```text
foxj1a versus ctrl-inj at 18 hpf
```

## The four input files

### 1. `contrast_plan.R`

This is the list of comparisons we want to run.

Each row gives:

```text
contrast name
group column
perturbation
control
time point
expected cross-RT-block status
control note
```

Edit this file when adding or changing a biological comparison.

The file contains two readable tables:

- `perturbation_contrast_plan` compares the original sequencing groups.
- `phenotype_contrast_plan` compares preferred MorphSeq phenotype labels.

### 2. `../cell_type_counts/per_embryo_celltype_counts.tsv`

This contains the observed number of cells of every cell type in every embryo.

It is used to ask simple questions such as:

```text
How many control embryos contained this cell type?
What was the mean number of cells per embryo?
```

### 3. `../0_shared/morphseq_embryo_sequence_map.csv`

This connects each preferred MorphSeq phenotype prediction to its sequencing
embryo ID.

### 4. `cell_type_gate_policy.R`

This contains the shared reliability rule.

A cell type is reliable when:

```text
at least half of the control embryos contain it
and
its modeled control abundance is at least -3
```

The perturbation does not need to contain the cell type. A real loss of a cell
type in the perturbation can therefore remain a biological result.

## What the notebook does

`differential_abundance.Rmd` follows five steps.

### Step 1: Check the contrast plan

For every contrast, the notebook checks:

- Both groups exist at the requested time point.
- Both groups have enough embryos.
- Each group has one RT block.
- The observed cross-RT-block status matches the plan.

If any check fails, the notebook stops with an explanation.

### Step 2: Fit differential abundance

The notebook compares the perturbation with its control using Hooke and Platt.

The effect is:

```text
perturbation abundance - control abundance
```

Therefore:

```text
positive value = more abundant in the perturbation
negative value = less abundant in the perturbation
```

### Step 3: Save the ungated model result

Each contrast is saved separately:

```text
output/raw_model_results/<contrast_name>.tsv
```

These files contain the model results before the reliability policy is applied.
Existing files are skipped, so fitting can resume without repeating a completed
contrast.

### Step 4: Add reliability columns

The notebook combines each model result with the observed embryo cell counts.

It adds columns including:

```text
control_detection_fraction
perturbation_detection_fraction
passes_control_detection
passes_control_abundance
is_reliable
```

No cell types are deleted.

### Step 5: Write one final table

All contrasts are combined into:

```text
output/differential_abundance.tsv
```

One row represents:

```text
one contrast x one cell type
```

The final table also distinguishes:

- `is_fdr_significant`: passes the reliability policy and the FDR threshold.
- `is_screening_hit`: passes the reliability policy and has raw `p < 0.10`.

The screening flag is exploratory. It is not called statistically significant.

## How to run it

Open `differential_abundance.Rmd` and choose **Knit with Parameters**.

Choose one mode:

| Mode | What it does |
|---|---|
| `check_plan` | Checks every planned contrast without fitting |
| `fit_one` | Fits the contrast named in `contrast_name` |
| `fit_group` | Loads the CDS once and fits every contrast in `fit_group` |
| `fit_all` | Fits every unfinished contrast, then assembles the final table |
| `assemble` | Combines previously completed model results |

The default is `check_plan`, so knitting the notebook without changing anything
is safe and does not start an expensive model.

Set `overwrite_existing = true` to recompute and replace existing raw
checkpoints. Otherwise completed contrasts are skipped.

## How to run every contrast on the cluster

From the repository root:

```bash
results/mcolon/20260727_gene14_clean/abundance_dact/submit_differential_abundance.sh
```

This submits one array task per gene-and-timepoint fit group, with at most three
groups running at once. Each task loads the cell-type CDS once and fits all
contrasts in its group. After every group finishes, a dependent job assembles
the final table. Cluster submissions recompute and overwrite raw checkpoints.

Cluster logs are written to:

```text
output/logs/
```

## Where to make changes

| If you want to change... | Edit... |
|---|---|
| Which groups are compared | `contrast_plan.R` |
| The minimum embryos required for a contrast | `contrast_plan.R` |
| The cell-type reliability rule | `cell_type_gate_policy.R` |
| How Hooke/Platt is fit | `differential_abundance.Rmd` |
