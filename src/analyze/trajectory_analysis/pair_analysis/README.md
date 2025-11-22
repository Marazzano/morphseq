# Pair Analysis Module

Reusable utilities for comparing trajectories across experimental groups (genotypes, pairs, treatments).

## Usage

```python
from src.analyze.trajectory_analysis.pair_analysis import (
    get_trajectories_for_group,
    compute_binned_mean,
    plot_genotypes_overlaid,
    plot_faceted_trajectories,
    GENOTYPE_COLORS,
    GENOTYPE_ORDER,
)
```

## Functions

### Data Utilities (`data_utils.py`)

**`get_trajectories_for_group(df, filter_dict, time_col, metric_col, embryo_id_col)`**

Extract trajectories for a specific group defined by filter conditions.

```python
trajectories, ids, n = get_trajectories_for_group(
    df,
    {'pair': 'cep290_pair_1', 'genotype': 'cep290_homozygous'},
    time_col='predicted_stage_hpf',
    metric_col='baseline_deviation_normalized'
)
```

**`compute_binned_mean(times, values, bin_width=0.5)`**

Compute binned mean of values over time for smoothing.

```python
bin_times, bin_means = compute_binned_mean(all_times, all_values, bin_width=0.5)
```

**`get_global_axis_ranges(all_trajectories, padding_fraction=0.1)`**

Compute global axis ranges from multiple trajectory lists for aligned plots.

### Plotting (`plotting.py`)

**`plot_genotypes_overlaid(df, groups, ...)`**

Create a 1xN plot with all genotypes overlaid on each subplot. Useful for comparing genotypes within each group.

```python
fig = plot_genotypes_overlaid(
    df,
    groups=['cep290_pair_1', 'cep290_pair_2', 'cep290_pair_3'],
    group_col='pair',
    genotype_col='genotype',
    output_path=Path('genotypes_by_pair.png'),
)
```

**`plot_faceted_trajectories(df, row_groups, col_groups, ...)`**

Create an NxM faceted grid. Rows and columns can be any grouping variables.

```python
fig = plot_faceted_trajectories(
    df,
    row_groups=['cep290_pair_1', 'cep290_pair_2', 'cep290_pair_3'],
    col_groups=['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous'],
    row_col='pair',
    col_col='genotype',
    output_path=Path('all_pairs_overview.png'),
)
```

## Default Configuration

```python
GENOTYPE_ORDER = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
GENOTYPE_COLORS = {
    'cep290_wildtype': '#2E7D32',      # Green
    'cep290_heterozygous': '#FFA500',  # Orange
    'cep290_homozygous': '#D32F2F',    # Red
}
```

Override by passing custom `genotype_order` and `genotype_colors` parameters.

## Example Script

See `results/mcolon/20251113_curvature_pair_analysis/analyze_pairs.py` for a complete example using these utilities.
