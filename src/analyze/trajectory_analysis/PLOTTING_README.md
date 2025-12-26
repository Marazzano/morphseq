# Trajectory Analysis Plotting Guide

Comprehensive plotting utilities for comparing trajectories across experimental groups. Supports generic faceted plots (any column grouping) and pair-specific convenience functions.

## Quick Start

### Level 2: Pair-Specific (Recommended for pair analysis)

```python
from src.analyze.trajectory_analysis import (
    plot_pairs_overview,
    plot_genotypes_by_pair,
    plot_single_genotype_across_pairs,
)

# NxM grid: pairs (rows) × genotypes (columns)
fig = plot_pairs_overview(df, backend='plotly')

# 1xN plot: pairs (columns) with genotypes overlaid
fig = plot_genotypes_by_pair(df, backend='plotly')

# 1xN plot: single genotype across all pairs
fig = plot_single_genotype_across_pairs(
    df,
    genotype='cep290_homozygous',
    backend='plotly'
)
```

### Level 1: Generic Faceting (For custom groupings)

```python
from src.analyze.trajectory_analysis import plot_trajectories_faceted

# Group by ANY columns
fig = plot_trajectories_faceted(
    df,
    row_by='experiment_id',      # Row facets
    col_by='genotype',           # Column facets
    overlay='condition',         # Overlay groups
    color_by='genotype',         # Color mapping
    backend='plotly'
)
```

## Architecture: Two-Level System

```
┌─────────────────────────────────────────────────────────────┐
│  Level 2: pair_analysis/plotting.py (Pair-Specific)         │
│  - plot_pairs_overview()                                    │
│  - plot_genotypes_by_pair()                                 │
│  - plot_single_genotype_across_pairs()                      │
│  - Auto: {genotype}_unknown_pair if pair column missing     │
└───────────────────────┬─────────────────────────────────────┘
                        │ calls
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 1: faceted_plotting.py (Generic)                     │
│  - plot_trajectories_faceted(row_by, col_by, overlay, ...) │
│  - Works with ANY column names                              │
│  - Both Plotly & Matplotlib backends                        │
└───────────────────────┬─────────────────────────────────────┘
                        │ uses
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  genotype_styling   _plotly()      _matplotlib()
   - suffix colors   - interactive  - static PNG
   - auto-detect     - hover info   - fast export
```

## Level 2: Pair-Specific Functions

### `plot_pairs_overview()`

Creates **NxM grid**: pairs (rows) × genotypes (columns). Each cell is one pair × one genotype.

**Signature:**
```python
fig = plot_pairs_overview(
    df: pd.DataFrame,
    pairs: Optional[List[str]] = None,           # Auto-detect if None
    genotypes: Optional[List[str]] = None,       # Auto-detect if None
    pair_col: str = 'pair',
    genotype_col: str = 'genotype',
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    backend: str = 'plotly',                     # 'plotly', 'matplotlib', 'both'
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    **kwargs
) -> Figure
```

**Example:**
```python
# Basic usage - all pairs and genotypes
fig = plot_pairs_overview(df[df['gene'] == 'cep290'])

# Specific subset
fig = plot_pairs_overview(
    df,
    pairs=['cep290_pair_1', 'cep290_pair_2'],
    genotypes=['cep290_wildtype', 'cep290_homozygous'],
    backend='both',  # Save both PNG and HTML
    output_path=Path('output/pairs_overview')
)
```

### `plot_genotypes_by_pair()`

Creates **1xN plot**: pairs (columns) with genotypes overlaid within each pair.

**Signature:**
```python
fig = plot_genotypes_by_pair(
    df: pd.DataFrame,
    pairs: Optional[List[str]] = None,
    pair_col: str = 'pair',
    genotype_col: str = 'genotype',
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    **kwargs
) -> Figure
```

**Example:**
```python
# All pairs with all genotypes overlaid
fig = plot_genotypes_by_pair(df[df['gene'] == 'b9d2'], backend='plotly')

# Just two pairs
fig = plot_genotypes_by_pair(
    df,
    pairs=['b9d2_pair_1', 'b9d2_pair_3'],
    backend='both',
    output_path=Path('output/genotypes_by_pair')
)
```

### `plot_single_genotype_across_pairs()`

Creates **1xN plot**: single genotype across multiple pairs. No overlay.

**Signature:**
```python
fig = plot_single_genotype_across_pairs(
    df: pd.DataFrame,
    genotype: str,                               # Required: which genotype to plot
    pairs: Optional[List[str]] = None,
    pair_col: str = 'pair',
    genotype_col: str = 'genotype',
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    **kwargs
) -> Figure
```

**Example:**
```python
# Homozygous genotype across all pairs
fig = plot_single_genotype_across_pairs(
    df[df['gene'] == 'cep290'],
    genotype='cep290_homozygous',
    backend='plotly'
)

# Wildtype across specific pairs
fig = plot_single_genotype_across_pairs(
    df,
    genotype='b9d2_wildtype',
    pairs=['b9d2_pair_1', 'b9d2_pair_2', 'b9d2_pair_4'],
    backend='matplotlib',  # Static PNG only
    output_path=Path('output/wildtype_across_pairs.png')
)
```

## Level 1: Generic Faceting

### `plot_trajectories_faceted()`

Universal faceted plotting with flexible grouping by any column(s).

**Signature:**
```python
fig = plot_trajectories_faceted(
    df: pd.DataFrame,
    # Data columns
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',                 # What defines individual lines

    # Faceting parameters
    row_by: Optional[str] = None,               # Row facets
    col_by: Optional[str] = None,               # Column facets
    overlay: Optional[str] = None,              # Overlay within subplots
    color_by: Optional[str] = None,             # Color mapping (auto for genotypes)
    facet_order: Optional[Dict[str, List]] = None,  # Custom order

    # Sizing
    height_per_row: int = 350,
    width_per_col: int = 400,

    # Backend and output
    backend: str = 'plotly',                    # 'plotly', 'matplotlib', 'both'
    output_path: Optional[Path] = None,

    # Labels and styling
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    y_label: str = 'Value',

    # Data processing
    bin_width: float = 0.5,                     # Mean trajectory binning
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
) -> Figure | Dict[str, Figure]
```

**Example: Group by Experiment × Genotype**
```python
fig = plot_trajectories_faceted(
    df,
    row_by='experiment_id',
    col_by='genotype',
    color_by='genotype',  # Auto-colors by suffix
    backend='plotly'
)
```

**Example: Custom Column Order**
```python
fig = plot_trajectories_faceted(
    df,
    col_by='treatment',
    overlay='cell_type',
    color_by='cell_type',
    facet_order={
        'treatment': ['control', 'treated_low', 'treated_high'],
        'cell_type': ['epiblast', 'mesoderm', 'ectoderm']
    },
    backend='plotly'
)
```

**Example: Dual Backend Output**
```python
result = plot_trajectories_faceted(
    df,
    row_by='gene',
    col_by='condition',
    overlay='genotype',
    color_by='genotype',
    backend='both',
    output_path=Path('output/analysis')
)

# Returns dict
plotly_fig = result['plotly']   # Interactive HTML
mpl_fig = result['matplotlib']  # Static PNG
```

## Facet Parameter Guide

| Parameter | Type | Purpose | Example |
|-----------|------|---------|---------|
| `row_by` | str | Create subplot rows | `'experiment_id'` |
| `col_by` | str | Create subplot columns | `'genotype'` |
| `overlay` | str | Draw multiple groups in same subplot | `'treatment'` |
| `color_by` | str | Determine line colors | `'genotype'` |
| `line_by` | str | Define individual trajectories | `'embryo_id'` |
| `x_col` | str | X-axis data | `'predicted_stage_hpf'` |
| `y_col` | str | Y-axis data | `'baseline_deviation_normalized'` |

**Validation Rules:**
- `row_by`, `col_by`, `overlay` should be **mutually unique** (no duplicates among these three)
- If duplicated → warning issued (but still works)
- `color_by` can match any of the facet params (e.g., color by genotype while faceting by genotype)
- Any parameter can be `None` (e.g., `row_by=None, col_by='pair'` → single row)

## Genotype Styling

### Automatic Color Mapping

Colors are based on **genotype suffix**, not gene prefix:
- **Wildtype** → `#2E7D32` (Green)
- **Heterozygous** → `#FFA500` (Orange)
- **Homozygous** → `#D32F2F` (Red)

**Examples:**
```python
# All use green (wildtype)
'cep290_wildtype'
'b9d2_wildtype'
'tmem67_wildtype'

# All use red (homozygous)
'cep290_homozygous'
'b9d2_homo'  # Abbreviation also works
'tmem67_homozygous'
```

### Manual Color Override

```python
from src.analyze.trajectory_analysis import plot_trajectories_faceted

custom_colors = {
    'condition_1': '#FF6B6B',
    'condition_2': '#4ECDC4',
    'condition_3': '#45B7D1',
}

fig = plot_trajectories_faceted(
    df,
    col_by='condition',
    color_by='condition',
    # Note: Custom coloring not yet in API, but suffix-based auto-coloring works
)
```

### Genotype Suffix Utility Functions

```python
from src.analyze.trajectory_analysis import (
    extract_genotype_suffix,
    extract_genotype_prefix,
    get_color_for_genotype,
    sort_genotypes_by_suffix,
    build_genotype_style_config,
)

# Extract components
suffix = extract_genotype_suffix('cep290_homozygous')  # 'homozygous'
prefix = extract_genotype_prefix('cep290_homozygous')  # 'cep290'

# Get color for any genotype
color = get_color_for_genotype('b9d2_het')  # '#FFA500' (orange)

# Sort genotypes by standard order
genotypes = ['b9d2_homo', 'b9d2_wt', 'b9d2_het']
sorted_genos = sort_genotypes_by_suffix(genotypes)
# Returns: ['b9d2_wt', 'b9d2_het', 'b9d2_homo']

# Build complete style config
config = build_genotype_style_config(genotypes)
# Returns: {
#   'order': [...],
#   'colors': {'b9d2_wt': '#2E7D32', ...},
#   'suffix_colors': {...},
#   'suffix_order': [...]
# }
```

## Backend Comparison

| Feature | Plotly | Matplotlib |
|---------|--------|-----------|
| **Output** | Interactive HTML | Static PNG |
| **Hover Info** | ✅ Embryo IDs | ❌ None |
| **Zoom/Pan** | ✅ Interactive | ❌ Static |
| **Export** | Easy (drag to save) | Automatic .png |
| **File Size** | Larger (interactive) | Smaller (raster) |
| **Best For** | Exploration | Papers, reports |

**Usage:**
```python
# Interactive exploration
fig = plot_pairs_overview(df, backend='plotly')
fig.show()  # In Jupyter

# Static output for publication
fig = plot_pairs_overview(df, backend='matplotlib', output_path='paper/fig1.png')

# Generate both
result = plot_pairs_overview(df, backend='both', output_path='output/analysis')
```

## Missing Pair Column Handling

If `pair` column is missing but `genotype` exists, automatically creates fallback pairs:

```python
df_no_pair = df.drop('pair', axis=1)

# Automatically creates: pair = '{genotype}_unknown_pair'
fig = plot_pairs_overview(df_no_pair)
# Pairs become: 'cep290_wildtype_unknown_pair', 'cep290_homo_unknown_pair', etc.
```

## Common Patterns

### Pattern 1: Compare Two Genes

```python
# Separate plots for each gene
for gene in ['cep290', 'b9d2']:
    df_gene = df[df['gene'] == gene]
    fig = plot_pairs_overview(df_gene, backend='plotly')
    fig.show()
```

### Pattern 2: Focus on Single Genotype

```python
# Wildtype across all pairs and genes
fig = plot_single_genotype_across_pairs(
    df[df['genotype'].str.contains('wildtype')],
    genotype='cep290_wildtype',  # Just pick one
    backend='plotly'
)
```

### Pattern 3: Time Course with Multiple Conditions

```python
fig = plot_trajectories_faceted(
    df,
    row_by='timepoint',        # Different development stages
    col_by='treatment',        # Different conditions
    overlay='genotype',        # Multiple genotypes overlaid
    color_by='genotype',       # Color by genotype
    backend='plotly'
)
```

### Pattern 4: Export Publication Figures

```python
from pathlib import Path

output_dir = Path('figures/manuscript')
output_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Overview grid
plot_pairs_overview(
    df[df['gene'] == 'cep290'],
    backend='matplotlib',
    output_path=output_dir / 'fig1_pairs_overview.png'
)

# Figure 2: Genotypes by pair
plot_genotypes_by_pair(
    df[df['gene'] == 'b9d2'],
    backend='matplotlib',
    output_path=output_dir / 'fig2_genotypes_by_pair.png'
)

# Supplementary: Interactive HTML for reviewers
plot_pairs_overview(
    df,
    backend='plotly',
    output_path=output_dir / 'supp_interactive.html'
)
```

## Troubleshooting

### "No data" in subplot
- Check filter conditions are correct
- Verify column names match your DataFrame
- Use `df.columns` to confirm available columns

### Colors not showing
- Ensure `color_by` column contains valid data
- Check for NaN values in the coloring column
- For genotype coloring, verify genotype names follow pattern `*_wildtype`, `*_het`, or `*_homo`

### Legend too crowded (Plotly)
- Legends only show first instance of each color (uses `legendgroup`)
- This is intentional to reduce clutter

### Different x/y limits per subplot
- All subplots share global x/y limits for fair comparison
- This prevents outliers from squishing other data

## API Deprecation Notes

Old function names still work (backward compatible):
- `plot_genotypes_overlaid()` → use `plot_genotypes_by_pair()`
- `plot_all_pairs_overview()` → use `plot_pairs_overview()`
- `plot_homozygous_across_pairs()` → use `plot_single_genotype_across_pairs()`
