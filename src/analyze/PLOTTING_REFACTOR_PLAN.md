# Plotting API Refactor Plan (Feature-First Naming)

This plan defines the new plotting API names, signatures, migration path, and
why the changes improve clarity and sustainability.

## Goals

- Use bioinformatics-standard vocabulary ("feature") instead of data-structure
  terms ("time_series").
- Keep generic plotting in `src/analyze/viz/plotting/` and trajectory-specific
  plotting in `src/analyze/trajectory_analysis/viz/plotting/`.
- Preserve backward compatibility via thin wrappers and deprecation warnings.

## New Canonical API (Generic)

### Single-panel

```python
def plot_feature_over_time(
    df: pd.DataFrame,
    feature: str = "metric_value",
    time_col: str = "hpf",
    id_col: str = "id",
    color_by: str = "group",
    show_individual: bool = True,
    trend_method: Optional[str] = "mean",
    show_trend: bool = True,
    show_sd_band: bool = False,
    smooth_window: Optional[int] = 5,
    alpha_individual: float = 0.3,
    alpha_trend: float = 0.8,
    linewidth_individual: float = 0.8,
    linewidth_trend: float = 2.5,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 100,
    palette: Optional[str] = None,
) -> plt.Figure:
    ...
```

### Faceted

```python
def plot_feature_over_time_faceted(
    df: pd.DataFrame,
    feature: str = "metric_value",
    time_col: str = "hpf",
    id_col: str = "id",
    color_by: str = "group",
    facet_by: str = "experiment",
    show_individual: bool = True,
    trend_method: Optional[str] = "mean",
    show_trend: bool = True,
    show_sd_band: bool = False,
    smooth_window: Optional[int] = None,
    alpha_individual: float = 0.3,
    alpha_trend: float = 0.9,
    linewidth_individual: float = 0.8,
    linewidth_trend: float = 2.5,
    figsize_per_panel: Tuple[float, float] = (6, 5),
    facet_ncols: Optional[int] = None,
    facet_sharex: bool = True,
    facet_sharey: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 100,
    palette: Optional[str] = None,
) -> plt.Figure:
    ...
```

### Optional: Multi-feature (generic wrapper)

```python
def plot_features_over_time(
    df: pd.DataFrame,
    features: List[str],
    time_col: str = "hpf",
    id_col: str = "id",
    color_by: str = "group",
    **kwargs,
) -> List[plt.Figure]:
    ...
```

## Backward Compatibility (Legacy Wrappers)

These wrappers keep old defaults and column names working:

```python
def plot_embryos_metric_over_time(
    df: pd.DataFrame,
    metric: str = "normalized_baseline_deviation",
    time_col: str = "predicted_stage_hpf",
    embryo_col: str = "embryo_id",
    color_by: str = "genotype",
    **kwargs,
) -> plt.Figure:
    ...

def plot_embryos_metric_over_time_faceted(
    df: pd.DataFrame,
    metric: str = "normalized_baseline_deviation",
    time_col: str = "predicted_stage_hpf",
    embryo_col: str = "embryo_id",
    color_by: str = "genotype",
    facet_by: str = "experiment_id",
    **kwargs,
) -> plt.Figure:
    ...
```

## Migration

### Old -> New

| Old Name | New Name |
|---|---|
| `plot_time_series_by_group` | `plot_feature_over_time` |
| `plot_time_series_faceted` | `plot_feature_over_time_faceted` |
| `plot_embryos_metric_over_time` | (wrapper -> `plot_feature_over_time`) |
| `plot_embryos_metric_over_time_faceted` | (wrapper -> `plotdo we currenlt have a handler that abstractifies and is defensive about facetting, how i imagine it is that _feature_over_time_faceted`) |

### Required Action

- New code should import from `src.analyze.viz.plotting`.
- Old names remain usable but emit `DeprecationWarning`.

## Why This Is Better

- **Clear intent:** "feature" is standard in bioinformatics tooling (Seurat,
  Scanpy). Users know immediately what is plotted.
- **Consistent modifiers:** core action is first, layout modifier last
  (`..._faceted`).
- **Scalable naming:** easy to add `plot_features_over_time`, or
  `plot_feature_distribution` later without rethinking the scheme.
- **Defensive API:** legacy wrappers keep existing notebooks functional while
  guiding users to the new names.

## Naming Convention (Required)

All new plotting functions should follow this pattern:

- **plot + feature + action**: e.g., `plot_feature_over_time`
- **layout modifier last**: e.g., `plot_feature_over_time_faceted`
- **plural for multi-feature**: e.g., `plot_features_over_time`

This makes autocomplete and documentation predictable and keeps a consistent
verb-noun structure across the plotting package.

## Adding New Plotting Functions

To add a new generic plotting function:

1) Place it in `src/analyze/viz/plotting/` if it is domain-agnostic.
2) Use the **feature-first** naming convention.
3) Provide explicit column parameters (`feature`, `time_col`, `id_col`,
   `color_by`) so the function is flexible across datasets.
4) If a legacy version exists, add a wrapper in the same module that maps
   old defaults to the new signature and emits a `DeprecationWarning`.

To add a trajectory-specific plot:

1) Place it in `src/analyze/trajectory_analysis/viz/plotting/`.
2) Keep the naming consistent, but allow domain-specific defaults
   (e.g., genotype colors, phenotype orders).

## Public Faceting Renderer (Proposed)

We already have an internal IR + renderer used by faceted plots:

- `TraceData`, `SubplotData`, `FigureData` (IR dataclasses)
- `_render_figure(...)` in `src/analyze/trajectory_analysis/viz/plotting/faceted/shared.py`

To improve separation of concerns and reduce duplication, add a **public**
entrypoint that renders a faceted plot from precomputed traces:

```python
def render_faceted_figure(
    figure_data: FigureData,
    backend: str = "plotly",
    output_path: Optional[Path] = None,
) -> Any:
    ...
```

This allows callers to build `FigureData` elsewhere (e.g., custom analysis)
and reuse the same rendering logic without reimplementing faceting.

Expose it via:
- `src/analyze/trajectory_analysis/viz/plotting/faceted/__init__.py`

## Why This Abstraction Level Is Correct

- **Feature-first** is domain language: users think in features (curvature,
  PCA, probability) rather than data structures.
- **Column-parameterized signatures** make the functions reusable across
  experiments without forcing a rigid schema.
- **Separation of generic vs trajectory-specific** avoids accidental coupling
  while keeping domain tools discoverable.

## Scope Boundaries

- Generic plotting lives in `src/analyze/viz/plotting/`.
- Trajectory-analysis-specific plotting remains in
  `src/analyze/trajectory_analysis/viz/plotting/`.
