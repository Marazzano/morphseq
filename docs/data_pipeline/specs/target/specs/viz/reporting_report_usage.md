# Reporting / `report.py` Usage Guide — worked examples

**Companion to `report_world.md`** (the doctrine — read that first for the *rules*). This doc is
the *how*, grounded in the 5 tier-1 reports actually built and wired this session
(`physical_embryo_registry`, `mask_geometry`, `death_detection_qc`, `surface_area_qc`, `snip_qc`).
Every example below is real code, not illustrative pseudocode — copy the pattern, don't reinvent it.

---

## The seam: review script → real `report.py`

Every report starts as a review-time script under
`results/mcolon/20260702_qc_reporting_v1/report_scripts/<product>/report.py`, driven by
`report_scripts/_loaders.py::merged(stage, product, suffix, ext="csv")` (globs per-well shards off
disk, ignores the registry). It moves to `src/data_pipeline/<stage>/<product>/report.py` by
**one mechanical change**: replace the `merged(...)` call with a plain path argument the caller
(the Snakemake rule, via `tasks.py`) hands in. Nothing else in the body changes.

```python
# review script (report_scripts/death_detection/report.py)
from .._loaders import merged

def build(output_dir: Path) -> list[Path]:
    fa = merged("feature_extraction", "fraction_alive", "fraction_alive")
    flags = merged("quality_control", "death_detection", "death_detection_qc")
    ...
```

```python
# real report.py (src/data_pipeline/quality_control/death_detection/report.py)
def build_death_detection_report(
    *,
    death_detection_qc_csv: Path,
    fraction_alive_csv: Path,
    output_experiment_png: Path,
    output_curtain_png: Path,
    output_death_time_png: Path,
) -> list[Path]:
    fa = pd.read_csv(fraction_alive_csv)
    flags = pd.read_csv(death_detection_qc_csv)
    ...
```

Naming convention: `build(output_dir)` (review) → `build_<step>_report(*, <input>_csv/parquet,
output_<artifact>_png, ...)` (real) — explicit path args, one per input and per output artifact,
matching `entrypoint.py`'s style (`run_<step>(*, ...)`). Keep the review-harness copy in sync after
editing the real one (or vice versa) — they should only ever differ in the loader seam.

**Gallery images need an `output_root`-relative path resolver**, not the review harness's
`snip_image_paths()` (which is loader-only glue). Reuse this exact pattern (lifted from
`unet_snip/entrypoint.py::_resolve_snip_paths`):

```python
def _resolve_snip_image_paths(snip_inventory: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    def _abs(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        p = Path(str(value))
        return str(p if p.is_absolute() else (output_root / p))
    resolved = snip_inventory[["snip_id", "processed_snip_path"]].copy()
    resolved["resolved_image_path"] = resolved["processed_snip_path"].map(_abs)
    return resolved[["snip_id", "resolved_image_path"]]
```
(See `mask_geometry/report.py` and `surface_area_qc/report.py` for the two real call sites.)

---

## Choosing a renderer — worked decision log

| Shape of your view | Renderer | Real example |
|---|---|---|
| One line per animal/well, ensemble shape is the point (many low-alpha strands) | `plot_grouped_traces` (A) | `death_detection`'s mortality curtain — `group_col="physical_embryo_id"` |
| A SMALL, NAMED set of categories, each line's IDENTITY matters | `plot_labeled_series_over_time` (new, this session) | `snip_qc`'s 8 exclusion reasons |
| A per-group row COUNT distribution (not a measured value) | `plot_count_per_group` (B) | `physical_embryo_registry`'s embryos-per-well |
| Raw-metric histogram, pass/fail split by a scalar cutoff | `plot_metric_histogram` (C) | any stock QC spec |
| One gridded PNG, one panel per feature column | `plot_histogram_grid` (D) | `mask_geometry`'s feature grid |
| Cutoff-relative or value-quartile image gallery | `render_quartile_gallery*` (E) | `mask_geometry`'s area_um2 gallery |
| Metric vs. a per-row reference BAND (not a fixed scalar) | `plot_metric_vs_reference` + `render_quartile_gallery_vs_band` (F) | `surface_area_qc` vs. stage-interpolated band |
| A single aggregate number over time | `plot_line` | `death_detection`'s whole-experiment survival curve |
| A per-well scalar laid out in physical plate geometry | `plot_plate_heatmap` | `physical_embryo_registry`'s plate view |

**The mistake to avoid** (made and caught this session): `plot_grouped_traces` (renderer A) is
built for *many same-colored, overplotted* strands — using it for 8 *named* exclusion-reason
categories produced 8 indistinguishable same-colored lines with no legend. If you need to tell
lines apart by what they ARE (not just see the ensemble shape), you need
`plot_labeled_series_over_time`, not A. When in doubt: "would a human need a legend to make sense
of this?" — if yes, renderer A is the wrong tool.

---

## `plot_labeled_series_over_time` — the newest renderer

Added this session for "a small, named vocabulary of categories over time" — the shape renderer A,
B, C, D, E, F did not cover. Lives in `viz/reporting.py` alongside the others.

```python
CATEGORICAL_COLORS: tuple[str, ...] = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)  # fixed order, never cycled — from the dataviz skill's palette.md categorical theme

def plot_labeled_series_over_time(
    df, value_col, *, series_col, time_col="time_index", title, output_path,
    ylabel=None, panels=None, smoothing_window=1,
) -> Path: ...
```

- Colors are assigned in **fixed categorical order** (never cycled/reassigned) — see the dataviz
  skill (`Skill: dataviz`) for why: identity must follow the entity, not its rank.
- **Every line is legended** — this renderer is for identity, so a legend is not optional.
- `smoothing_window > 1` applies the same centered rolling-mean smoothing as `plot_grouped_traces`
  — a per-time_index fraction over a modest cohort is noisy; the trend is the point, not every
  jagged tick. `snip_qc` uses `smoothing_window=9`.
- **`panels=[(panel_title, panel_df), ...]`** renders N related views as side-by-side subplots
  sharing ONE y-axis and ONE legend, in ONE PNG — use this whenever the report's whole point is a
  *comparison* between two slices of the same data (e.g. all-snips vs. not-dead-only). Two
  standalone PNGs are strictly worse for a comparison than one shared-axis figure: colors are
  guaranteed identical across panels (computed once, from the union of all panels' categories,
  before any panel is drawn), so "is the pink line higher on the left or the right" is a single
  eye movement, not a mental cross-reference between two files.

```python
# snip_qc/report.py — the real call, two panels, one artifact
combined = plot_labeled_series_over_time(
    pd.DataFrame(), "fraction",             # ignored when panels= is given
    series_col="reason", time_col="time_index",
    title="snip_qc — exclusion reason fraction over time",
    ylabel="fraction of snips excluded for reason",
    output_path=output_exclusion_reasons_png,
    panels=[
        ("all snips", _reason_fractions_over_time(qc)),
        ("not-dead snips only", _reason_fractions_over_time(qc[~is_dead])),
    ],
    smoothing_window=9,
)
return [combined]  # ONE artifact, not two
```

**Gotcha (hit and fixed this session): `fig.legend(bbox_to_anchor=...)` + `fig.tight_layout()`
silently CLIPS the legend at save time.** `tight_layout` recomputes the axes layout without
accounting for a figure-level legend placed outside the axes rect. Fix: always pass
`bbox_inches="tight"` to `fig.savefig(...)` when using a `fig.legend(...)` (not `ax.legend(...)`)
positioned via `bbox_to_anchor`. `plot_labeled_series_over_time` already does this — if you write a
new multi-panel renderer with an external legend, copy this, don't rediscover it.

---

## Deriving values a report needs without a new input (the recurring trick)

**`time_index` is not always a column — parse it from `snip_id` instead of joining.**
`snip_qc`'s own contract (`SNIP_QC_TABLE_COLUMNS`) does not carry `time_index` — only the identity
spine (`experiment_id, well_id, physical_embryo_id, embryo_id, snip_id`) plus `use_snip` /
`qc_fail_reasons`. Rather than adding a new join (which would violate report_world.md's scope
rule — a report may only use what its step already consumes), `snip_id` itself encodes
`time_index` and is already there:

```python
from data_pipeline.shared.identifiers import parse_snip_id
qc["time_index"] = qc["snip_id"].map(lambda s: parse_snip_id(s)[1])
```

This generalizes: before reaching for a join or a new input, check whether the value is already
*encodable* in a column you have (an ID that embeds it) via an existing parser in
`shared/identifiers/`. A report may derive freely from its own inputs (report_world.md: "`report.py`
may compute report-only values") — a derivation is not a scope violation; a new upstream read is.

**Dead/not-dead split from a pipe-joined reasons string, no join:**

```python
_DEATH_FLAGS = ("viability_dead_flag", "persistence_dead_flag")
is_dead = qc["qc_fail_reasons"].apply(lambda reasons: any(f in reasons for f in _DEATH_FLAGS))
not_dead_view = qc[~is_dead]
```
`qc_fail_reasons` is a pipe-joined string of fired flag-column names (`"edge_flag|focus_flag"`);
substring containment on the raw string is enough — no need to explode/split it first.

---

## Wiring checklist (concrete, not aspirational — do these 5 things in this order)

1. **`PIPELINE_STEPS` row** in `orchestration/paths.py`, named `<step>_report`,
   `product_dir: <product>/report`, `fanout: EXPERIMENT`,
   `execution: EXECUTION_PER_WELL` (yes, even though it's not per-well — see the tech-debt note
   below), one dict entry per artifact filename (`{experiment_id}_whatever.png`).
2. **`report.py`** in the product folder (`src/data_pipeline/<stage>/<product>/report.py`) — ported
   from the review script per the seam above.
3. **`tasks.py`**: a `cmd_<step>_report(args)` function (thin — reads `args.*`, calls
   `build_<step>_report(...)`) + an `add_parser("<step>-report")` block with one `--flag` per
   input/output path, right after the step's other subcommands.
4. **`.smk` rule** in the product's existing rules file: `input:` = the step's merged artifact(s)
   via `rule_artifact(..., path_mode=PATH_MODE_MERGED)` (plain, not wrapped in `ancient()` — see
   the "don't do this" note below), `output:` = the report artifacts, `shell:` = the new
   `tasks.py` verb.
5. Nothing else. The `reports` aggregate target in the Snakefile
   (`_reports_targets` + `rule reports`) is **generic** — it walks every `PIPELINE_STEPS` key
   ending in `_report` automatically. A new report step needs NO edit there.

Verify with the repo-wide guard test (`tests/data_pipeline/viz/test_reports_are_terminal.py`) —
it asserts (a) no module imports a `*/report.py`, (b) no non-report rule's `input:` names a
`*_report` artifact. Both assertions are structural, not per-product; a new report just needs to
follow the 5 steps above to pass automatically.

---

## Two mistakes made (and reverted) this session — don't repeat them

**1. Do not wrap a report's Snakemake `input:` in `ancient()` to avoid triggering a rebuild.**
The instinct: "I don't want `snakemake reports` to recompute the whole per-well fan just to draw a
picture." `ancient()` looks like the fix (it ignores an input's timestamp) but is wrong: it does
nothing when the merged file is *absent* (still cascades into the full per-well fan on a fresh
build) and, worse, makes an EXISTING report silently stale — regenerate the merged table with a bug
fix, and `ancient()` tells Snakemake the report is still up to date, so it never rebuilds. A report
should depend on its merged input the ordinary way; if you don't want a `reports` invocation to
recompute a stale merge, that's an invocation-time concern (run reports after a normal build, so
the merge is already current), not something to encode by lying about timestamps.

**2. Running a report target under a `--configfile` SMOKE overlay (e.g.
`config_smoke_through_line_20250912_B01.yaml`) is dangerous if a merged experiment-level file
already exists from a full run.** That smoke config restricts BOTH the well set (`target_wells`)
AND the timepoint cap (`smoke_max_time_indices: 1`). Snakemake's provenance trigger will happily
recompute the restricted well's entire per-well chain (frame_masks → registry → snip_inventory →
mask_geometry → ...) under that 1-timepoint cap, **overwriting the good full-timepoint per-well
shard on disk**, and then the merge rule re-derives the experiment-level merged file from the now-
degraded shard — silently shrinking real data. If you need to smoke-test a report body cheaply,
call the `tasks.py` subcommand directly against a hand-built or scratch-merged CSV (see
`report_scripts/_loaders.py::merged()` for the pattern) — never invoke the real Snakefile with a
narrower config than the one that produced the on-disk data you care about.

---

## Recompute-without-Snakemake escape hatch (for iterating on a report body quickly)

Once a step's merge already exists on disk, you do not need Snakemake to re-render a report while
you're tuning its plot code — call `tasks.py` directly:

```bash
PYTHONPATH=src conda run -n segmentation_grounded_sam --no-capture-output \
  python -m data_pipeline.pipeline_orchestrator.tasks snip-qc-report \
    --snip-qc-path data_pipeline_output/quality_control/20250912/snip_qc/20250912_snip_qc.parquet \
    --output-exclusion-reasons-png /tmp/scratch_preview.png
```

This is exactly what `report_world.md`'s DAG-wiring section calls the `tasks.py` "thin CLI adapter"
— it's also just a normal CLI tool you can run standalone. Iterate here; only run the real
`.smk` rule (or `snakemake reports`) once the body is right, and only against a config that matches
what actually produced the on-disk merged data.
