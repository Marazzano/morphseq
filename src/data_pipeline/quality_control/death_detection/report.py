"""death_detection report — mortality views at three grains. TERMINAL leaf (report_world.md).

Terminal report, THREE artifacts under one build, all recomputed from death_detection's own inputs
(death_detection_qc flags + fraction_alive trace + time_index):
  1. mortality_curtain          — per-EMBRYO: fraction_alive per animal, blue→red at called death;
  2. alive_embryos_experiment   — whole-EXPERIMENT: total alive embryos over time (survival curve);
  3. death_time_histogram       — distribution of called-death time_index.

(A per-WELL alive-count view is stubbed/retired — see build(); the derivation _alive_counts remains.)

Grain is per physical_embryo_id (one animal) — NOT embryo_id (that would split an animal across
channels). All derived values are REPORT-ONLY (never persisted to the flag table).

Consumed by nothing; imported by nothing but its own tasks.py subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.viz.reporting import (
    plot_grouped_traces,
    plot_line,
    plot_metric_histogram,
    plot_well_survival_over_time,
)


def _trace_with_flags(fraction_alive_csv: Path, death_detection_qc_csv: Path) -> pd.DataFrame:
    """fraction_alive trace (carries time_index) joined to the per-snip death flags on snip_id."""
    fa = pd.read_csv(fraction_alive_csv)
    flags = pd.read_csv(death_detection_qc_csv)
    return fa.merge(
        flags[["snip_id", "viability_dead_flag", "persistence_dead_flag"]],
        on="snip_id", how="left",
    )


# Viability cutoff used only to LOCATE the death inflection for the mark. This is a report-only
# visualization aid (where did the trace collapse?), not the pipeline's death gate.
_VIABILITY_CUTOFF = 0.5


def _called_death_marks(trace: pd.DataFrame) -> pd.DataFrame:
    """Report-only derived value: per physical_embryo_id flagged persistence-dead, the death
    INFLECTION — the first time_index where fraction_alive crosses below the viability cutoff.

    NOTE: persistence_dead_flag is BROADCAST to every snip of a dead animal (including its early
    healthy frames), so "earliest flagged time" would just be the animal's first frame (still
    alive). The visually meaningful called-death is where the trace actually collapses, so we find
    the first sub-cutoff crossing within each dead embryo's own trace.
    """
    dead_ids = trace.loc[trace["persistence_dead_flag"].astype(bool), "physical_embryo_id"].unique()
    dead = trace[trace["physical_embryo_id"].isin(dead_ids)]
    crossed = dead[dead["fraction_alive"] < _VIABILITY_CUTOFF]
    if crossed.empty:
        return crossed[["physical_embryo_id", "time_index", "fraction_alive"]]
    idx = crossed.groupby("physical_embryo_id")["time_index"].idxmin()
    return crossed.loc[idx, ["physical_embryo_id", "time_index", "fraction_alive"]]


def _alive_counts(trace: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """Per (well_id, time_index): count of embryos still alive = present that frame AND not yet past
    their called death (death_time is null, or death_time > t). Report-only derived value.

    Zero-fill is essential: when a well's LAST embryo dies there are no alive rows for that
    (well, time), so a naive groupby would drop the well from the plot instead of recording
    n_alive = 0. We build the full (well_id x time_index) grid and fill absent counts with 0, so an
    emptied well draws down to zero and stays there — the honest survival floor.
    """
    death_time = marks.set_index("physical_embryo_id")["time_index"] if len(marks) else pd.Series(dtype=float)
    present = trace[["well_id", "physical_embryo_id", "time_index"]].drop_duplicates()
    present["death_time"] = present["physical_embryo_id"].map(death_time)
    present["is_alive"] = present["death_time"].isna() | (present["time_index"] < present["death_time"])

    alive = (
        present[present["is_alive"]]
        .groupby(["well_id", "time_index"]).size().reset_index(name="n_alive")
    )
    # Full (well x time) grid over every well and every observed time_index, 0-filled.
    wells = present["well_id"].unique()
    times = sorted(present["time_index"].unique())
    grid = pd.MultiIndex.from_product([wells, times], names=["well_id", "time_index"]).to_frame(index=False)
    return grid.merge(alive, on=["well_id", "time_index"], how="left").fillna({"n_alive": 0})


def _experiment_survival(alive_per_well: pd.DataFrame) -> pd.DataFrame:
    """Whole-experiment total alive embryos over time_index (sum across wells)."""
    return alive_per_well.groupby("time_index")["n_alive"].sum().reset_index()


def _fraction_alive_per_well_time(trace: pd.DataFrame) -> pd.DataFrame:
    """Per (well_id, time_index): FRACTION of embryos alive = mean(~persistence_dead_flag).

    This is the REAL survival signal (persistence flag), distinct from the fraction_alive-cutoff
    marks used for the mortality curtain. One row per (well_id, time_index) with `frac_alive` in
    [0, 1], ready for the shared well x time heatmap helper.
    """
    t = trace[["well_id", "time_index", "persistence_dead_flag"]].copy()
    t["alive"] = ~t["persistence_dead_flag"].astype(bool)
    return (
        t.groupby(["well_id", "time_index"])["alive"]
        .mean()
        .reset_index(name="frac_alive")
    )


def build_death_detection_report(
    *,
    death_detection_qc_csv: Path,
    fraction_alive_csv: Path,
    output_experiment_png: Path,
    output_curtain_png: Path,
    output_death_time_png: Path,
    output_well_survival_png: Path,
) -> list[Path]:
    trace = _trace_with_flags(fraction_alive_csv, death_detection_qc_csv)
    marks = _called_death_marks(trace)  # one row per dead embryo: physical_embryo_id, time_index

    alive_per_well = _alive_counts(trace, marks)
    survival = _experiment_survival(alive_per_well)
    survival_overlay = (survival["time_index"], survival["n_alive"], "total alive (experiment)")

    for output_png in (
        output_experiment_png, output_curtain_png, output_death_time_png, output_well_survival_png,
    ):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    # 1. Whole-experiment survival curve (standalone) — the clearest entry point into this report.
    experiment = plot_line(
        survival["time_index"], survival["n_alive"],
        title="death_detection — total alive embryos over time (experiment)",
        output_path=output_experiment_png,
        xlabel="time_index", ylabel="total alive embryos",
    )

    # 2. Per-embryo mortality curtain, with the experiment survival curve overlaid on a secondary
    #    right axis scaled 0 -> total embryo count (the curtain's own y is fraction_alive, 0-1).
    curtain = plot_grouped_traces(
        trace, "fraction_alive",
        group_col="physical_embryo_id", time_col="time_index",
        title="death_detection — mortality curtain (fraction_alive per animal; red after called death)",
        ylabel="fraction_alive",
        output_path=output_curtain_png,
        alpha=0.3, linewidth=1.6, smoothing_window=9,
        event_times=marks, event_time_col="time_index",
        overlay=survival_overlay,
        overlay_on_secondary_axis=True,
        overlay_ylabel="total alive embryos (experiment)",
    )

    # 3. Per-well alive-embryo count over time — STUB (retired). The per-well lines overplot onto a
    #    handful of integer levels (3/2/1/0) so individual wells aren't distinguishable, and smoothing
    #    a discrete count introduces artifacts. The experiment survival curve (#1) and the curtain (#2)
    #    already carry the mortality story. Kept as a stub: _alive_counts(...) is the derivation if a
    #    future per-well view (e.g. a small-multiple grid, or a well x time heatmap) is wanted.

    # 4. Death-time distribution.
    death_time_hist = plot_metric_histogram(
        marks["time_index"] if len(marks) else pd.Series([], dtype=float),
        cutoff=float("-inf"),  # pure distribution, no pass/fail split
        fail_direction="below",
        title="death_detection — called-death time distribution",
        output_path=output_death_time_png,
        xlabel="called-death time_index",
    )

    # 5. Per-well fraction-alive over time_index — the REAL survival heatmap (persistence flag),
    #    via the shared well x time helper (same plot as registry proxy / analysis_ready stage view).
    well_survival = plot_well_survival_over_time(
        _fraction_alive_per_well_time(trace),
        time_col="time_index",
        value_col="frac_alive",
        value_label="fraction alive",
        title="death_detection — fraction alive per well over time (persistence flag)",
        output_path=output_well_survival_png,
        vmin=0.0,
        vmax=1.0,
    )
    return [experiment, curtain, death_time_hist, well_survival]
