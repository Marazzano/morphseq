"""snip_qc report — exclusion reasons over time, dead vs not-dead. TERMINAL leaf (report_world.md).

The whole-experiment health view: for each qc_fail_reasons flag, what FRACTION of snips at each
time_index carry that reason. Death dominates exclusions (most of the plate is excluded because it
died), so a single reason-breakdown is misleading. ONE artifact, two side-by-side panels sharing a
y-axis and one legend (per report_world.md's snip_qc note) — a side-by-side comparison is the point,
so the two views belong in one PNG, not two separate files:
  left  — all_exclusions:    every snip, every reason, fraction over time_index;
  right — not_dead_exclusions: snips NOT flagged dead (viability_dead_flag / persistence_dead_flag)
                              dropped first, then the remaining reasons broken down. Surfaces
                              embryos filtered for non-death reasons (edge, focus, SA outlier)
                              that the all-exclusions panel buries under mortality.

Both panels use plot_labeled_series_over_time — a small NAMED set of categories (the 8 exclusion
reasons) each drawn as its own distinctly-colored, legended line, not the animal-curtain renderer A
(which overplots many same-colored strands where the ensemble shape is the point, not each line's
identity); the SAME reason gets the SAME color in both panels, which is what makes the comparison
legible. No new input: qc_fail_reasons + snip_id (which encodes time_index) are already in snip_qc's
own merged output — time_index is parsed from snip_id via parse_snip_id, not joined.

Consumed by nothing; imported by nothing but its own tasks.py subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.quality_control.snip_qc.contract import SNIP_QC_EXCLUSION_FLAGS
from data_pipeline.shared.identifiers import parse_snip_id
from data_pipeline.viz.reporting import plot_labeled_series_over_time

_DEATH_FLAGS = ("viability_dead_flag", "persistence_dead_flag")


def _reason_fractions_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form (reason, time_index, fraction): for each reason, the fraction of snips AT THAT
    time_index whose qc_fail_reasons contains it. Report-only derived value (never persisted)."""
    total_per_time = df.groupby("time_index").size()
    rows = []
    for reason in SNIP_QC_EXCLUSION_FLAGS:
        has_reason = df["qc_fail_reasons"].str.contains(reason, regex=False)
        count_per_time = df[has_reason].groupby("time_index").size()
        fraction = (count_per_time / total_per_time).fillna(0.0)
        for time_index, frac in fraction.items():
            rows.append({"reason": reason, "time_index": time_index, "fraction": frac})
    return pd.DataFrame(rows, columns=["reason", "time_index", "fraction"])


def build_snip_qc_report(
    *,
    snip_qc_path: Path,
    output_exclusion_reasons_png: Path,
) -> list[Path]:
    snip_qc_path = Path(snip_qc_path)
    qc = pd.read_parquet(snip_qc_path) if snip_qc_path.suffix == ".parquet" else pd.read_csv(snip_qc_path)
    qc = qc.copy()
    qc["qc_fail_reasons"] = qc["qc_fail_reasons"].fillna("")
    qc["time_index"] = qc["snip_id"].map(lambda s: parse_snip_id(s)[1])

    is_dead = qc["qc_fail_reasons"].apply(
        lambda reasons: any(flag in reasons for flag in _DEATH_FLAGS)
    )

    output_path = Path(output_exclusion_reasons_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = plot_labeled_series_over_time(
        pd.DataFrame(), "fraction",
        series_col="reason", time_col="time_index",
        title="snip_qc — exclusion reason fraction over time",
        ylabel="fraction of snips excluded for reason",
        output_path=output_path,
        panels=[
            ("all snips", _reason_fractions_over_time(qc)),
            ("not-dead snips only", _reason_fractions_over_time(qc[~is_dead])),
        ],
        smoothing_window=9,
    )
    return [combined]
