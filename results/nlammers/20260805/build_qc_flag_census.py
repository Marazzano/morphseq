#!/usr/bin/env python
"""Audit exclusion flags in completed Keyence/YX1 merged snip-QC tables.

This is an observational audit only. It never changes pipeline products. A dataset enters the
snapshot when its experiment-level ``*_snip_qc.parquet`` exists and can be read.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
)
MANIFEST = (
    REPO_ROOT
    / "src/data_pipeline/pipeline_orchestrator/manifests/front_half_archive.txt"
)
DEFAULT_OUTPUT_DIR = HERE / "corpus_census_data/qc_flag_census"

KNOWN_FLAGS = (
    "viability_dead_flag",
    "persistence_dead_flag",
    "sa_outlier_flag",
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
    "focus_flag",
    "motion_blur_flag",
)

MASK_GEOMETRY_FLAGS = {
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
}
DEATH_FLAGS = {"viability_dead_flag", "persistence_dead_flag"}
POLICIES = {
    "strict_current": set(KNOWN_FLAGS),
    "death_only": DEATH_FLAGS,
    "death_and_mask_geometry": DEATH_FLAGS | MASK_GEOMETRY_FLAGS,
    "mask_geometry_only": MASK_GEOMETRY_FLAGS,
    "ignore_surface_area": set(KNOWN_FLAGS) - {"sa_outlier_flag"},
    "ignore_focus_motion": set(KNOWN_FLAGS) - {"focus_flag", "motion_blur_flag"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def manifest_scope_map() -> dict[str, str]:
    result = {}
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1] in {"Keyence", "YX1"}:
            result[fields[0]] = fields[1]
    return result


def merged_qc_path(experiment_id: str) -> Path:
    return (
        PIPELINE_ROOT
        / "quality_control"
        / experiment_id
        / "snip_qc"
        / f"{experiment_id}_snip_qc.parquet"
    )


def reason_set(value: object) -> frozenset[str]:
    if value is None or pd.isna(value):
        return frozenset()
    return frozenset(part for part in str(value).split("|") if part)


def entity_summary(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    pass_column: str,
) -> tuple[int, int, int]:
    by_entity = frame.groupby(keys, dropna=False)[pass_column].any()
    total = len(by_entity)
    passing = int(by_entity.sum())
    return total, passing, total - passing


def scope_groups(frame: pd.DataFrame):
    for scope, group in frame.groupby("scope", sort=True):
        yield scope, group
    yield "All", frame


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    formatters: dict[str, object] | None = None,
) -> str:
    formatters = formatters or {}
    labels = [column.replace("_", " ").title() for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            formatter = formatters.get(column)
            if callable(formatter):
                value = formatter(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone()

    scope_by_experiment = manifest_scope_map()
    snapshot_rows = []
    for experiment_id, scope in scope_by_experiment.items():
        path = merged_qc_path(experiment_id)
        present = path.is_file()
        stat = path.stat() if present else None
        snapshot_rows.append(
            {
                "scope": scope,
                "experiment_id": experiment_id,
                "qc_path": str(path),
                "qc_present_at_snapshot": present,
                "qc_size_bytes": stat.st_size if stat else 0,
                "qc_mtime": (
                    datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()
                    if stat
                    else ""
                ),
                "read_status": "not_attempted" if present else "missing",
                "read_error": "",
                "raw_row_count": 0,
                "analyzed_row_count": 0,
            }
        )
    snapshot = pd.DataFrame(snapshot_rows)

    frames = []
    for index, record in snapshot.loc[snapshot["qc_present_at_snapshot"]].iterrows():
        path = Path(record["qc_path"])
        try:
            available = set(pq.read_schema(path).names)
            desired = {
                "snip_id",
                "well_id",
                "physical_embryo_id",
                "time_index",
                "channel_id",
                "use_snip",
                "qc_fail_reasons",
                *KNOWN_FLAGS,
            }
            required = {
                "snip_id",
                "well_id",
                "physical_embryo_id",
                "use_snip",
                "qc_fail_reasons",
            }
            missing = sorted(required - available)
            if missing:
                raise ValueError(f"missing required columns: {missing}")
            frame = pd.read_parquet(path, columns=sorted(desired & available))
            raw_count = len(frame)
            if "channel_id" in frame and frame["channel_id"].astype(str).str.upper().eq("BF").any():
                frame = frame.loc[
                    frame["channel_id"].astype(str).str.upper().eq("BF")
                ].copy()
            elif frame["snip_id"].astype(str).str.contains("_BF_").any():
                frame = frame.loc[
                    frame["snip_id"].astype(str).str.contains("_BF_")
                ].copy()
            frame["scope"] = record["scope"]
            frame["experiment_id"] = record["experiment_id"]
            frame["reason_set"] = frame["qc_fail_reasons"].map(reason_set)
            frame["strict_pass"] = frame["reason_set"].map(len).eq(0)
            frame["use_snip"] = frame["use_snip"].fillna(False).astype(bool)
            frame["verdict_mismatch"] = frame["use_snip"].ne(frame["strict_pass"])
            frames.append(frame)
            snapshot.loc[index, "read_status"] = "read"
            snapshot.loc[index, "raw_row_count"] = raw_count
            snapshot.loc[index, "analyzed_row_count"] = len(frame)
        except Exception as exc:
            snapshot.loc[index, "read_status"] = "error"
            snapshot.loc[index, "read_error"] = f"{type(exc).__name__}: {exc}"

    if not frames:
        raise RuntimeError("No merged snip-QC tables could be read.")
    qc = pd.concat(frames, ignore_index=True)
    observed_flags = sorted(set().union(*qc["reason_set"]))
    unknown_flags = sorted(set(observed_flags) - set(KNOWN_FLAGS))

    dataset_rows = []
    zero_well_rows = []
    zero_embryo_rows = []
    for (scope, experiment_id), group in qc.groupby(
        ["scope", "experiment_id"], sort=True
    ):
        well_total, well_pass, well_zero = entity_summary(
            group, ["well_id"], pass_column="strict_pass"
        )
        embryo_total, embryo_pass, embryo_zero = entity_summary(
            group, ["physical_embryo_id"], pass_column="strict_pass"
        )
        dataset_rows.append(
            {
                "scope": scope,
                "experiment_id": experiment_id,
                "timepoint_count": len(group),
                "passing_timepoint_count": int(group["strict_pass"].sum()),
                "timepoint_pass_rate": float(group["strict_pass"].mean()),
                "well_count": well_total,
                "wells_with_any_pass": well_pass,
                "zero_pass_well_count": well_zero,
                "well_any_pass_rate": well_pass / well_total if well_total else float("nan"),
                "physical_embryo_id_count": embryo_total,
                "embryos_with_any_pass": embryo_pass,
                "zero_pass_embryo_count": embryo_zero,
                "embryo_any_pass_rate": (
                    embryo_pass / embryo_total if embryo_total else float("nan")
                ),
                "verdict_mismatch_count": int(group["verdict_mismatch"].sum()),
            }
        )
        by_well = group.groupby("well_id", dropna=False)
        for well_id, entity in by_well:
            if not entity["strict_pass"].any():
                zero_well_rows.append(
                    {
                        "scope": scope,
                        "experiment_id": experiment_id,
                        "well_id": well_id,
                        "timepoint_count": len(entity),
                        "physical_embryo_id_count": entity[
                            "physical_embryo_id"
                        ].nunique(),
                        "most_common_failure_combination": (
                            entity["qc_fail_reasons"].value_counts().index[0]
                        ),
                    }
                )
        by_embryo = group.groupby("physical_embryo_id", dropna=False)
        for embryo_id, entity in by_embryo:
            if not entity["strict_pass"].any():
                zero_embryo_rows.append(
                    {
                        "scope": scope,
                        "experiment_id": experiment_id,
                        "well_id": entity["well_id"].astype(str).iloc[0],
                        "physical_embryo_id": embryo_id,
                        "timepoint_count": len(entity),
                        "most_common_failure_combination": (
                            entity["qc_fail_reasons"].value_counts().index[0]
                        ),
                    }
                )
    dataset_summary = pd.DataFrame(dataset_rows)

    flag_rows = []
    dataset_flag_rows = []
    combination_rows = []
    policy_rows = []
    for scope, group in scope_groups(qc):
        failed_count = int((~group["strict_pass"]).sum())
        for flag in observed_flags:
            triggered = group["reason_set"].map(lambda values: flag in values)
            exclusive = group["reason_set"].map(lambda values: values == {flag})
            flag_rows.append(
                {
                    "scope": scope,
                    "flag": flag,
                    "triggered_timepoint_count": int(triggered.sum()),
                    "trigger_rate_all_timepoints": float(triggered.mean()),
                    "share_of_failed_timepoints": (
                        float(triggered.sum() / failed_count)
                        if failed_count
                        else float("nan")
                    ),
                    "exclusive_timepoint_count": int(exclusive.sum()),
                    "exclusive_rate_all_timepoints": float(exclusive.mean()),
                    "dataset_count_triggered": group.loc[
                        triggered, "experiment_id"
                    ].nunique(),
                    "well_count_triggered": group.loc[
                        triggered, ["experiment_id", "well_id"]
                    ].drop_duplicates().shape[0],
                    "embryo_count_triggered": group.loc[
                        triggered, ["experiment_id", "physical_embryo_id"]
                    ].drop_duplicates().shape[0],
                }
            )
        for combination, count in group.loc[
            ~group["strict_pass"], "qc_fail_reasons"
        ].value_counts().items():
            combination_rows.append(
                {
                    "scope": scope,
                    "failure_combination": combination,
                    "timepoint_count": int(count),
                    "rate_all_timepoints": float(count / len(group)),
                    "share_of_failed_timepoints": float(count / failed_count),
                    "flag_count": len(reason_set(combination)),
                }
            )
        for policy, exclusion_flags in POLICIES.items():
            passes = group["reason_set"].map(
                lambda values: not bool(values & exclusion_flags)
            )
            working = group.assign(policy_pass=passes)
            well_total, well_pass, well_zero = entity_summary(
                working, ["experiment_id", "well_id"], pass_column="policy_pass"
            )
            embryo_total, embryo_pass, embryo_zero = entity_summary(
                working,
                ["experiment_id", "physical_embryo_id"],
                pass_column="policy_pass",
            )
            policy_rows.append(
                {
                    "scope": scope,
                    "policy": policy,
                    "exclusion_flags": "|".join(
                        flag for flag in KNOWN_FLAGS if flag in exclusion_flags
                    ),
                    "passing_timepoint_count": int(passes.sum()),
                    "timepoint_pass_rate": float(passes.mean()),
                    "well_count": well_total,
                    "wells_with_any_pass": well_pass,
                    "zero_pass_well_count": well_zero,
                    "well_any_pass_rate": well_pass / well_total,
                    "physical_embryo_id_count": embryo_total,
                    "embryos_with_any_pass": embryo_pass,
                    "zero_pass_embryo_count": embryo_zero,
                    "embryo_any_pass_rate": embryo_pass / embryo_total,
                }
            )

    for (scope, experiment_id), group in qc.groupby(
        ["scope", "experiment_id"], sort=True
    ):
        for flag in observed_flags:
            triggered = group["reason_set"].map(lambda values: flag in values)
            dataset_flag_rows.append(
                {
                    "scope": scope,
                    "experiment_id": experiment_id,
                    "flag": flag,
                    "triggered_timepoint_count": int(triggered.sum()),
                    "trigger_rate_all_timepoints": float(triggered.mean()),
                }
            )

    flag_summary = pd.DataFrame(flag_rows).sort_values(
        ["scope", "triggered_timepoint_count"], ascending=[True, False]
    )
    dataset_flag_summary = pd.DataFrame(dataset_flag_rows)
    combination_summary = pd.DataFrame(combination_rows).sort_values(
        ["scope", "timepoint_count"], ascending=[True, False]
    )
    policy_summary = pd.DataFrame(policy_rows)
    zero_pass_wells = pd.DataFrame(zero_well_rows)
    zero_pass_embryos = pd.DataFrame(zero_embryo_rows)

    outputs = {
        "input_snapshot.csv": snapshot,
        "dataset_summary.csv": dataset_summary,
        "flag_summary.csv": flag_summary,
        "dataset_flag_summary.csv": dataset_flag_summary,
        "failure_combination_summary.csv": combination_summary,
        "policy_sensitivity_summary.csv": policy_summary,
        "zero_pass_wells.csv": zero_pass_wells,
        "zero_pass_embryos.csv": zero_pass_embryos,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    metadata = {
        "generated_at": generated_at.isoformat(),
        "pipeline_root": str(PIPELINE_ROOT),
        "manifest": str(MANIFEST),
        "manifest_dataset_count": int(len(snapshot)),
        "read_dataset_count": int(snapshot["read_status"].eq("read").sum()),
        "error_dataset_count": int(snapshot["read_status"].eq("error").sum()),
        "analyzed_timepoint_count": int(len(qc)),
        "observed_flags": observed_flags,
        "unknown_flags": unknown_flags,
        "bf_channel_policy": (
            "Use BF rows when an explicit BF channel or _BF_ snip IDs are present; "
            "otherwise retain all rows."
        ),
        "limitations": [
            "Only datasets with a readable merged snip-QC table at snapshot time are included.",
            "Completed-QC datasets may not represent unfinished datasets.",
            "Persisted QC artifacts may have been produced under different historical configs.",
            "physical_embryo_id can include tracking fragments or spurious detections.",
            "Counterfactual policies are descriptive sensitivity analyses, not recommendations.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    scope_overview_rows = []
    for scope, group in scope_groups(qc):
        well_total, well_pass, well_zero = entity_summary(
            group, ["experiment_id", "well_id"], pass_column="strict_pass"
        )
        embryo_total, embryo_pass, embryo_zero = entity_summary(
            group,
            ["experiment_id", "physical_embryo_id"],
            pass_column="strict_pass",
        )
        scope_overview_rows.append(
            {
                "scope": scope,
                "dataset_count": group["experiment_id"].nunique(),
                "timepoint_count": len(group),
                "timepoint_pass_rate": group["strict_pass"].mean(),
                "well_count": well_total,
                "well_any_pass_rate": well_pass / well_total,
                "zero_pass_wells": well_zero,
                "physical_embryo_id_count": embryo_total,
                "embryo_any_pass_rate": embryo_pass / embryo_total,
                "zero_pass_embryos": embryo_zero,
            }
        )
    scope_overview = pd.DataFrame(scope_overview_rows)
    scope_overview.to_csv(output_dir / "scope_overview.csv", index=False)

    report_parts = [
        "# QC flag census",
        "",
        f"Generated: `{generated_at.isoformat()}`",
        "",
        (
            "Snapshot definition: readable experiment-level merged `snip_qc` tables. "
            "Counts use BF rows when BF is identifiable."
        ),
        "",
        "## Strict current verdict",
        "",
        markdown_table(
            scope_overview,
            [
                "scope",
                "dataset_count",
                "timepoint_count",
                "timepoint_pass_rate",
                "well_count",
                "well_any_pass_rate",
                "zero_pass_wells",
                "physical_embryo_id_count",
                "embryo_any_pass_rate",
                "zero_pass_embryos",
            ],
            formatters={
                "timepoint_pass_rate": lambda value: f"{value:.1%}",
                "well_any_pass_rate": lambda value: f"{value:.1%}",
                "embryo_any_pass_rate": lambda value: f"{value:.1%}",
            },
        ),
        "",
        "## Most frequent exclusion flags",
    ]
    for scope in ["Keyence", "YX1", "All"]:
        table = flag_summary.loc[flag_summary["scope"].eq(scope)].head(8)
        report_parts.extend(
            [
                "",
                f"### {scope}",
                "",
                markdown_table(
                    table,
                    [
                        "flag",
                        "triggered_timepoint_count",
                        "trigger_rate_all_timepoints",
                        "exclusive_timepoint_count",
                        "dataset_count_triggered",
                    ],
                    formatters={
                        "trigger_rate_all_timepoints": lambda value: f"{value:.1%}",
                    },
                ),
            ]
        )
    report_parts.extend(["", "## Policy sensitivity", ""])
    report_parts.append(
        markdown_table(
            policy_summary,
            [
                "scope",
                "policy",
                "timepoint_pass_rate",
                "well_any_pass_rate",
                "embryo_any_pass_rate",
                "zero_pass_well_count",
            ],
            formatters={
                "timepoint_pass_rate": lambda value: f"{value:.1%}",
                "well_any_pass_rate": lambda value: f"{value:.1%}",
                "embryo_any_pass_rate": lambda value: f"{value:.1%}",
            },
        )
    )
    report_parts.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- Flag trigger counts overlap; a timepoint may contribute to several flags.",
            "- A physical embryo ID is a tracking product, not necessarily one true embryo.",
            "- A well with any passing timepoint is a better test of gross acquisition failure.",
            "- Existing completed-QC datasets are a selected subset of the corpus.",
            "- Historical QC artifacts may reflect different flag configurations.",
            "- Policy sensitivities are diagnostic counterfactuals, not proposed training gates.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(report_parts))

    print(scope_overview.to_string(index=False))
    print()
    print(
        flag_summary.loc[flag_summary["scope"].eq("YX1")]
        .head(8)[
            ["flag", "triggered_timepoint_count", "trigger_rate_all_timepoints"]
        ]
        .to_string(index=False)
    )
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
