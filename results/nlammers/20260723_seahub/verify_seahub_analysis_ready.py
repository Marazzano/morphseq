"""Verify that every planned SeaHub shard produced analysis-ready output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--pipeline-output-root", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.experiment_manifest.expanduser().resolve())
    output_root = args.pipeline_output_root.expanduser().resolve()
    report_rows: list[dict] = []
    failures: list[str] = []

    for experiment_id in manifest["experiment_id"].astype(str):
        parquet_path = (
            output_root
            / "analysis_ready"
            / experiment_id
            / "analysis_ready"
            / f"{experiment_id}_analysis_ready.parquet"
        )
        record = {
            "experiment_id": experiment_id,
            "analysis_ready_parquet": str(parquet_path),
            "exists": parquet_path.is_file(),
            "row_count": 0,
            "source_scope_all_seahub": False,
            "error": None,
        }
        if not parquet_path.is_file():
            record["error"] = "missing_analysis_ready"
            failures.append(f"{experiment_id}: missing {parquet_path}")
            report_rows.append(record)
            continue
        try:
            frame = pd.read_parquet(parquet_path)
            record["row_count"] = len(frame)
            if "source_scope" not in frame.columns:
                record["error"] = "missing_source_scope"
            else:
                scopes = frame["source_scope"].dropna().astype(str).str.casefold()
                record["source_scope_all_seahub"] = bool(
                    not scopes.empty and scopes.eq("seahub").all()
                )
                if not record["source_scope_all_seahub"]:
                    record["error"] = "source_scope_not_seahub"
            if frame.empty and record["error"] is None:
                record["error"] = "empty_analysis_ready"
        except Exception as exc:  # surfaced in the report and final nonzero exit
            record["error"] = f"unreadable: {type(exc).__name__}: {exc}"
        if record["error"] is not None:
            failures.append(f"{experiment_id}: {record['error']}")
        report_rows.append(record)

    report = pd.DataFrame.from_records(report_rows)
    report_path = args.report_csv.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"{len(failures)} of {len(report)} SeaHub shards failed verification:\n"
            f"{preview}"
        )

    report_path.with_suffix(".success").write_text(
        f"verified_shards={len(report)}\n"
        f"analysis_ready_rows={int(report['row_count'].sum())}\n",
        encoding="utf-8",
    )
    print(
        f"Verified {len(report)} SeaHub analysis-ready shards with "
        f"{int(report['row_count'].sum())} total rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
