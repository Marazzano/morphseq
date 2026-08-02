"""Run one source-experiment partition of full-corpus SeaHub detection.

The SGE array task id selects one experiment from the sorted set of
policy-included FOVs. GroundingDINO is loaded once and reused for every FOV in
that experiment. Each task writes to its own directory so tasks are isolated
and individually rerunnable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from seahub_workflow import run_grounding_dino_segmentation


HERE = Path(__file__).resolve().parent


def _task_id(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    raw = os.environ.get("SGE_TASK_ID")
    if raw is None or not raw.isdigit():
        raise SystemExit("Provide --task-id or run as an SGE array task.")
    return int(raw)


def _included_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame["include_for_seahub"]
    if values.dtype == bool:
        return values
    return values.astype(str).str.casefold().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciled-fovs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate task selection without loading the detector.",
    )
    args = parser.parse_args()

    reconciled_path = args.reconciled_fovs_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"Output must remain under {HERE}: {output_root}") from exc

    reconciled = pd.read_csv(reconciled_path)
    required = {
        "include_for_seahub",
        "experiment_id",
        "image_id",
        "image_path",
    }
    missing = required.difference(reconciled.columns)
    if missing:
        raise ValueError(f"Reconciled FOV table is missing columns: {sorted(missing)}")

    included = reconciled[_included_mask(reconciled)].copy()
    experiments = sorted(included["experiment_id"].dropna().astype(str).unique())
    task_id = _task_id(args.task_id)
    if task_id < 1 or task_id > len(experiments):
        raise ValueError(
            f"Task {task_id} is outside the valid range 1..{len(experiments)}"
        )

    experiment_id = experiments[task_id - 1]
    partition = included[
        included["experiment_id"].astype(str).eq(experiment_id)
    ].copy()
    missing_images = [
        path for path in partition["image_path"].astype(str) if not Path(path).is_file()
    ]
    if missing_images:
        preview = "\n".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} source images are missing for {experiment_id}:\n"
            f"{preview}"
        )

    task_output = output_root / "tasks" / experiment_id
    if task_output.exists():
        existing = sorted(task_output.iterdir())
        if existing:
            preview = ", ".join(path.name for path in existing[:5])
            raise FileExistsError(
                "Refusing to reuse a nonempty SeaHub detection partition "
                f"{task_output}. Existing entries include: {preview}. Use a fresh "
                "SEAHUB_RUN_ID; do not mix partitions across attempts."
            )
    print(
        f"Task {task_id}/{len(experiments)}: {experiment_id}, "
        f"{len(partition)} included FOVs -> {task_output}",
        flush=True,
    )
    if args.dry_run:
        return

    # With array concurrency fixed at one, task 1 is a cheap systemic smoke
    # gate. If model loading or GPU inference fails there, later tasks stop
    # before loading the 2 GB checkpoint rather than repeating the same failure.
    if task_id > 1:
        smoke_marker = output_root / "tasks" / experiments[0] / "_SUCCESS"
        if not smoke_marker.is_file():
            raise RuntimeError(
                "Task 1 did not leave its _SUCCESS marker; refusing to run "
                "later partitions after a failed GPU/environment smoke test."
            )

    manifest, qc = run_grounding_dino_segmentation(
        partition,
        task_output,
        device="cuda",
    )
    if len(qc) != len(partition):
        raise RuntimeError(
            f"Detection returned {len(qc)} QC rows for {len(partition)} FOVs."
        )
    pass_count = int(qc["segmentation_qc_status"].eq("pass").sum())
    expected_manifest_rows = pass_count * 8
    if len(manifest) != expected_manifest_rows:
        raise RuntimeError(
            f"Detection returned {len(manifest)} embryo rows; expected "
            f"{expected_manifest_rows} from {pass_count} passing FOVs."
        )

    (task_output / "_SUCCESS").write_text(
        f"experiment_id={experiment_id}\n"
        f"included_fovs={len(partition)}\n"
        f"passing_fovs={pass_count}\n"
        f"embryos={len(manifest)}\n",
        encoding="utf-8",
    )
    print(
        f"Completed {experiment_id}: {pass_count}/{len(partition)} FOVs passed, "
        f"{len(manifest)} embryo boxes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
