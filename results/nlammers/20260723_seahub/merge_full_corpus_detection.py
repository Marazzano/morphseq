"""Merge and validate all source-experiment SeaHub detection partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _included_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame["include_for_seahub"]
    if values.dtype == bool:
        return values
    return values.astype(str).str.casefold().isin({"true", "1", "yes"})


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciled-fovs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    reconciled = pd.read_csv(args.reconciled_fovs_csv.expanduser().resolve())
    included = reconciled[_included_mask(reconciled)].copy()
    experiments = sorted(included["experiment_id"].dropna().astype(str).unique())
    output_root = args.output_root.expanduser().resolve()

    stale_merge_outputs = [
        output_root / "embryo_manifest.csv",
        output_root / "segmentation_qc.csv",
        output_root / "_MERGE_SUCCESS",
    ]
    existing_merge_outputs = [path for path in stale_merge_outputs if path.exists()]
    if existing_merge_outputs:
        raise FileExistsError(
            "Refusing to overwrite prior SeaHub merged detection products: "
            f"{existing_merge_outputs}. Use a fresh SEAHUB_RUN_ID."
        )

    manifests: list[pd.DataFrame] = []
    qcs: list[pd.DataFrame] = []
    for experiment_id in experiments:
        task_dir = output_root / "tasks" / experiment_id
        required = [
            task_dir / "_SUCCESS",
            task_dir / "embryo_manifest.csv",
            task_dir / "segmentation_qc.csv",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete detection partition {experiment_id}: {missing}"
            )
        manifests.append(_read_optional_csv(task_dir / "embryo_manifest.csv"))
        qc = pd.read_csv(task_dir / "segmentation_qc.csv")
        qc["detection_partition"] = experiment_id
        qcs.append(qc)

    nonempty_manifests = [frame for frame in manifests if not frame.empty]
    if not nonempty_manifests:
        raise RuntimeError("Every detection partition produced an empty manifest.")
    manifest = pd.concat(nonempty_manifests, ignore_index=True, sort=False)
    qc = pd.concat(qcs, ignore_index=True, sort=False)

    expected_ids = set(included["image_id"].astype(str))
    qc_ids = qc["image_id"].astype(str)
    if qc_ids.duplicated().any():
        duplicates = sorted(qc_ids[qc_ids.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate FOVs across detection partitions: {duplicates[:10]}")
    if set(qc_ids) != expected_ids:
        missing_ids = sorted(expected_ids.difference(qc_ids))
        extra_ids = sorted(set(qc_ids).difference(expected_ids))
        raise ValueError(
            f"Detection coverage mismatch: missing={missing_ids[:10]}, "
            f"extra={extra_ids[:10]}"
        )

    manifest_counts = manifest.groupby("image_id").size()
    bad_counts = manifest_counts[manifest_counts.ne(8)]
    if not bad_counts.empty:
        raise ValueError(
            "Passing FOVs without exactly eight embryo rows: "
            f"{bad_counts.head(10).to_dict()}"
        )
    passing_ids = set(
        qc.loc[qc["segmentation_qc_status"].eq("pass"), "image_id"].astype(str)
    )
    if set(manifest_counts.index.astype(str)) != passing_ids:
        raise ValueError("Manifest FOVs do not match the passing detection QC rows.")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_root / "embryo_manifest.csv", index=False)
    qc.to_csv(output_root / "segmentation_qc.csv", index=False)
    (output_root / "_MERGE_SUCCESS").write_text(
        f"included_fovs={len(included)}\n"
        f"passing_fovs={len(passing_ids)}\n"
        f"failed_fovs={len(included) - len(passing_ids)}\n"
        f"embryos={len(manifest)}\n",
        encoding="utf-8",
    )
    print(
        f"Merged {len(included)} FOVs across {len(experiments)} experiments: "
        f"{len(passing_ids)} passed, {len(manifest)} embryo boxes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
