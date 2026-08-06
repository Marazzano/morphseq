"""Render a deterministic legacy / pre-fix / post-fix snip comparison.

The pre-fix tree is an immutable snapshot made before the 6.5 um/px, 75 um
compatibility rerun.  The post-fix tree defaults to the live pipeline product,
so this script deliberately refuses to run until that inventory is newer than
the snapshot and all of its snip rasters have been quiet for a configurable
minimum age.

Rows are not hand-picked.  The script pairs the primary embryo in each well to
the legacy ``e00`` raster, orders wells by pre-fix foreground saturation, and
samples evenly spaced ranks across that distribution.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT = "20250612_24hpf_ctrl_atf6"
RESULT_DIR = Path(__file__).resolve().parent
DEFAULT_PRE_FIX_ROOT = RESULT_DIR / "before" / "snips"
DEFAULT_POST_FIX_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output/"
    f"object_extraction/{EXPERIMENT}/snips"
)
DEFAULT_LEGACY_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/"
    f"bf_embryo_snips/{EXPERIMENT}"
)
DEFAULT_OUTPUT = RESULT_DIR / "snip_compatibility_montage.png"

FOREGROUND_THRESHOLD = 30
SATURATION_THRESHOLD = 250


def _inventory_path(snips_root: Path) -> Path:
    return snips_root / f"{EXPERIMENT}_snip_inventory.csv"


def _valid_rows(inventory_path: Path) -> pd.DataFrame:
    if not inventory_path.is_file():
        raise FileNotFoundError(f"snip inventory not found: {inventory_path}")

    frame = pd.read_csv(inventory_path)
    required = {
        "experiment_id",
        "well_id",
        "physical_embryo_id",
        "snip_id",
        "processed_snip_path",
        "is_valid_snip",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{inventory_path} is missing required columns: {missing}")

    experiment_ids = set(frame["experiment_id"].dropna().astype(str))
    if experiment_ids != {EXPERIMENT}:
        raise ValueError(
            f"{inventory_path} contains experiment ids {sorted(experiment_ids)}, "
            f"expected only {EXPERIMENT}"
        )

    valid = (
        frame["is_valid_snip"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    result = frame.loc[valid].copy()
    if result.empty:
        raise ValueError(f"{inventory_path} has no valid snips")
    return result.sort_values(
        ["well_id", "physical_embryo_id", "snip_id"], kind="stable"
    ).reset_index(drop=True)


def _primary_by_well(rows: pd.DataFrame) -> pd.DataFrame:
    """Choose the first physical embryo in each well under the canonical id order."""
    return rows.drop_duplicates("well_id", keep="first").set_index("well_id")


def _identity_set(rows: pd.DataFrame, inventory_path: Path) -> set[tuple[str, ...]]:
    columns = ["well_id", "physical_embryo_id", "snip_id"]
    if rows.duplicated(columns).any():
        duplicates = rows.loc[rows.duplicated(columns, keep=False), columns]
        raise ValueError(
            f"{inventory_path} contains duplicate snip identities: "
            f"{duplicates.head().to_dict(orient='records')}"
        )
    return set(rows[columns].astype(str).itertuples(index=False, name=None))


def _resolve_snapshot_path(snips_root: Path, inventory_value: object) -> Path:
    """Map a production inventory path into either a snapshot or live snips root."""
    source = Path(str(inventory_value))
    parts = source.parts
    try:
        snips_index = parts.index("snips")
    except ValueError as exc:
        raise ValueError(
            f"processed_snip_path has no 'snips' path component: {source}"
        ) from exc
    relative = Path(*parts[snips_index + 1 :])
    if not relative.parts or relative.parts[0] != "per_well":
        raise ValueError(
            f"processed_snip_path does not resolve beneath snips/per_well: {source}"
        )
    return snips_root / relative


def _well_slug(well_id: str) -> str:
    prefix = f"{EXPERIMENT}_"
    if not str(well_id).startswith(prefix):
        raise ValueError(f"invalid well_id for {EXPERIMENT}: {well_id!r}")
    return str(well_id)[len(prefix) :]


def _legacy_path(legacy_root: Path, well_id: str) -> Path:
    return legacy_root / f"{EXPERIMENT}_{_well_slug(well_id)}_e00_t0000.jpg"


def _load_gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"snip raster not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _image_metrics(image: np.ndarray) -> dict[str, float | int]:
    foreground = image > FOREGROUND_THRESHOLD
    n_foreground = int(foreground.sum())
    if n_foreground == 0:
        raise ValueError("snip contains no pixels above the foreground threshold")
    values = image[foreground]
    return {
        "foreground_pixels": n_foreground,
        "foreground_mean": float(values.mean()),
        "foreground_p95": float(np.percentile(values, 95)),
        "foreground_fraction_ge_250": float(
            np.mean(values >= SATURATION_THRESHOLD)
        ),
    }


def _post_fix_is_stable(
    pre_fix_inventory: Path,
    post_fix_inventory: Path,
    post_fix_paths: list[Path],
    minimum_age_seconds: float,
) -> None:
    if post_fix_inventory.stat().st_mtime_ns <= pre_fix_inventory.stat().st_mtime_ns:
        raise RuntimeError(
            "post-fix inventory is not newer than the pre-fix snapshot; the compatibility "
            "rerun has not completed (or this points at the pre-fix product)"
        )

    newest_path = max(
        [post_fix_inventory, *post_fix_paths], key=lambda path: path.stat().st_mtime_ns
    )
    age_seconds = time.time() - newest_path.stat().st_mtime
    if age_seconds < minimum_age_seconds:
        raise RuntimeError(
            f"post-fix files are still too recent: {newest_path} is only "
            f"{age_seconds:.1f}s old, below --minimum-post-fix-age-seconds "
            f"{minimum_age_seconds:.1f}. Wait for the rerun to settle and try again."
        )


def _build_pair_table(
    pre_fix_root: Path,
    post_fix_root: Path,
    legacy_root: Path,
) -> pd.DataFrame:
    pre_fix_inventory = _inventory_path(pre_fix_root)
    post_fix_inventory = _inventory_path(post_fix_root)
    pre_fix_rows = _valid_rows(pre_fix_inventory)
    post_fix_rows = _valid_rows(post_fix_inventory)
    pre_fix_identities = _identity_set(pre_fix_rows, pre_fix_inventory)
    post_fix_identities = _identity_set(post_fix_rows, post_fix_inventory)
    if pre_fix_identities != post_fix_identities:
        missing = sorted(pre_fix_identities - post_fix_identities)
        unexpected = sorted(post_fix_identities - pre_fix_identities)
        raise ValueError(
            "pre/post inventories do not contain the same valid snip identities; "
            f"missing post-fix={missing[:5]}, unexpected post-fix={unexpected[:5]}"
        )

    pre_fix = _primary_by_well(pre_fix_rows)
    post_fix = _primary_by_well(post_fix_rows)

    common_wells = sorted(pre_fix.index)

    records: list[dict[str, object]] = []
    for well_id in common_wells:
        pre_fix_row = pre_fix.loc[well_id]
        post_fix_row = post_fix.loc[well_id]
        if str(pre_fix_row["physical_embryo_id"]) != str(
            post_fix_row["physical_embryo_id"]
        ):
            raise ValueError(
                f"primary embryo identity changed for {well_id}: "
                f"{pre_fix_row['physical_embryo_id']} -> "
                f"{post_fix_row['physical_embryo_id']}"
            )

        paths = {
            "legacy_path": _legacy_path(legacy_root, well_id),
            "pre_fix_path": _resolve_snapshot_path(
                pre_fix_root, pre_fix_row["processed_snip_path"]
            ),
            "post_fix_path": _resolve_snapshot_path(
                post_fix_root, post_fix_row["processed_snip_path"]
            ),
        }
        # The legacy archive is incomplete for E07 on this plate.  It cannot form a
        # three-way comparison, so omit it instead of failing the entire montage.
        if not paths["legacy_path"].is_file():
            continue
        missing = [
            str(paths[key])
            for key in ("pre_fix_path", "post_fix_path")
            if not paths[key].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"paired primary raster(s) missing for {well_id}: {missing}"
            )

        pre_fix_image = _load_gray(paths["pre_fix_path"])
        pre_fix_metrics = _image_metrics(pre_fix_image)
        records.append(
            {
                "well_id": well_id,
                "well": _well_slug(well_id),
                "physical_embryo_id": str(pre_fix_row["physical_embryo_id"]),
                **{key: str(value) for key, value in paths.items()},
                "pre_fix_foreground_pixels": pre_fix_metrics["foreground_pixels"],
                "pre_fix_foreground_mean": pre_fix_metrics["foreground_mean"],
                "pre_fix_foreground_p95": pre_fix_metrics["foreground_p95"],
                "pre_fix_foreground_fraction_ge_250": pre_fix_metrics[
                    "foreground_fraction_ge_250"
                ],
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        ["pre_fix_foreground_fraction_ge_250", "well_id"], kind="stable"
    ).reset_index(drop=True)


def _select_saturation_quantiles(pairs: pd.DataFrame, n_wells: int) -> pd.DataFrame:
    if n_wells < 1:
        raise ValueError("--n-wells must be positive")
    if len(pairs) < n_wells:
        raise ValueError(
            f"requested {n_wells} montage wells, but only {len(pairs)} complete pairs exist"
        )
    indices = np.linspace(0, len(pairs) - 1, num=n_wells, dtype=int)
    selected = pairs.iloc[indices].copy().reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected.insert(1, "source_saturation_rank", indices + 1)
    return selected


def _metric_caption(metrics: dict[str, float | int]) -> str:
    saturation = 100.0 * float(metrics["foreground_fraction_ge_250"])
    return (
        f"fg={int(metrics['foreground_pixels']):,} px   "
        f">=250: {saturation:.2f}%"
    )


def render_montage(selected: pd.DataFrame, output_png: Path) -> pd.DataFrame:
    columns = (
        ("legacy_path", "Legacy\n6.5 µm/px, 75 µm"),
        ("pre_fix_path", "Pre-fix\n7.8 µm/px, 20 µm"),
        ("post_fix_path", "Post-fix\n6.5 µm/px, 75 µm"),
    )
    figure, axes = plt.subplots(
        nrows=len(selected),
        ncols=len(columns),
        figsize=(9.0, 2.75 * len(selected)),
        squeeze=False,
    )

    metric_records: list[dict[str, object]] = []
    any_changed = False
    for row_index, row in selected.iterrows():
        loaded: dict[str, np.ndarray] = {}
        record = dict(row)
        for column_index, (path_key, heading) in enumerate(columns):
            image = _load_gray(Path(str(row[path_key])))
            loaded[path_key] = image
            metrics = _image_metrics(image)
            prefix = path_key.removesuffix("_path")
            for metric_name, value in metrics.items():
                record[f"{prefix}_{metric_name}"] = value

            axis = axes[row_index, column_index]
            axis.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(_metric_caption(metrics), fontsize=8, pad=3)
            if row_index == 0:
                axis.text(
                    0.5,
                    1.17,
                    heading,
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )
            if column_index == 0:
                pre_fix_saturation = (
                    100.0 * float(row["pre_fix_foreground_fraction_ge_250"])
                )
                axis.set_ylabel(
                    f"{row['well']}\npre-fix >=250: {pre_fix_saturation:.2f}%",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=34,
                    fontsize=9,
                )

        shapes = {image.shape for image in loaded.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"legacy/pre/post shape mismatch for {row['well_id']}: "
                f"{sorted(shapes)}"
            )
        any_changed |= not np.array_equal(
            loaded["pre_fix_path"], loaded["post_fix_path"]
        )
        metric_records.append(record)

    if not any_changed:
        plt.close(figure)
        raise RuntimeError(
            "all selected post-fix rasters are pixel-identical to pre-fix rasters; "
            "refusing to label them as a completed compatibility rerun"
        )

    figure.suptitle(
        f"{EXPERIMENT}: deterministic primary-embryo compatibility check\n"
        "Rows sample evenly spaced ranks of the pre-fix foreground-saturation distribution; "
        "all panels use the same 0-255 display range.",
        fontsize=12,
        y=0.998,
    )
    figure.tight_layout(rect=(0.06, 0.0, 1.0, 0.975), h_pad=1.0, w_pad=0.5)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pd.DataFrame.from_records(metric_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-fix-snips-root", type=Path, default=DEFAULT_PRE_FIX_ROOT)
    parser.add_argument("--post-fix-snips-root", type=Path, default=DEFAULT_POST_FIX_ROOT)
    parser.add_argument("--legacy-snips-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-csv", type=Path)
    parser.add_argument("--n-wells", type=int, default=8)
    parser.add_argument(
        "--minimum-post-fix-age-seconds",
        type=float,
        default=300.0,
        help="refuse live post-fix rasters newer than this many seconds (default: 300)",
    )
    args = parser.parse_args()

    pre_fix_inventory = _inventory_path(args.pre_fix_snips_root)
    post_fix_inventory = _inventory_path(args.post_fix_snips_root)
    pairs = _build_pair_table(
        args.pre_fix_snips_root,
        args.post_fix_snips_root,
        args.legacy_snips_root,
    )
    post_fix_paths = [
        _resolve_snapshot_path(args.post_fix_snips_root, inventory_path)
        for inventory_path in _valid_rows(post_fix_inventory)["processed_snip_path"]
    ]
    missing_post_fix_paths = [path for path in post_fix_paths if not path.is_file()]
    if missing_post_fix_paths:
        raise FileNotFoundError(
            "post-fix inventory references missing raster(s): "
            f"{[str(path) for path in missing_post_fix_paths[:5]]}"
        )
    _post_fix_is_stable(
        pre_fix_inventory,
        post_fix_inventory,
        post_fix_paths,
        args.minimum_post_fix_age_seconds,
    )

    selected = _select_saturation_quantiles(pairs, args.n_wells)
    metrics = render_montage(selected, args.output_png)
    selection_csv = args.selection_csv or args.output_png.with_suffix(".selection.csv")
    selection_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(selection_csv, index=False)
    print(f"wrote montage: {args.output_png}")
    print(f"wrote selection and metrics: {selection_csv}")


if __name__ == "__main__":
    main()
