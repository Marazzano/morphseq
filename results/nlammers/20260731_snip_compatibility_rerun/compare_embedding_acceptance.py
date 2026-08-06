"""Compare pre/post-fix VAE embeddings with the legacy checkpoint table.

Run this only after the post-fix merged latent table has finished writing.  The
legacy encoder dimension mapping is the one validated in
``root_analysis/embedding_impact_probe.py``:

* current ``z_mu_00..99`` maps directly to encoder indices 0..99;
* legacy indices 0..19 are ``z_mu_n_00..19``;
* legacy indices 20..99 are ``z_mu_b_20..99``.

Current pipeline embryo ``e01`` is paired by well with legacy ``e00``, matching
the previously validated comparison and avoiding extra current embryos in a
small number of wells.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


EXPERIMENT_ID = "20250612_24hpf_ctrl_atf6"
N_DIMENSIONS = 100
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")

DEFAULT_PRE_PATH = (
    HERE / "before" / f"{EXPERIMENT_ID}_latents.parquet"
)
DEFAULT_POST_PATH = (
    DATA_ROOT
    / "pipeline"
    / "output"
    / "feature_extraction"
    / EXPERIMENT_ID
    / "latent_embeddings"
    / f"{EXPERIMENT_ID}_latents.parquet"
)
DEFAULT_LEGACY_PATH = (
    DATA_ROOT
    / "legacy"
    / "20241107_ds_sweep01_optimum"
    / f"morph_latents_{EXPERIMENT_ID}.csv"
)

WELL_RE = re.compile(r"_([A-H]\d{2})_e\d+(?:_[^_]+)?_t\d+$")
CURRENT_PRIMARY_EMBRYO_RE = re.compile(r"_e01(?:_|$)")
LEGACY_PRIMARY_EMBRYO_RE = re.compile(r"_e00(?:_|$)")


def _legacy_column(index: int) -> str:
    family = "n" if index < 20 else "b"
    return f"z_mu_{family}_{index:02d}"


def _current_column(index: int) -> str:
    return f"z_mu_{index:02d}"


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported table format for {path}; expected CSV or Parquet")


def _prepare_table(path: Path, *, legacy: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    latent_columns = [
        _legacy_column(index) if legacy else _current_column(index)
        for index in range(N_DIMENSIONS)
    ]
    frame = _read_columns(path, ["snip_id", *latent_columns])
    input_rows = len(frame)

    selector = LEGACY_PRIMARY_EMBRYO_RE if legacy else CURRENT_PRIMARY_EMBRYO_RE
    frame = frame[frame["snip_id"].astype(str).str.contains(selector)].copy()
    selected_rows = len(frame)
    frame["well"] = frame["snip_id"].astype(str).map(
        lambda value: WELL_RE.search(value).group(1) if WELL_RE.search(value) else None
    )
    if frame["well"].isna().any():
        examples = frame.loc[frame["well"].isna(), "snip_id"].head(5).tolist()
        raise ValueError(f"Could not extract well from snip_id values: {examples}")
    if frame["well"].duplicated().any():
        duplicates = sorted(frame.loc[frame["well"].duplicated(False), "well"].unique())
        raise ValueError(f"Primary-embryo rows are not one-to-one by well: {duplicates[:10]}")

    values = frame[latent_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite latent value(s) found in {path}")

    canonical_columns = [_current_column(index) for index in range(N_DIMENSIONS)]
    canonical = pd.DataFrame(values, index=frame["well"], columns=canonical_columns)
    canonical.index.name = "well"
    return canonical.sort_index(), {
        "input_rows": int(input_rows),
        "primary_embryo_rows": int(selected_rows),
        "unique_primary_wells": int(len(canonical)),
    }


def _dimension_correlations(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    reference_centered = reference - reference.mean(axis=0)
    other_centered = other - other.mean(axis=0)
    denominator = np.linalg.norm(reference_centered, axis=0) * np.linalg.norm(
        other_centered, axis=0
    )
    correlations = np.full(reference.shape[1], np.nan, dtype=float)
    valid = denominator > 0
    correlations[valid] = np.sum(
        reference_centered[:, valid] * other_centered[:, valid], axis=0
    ) / denominator[valid]
    return correlations


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _compare(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    reference_name: str,
) -> tuple[dict[str, object], np.ndarray]:
    shared_wells = sorted(set(reference.index) & set(other.index))
    if len(shared_wells) < 3:
        raise ValueError(
            f"Need at least three paired wells; found {len(shared_wells)} for {reference_name}"
        )

    reference_array = reference.loc[shared_wells].to_numpy(dtype=float)
    other_array = other.loc[shared_wells].to_numpy(dtype=float)
    correlations = _dimension_correlations(reference_array, other_array)

    center = reference_array.mean(axis=0)
    scale = reference_array.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    reference_z = (reference_array - center) / scale
    other_z = (other_array - center) / scale

    reference_distances = pdist(reference_z)
    other_distances = pdist(other_z)
    finite_correlations = correlations[np.isfinite(correlations)]
    if not finite_correlations.size:
        raise ValueError(f"No finite per-dimension correlations for {reference_name}")

    metrics: dict[str, object] = {
        "n_paired_wells": int(len(shared_wells)),
        "n_dimensions": int(reference_array.shape[1]),
        "n_dimensions_with_finite_correlation": int(finite_correlations.size),
        "median_per_dimension_correlation": float(np.median(finite_correlations)),
        "standardized_rmse": float(np.sqrt(np.mean((other_z - reference_z) ** 2))),
        "pairwise_distance_correlation": _safe_correlation(
            reference_distances, other_distances
        ),
        "standardization_reference": reference_name,
    }
    return metrics, correlations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, default=DEFAULT_PRE_PATH)
    parser.add_argument("--post", type=Path, default=DEFAULT_POST_PATH)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument(
        "--output-json", type=Path, default=HERE / "embedding_acceptance_summary.json"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=HERE / "embedding_acceptance_per_dimension.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pre.resolve() == args.post.resolve():
        raise ValueError("--pre and --post must refer to different files")

    pre, pre_inventory = _prepare_table(args.pre, legacy=False)
    post, post_inventory = _prepare_table(args.post, legacy=False)
    legacy, legacy_inventory = _prepare_table(args.legacy, legacy=True)

    pre_legacy_metrics, pre_legacy_correlations = _compare(
        legacy, pre, reference_name="legacy"
    )
    post_legacy_metrics, post_legacy_correlations = _compare(
        legacy, post, reference_name="legacy"
    )
    pre_post_metrics, pre_post_correlations = _compare(
        pre, post, reference_name="pre_fix"
    )
    biological = slice(20, N_DIMENSIONS)
    pre_legacy_biological_metrics, _ = _compare(
        legacy.iloc[:, biological],
        pre.iloc[:, biological],
        reference_name="legacy_biological_20_99",
    )
    post_legacy_biological_metrics, _ = _compare(
        legacy.iloc[:, biological],
        post.iloc[:, biological],
        reference_name="legacy_biological_20_99",
    )
    pre_post_biological_metrics, _ = _compare(
        pre.iloc[:, biological],
        post.iloc[:, biological],
        reference_name="pre_fix_biological_20_99",
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "inputs": {
            "pre_fix_merged_latents": str(args.pre.resolve()),
            "post_fix_merged_latents": str(args.post.resolve()),
            "legacy_checkpoint_table": str(args.legacy.resolve()),
        },
        "dimension_mapping": {
            "current": "z_mu_00..z_mu_99",
            "legacy_0_19": "z_mu_n_00..z_mu_n_19",
            "legacy_20_99": "z_mu_b_20..z_mu_b_99",
            "pairing": "current e01 to legacy e00 by A01-H12 well",
        },
        "dataset_inventory": {
            "pre_fix": pre_inventory,
            "post_fix": post_inventory,
            "legacy": legacy_inventory,
        },
        "comparisons": {
            "pre_vs_legacy": pre_legacy_metrics,
            "post_vs_legacy": post_legacy_metrics,
            "pre_vs_post": pre_post_metrics,
        },
        "comparisons_biological_20_99": {
            "pre_vs_legacy": pre_legacy_biological_metrics,
            "post_vs_legacy": post_legacy_biological_metrics,
            "pre_vs_post": pre_post_biological_metrics,
        },
    }

    per_dimension = pd.DataFrame(
        {
            "dimension": np.arange(N_DIMENSIONS),
            "family": [
                "nuisance" if index < 20 else "biological"
                for index in range(N_DIMENSIONS)
            ],
            "pre_vs_legacy_correlation": pre_legacy_correlations,
            "post_vs_legacy_correlation": post_legacy_correlations,
            "pre_vs_post_correlation": pre_post_correlations,
        }
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    per_dimension.to_csv(args.output_csv, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
