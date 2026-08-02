"""Quantify the aggregate VAE impact of the legacy/current snip regression."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


EXPERIMENT_ID = "20250612_30hpf_ctrl_atf6"
DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
LEGACY_PATH = (
    DATA_ROOT
    / "legacy"
    / "20241107_ds_sweep01_optimum"
    / f"morph_latents_{EXPERIMENT_ID}.csv"
)
CURRENT_PATH = (
    DATA_ROOT
    / "pipeline"
    / "output"
    / "analysis_ready"
    / EXPERIMENT_ID
    / "analysis_ready"
    / f"{EXPERIMENT_ID}_analysis_ready.parquet"
)
HERE = Path(__file__).resolve().parent
WELL_RE = re.compile(r"_([A-H]\d{2})_e\d+_t\d+$")


def _legacy_column(index: int) -> str:
    family = "n" if index < 20 else "b"
    return f"z_mu_{family}_{index:02d}"


def _current_column(index: int) -> str:
    return f"z_mu_{index:02d}"


def _standardize(
    reference: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center = reference.mean(axis=0)
    scale = reference.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    return (reference - center) / scale, (other - center) / scale


def _correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    a0 /= np.linalg.norm(a0, axis=0, keepdims=True)
    b0 /= np.linalg.norm(b0, axis=0, keepdims=True)
    return a0.T @ b0


def _nearest_reference_accuracy(
    legacy: np.ndarray, current: np.ndarray
) -> float:
    distances = np.sqrt(
        np.sum((current[:, None, :] - legacy[None, :, :]) ** 2, axis=2)
    )
    return float(np.mean(np.argmin(distances, axis=1) == np.arange(len(legacy))))


def _multivariate_group_r2(x: np.ndarray, groups: np.ndarray) -> float:
    overall = x.mean(axis=0)
    total = float(np.sum((x - overall) ** 2))
    between = 0.0
    for group in pd.unique(groups):
        subset = x[groups == group]
        between += float(len(subset) * np.sum((subset.mean(axis=0) - overall) ** 2))
    return between / total if total > 0 else float("nan")


def _space_metrics(
    legacy: np.ndarray,
    current: np.ndarray,
    *,
    temperature: np.ndarray,
    genotype: np.ndarray,
) -> dict[str, object]:
    legacy_z, current_z = _standardize(legacy, current)
    difference = current_z - legacy_z
    legacy_distances = pdist(legacy_z)
    current_distances = pdist(current_z)

    n_components = min(10, legacy_z.shape[1], legacy_z.shape[0] - 1)
    _, singular_values, vt = np.linalg.svd(legacy_z, full_matrices=False)
    components = vt[:n_components]
    legacy_pc = legacy_z @ components.T
    current_pc = current_z @ components.T
    explained_variance = singular_values**2 / max(legacy_z.shape[0] - 1, 1)
    explained_ratio = explained_variance / explained_variance.sum()
    pc_correlations = [
        float(pearsonr(legacy_pc[:, i], current_pc[:, i]).statistic)
        for i in range(legacy_pc.shape[1])
    ]

    return {
        "standardized_rmse": float(np.sqrt(np.mean(difference**2))),
        "median_sample_cosine_similarity": float(
            np.median(
                np.sum(legacy_z * current_z, axis=1)
                / (
                    np.linalg.norm(legacy_z, axis=1)
                    * np.linalg.norm(current_z, axis=1)
                )
            )
        ),
        "nearest_legacy_self_match_fraction": _nearest_reference_accuracy(
            legacy_z, current_z
        ),
        "pairwise_distance_pearson": float(
            pearsonr(legacy_distances, current_distances).statistic
        ),
        "pairwise_distance_spearman": float(
            spearmanr(legacy_distances, current_distances).statistic
        ),
        "legacy_temperature_r2": _multivariate_group_r2(
            legacy_z, temperature
        ),
        "current_temperature_r2": _multivariate_group_r2(
            current_z, temperature
        ),
        "legacy_genotype_r2": _multivariate_group_r2(legacy_z, genotype),
        "current_genotype_r2": _multivariate_group_r2(current_z, genotype),
        "legacy_pca_explained_variance_first_10": [
            float(x) for x in explained_ratio[:n_components]
        ],
        "legacy_vs_current_pc_score_correlations": pc_correlations,
    }


def main() -> None:
    legacy = pd.read_csv(LEGACY_PATH)
    current_read_columns = [
        "well_id",
        "physical_embryo_id",
        "temperature",
        "genotype",
        "embedding_model_name",
    ] + [_current_column(i) for i in range(100)]
    current = pd.read_parquet(CURRENT_PATH, columns=current_read_columns)

    legacy["well"] = legacy["snip_id"].astype(str).map(
        lambda value: (
            WELL_RE.search(value).group(1) if WELL_RE.search(value) else None
        )
    )
    current["well"] = current["well_id"].astype(str).str.rsplit("_", n=1).str[-1]

    # Current IDs are 1-based.  The first registered embryo is the closest
    # counterpart to legacy e00 in the two multi-embryo wells.
    if "physical_embryo_id" in current.columns:
        current = current[
            current["physical_embryo_id"].astype(str).str.endswith("_e01")
        ].copy()
    current = current.drop_duplicates("well", keep="first")

    metadata_columns = [
        column
        for column in ("well", "temperature", "genotype", "embedding_model_name")
        if column in current.columns
    ]
    legacy_columns = ["well"] + [_legacy_column(i) for i in range(100)]
    current_columns = metadata_columns + [_current_column(i) for i in range(100)]
    paired = legacy[legacy_columns].merge(
        current[current_columns], on="well", how="inner", validate="one_to_one"
    )

    legacy_array = paired[[_legacy_column(i) for i in range(100)]].to_numpy(
        dtype=float
    )
    current_array = paired[[_current_column(i) for i in range(100)]].to_numpy(
        dtype=float
    )
    correlation = _correlation_matrix(legacy_array, current_array)
    diagonal = np.diag(correlation)
    row_ind, col_ind = linear_sum_assignment(-np.abs(correlation))
    assignment = dict(zip(row_ind.tolist(), col_ind.tolist()))

    per_dimension = pd.DataFrame(
        {
            "dimension": np.arange(100),
            "family": ["nuisance" if i < 20 else "biological" for i in range(100)],
            "direct_correlation": diagonal,
            "best_matching_current_dimension": np.argmax(
                np.abs(correlation), axis=1
            ),
            "best_matching_abs_correlation": np.max(
                np.abs(correlation), axis=1
            ),
            "hungarian_current_dimension": [
                assignment[i] for i in range(100)
            ],
        }
    )

    temperature = paired["temperature"].astype(str).to_numpy()
    genotype = paired["genotype"].astype(str).to_numpy()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "n_shared_wells": int(len(paired)),
        "current_model_names": sorted(
            paired.get("embedding_model_name", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "dimension_ordering": {
            "hungarian_identity_matches": int(
                sum(index == mapped for index, mapped in assignment.items())
            ),
            "argmax_identity_matches": int(
                np.sum(np.argmax(np.abs(correlation), axis=1) == np.arange(100))
            ),
            "median_direct_correlation": float(np.median(diagonal)),
            "direct_correlation_q05_q95": [
                float(x) for x in np.quantile(diagonal, [0.05, 0.95])
            ],
            "n_direct_correlation_ge_0_9": int(np.sum(diagonal >= 0.9)),
            "n_direct_correlation_ge_0_75": int(np.sum(diagonal >= 0.75)),
            "n_direct_correlation_lt_0_5": int(np.sum(diagonal < 0.5)),
        },
        "all_100_dimensions": _space_metrics(
            legacy_array,
            current_array,
            temperature=temperature,
            genotype=genotype,
        ),
        "biological_dimensions_20_99": _space_metrics(
            legacy_array[:, 20:],
            current_array[:, 20:],
            temperature=temperature,
            genotype=genotype,
        ),
        "nuisance_dimensions_0_19": _space_metrics(
            legacy_array[:, :20],
            current_array[:, :20],
            temperature=temperature,
            genotype=genotype,
        ),
    }

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "embedding_impact_probe.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    per_dimension.to_csv(HERE / "embedding_impact_per_dimension.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
