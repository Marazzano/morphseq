"""Refit the WT morphology reference using the August 2026 pipeline embeddings.

This is the updated-embedding counterpart of the reusable machinery in
``results/nlammers/20260730_GENE7``.  It preserves the established analysis
recipe while making the image-generation boundary explicit:

* read ``feature_extraction/*/latent_embeddings/*_latents.parquet`` directly;
* use dimensions 20--99 (the checkpoint's biological block);
* fit PCA on qualifying longitudinal WT references plus all 20240813 Hotfish WT;
* fit weighted and unweighted local-principal-curve WT splines;
* restore the degree-3 morphology-to-stage model from the 2025-Q1 notebook;
* project GENE7 without allowing it to influence PCA, staging, or the spline.

The legacy companion table supplies descriptive/staging metadata only.  None of
its historical latent values are used.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from core.functions.spline_fitting_v2 import spline_fit_wrapper  # noqa: E402
from morphseq_integration import build_master_table  # noqa: E402


PIPELINE_OUTPUT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
)
PIPELINE_INPUT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/input"
)
COMPANION_PATH = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/legacy/"
    "20241107_ds_sweep01_optimum/embryo_stats_df.csv"
)

DATA_DIR = HERE / "data/morphology_reference"
MODEL_DIR = HERE / "models/morphology_reference"

REFERENCE_EXPERIMENTS = ("20240626", "20240812", "20250215")
HOTFISH_EXPERIMENTS = (
    "20240813_24hpf",
    "20240813_30hpf",
    "20240813_36hpf",
)
GENE7_EXPERIMENTS = (
    "20250612_24hpf_ctrl_atf6",
    "20250612_24hpf_wfs1_ctcf",
    "20250612_30hpf_ctrl_atf6",
    "20250612_30hpf_wfs1_ctcf",
    "20250612_36hpf_ctrl_atf6",
    "20250612_36hpf_wfs1_ctcf",
)

REFERENCE_PERTURBATION = "wt_ab"
REFERENCE_START_STAGE = 18.0
REFERENCE_TARGET_STAGE = 42.0
HOTFISH_CONTROL_TEMPERATURE = 28.5
HOTFISH_OUTLIERS = {
    "20240813_24hpf_F06_e00_t0000",
    "20240813_36hpf_D03_e00_t0000",
    "20240813_36hpf_C03_e00_t0000",
}

BIOLOGICAL_DIMS = tuple(range(20, 100))
LATENT_COLUMNS = tuple(f"z_mu_b_{index:02d}" for index in BIOLOGICAL_DIMS)
FLAT_LATENT_COLUMNS = tuple(f"z_mu_{index:02d}" for index in BIOLOGICAL_DIMS)

N_COMPONENTS = 10
PCA_COLUMNS = tuple(f"PCA_{index:02d}_bio" for index in range(N_COMPONENTS))
STAGE_MODEL_DEGREE = 3
RANDOM_SEED = 42

SPLINE_ALPHA = 0.25
SPLINE_N_BOOTS = 50
SPLINE_N_POINTS = 2500
SPLINE_BOOT_SIZE = 1000

_CURRENT_SNIP_RE = re.compile(
    r"^(?P<experiment_id>.+)_(?P<well_index>[A-H]\d{2})_e(?P<embryo>\d+)_BF_t(?P<time>\d+)$"
)
_LEGACY_SNIP_RE = re.compile(
    r"^(?P<experiment_id>.+)_(?P<well_index>[A-H]\d{2})_e(?P<embryo>\d+)_t(?P<time>\d+)$"
)


def _parse_snip_ids(series: pd.Series, pattern: re.Pattern[str]) -> pd.DataFrame:
    parsed = series.astype(str).str.extract(pattern)
    if parsed.isna().any(axis=None):
        examples = series.loc[parsed.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"Could not parse {len(examples)}+ snip ids, e.g. {examples}")
    parsed["time_index"] = parsed.pop("time").astype(int)
    parsed["local_embryo_index"] = parsed.pop("embryo").astype(int)
    return parsed


def _latent_path(experiment_id: str) -> Path:
    return (
        PIPELINE_OUTPUT
        / "feature_extraction"
        / experiment_id
        / "latent_embeddings"
        / f"{experiment_id}_latents.parquet"
    )


def load_current_latents(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for experiment_id in experiment_ids:
        path = _latent_path(experiment_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["snip_id", *FLAT_LATENT_COLUMNS])
        parsed = _parse_snip_ids(frame["snip_id"], _CURRENT_SNIP_RE)
        if set(parsed["experiment_id"]) != {experiment_id}:
            raise ValueError(f"{path.name}: embedded experiment id disagrees with filename")
        frame = pd.concat([frame, parsed], axis=1)
        frame = frame.rename(
            columns={
                flat: biological
                for flat, biological in zip(FLAT_LATENT_COLUMNS, LATENT_COLUMNS)
            }
        )
        frame = frame.rename(columns={"snip_id": "pipeline_snip_id"})
        frame["well_id"] = frame["experiment_id"] + "_" + frame["well_index"]
        frame["physical_embryo_id"] = (
            frame["well_id"]
            + "_e"
            + frame["local_embryo_index"].astype(str).str.zfill(2)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_companion_metadata(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    wanted = [
        "snip_id",
        "embryo_id",
        "experiment_time",
        "temperature",
        "medium",
        "short_pert_name",
        "control_flag",
        "phenotype",
        "predicted_stage_hpf",
        "surface_area_um",
        "length_um",
        "width_um",
        "train_cat",
        "recon_mse",
    ]
    header = pd.read_csv(COMPANION_PATH, nrows=0)
    wanted = [column for column in wanted if column in header.columns]
    companion = pd.read_csv(COMPANION_PATH, usecols=wanted, low_memory=False)
    parsed = _parse_snip_ids(companion["snip_id"], _LEGACY_SNIP_RE)
    companion = pd.concat([companion, parsed], axis=1)
    companion = companion.loc[companion["experiment_id"].isin(experiment_ids)].copy()
    return companion.rename(columns={"snip_id": "legacy_snip_id"})


def attach_companion_metadata(
    current: pd.DataFrame, companion: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["experiment_id", "well_index", "time_index"]
    if companion.duplicated(keys).any():
        duplicate = companion.loc[companion.duplicated(keys, keep=False), keys].head()
        raise ValueError(f"Companion metadata is not unique on {keys}:\n{duplicate}")
    merged = current.merge(
        companion.drop(columns=["local_embryo_index"], errors="ignore"),
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    audit = (
        merged.groupby("experiment_id", sort=False)
        .agg(
            pipeline_rows=("pipeline_snip_id", "size"),
            matched_metadata=("_merge", lambda x: int((x == "both").sum())),
            pipeline_wells=("well_id", "nunique"),
        )
        .reset_index()
    )
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge")
    merged["predicted_stage_hpf"] = pd.to_numeric(
        merged["predicted_stage_hpf"], errors="coerce"
    )
    return merged, audit


def select_reference(frame: pd.DataFrame) -> pd.DataFrame:
    spans = (
        frame.groupby(["experiment_id", "embryo_id", "short_pert_name"], dropna=False)[
            "predicted_stage_hpf"
        ]
        .agg(["min", "max"])
        .reset_index()
    )
    qualifying = spans.loc[
        (spans["short_pert_name"] == REFERENCE_PERTURBATION)
        & (spans["min"] <= REFERENCE_START_STAGE)
        & (spans["max"] >= REFERENCE_TARGET_STAGE),
        ["experiment_id", "embryo_id"],
    ]
    selected = frame.merge(
        qualifying, on=["experiment_id", "embryo_id"], how="inner", validate="many_to_one"
    )
    return selected.dropna(subset=["predicted_stage_hpf", *LATENT_COLUMNS]).reset_index(drop=True)


def load_workbook_temperatures(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    """Load workbook-derived temperatures from the pipeline's ingested CSV product."""

    frames = []
    for experiment_id in experiment_ids:
        plate_metadata = (
            PIPELINE_OUTPUT
            / "acquisition"
            / experiment_id
            / "ingest_metadata"
            / "plate_metadata.csv"
        )
        frame = pd.read_csv(
            plate_metadata,
            usecols=["experiment_id", "well_index", "temperature"],
        ).dropna(subset=["temperature"])
        frame["experiment_id"] = experiment_id
        frames.append(frame.rename(columns={"temperature": "workbook_temperature"}))
    return pd.concat(frames, ignore_index=True)


def select_hotfish(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[~frame["legacy_snip_id"].isin(HOTFISH_OUTLIERS)].copy()
    temperatures = load_workbook_temperatures(HOTFISH_EXPERIMENTS)
    selected = selected.drop(columns="temperature", errors="ignore").merge(
        temperatures,
        on=["experiment_id", "well_index"],
        how="left",
        validate="many_to_one",
    )
    if selected["workbook_temperature"].isna().any():
        raise ValueError("Hotfish workbook-temperature join left unmatched rows")
    selected = selected.rename(columns={"workbook_temperature": "temperature"})
    return selected.dropna(subset=[*LATENT_COLUMNS]).reset_index(drop=True)


def _base_perturbation(value: object) -> str:
    parts = [part.strip() for part in str(value).split(",")]
    parts = [part for part in parts if part.lower() not in {"hot", "cold"}]
    label = ",".join(parts) if parts else str(value)
    return "Control" if label.lower() in {"control", "ctrl-inj", "ctrl"} else label


def build_gene7() -> tuple[pd.DataFrame, pd.DataFrame]:
    current = load_current_latents(GENE7_EXPERIMENTS)
    # The paired sequencing unit is one imaging well. Match the established bridge by selecting
    # the lowest current embryo index per well, recording the multiplicity in the audit.
    multiplicity = (
        current.groupby(["experiment_id", "well_id"])
        .agg(n_pipeline_embryos=("physical_embryo_id", "nunique"))
        .reset_index()
    )
    current = (
        current.sort_values(["well_id", "time_index", "local_embryo_index"])
        .drop_duplicates("well_id", keep="first")
        .merge(multiplicity, on=["experiment_id", "well_id"], how="left")
    )

    master = build_master_table(
        experiment_ids=list(GENE7_EXPERIMENTS), exclusions="drop", verify_pairing=True
    )
    metadata_columns = [
        "experiment_id",
        "well_id",
        "well_index",
        "seq_sample_id",
        "pairing_status",
        "has_seq",
        "embryo_ID",
        "perturbation",
        "target",
        "type",
        "allele",
        "strain",
        "expt",
        "sci_batch",
        "timepoint",
        "stage",
        "temp",
        "pheno",
    ]
    metadata_columns = [column for column in metadata_columns if column in master.columns]
    metadata = master.loc[:, metadata_columns].drop_duplicates("well_id")
    merged = metadata.merge(
        current,
        on=["experiment_id", "well_id", "well_index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    audit = (
        merged.groupby("experiment_id", sort=False)
        .agg(
            curated_paired_wells=("well_id", "size"),
            wells_with_new_embedding=("_merge", lambda x: int((x == "both").sum())),
            wells_with_sequence=("has_seq", "sum"),
        )
        .reset_index()
    )
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge")
    merged["temperature"] = pd.to_numeric(merged["temp"], errors="coerce")
    merged["collection_stage_hpf"] = pd.to_numeric(merged["timepoint"], errors="coerce")
    merged["perturbation_group"] = merged["target"].map(_base_perturbation)
    merged["embryo_id"] = merged["physical_embryo_id"]
    return merged.reset_index(drop=True), audit


def project(pca: PCA, frame: pd.DataFrame, source: str) -> pd.DataFrame:
    coordinates = pca.transform(frame.loc[:, LATENT_COLUMNS])
    projected = frame.copy()
    projected.loc[:, PCA_COLUMNS] = coordinates
    projected["source"] = source
    return projected


def spline_weights(fitting_set: pd.DataFrame) -> np.ndarray:
    hotfish = (fitting_set["source"] == "hotfish_control").to_numpy()
    weights = np.empty(len(fitting_set), dtype=float)
    weights[hotfish] = SPLINE_ALPHA / hotfish.sum()
    weights[~hotfish] = (1.0 - SPLINE_ALPHA) / (~hotfish).sum()
    return weights / weights.sum()


def fit_splines(reference: pd.DataFrame, hotfish: pd.DataFrame) -> dict[str, pd.DataFrame]:
    controls = hotfish.loc[
        np.isclose(hotfish["temperature"], HOTFISH_CONTROL_TEMPERATURE)
    ].copy()
    controls["source"] = "hotfish_control"
    fitting_set = pd.concat([reference, controls], ignore_index=True)
    fitting_set["spline_stage"] = np.floor(fitting_set["predicted_stage_hpf"])
    shared = dict(
        fit_cols=list(PCA_COLUMNS),
        stage_col="spline_stage",
        n_boots=SPLINE_N_BOOTS,
        n_spline_points=SPLINE_N_POINTS,
        boot_size=SPLINE_BOOT_SIZE,
    )
    np.random.seed(RANDOM_SEED)
    unweighted = spline_fit_wrapper(fitting_set, obs_weights=None, **shared)
    np.random.seed(RANDOM_SEED)
    weighted = spline_fit_wrapper(
        fitting_set, obs_weights=spline_weights(fitting_set), **shared
    )
    return {"weighted": weighted, "unweighted": unweighted}


def new_stage_model() -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=STAGE_MODEL_DEGREE, include_bias=True)),
            ("linear", LinearRegression()),
        ]
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, object]:
    return {
        "validation": label,
        "n": len(y_true),
        "mae_hpf": mean_absolute_error(y_true, y_pred),
        "rmse_hpf": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def validate_stage_model(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = reference.loc[:, PCA_COLUMNS].to_numpy(float)
    y = reference["predicted_stage_hpf"].to_numpy(float)
    embryos = reference["embryo_id"].astype(str).to_numpy()
    experiments = reference["experiment_id"].astype(str).to_numpy()
    records = []
    predictions = []

    group_cv = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(group_cv.split(X, y, groups=embryos), start=1):
        model = clone(new_stage_model()).fit(X[train], y[train])
        pred = model.predict(X[test])
        records.append(_metrics(y[test], pred, f"embryo_group_5fold:{fold}"))
        predictions.append(
            pd.DataFrame(
                {
                    "validation": "embryo_group_5fold",
                    "fold": fold,
                    "row_index": reference.index[test],
                    "experiment_id": experiments[test],
                    "embryo_id": embryos[test],
                    "observed_stage_hpf": y[test],
                    "predicted_stage_hpf": pred,
                }
            )
        )

    logo = LeaveOneGroupOut()
    for train, test in logo.split(X, y, groups=experiments):
        held_out = str(np.unique(experiments[test])[0])
        model = clone(new_stage_model()).fit(X[train], y[train])
        pred = model.predict(X[test])
        records.append(_metrics(y[test], pred, f"leave_experiment_out:{held_out}"))
        predictions.append(
            pd.DataFrame(
                {
                    "validation": "leave_experiment_out",
                    "fold": held_out,
                    "row_index": reference.index[test],
                    "experiment_id": experiments[test],
                    "embryo_id": embryos[test],
                    "observed_stage_hpf": y[test],
                    "predicted_stage_hpf": pred,
                }
            )
        )

    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True)


def stage_spline(spline: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    out = spline.copy()
    raw = model.predict(out.loc[:, PCA_COLUMNS])
    if np.nanmedian(raw[: max(5, len(raw) // 20)]) > np.nanmedian(
        raw[-max(5, len(raw) // 20) :]
    ):
        out = out.iloc[::-1].reset_index(drop=True)
        raw = raw[::-1]
    order = np.arange(len(out), dtype=float)
    out["stage_model_raw_hpf"] = raw
    out["stage_hpf"] = IsotonicRegression(increasing=True).fit_transform(order, raw)
    out["spline_point"] = np.arange(len(out))
    return out


def annotate_distance_and_stage(frame: pd.DataFrame, spline: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    nearest = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        spline.loc[:, PCA_COLUMNS]
    )
    distance, index = nearest.kneighbors(result.loc[:, PCA_COLUMNS])
    result["dist_to_wt_spline"] = distance[:, 0]
    result["nearest_spline_point"] = index[:, 0]
    result["spline_stage_hpf"] = spline.iloc[index[:, 0]]["stage_hpf"].to_numpy()
    return result


def add_stage_predictions(frame: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    result = frame.copy()
    result["morph_stage_hpf"] = model.predict(result.loc[:, PCA_COLUMNS])
    if "collection_stage_hpf" in result:
        result["morph_stage_shift_hpf"] = (
            result["morph_stage_hpf"] - result["collection_stage_hpf"]
        )
    return result


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] Load updated reference and Hotfish embeddings + legacy metadata")
    reference_current = load_current_latents(REFERENCE_EXPERIMENTS)
    hotfish_current = load_current_latents(HOTFISH_EXPERIMENTS)
    companion = load_companion_metadata(REFERENCE_EXPERIMENTS + HOTFISH_EXPERIMENTS)
    reference_joined, reference_audit = attach_companion_metadata(reference_current, companion)
    hotfish_joined, hotfish_audit = attach_companion_metadata(hotfish_current, companion)
    reference = select_reference(reference_joined)
    hotfish = select_hotfish(hotfish_joined)

    print("[2/7] Load updated GENE7 embeddings + paired sequencing metadata")
    gene7, gene7_audit = build_gene7()

    print("[3/7] Fit WT PCA; project WT, Hotfish, and GENE7")
    fitting_latents = pd.concat(
        [reference.loc[:, LATENT_COLUMNS], hotfish.loc[:, LATENT_COLUMNS]], ignore_index=True
    )
    pca = PCA(n_components=N_COMPONENTS).fit(fitting_latents)
    reference_pca = project(pca, reference, "reference")
    hotfish_pca = project(pca, hotfish, "hotfish")
    gene7_pca = project(pca, gene7, "GENE7")

    print("[4/7] Validate and refit degree-3 WT morphology staging model")
    stage_metrics, stage_cv = validate_stage_model(reference_pca)
    stage_model = new_stage_model().fit(
        reference_pca.loc[:, PCA_COLUMNS], reference_pca["predicted_stage_hpf"]
    )
    reference_pca = add_stage_predictions(reference_pca, stage_model)
    hotfish_pca = add_stage_predictions(hotfish_pca, stage_model)
    gene7_pca = add_stage_predictions(gene7_pca, stage_model)

    print("[5/7] Fit weighted and unweighted WT reference splines")
    splines = fit_splines(reference_pca, hotfish_pca)
    splines = {name: stage_spline(frame, stage_model) for name, frame in splines.items()}

    print("[6/7] Annotate WT and GENE7 staging/severity readouts")
    primary_spline = splines["weighted"]
    reference_pca = annotate_distance_and_stage(reference_pca, primary_spline)
    hotfish_pca = annotate_distance_and_stage(hotfish_pca, primary_spline)
    gene7_pca = annotate_distance_and_stage(gene7_pca, primary_spline)

    print("[7/7] Write fitted artifacts and provenance")
    reference_pca.to_parquet(DATA_DIR / "reference_pca.parquet", index=False)
    hotfish_pca.to_parquet(DATA_DIR / "hotfish_pca.parquet", index=False)
    gene7_pca.to_parquet(DATA_DIR / "gene7_pca.parquet", index=False)
    for name, spline in splines.items():
        spline.to_csv(DATA_DIR / f"wt_spline_{name}.csv", index=False)
    stage_metrics.to_csv(DATA_DIR / "stage_model_validation.csv", index=False)
    stage_cv.to_parquet(DATA_DIR / "stage_model_cv_predictions.parquet", index=False)

    variance = pd.DataFrame(
        {
            "component": np.arange(1, N_COMPONENTS + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance.to_csv(DATA_DIR / "pca_variance.csv", index=False)

    reference_selected = (
        reference_pca.groupby("experiment_id")
        .agg(
            selected_rows=("pipeline_snip_id", "size"),
            selected_embryos=("embryo_id", "nunique"),
            stage_min=("predicted_stage_hpf", "min"),
            stage_max=("predicted_stage_hpf", "max"),
        )
        .reset_index()
    )
    coverage = pd.concat(
        [
            reference_audit.assign(cohort="reference_pipeline"),
            hotfish_audit.assign(cohort="hotfish_pipeline"),
        ],
        ignore_index=True,
    )
    coverage.to_csv(DATA_DIR / "embedding_metadata_join_audit.csv", index=False)
    gene7_audit.to_csv(DATA_DIR / "gene7_coverage_audit.csv", index=False)
    reference_selected.to_csv(DATA_DIR / "reference_selection_summary.csv", index=False)

    joblib.dump(pca, MODEL_DIR / "wt_morphology_pca.joblib")
    joblib.dump(stage_model, MODEL_DIR / "wt_morphology_stage_model.joblib")

    provenance = {
        "embedding_model": "20241107_ds_sweep01_optimum",
        "embedding_source": "pipeline/output/feature_extraction/*/latent_embeddings",
        "reference_experiments": list(REFERENCE_EXPERIMENTS),
        "hotfish_experiments": list(HOTFISH_EXPERIMENTS),
        "gene7_experiments": list(GENE7_EXPERIMENTS),
        "biological_dimensions": [min(BIOLOGICAL_DIMS), max(BIOLOGICAL_DIMS)],
        "n_pca_components": N_COMPONENTS,
        "stage_model": f"PolynomialFeatures(degree={STAGE_MODEL_DEGREE}) + LinearRegression",
        "reference_filter": {
            "short_pert_name": REFERENCE_PERTURBATION,
            "min_stage_at_or_before": REFERENCE_START_STAGE,
            "max_stage_at_or_after": REFERENCE_TARGET_STAGE,
        },
        "spline": {
            "primary": "weighted",
            "hotfish_control_temperature": HOTFISH_CONTROL_TEMPERATURE,
            "hotfish_sampling_mass": SPLINE_ALPHA,
            "n_bootstraps": SPLINE_N_BOOTS,
            "points": SPLINE_N_POINTS,
            "bootstrap_size": SPLINE_BOOT_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "hotfish_temperature_source": (
            "pipeline/output/acquisition/*/ingest_metadata/plate_metadata.csv"
        ),
        "gene7_embryo_selection": "lowest current local_embryo_index per curated paired well",
    }
    (DATA_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print("\nReference selection")
    print(reference_selected.to_string(index=False))
    print("\nStage validation")
    print(stage_metrics.to_string(index=False))
    print(f"\nGENE7 projected rows: {len(gene7_pca)}")
    print(f"Artifacts: {DATA_DIR}")


if __name__ == "__main__":
    main()
