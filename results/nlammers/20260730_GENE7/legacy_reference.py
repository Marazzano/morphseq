"""Reference-set construction and QC, recapitulating the 2025-Q1 chain.

Source of truth for both the wildtype reference and the 20240813 hotfish cohort is the legacy
training stats table::

    models/legacy/20241107_ds_sweep01_optimum/embryo_stats_df.csv

which carries the descriptive metadata AND the 100-dim latent block inline. This is the ``morph_df``
of ``results/nlammers/20250310/fit_morph_spline_v2.ipynb`` and
``results/nlammers/20250217/add_temperature_info.ipynb``, and the recipes below are ported from those
notebooks rather than re-invented.

Two departures from the notebooks, both forced by the cluster environment:

1. **Temperature is read from the plate workbooks at load time instead of being patched into
   ``embryo_stats_df.csv`` in place.** ``add_temperature_info.ipynb`` overwrote the ``temperature``
   column and re-saved the CSV; that ran against the Dropbox copy, so the cluster copy still carries
   the pre-patch placeholder (a uniform 22.0 for every 20240813 well, which is not a real
   temperature). ``load_workbook_temperatures`` reads the same ``temperature`` sheet the notebook
   read, and ``attach_workbook_temperature`` applies it — but nothing is written back to the shared
   CSV. Mutating a 415 MB shared input as a side effect of an analysis is what made this discrepancy
   invisible in the first place.

2. **No polynomial surface / ``mdl_stage_hpf``.** The notebook fit a degree-2 ``PolynomialFeatures``
   regression from PCA coordinates to stage, then used it to re-parameterize the spline. That is
   deliberately omitted here, so ``predicted_stage_hpf`` is the stage axis throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMPANION_PATH = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/legacy/"
    "20241107_ds_sweep01_optimum/embryo_stats_df.csv"
)
PLATE_METADATA_DIR = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/input/plate_metadata"
)

# --- Reference-set recipe (20250310/fit_morph_spline_v2.ipynb, cell 7) --------------------------
# "search for reference embryos from timelapse data that closely overlap with 28C, but which also
#  extend out into later timepoints"
REFERENCE_PERT_NAME = "wt_ab"
REFERENCE_START_STAGE = 18.0  # embryo must be imaged at or before this stage
REFERENCE_TARGET_STAGE = 42.0  # ...and still alive at or after this one

# --- 20240813 hotfish cohort (same notebook, cell 5) -------------------------------------------
HOTFISH_EXPERIMENTS: tuple[str, ...] = (
    "20240813_24hpf",
    "20240813_30hpf",
    "20240813_36hpf",
)
# 20240813_extras is deliberately excluded: it has no sequencing counterpart and no hash map, and the
# notebook commented it out of HF_experiments too.

# Hand-curated outlier snips dropped by the notebook (cell 5). Kept as an explicit list rather than
# folded into a numeric filter, because that is what it is: three embryos inspected and rejected.
HOTFISH_OUTLIER_SNIPS: tuple[str, ...] = (
    "20240813_24hpf_F06_e00_t0000",
    "20240813_36hpf_D03_e00_t0000",
    "20240813_36hpf_C03_e00_t0000",
)

HOTFISH_CONTROL_TEMPERATURE = 28.5  # the 20240813 control arm; anchors the weighted spline

TEMPERATURE_SHEET = "temperature"
_PLATE_ROWS = "ABCDEFGH"

# Metadata columns carried alongside the latent block.
META_COLUMNS: tuple[str, ...] = (
    "snip_id",
    "embryo_id",
    "experiment_date",
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
)

BIOLOGICAL_LATENT_PATTERN = r"z_mu_b"


def biological_latent_columns(columns: "list[str]") -> list[str]:
    """The ``z_mu_b_*`` columns, in ascending dimension order.

    The notebooks selected these with ``re.search("z_mu_b", col)``. Sorting by the embedded index
    matters: these names run ``z_mu_b_20`` .. ``z_mu_b_99`` and a lexical sort would order
    ``z_mu_b_100`` before ``z_mu_b_20`` if the checkpoint ever grew past 100 dims.
    """
    hits = []
    for column in columns:
        if re.search(BIOLOGICAL_LATENT_PATTERN, str(column)):
            match = re.search(r"(\d+)$", str(column))
            hits.append((int(match.group(1)) if match else -1, str(column)))
    return [name for _, name in sorted(hits)]


@dataclass(frozen=True)
class ReferenceSets:
    """The three cohorts the downstream PCA and spline consume.

    ``reference`` is the wildtype timelapse population; ``hotfish`` is 20240813 with workbook
    temperatures attached and curated outliers removed; ``latent_columns`` is the shared
    ``z_mu_b_*`` block both are expressed in.
    """

    reference: pd.DataFrame
    hotfish: pd.DataFrame
    latent_columns: tuple[str, ...]

    @property
    def hotfish_controls(self) -> pd.DataFrame:
        """The 20240813 embryos at the control temperature — the weighted spline's anchor."""
        return self.hotfish.loc[
            np.isclose(
                pd.to_numeric(self.hotfish["temperature"], errors="coerce"),
                HOTFISH_CONTROL_TEMPERATURE,
            )
        ]


def load_workbook_temperatures(
    experiment_ids: "tuple[str, ...] | list[str]" = HOTFISH_EXPERIMENTS,
    *,
    plate_metadata_dir: Path | None = None,
) -> pd.DataFrame:
    """Flatten each experiment's ``temperature`` sheet to one row per well.

    Ported from ``20250217/add_temperature_info.ipynb`` cell 6: read the 8xN grid, walk it
    row-major, and mint ``{row_letter}{col:02d}`` well labels.

    Returns:
        ``experiment_date``, ``well_index``, ``temperature`` — blanks dropped.
    """
    directory = plate_metadata_dir or PLATE_METADATA_DIR
    frames = []
    for experiment_id in experiment_ids:
        workbook = directory / f"{experiment_id}_well_metadata.xlsx"
        if not workbook.is_file():
            raise FileNotFoundError(f"plate workbook not found: {workbook}")
        grid = pd.read_excel(workbook, sheet_name=TEMPERATURE_SHEET, index_col=0)
        records = []
        for row_position in range(grid.shape[0]):
            for col_position in range(grid.shape[1]):
                value = grid.iloc[row_position, col_position]
                if pd.isna(value):
                    continue
                records.append(
                    {
                        "experiment_date": experiment_id,
                        "well_index": f"{_PLATE_ROWS[row_position]}{col_position + 1:02d}",
                        "temperature": float(value),
                    }
                )
        frames.append(pd.DataFrame.from_records(records))
    return pd.concat(frames, ignore_index=True)


def _well_index_from_embryo_id(embryo_id: str) -> str:
    """The well label embedded in a legacy embryo_id (``..._A01_e00`` -> ``A01``).

    Matches the notebook's ``eid.split("_")[-2]``. Legacy ids carry no channel token, so the well is
    always the second-to-last underscore token.
    """
    return str(embryo_id).split("_")[-2]


def attach_workbook_temperature(
    frame: pd.DataFrame, temperatures: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Replace ``temperature`` with the authoritative per-well workbook value.

    The companion's stored ``temperature`` for 20240813 is a uniform placeholder, so it is
    overwritten rather than merged into. Fails loud if any row goes unmatched: a silent NaN here
    would quietly drop embryos from the control arm and shift the spline.
    """
    resolved = (
        temperatures
        if temperatures is not None
        else load_workbook_temperatures(tuple(frame["experiment_date"].unique()))
    )
    out = frame.copy()
    out["well_index"] = [_well_index_from_embryo_id(value) for value in out["embryo_id"]]
    out = out.drop(columns=["temperature"], errors="ignore").merge(
        resolved, on=["experiment_date", "well_index"], how="left"
    )

    unmatched = out.loc[out["temperature"].isna(), ["experiment_date", "well_index"]]
    if not unmatched.empty:
        preview = unmatched.drop_duplicates().head(8).to_dict(orient="records")
        raise ValueError(
            f"{len(unmatched)} row(s) have no workbook temperature, e.g. {preview}. "
            "The temperature sheet and the imaged wells disagree."
        )
    return out


def load_companion(
    *,
    companion_path: Path | None = None,
    columns: "list[str] | None" = None,
) -> pd.DataFrame:
    """Read the legacy stats table: metadata plus the ``z_mu_b_*`` latent block.

    The file is ~415 MB and 231 columns wide, so the column subset is resolved from the header first
    and only what is needed is parsed.
    """
    path = companion_path or COMPANION_PATH
    if not path.is_file():
        raise FileNotFoundError(f"legacy companion not found: {path}")

    header = pd.read_csv(path, nrows=0)
    latent_columns = biological_latent_columns(list(header.columns))
    wanted = columns or [c for c in META_COLUMNS if c in header.columns] + latent_columns
    return pd.read_csv(path, usecols=wanted, low_memory=False)


def select_reference_embryos(companion: pd.DataFrame) -> pd.DataFrame:
    """The wildtype reference population, per ``fit_morph_spline_v2.ipynb`` cell 7.

    An embryo qualifies when its genotype is ``wt_ab`` AND its imaged stage range spans from at or
    before ``REFERENCE_START_STAGE`` to at or beyond ``REFERENCE_TARGET_STAGE`` — i.e. a timelapse
    embryo that both overlaps the early window and survives late enough to define the trajectory's
    far end. Selection is at EMBRYO level, then all of that embryo's snips are returned.

    Note this is a genotype+longevity filter, not a per-experiment one: it draws from every timelapse
    that contains qualifying embryos, so the reference is not synonymous with any single experiment.
    """
    spans = (
        companion.loc[:, ["experiment_date", "embryo_id", "predicted_stage_hpf", "short_pert_name"]]
        .groupby(["experiment_date", "embryo_id", "short_pert_name"])["predicted_stage_hpf"]
        .agg(["min", "max"])
        .reset_index()
    )
    qualifying = spans.loc[
        (spans["short_pert_name"] == REFERENCE_PERT_NAME)
        & (spans["min"] <= REFERENCE_START_STAGE)
        & (spans["max"] >= REFERENCE_TARGET_STAGE)
    ]
    return companion.merge(qualifying.loc[:, ["embryo_id"]], on="embryo_id", how="inner")


def select_hotfish(
    companion: pd.DataFrame,
    *,
    experiments: "tuple[str, ...]" = HOTFISH_EXPERIMENTS,
    drop_outliers: bool = True,
) -> pd.DataFrame:
    """The 20240813 cohort with workbook temperatures and curated outliers removed.

    Per ``fit_morph_spline_v2.ipynb`` cell 5, plus the temperature repair from
    ``add_temperature_info.ipynb``.
    """
    cohort = companion.loc[companion["experiment_date"].isin(experiments)].reset_index(drop=True)
    if drop_outliers:
        cohort = cohort.loc[~cohort["snip_id"].isin(HOTFISH_OUTLIER_SNIPS)]
    return attach_workbook_temperature(cohort).reset_index(drop=True)


def build_reference_sets(
    *, companion: pd.DataFrame | None = None, companion_path: Path | None = None
) -> ReferenceSets:
    """Load the companion once and derive both cohorts from it."""
    resolved = companion if companion is not None else load_companion(companion_path=companion_path)
    latent_columns = tuple(biological_latent_columns(list(resolved.columns)))
    if not latent_columns:
        raise ValueError("no z_mu_b_* latent columns found in the companion table.")
    return ReferenceSets(
        reference=select_reference_embryos(resolved),
        hotfish=select_hotfish(resolved),
        latent_columns=latent_columns,
    )
