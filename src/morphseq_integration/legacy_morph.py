"""Read the legacy VAE morphology latents into the canonical identity spine.

The legacy embeddings live at::

    <legacy_root>/morph_latents_{experiment_id}.csv

and are the output of the ``20241107_ds_sweep01_optimum`` checkpoint. They are used in preference to
the current pipeline's ``legacy_embeddings`` because the pipeline's snip raster is degraded relative
to the build that trained this checkpoint (``target_pixel_size_um`` 7.8 vs 6.5, plus an unresolved
saturation difference), so the pipeline's latents are not interchangeable with these.

Why this module exists rather than a call to the pipeline's parsers
------------------------------------------------------------------
Legacy snip ids predate the current identifier contract and are NOT parseable by
``data_pipeline.shared.identifiers``:

    legacy   20250612_30hpf_ctrl_atf6_A01_e00_t0000     no channel token, zero-based embryo
    current  20250612_30hpf_ctrl_atf6_A01_e01_BF_t0000  channel token, one-based embryo

``parse_physical_embryo_id`` rejects ``_e00`` outright (the contract is one-based: "embryo 1 is the
first embryo, not embryo 0") and ``parse_embryo_id`` requires a channel from the canonical
vocabulary. Feeding legacy ids to either raises. So the legacy grammar is parsed here, by a regex
that is deliberately scoped to this one historical format, and the result is re-minted through the
canonical ``build_well_id`` / ``build_physical_embryo_id`` constructors. Downstream never sees a
hand-assembled identifier.

The zero-to-one embryo index shift is applied via the pipeline's own
``track_index_to_embryo_index``, so ``e00 -> local_embryo_index 1`` uses the same conversion the
pipeline uses for backend track ids rather than an inline ``+ 1``.

**The embryo index is not a join key.** Legacy ``e00`` and current ``e01`` denote the same animal
only when a well resolved exactly one embryo, which is not universal — three of the 38 legacy files
(``20250624_chem02_35C_T00_1216``, ``20250624_chem02_35C_T01_1711``, ``20250625_chem02_35C_T02_1228``)
carry ``e01`` as well, and the current pipeline resolves 2 embryos in some wells where legacy found 1.
Join on ``well_id``.

Latent naming
-------------
Both legacy tables number the 100 latent dimensions continuously ``00``–``99`` and use the prefix to
mark the disentanglement split at 20: ``z_mu_n_00``–``z_mu_n_19`` (nuisance) then
``z_mu_b_20``–``z_mu_b_99`` (biological), and likewise for ``z_sigma_``. The flat pipeline name
``z_mu_{ii}`` therefore corresponds to legacy index ``ii`` directly. ``flat_latent_name`` and
``legacy_latent_name`` are the two directions of that correspondence; the split point is read off
the observed column names, never assumed.

The metadata companion
----------------------
``<models_root>/20241107_ds_sweep01_optimum/embryo_stats_df.csv`` (~415 MB) carries the descriptive
metadata the latent CSVs lack. It is the TRAINING-set stats table for the Nov 2024 run and covers 44
experiments up to ``20250215`` only — it has **no rows for the 2025 experiments** (including all six
20250612/GENE7 plates), which were embedded later by applying the checkpoint to new data. So
``attach_legacy_metadata`` is a no-op-with-indicator for GENE7 and genuinely useful for
``20240812`` / ``20240813_*``. It is read with an explicit column subset and a row filter so the full
file is never materialized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.shared.identifiers.constructors import (
    build_physical_embryo_id,
    build_well_id,
)
from data_pipeline.shared.identifiers.parsers import track_index_to_embryo_index
from data_pipeline.shared.identifiers.validators import validate_well_index

LEGACY_MODEL_NAME = "20241107_ds_sweep01_optimum"

_DEFAULT_LEGACY_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/legacy"
) / LEGACY_MODEL_NAME

_DEFAULT_MODELS_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/legacy"
) / LEGACY_MODEL_NAME

LATENTS_FILE_PREFIX = "morph_latents_"
LATENTS_FILE_SUFFIX = ".csv"
COMPANION_FILENAME = "embryo_stats_df.csv"

SOURCE_LEGACY = "legacy"

# Legacy snip_id grammar: {experiment_id}_{well_index}_e{NN}_t{NNNN}. No channel token, and the
# embryo index is ZERO-based. Anchored and non-greedy on the tail tokens so the (possibly
# underscore-bearing) experiment_id absorbs only what precedes the well label.
_LEGACY_SNIP_ID_RE = re.compile(
    r"^(?P<experiment_id>.+)_(?P<well_index>[A-Za-z]\d{1,3})_e(?P<embryo>\d+)_t(?P<time>\d+)$"
)

# Latent column grammar, both generations.
_LEGACY_LATENT_RE = re.compile(r"^(?P<kind>z_mu|z_sigma)_(?P<family>[nb])_(?P<index>\d+)$")
_FLAT_LATENT_RE = re.compile(r"^(?P<kind>z_mu|z_sigma)_(?P<index>\d+)$")

NUISANCE_FAMILY = "n"
BIOLOGICAL_FAMILY = "b"
EXPECTED_LATENT_DIMS = 100

SPINE_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "well_index",
    "physical_embryo_id",
    "local_embryo_index",
    "time_index",
    "snip_id",
    "source",
)

# Descriptive columns of the companion worth carrying. Deliberately excludes its own latent block
# (identical to the latent CSVs') and its UMAP coords (fit on the training population, not portable).
COMPANION_COLUMNS: tuple[str, ...] = (
    "snip_id",
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


def default_legacy_root(legacy_root: str | Path | None = None) -> Path:
    """Directory holding the ``morph_latents_*.csv`` files."""
    root = Path(legacy_root) if legacy_root is not None else _DEFAULT_LEGACY_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"[morphseq_integration] legacy latents directory {root} does not exist."
        )
    return root


def default_companion_path(models_root: str | Path | None = None) -> Path:
    """Path to ``embryo_stats_df.csv``, the legacy metadata companion."""
    root = Path(models_root) if models_root is not None else _DEFAULT_MODELS_ROOT
    return root / COMPANION_FILENAME


def available_experiments(legacy_root: str | Path | None = None) -> list[str]:
    """Every experiment_id with a legacy latents file, derived from the FILENAME.

    The filename is the contract, not the ``experiment_date`` column: the two agree for the GENE7
    plates but ``experiment_date`` is a weaker key (it drops the descriptive suffix for some
    experiments and is not unique across files).
    """
    root = default_legacy_root(legacy_root)
    return sorted(
        path.name[len(LATENTS_FILE_PREFIX) : -len(LATENTS_FILE_SUFFIX)]
        for path in root.glob(f"{LATENTS_FILE_PREFIX}*{LATENTS_FILE_SUFFIX}")
    )


def latents_path(experiment_id: str, legacy_root: str | Path | None = None) -> Path:
    """Path to one experiment's legacy latents CSV."""
    root = default_legacy_root(legacy_root)
    return root / f"{LATENTS_FILE_PREFIX}{experiment_id}{LATENTS_FILE_SUFFIX}"


# ---------------------------------------------------------------------------
# Latent-name correspondence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentColumns:
    """The latent block of one table, classified by generation and family.

    ``mu_columns`` / ``sigma_columns`` are in ascending dimension order. ``split_index`` is the first
    biological dimension (20 for this checkpoint), read off the observed names rather than assumed;
    it is ``None`` for a flat table, which carries no family information.
    """

    mu_columns: tuple[str, ...]
    sigma_columns: tuple[str, ...]
    is_legacy_naming: bool
    split_index: int | None

    @property
    def n_dims(self) -> int:
        return len(self.mu_columns)


def classify_latent_columns(columns: "list[str] | tuple[str, ...]") -> LatentColumns:
    """Identify and order the latent columns of either naming generation.

    Raises:
        ValueError: if both namings appear at once (an already-merged frame, where dimension
            identity would be ambiguous), or if mu/sigma dimension sets disagree.
    """
    legacy_hits: dict[str, list[tuple[int, str]]] = {"z_mu": [], "z_sigma": []}
    flat_hits: dict[str, list[tuple[int, str]]] = {"z_mu": [], "z_sigma": []}
    families: dict[int, str] = {}

    for column in columns:
        legacy_match = _LEGACY_LATENT_RE.match(str(column))
        if legacy_match:
            index = int(legacy_match.group("index"))
            legacy_hits[legacy_match.group("kind")].append((index, str(column)))
            families[index] = legacy_match.group("family")
            continue
        flat_match = _FLAT_LATENT_RE.match(str(column))
        if flat_match:
            flat_hits[flat_match.group("kind")].append(
                (int(flat_match.group("index")), str(column))
            )

    legacy_present = bool(legacy_hits["z_mu"])
    flat_present = bool(flat_hits["z_mu"])
    if legacy_present and flat_present:
        raise ValueError(
            "[morphseq_integration] table carries BOTH legacy (z_mu_n_/z_mu_b_) and flat (z_mu_) "
            "latent names. Dimension identity is ambiguous — split the frame by source first."
        )
    if not legacy_present and not flat_present:
        raise ValueError(
            "[morphseq_integration] no z_mu_* latent columns found. Expected either legacy "
            "z_mu_n_NN/z_mu_b_NN or flat z_mu_NN."
        )

    hits = legacy_hits if legacy_present else flat_hits
    mu = tuple(name for _, name in sorted(hits["z_mu"]))
    sigma = tuple(name for _, name in sorted(hits["z_sigma"]))

    mu_indices = sorted(index for index, _ in hits["z_mu"])
    sigma_indices = sorted(index for index, _ in hits["z_sigma"])
    if sigma_indices and mu_indices != sigma_indices:
        raise ValueError(
            "[morphseq_integration] z_mu_* and z_sigma_* cover different dimension indices "
            f"({len(mu_indices)} vs {len(sigma_indices)}); the latent block is malformed."
        )

    split_index = None
    if legacy_present:
        biological = sorted(index for index, family in families.items() if family == BIOLOGICAL_FAMILY)
        split_index = biological[0] if biological else None

    return LatentColumns(
        mu_columns=mu,
        sigma_columns=sigma,
        is_legacy_naming=legacy_present,
        split_index=split_index,
    )


def legacy_latent_name(kind: str, index: int, *, split_index: int = 20) -> str:
    """The legacy name for a latent dimension: ``("z_mu", 5) -> "z_mu_n_05"``.

    Dimensions below ``split_index`` are nuisance (``_n_``), at or above are biological (``_b_``).
    """
    family = NUISANCE_FAMILY if int(index) < int(split_index) else BIOLOGICAL_FAMILY
    return f"{kind}_{family}_{int(index):02d}"


def flat_latent_name(kind: str, index: int) -> str:
    """The flat pipeline name for a latent dimension: ``("z_mu", 5) -> "z_mu_05"``."""
    return f"{kind}_{int(index):02d}"


def to_flat_latent_names(columns: "list[str] | tuple[str, ...]") -> dict[str, str]:
    """Rename map from legacy latent names to flat pipeline names, preserving dimension index.

    ``z_mu_n_05 -> z_mu_05``, ``z_mu_b_37 -> z_mu_37``. Non-latent columns are absent from the map.
    Use when a legacy frame must be stacked with pipeline output; prefer keeping legacy names
    otherwise, since they carry the nuisance/biological split that the flat names lose.
    """
    renames: dict[str, str] = {}
    for column in columns:
        match = _LEGACY_LATENT_RE.match(str(column))
        if match:
            renames[str(column)] = flat_latent_name(
                match.group("kind"), int(match.group("index"))
            )
    return renames


# ---------------------------------------------------------------------------
# snip_id parsing
# ---------------------------------------------------------------------------


def parse_legacy_snip_id(snip_id: str) -> tuple[str, str, int, int]:
    """Decompose a legacy snip_id into ``(experiment_id, well_index, local_embryo_index, time_index)``.

    Accepts the legacy grammar ``{experiment_id}_{well_index}_e{NN}_t{NNNN}`` (zero-based embryo, no
    channel token) and returns a ONE-based ``local_embryo_index`` converted through the pipeline's
    own ``track_index_to_embryo_index``, so ``e00 -> 1``.

    The well label must already be canonical (``A01``, not ``A1``) — every one of the 17,174 rows
    across the 38 embedded experiments is zero-padded, so an unpadded label means a corrupted or
    hand-edited id and is rejected rather than quietly repaired.

    Raises:
        ValueError: on an unparseable id, or a well label outside the canonical 8x12 grammar.
    """
    text = str(snip_id).strip()
    match = _LEGACY_SNIP_ID_RE.match(text)
    if match is None:
        raise ValueError(
            f"[morphseq_integration] cannot parse legacy snip_id {snip_id!r}. Expected "
            "{experiment_id}_{well_index}_e{NN}_t{NNNN} (e.g. 20250612_30hpf_ctrl_atf6_A01_e00_t0000)."
        )
    well_index = validate_well_index(match.group("well_index"))
    local_embryo_index = track_index_to_embryo_index(int(match.group("embryo")))
    return (
        match.group("experiment_id"),
        well_index,
        local_embryo_index,
        int(match.group("time")),
    )


def _build_spine(snip_ids: "pd.Series", experiment_id: str) -> pd.DataFrame:
    """The canonical identity columns for a set of legacy snip ids.

    ``experiment_id`` comes from the caller (the filename) and is cross-checked against what each
    snip_id encodes, so a mislabelled or mis-copied file is caught rather than silently relabelled.
    """
    records = []
    mismatched: list[str] = []
    for snip_id in snip_ids:
        parsed_experiment, well_index, local_embryo_index, time_index = parse_legacy_snip_id(snip_id)
        if parsed_experiment != experiment_id:
            mismatched.append(f"{snip_id} (encodes {parsed_experiment!r})")
            continue
        well_id = build_well_id(experiment_id, well_index)
        records.append(
            {
                "experiment_id": experiment_id,
                "well_id": well_id,
                "well_index": well_index,
                "physical_embryo_id": build_physical_embryo_id(well_id, local_embryo_index),
                "local_embryo_index": local_embryo_index,
                "time_index": time_index,
                "snip_id": str(snip_id),
                "source": SOURCE_LEGACY,
            }
        )

    if mismatched:
        raise ValueError(
            f"[morphseq_integration] {len(mismatched)} snip_id(s) in the "
            f"{experiment_id!r} latents file encode a different experiment_id, e.g. "
            f"{mismatched[:3]}. The filename and the ids disagree."
        )

    return pd.DataFrame(records, columns=list(SPINE_COLUMNS))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_legacy_latents(
    experiment_id: str,
    *,
    legacy_root: str | Path | None = None,
    flat_latent_names: bool = False,
) -> pd.DataFrame:
    """Load one experiment's legacy latents with the canonical identity spine prepended.

    Args:
        experiment_id: Pipeline experiment id, e.g. ``20250612_30hpf_ctrl_atf6``.
        legacy_root: Override the legacy latents directory.
        flat_latent_names: Rename latents to the flat pipeline convention (``z_mu_05``). Default
            keeps the legacy names, which retain the nuisance/biological split.

    Returns:
        ``SPINE_COLUMNS`` followed by the 200 latent columns, one row per legacy snip.

    Raises:
        FileNotFoundError: if the experiment has no legacy latents file.
        ValueError: on an unparseable snip_id, a duplicate identity, or a latent block whose
            dimension count is not 100.
    """
    path = latents_path(experiment_id, legacy_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"[morphseq_integration] no legacy latents for {experiment_id!r} at {path}. "
            "Call available_experiments() to list what is embedded."
        )

    raw = pd.read_csv(path)
    if "snip_id" not in raw.columns:
        raise ValueError(f"[morphseq_integration] {path.name} has no snip_id column.")

    latents = classify_latent_columns(list(raw.columns))
    if latents.n_dims != EXPECTED_LATENT_DIMS:
        raise ValueError(
            f"[morphseq_integration] {path.name} carries {latents.n_dims} latent dimensions, "
            f"expected {EXPECTED_LATENT_DIMS}. This is not the {LEGACY_MODEL_NAME} checkpoint's "
            "output, or the file is truncated."
        )

    spine = _build_spine(raw["snip_id"], experiment_id)
    latent_block = raw.loc[:, [*latents.mu_columns, *latents.sigma_columns]]
    out = pd.concat([spine, latent_block.reset_index(drop=True)], axis=1)

    _require_unique_identity(out, source_name=path.name)

    if flat_latent_names:
        out = out.rename(columns=to_flat_latent_names(list(out.columns)))
    return out


def _require_unique_identity(frame: pd.DataFrame, *, source_name: str) -> None:
    """Fail if any (well_id, local_embryo_index, time_index) appears twice."""
    keys = ["well_id", "local_embryo_index", "time_index"]
    duplicated = frame.loc[frame.duplicated(subset=keys, keep=False), [*keys, "snip_id"]]
    if not duplicated.empty:
        raise ValueError(
            f"[morphseq_integration] {source_name} has duplicate "
            f"(well_id, local_embryo_index, time_index): "
            f"{duplicated.head(6).to_dict(orient='records')}. Each identity must appear once."
        )


def load_many_legacy_latents(
    experiment_ids: "list[str]",
    *,
    legacy_root: str | Path | None = None,
    flat_latent_names: bool = False,
) -> pd.DataFrame:
    """Load and stack several experiments' legacy latents.

    Every requested experiment must have a file; a silent skip would make a missing experiment look
    like an experiment with no embryos.
    """
    frames = [
        load_legacy_latents(
            experiment_id, legacy_root=legacy_root, flat_latent_names=flat_latent_names
        )
        for experiment_id in experiment_ids
    ]
    if not frames:
        return pd.DataFrame(columns=list(SPINE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Metadata companion
# ---------------------------------------------------------------------------


def load_companion_metadata(
    snip_ids: "set[str] | list[str] | None" = None,
    *,
    companion_path: str | Path | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read the descriptive columns of ``embryo_stats_df.csv``, optionally filtered to ``snip_ids``.

    The file is ~415 MB and 231 columns wide, so it is read in chunks with an explicit column
    subset. Passing ``snip_ids`` keeps only matching rows.

    Note this table covers the Nov 2024 TRAINING population (44 experiments, through ``20250215``)
    and has no rows for the 2025 experiments — see the module docstring.
    """
    path = Path(companion_path) if companion_path is not None else default_companion_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"[morphseq_integration] legacy metadata companion not found: {path}"
        )

    wanted = None if snip_ids is None else set(map(str, snip_ids))
    collected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, usecols=list(COMPANION_COLUMNS), chunksize=chunksize, low_memory=False
    ):
        if wanted is not None:
            chunk = chunk.loc[chunk["snip_id"].astype(str).isin(wanted)]
        if not chunk.empty:
            collected.append(chunk)

    if not collected:
        return pd.DataFrame(columns=list(COMPANION_COLUMNS))
    return pd.concat(collected, ignore_index=True)


def attach_legacy_metadata(
    latents: pd.DataFrame,
    *,
    companion_path: str | Path | None = None,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join the companion metadata onto a latents frame, keyed on ``snip_id``.

    Adds ``has_legacy_metadata`` (bool) so an unmatched row is visible rather than a silent block of
    NaNs. Rows are never dropped: for the 2025 experiments the companion has no coverage at all and
    the indicator is uniformly ``False``.

    Raises:
        ValueError: if the companion carries duplicate ``snip_id`` rows, which would fan out the join.
    """
    if metadata is None:
        metadata = load_companion_metadata(
            set(latents["snip_id"].astype(str)), companion_path=companion_path
        )

    if not metadata.empty:
        duplicated = metadata.loc[metadata.duplicated(subset=["snip_id"], keep=False), "snip_id"]
        if not duplicated.empty:
            raise ValueError(
                "[morphseq_integration] companion metadata has duplicate snip_id rows: "
                f"{sorted(set(duplicated.astype(str)))[:6]}. The join would fan out."
            )

    if metadata.empty:
        out = latents.copy()
        for column in COMPANION_COLUMNS:
            if column != "snip_id":
                out[column] = pd.NA
        out["has_legacy_metadata"] = False
        return out

    # Carry whatever payload columns this metadata frame actually has. A caller may pass a subset
    # (tests, or a narrowed read), and demanding the full set would raise instead of joining.
    payload = [
        column
        for column in metadata.columns
        if column != "snip_id" and column not in latents.columns
    ]
    out = latents.merge(
        metadata.loc[:, ["snip_id", *payload]],
        on="snip_id",
        how="left",
        indicator="_companion_match",
    )
    out["has_legacy_metadata"] = out["_companion_match"] == "both"
    return out.drop(columns=["_companion_match"])


__all__ = [
    "LEGACY_MODEL_NAME",
    "SOURCE_LEGACY",
    "SPINE_COLUMNS",
    "COMPANION_COLUMNS",
    "EXPECTED_LATENT_DIMS",
    "LatentColumns",
    "attach_legacy_metadata",
    "available_experiments",
    "classify_latent_columns",
    "default_companion_path",
    "default_legacy_root",
    "flat_latent_name",
    "latents_path",
    "legacy_latent_name",
    "load_companion_metadata",
    "load_legacy_latents",
    "load_many_legacy_latents",
    "parse_legacy_snip_id",
    "to_flat_latent_names",
]
