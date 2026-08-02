"""Assemble the morph-seq master table.

One row per **paired imaging well** of the requested experiments, carrying:

    identity spine   experiment_id, well_id, well_index, seq_sample_id, pairing_status
    morphology       the legacy VAE latents (z_mu_n_/z_mu_b_, z_sigma_*) for that well
    sequencing       the per-embryo sequencing metadata for the matching hash well
    coverage flags   has_morph, has_seq

The well is the unit of pairing: a hash well identifies a physical well of embryos, so the
sequencing side is well-resolved by construction. Morphology can be finer (the pipeline resolves
multiple embryos per well), so ``well_embryo_selection`` states how that is collapsed and
``n_morph_embryos`` records what was collapsed away. Nothing is silently inner-joined: unpaired and
uncovered rows survive with their flags set, and ``coverage_summary`` reports the counts.

Deliberately absent
-------------------
**No QC from the current pipeline.** The pipeline's ``use_snip`` / ``sa_outlier_flag`` were computed
on the degraded snip raster (smaller, more saturated) and on a reference population that
mis-calibrates cold-reared embryos, so they do not apply to the legacy morphology this table is built
on. Exclusions are curated by inspection instead. ``notes``/exclusion columns can join on ``well_id``
when that curation exists.

**No transcriptional pseudostage.** That needs the ``project_ccs_data.py`` port and a Hooke refit
whose current model excludes GENE7 outright.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import legacy_morph as lm
from .crosswalk import STATUS_PAIRED, build_crosswalk
from .exclusions import apply_exclusions
from .experiment_key import ExperimentKey
from .identifiers import strip_rt_block
from .paths import PipelinePaths

_SEAHUB_PREPROCESSED = Path("/net/seahub_zfish/vol1/data/preprocessed")

SEQ_METADATA_TEMPLATE = "{sci_expt}_embryo_metadata.tsv"

# Sequencing columns carried onto the master table. embryo_ID is kept verbatim for provenance
# alongside the RT-block-stripped sample key it reduces to.
SEQ_COLUMNS: tuple[str, ...] = (
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
    "imaging",
    "hash_plate",
    "hash_well",
    "rt_block",
    "pheno",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "well_index",
    "sci_expt",
    "hash_plate",
    "hash_well",
    "seq_sample_id",
    "pairing_status",
    "has_morph",
    "has_seq",
    "n_morph_embryos",
    "well_embryo_selection",
)

SELECT_FIRST_EMBRYO = "first_embryo_index"


def seq_metadata_path(sci_expt: str, *, seahub_root: str | Path | None = None) -> Path:
    """Path to a sequencing experiment's embryo metadata TSV."""
    root = Path(seahub_root) if seahub_root is not None else _SEAHUB_PREPROCESSED
    return root / sci_expt / SEQ_METADATA_TEMPLATE.format(sci_expt=sci_expt)


def load_seq_metadata(
    sci_expt: str,
    *,
    seahub_root: str | Path | None = None,
    imaging_only: bool = True,
) -> pd.DataFrame:
    """Load one sequencing experiment's embryo metadata, keyed by ``seq_sample_id``.

    ``embryo_ID`` carries an RT-block suffix (``GENE7_P18_A1_Bl1``) that the sample-level key drops;
    ``strip_rt_block`` reduces it to ``seq_sample_id``.

    Args:
        imaging_only: Keep only rows flagged ``imaging`` — the embryos that have an imaging
            counterpart. The rest cannot pair with morphology by definition.

    Raises:
        FileNotFoundError: if the metadata TSV is absent.
        ValueError: if ``seq_sample_id`` is not unique after filtering, which would fan out the join.
    """
    path = seq_metadata_path(sci_expt, seahub_root=seahub_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"[morphseq_integration] sequencing metadata for {sci_expt!r} not found: {path}"
        )

    seq = pd.read_csv(path, sep="\t")
    if "embryo_ID" not in seq.columns:
        raise ValueError(f"[morphseq_integration] {path.name} has no embryo_ID column.")

    if imaging_only:
        if "imaging" not in seq.columns:
            raise ValueError(
                f"[morphseq_integration] {path.name} has no 'imaging' column; pass "
                "imaging_only=False to load it anyway."
            )
        seq = seq.loc[seq["imaging"].astype(str).str.lower().isin(["true", "1"])]

    seq = seq.copy()
    seq["seq_sample_id"] = seq["embryo_ID"].map(strip_rt_block)

    duplicated = seq.loc[seq.duplicated(subset=["seq_sample_id"], keep=False), "seq_sample_id"]
    if not duplicated.empty:
        raise ValueError(
            f"[morphseq_integration] {path.name} has duplicate seq_sample_id after stripping RT "
            f"blocks: {sorted(set(duplicated))[:6]}. One sample key must name one embryo."
        )

    present = [column for column in SEQ_COLUMNS if column in seq.columns]
    return seq.loc[:, ["seq_sample_id", *present]].reset_index(drop=True)


def verify_seq_pairing(
    crosswalk: pd.DataFrame,
    *,
    seahub_root: str | Path | None = None,
) -> pd.DataFrame:
    """Assert the minted ``seq_sample_id``s exactly equal the sequencing side's imaging embryos.

    This is the gate that makes the sample-id grammar *correct* rather than merely plausible: it
    checks set equality in BOTH directions, per sequencing experiment. A minted id absent from the
    sequencing corpus means a wrong ``hash_plate_num``, ``sci_expt``, or well convention; a
    sequencing embryo with no minted id means an imaging well was missed.

    Returns:
        Per-``sci_expt`` counts: ``n_minted``, ``n_seq_imaging``, ``n_shared``, ``equal``.

    Raises:
        ValueError: on any asymmetry, naming the offending ids.
    """
    paired = crosswalk.loc[crosswalk["pairing_status"] == STATUS_PAIRED]
    rows = []
    for sci_expt, group in paired.groupby("sci_expt", sort=True):
        minted = set(group["seq_sample_id"])
        seq_keys = set(
            load_seq_metadata(str(sci_expt), seahub_root=seahub_root, imaging_only=True)[
                "seq_sample_id"
            ]
        )
        minted_only = sorted(minted - seq_keys)
        seq_only = sorted(seq_keys - minted)
        if minted_only or seq_only:
            raise ValueError(
                f"[morphseq_integration] {sci_expt}: minted sample ids do not match the "
                f"sequencing corpus. {len(minted_only)} minted-but-absent "
                f"(e.g. {minted_only[:5]}), {len(seq_only)} sequenced-but-unminted "
                f"(e.g. {seq_only[:5]}). Check hash_plate_num, sci_expt, and the well convention."
            )
        rows.append(
            {
                "sci_expt": str(sci_expt),
                "n_minted": len(minted),
                "n_seq_imaging": len(seq_keys),
                "n_shared": len(minted & seq_keys),
                "equal": True,
            }
        )
    return pd.DataFrame(rows)


def _well_resolved_morph(
    experiment_ids: "list[str]", *, legacy_root: str | Path | None = None
) -> pd.DataFrame:
    """Legacy morphology collapsed to one row per well, with the collapse recorded.

    Legacy is one row per well for almost every experiment, but not all: a few carry a second
    embryo, and the timelapses carry many timepoints. The earliest timepoint and lowest embryo index
    are selected deterministically, and ``n_morph_embryos`` / ``n_morph_timepoints`` record what was
    set aside so the collapse is never invisible.
    """
    morph = lm.load_many_legacy_latents(experiment_ids, legacy_root=legacy_root)
    if morph.empty:
        return morph

    counts = (
        morph.groupby("well_id")
        .agg(
            n_morph_embryos=("local_embryo_index", "nunique"),
            n_morph_timepoints=("time_index", "nunique"),
        )
        .reset_index()
    )

    ordered = morph.sort_values(["well_id", "time_index", "local_embryo_index"])
    selected = ordered.drop_duplicates(subset=["well_id"], keep="first").copy()
    selected["well_embryo_selection"] = SELECT_FIRST_EMBRYO
    return selected.merge(counts, on="well_id", how="left")


def build_master_table(
    experiment_ids: "list[str] | None" = None,
    *,
    key: ExperimentKey | None = None,
    paths: PipelinePaths | None = None,
    legacy_root: str | Path | None = None,
    seahub_root: str | Path | None = None,
    verify_pairing: bool = True,
    exclusions: str = "drop",
) -> pd.DataFrame:
    """Build the master morph-seq table for the given (default: all keyed) experiments.

    Args:
        experiment_ids: Restrict to these imaging experiments; each needs a key entry.
        verify_pairing: Run ``verify_seq_pairing`` first. Leave on — it is cheap and it is the check
            that validates the whole identity bridge.
        exclusions: How to treat curated image-QC exclusions (``excluded_wells.csv``).
            ``"drop"`` (default) removes them, ``"flag"`` adds a ``curated_excluded`` column,
            ``"keep"`` ignores the table entirely. Dropping is the default so every downstream
            analysis — morphology and sequencing alike — sees the same well set without having to
            remember to filter.

    Returns:
        ``IDENTITY_COLUMNS``, then the sequencing payload, then the legacy latent block. One row per
        imaging well of every requested experiment.
    """
    if exclusions not in ("drop", "flag", "keep"):
        raise ValueError(
            f"[morphseq_integration] exclusions must be 'drop', 'flag', or 'keep', got {exclusions!r}."
        )
    crosswalk = build_crosswalk(experiment_ids, key=key, paths=paths)
    if verify_pairing:
        verify_seq_pairing(crosswalk, seahub_root=seahub_root)

    targets = sorted(set(crosswalk["experiment_id"]))
    embedded = set(lm.available_experiments(legacy_root))
    with_morph = [experiment_id for experiment_id in targets if experiment_id in embedded]
    morph = _well_resolved_morph(with_morph, legacy_root=legacy_root)

    seq_frames = [
        load_seq_metadata(str(sci_expt), seahub_root=seahub_root, imaging_only=True)
        for sci_expt in sorted(set(crosswalk["sci_expt"]))
    ]
    seq = (
        pd.concat(seq_frames, ignore_index=True)
        if seq_frames
        else pd.DataFrame(columns=["seq_sample_id"])
    )

    # Suffix the sequencing side's own hash columns: the crosswalk already carries hash_plate /
    # hash_well minted from the imaging workbook, and keeping both lets them be compared.
    seq = seq.rename(
        columns={"hash_plate": "seq_hash_plate", "hash_well": "seq_hash_well"}
    )

    master = crosswalk.merge(morph, on=["experiment_id", "well_id", "well_index"], how="left")
    master["has_morph"] = master["snip_id"].notna()

    master = master.merge(seq, on="seq_sample_id", how="left", indicator="_seq_match")
    master["has_seq"] = master["_seq_match"] == "both"
    master = master.drop(columns=["_seq_match"])

    master["n_morph_embryos"] = master["n_morph_embryos"].fillna(0).astype(int)
    master["well_embryo_selection"] = master["well_embryo_selection"].fillna("")

    # Applied AFTER verify_seq_pairing: that gate asserts all 576 minted ids equal the sequencing
    # corpus, and filtering first would make it fail for the wrong reason.
    if exclusions != "keep":
        master = apply_exclusions(master, mode=exclusions)

    leading = [column for column in IDENTITY_COLUMNS if column in master.columns]
    rest = [column for column in master.columns if column not in leading]
    return master.loc[:, [*leading, *rest]].sort_values(
        ["experiment_id", "well_index"]
    ).reset_index(drop=True)


def coverage_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Per-experiment coverage: wells, paired, morph, seq, both, neither.

    The honest view of what the table actually contains — report this rather than the row count,
    which counts uncovered wells too.
    """
    # Derive the combination flags as plain columns first: groupby.apply's signature for excluding
    # grouping columns differs across pandas versions, and none of this needs a Python-level apply.
    flags = master.loc[:, ["experiment_id", "sci_expt", "has_morph", "has_seq"]].copy()
    flags["is_paired"] = master["pairing_status"] == STATUS_PAIRED
    flags["is_both"] = flags["has_morph"] & flags["has_seq"]
    flags["is_neither"] = ~flags["has_morph"] & ~flags["has_seq"]

    grouped = flags.groupby(["experiment_id", "sci_expt"], sort=True)
    summary = grouped.agg(
        n_paired=("is_paired", "sum"),
        n_morph=("has_morph", "sum"),
        n_seq=("has_seq", "sum"),
        n_both=("is_both", "sum"),
        n_neither=("is_neither", "sum"),
    )
    summary.insert(0, "n_wells", grouped.size())
    return summary.astype(int).reset_index()


def write_master_table(master: pd.DataFrame, out_path: str | Path) -> Path:
    """Write the master table to parquet, creating the parent directory."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(path, index=False)
    return path


__all__ = [
    "IDENTITY_COLUMNS",
    "SEQ_COLUMNS",
    "build_master_table",
    "coverage_summary",
    "load_seq_metadata",
    "seq_metadata_path",
    "verify_seq_pairing",
    "write_master_table",
]
