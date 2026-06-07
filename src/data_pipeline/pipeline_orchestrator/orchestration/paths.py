"""Pipeline-wide path registry (the ORCHESTRATION kingdom).

One ``STAGES`` table is the single source of truth for *where every stage's artifacts live*
under ``output_root``. Both the Snakefile (at parse time) and the Python entrypoints (at run
time) import from here, so rule-output paths and code-output paths are guaranteed identical —
"add a feature" becomes "add one registry row".

Three things are kept deliberately separate (see
docs/refactors/streamline-snakemake/well_id_throughline_refactor_plan.md, the 2026-06-02
DECISION block, and target/front_end_naming_and_flow.md):

    Root      output_root           env.yaml (machine-specific)   -> passed in here as `root`
    Layout    {family}/{exp}/...    THIS FILE (a code contract)   -> the STAGES table + helpers
    Selection which wells/features  config.yaml (a science choice) -> NOT here

This file knows ``output_root``, families, grain, and the ``per_well/{well_id}`` shape. It does
NOT mint identifiers — that is the IDENTITY kingdom (``shared/identifiers/``). Identity flows
*into* orchestration, never the other way.

⚠️ FORWARD DECLARATION (2026-06-06). This registry names the **TARGET** artifacts
(``frame_inventory``, ``discovered_wells.txt``, the ``ingest_*``/``join_*`` stage keys). Several
of these names do not exist on disk yet — they are produced once Scope 2 (the data-semantics
migration) and Scope 5 (per-well back half) reconstruct each stage. The registry is the
**contract those scopes converge onto**; it is intentionally NOT yet wired into the live
Snakefile (which still emits the legacy ``frame_contract.csv`` / ``wells.txt`` names). Wire each
stage to this registry as it is reconstructed.

Coverage: **front-end stages only** for now (the two metadata ingest lineages, well discovery,
and the post-fan frame-inventory tail). Back-half stages (segmentation, snips, aux, features,
QC, analysis_ready) get rows when Scope 5 specifies their grain/family. Spec:
target/front_end_naming_and_flow.md (§ PATHS.PY REGISTRY ROWS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# Grain of a stage's outputs:
#   "experiment"          -> one artifact per experiment (pre-fan, or an off-spine merged view)
#   "per_well_then_merge" -> a per-well shard at per_well/{well_id}/, concatenated to an
#                            experiment-level merged view
EXPERIMENT = "experiment"
PER_WELL_THEN_MERGE = "per_well_then_merge"

# How to resolve a path for a given call (independent of the stage's declared grain):
#   "experiment" -> {family}/{exp}/{file}                      (the experiment-grain artifact)
#   "per_well"   -> {family}/{exp}/per_well/{well_id}/{file}   (one well's shard)
#   "merged"     -> {family}/{exp}/{file}                      (the concatenated experiment view)
PATH_MODE_EXPERIMENT = "experiment"
PATH_MODE_PER_WELL = "per_well"
PATH_MODE_MERGED = "merged"


# ──────────────────────────────────────────────────────────────────────────────────────────
# THE REGISTRY (front-end stages — TARGET names)
# ──────────────────────────────────────────────────────────────────────────────────────────
# Per row:
#   family    : top-level folder under output_root that namespaces the domain
#   fanout    : the stage's grain (EXPERIMENT | PER_WELL_THEN_MERGE)
#   artifacts : {artifact_key: filename_template}; templates may reference {well_id} and any
#               key passed via `format_vars` (e.g. {scope}). Sentinels (.validated,
#               .provenance.json) are DERIVED by helpers below, not listed as artifacts.
STAGES: dict[str, dict] = {
    # ── PLATE LINEAGE (Excel — authored design + plate geometry) ──────────────
    "ingest_plate_metadata": {
        "family": "experiment_metadata",
        "fanout": EXPERIMENT,
        "artifacts": {"csv": "plate_metadata.csv"},
    },

    # ── SCOPE LINEAGE (raw microscope file — acquisition facts) ───────────────
    "ingest_scope_metadata": {
        "family": "experiment_metadata",
        "fanout": EXPERIMENT,
        # {scope} -> format_vars={"scope": "yx1" | "keyence"}; the ONLY raw read.
        "artifacts": {"raw": "scope_metadata__{scope}.csv"},
    },
    "map_series_to_wells": {
        "family": "experiment_metadata",
        "fanout": EXPERIMENT,
        # .provenance.json via provenance_path().
        "artifacts": {"mapping": "series_well_mapping.csv"},
    },
    "join_series_mapping_to_scope_metadata": {  # CONVERGENCE LINE; well_id minted here
        "family": "experiment_metadata",
        "fanout": EXPERIMENT,
        # .validated via validated_path().
        "artifacts": {"mapped": "scope_metadata_mapped.csv"},
    },

    # ── FAN POINT (well discovery — checkpoint) ───────────────────────────────
    "discover_wells": {
        "family": "experiment_metadata",
        "fanout": EXPERIMENT,
        "artifacts": {"wells": "discovered_wells.txt"},  # one well_id per line
    },

    # ── POST-FAN (per well_id) — frame inventory joins the spine ──────────────
    # build_frame_inventory_well writes the shard; validate_frame_inventory_well writes only the
    # .validated sentinel + report (one-file+sentinel model — see stitched_handoff_contract.md).
    "frame_inventory_well": {
        "family": "experiment_metadata",  # <frame-inventory-family> — OPEN (findings #5 leans
                                          # a dedicated frame_inventory/ family); placeholder.
        "fanout": PER_WELL_THEN_MERGE,
        "artifacts": {"inventory": "{well_id}_frame_inventory.csv"},  # well_id IN the filename
    },
}


def _spec(stage: str) -> dict:
    try:
        return STAGES[stage]
    except KeyError:
        raise KeyError(
            f"Unknown stage {stage!r}. Known front-end stages: {sorted(STAGES)}. "
            f"Back-half stages are not in the registry yet (added in Scope 5)."
        )


def _filename(stage: str, artifact: str, format_vars: Optional[dict] = None) -> str:
    spec = _spec(stage)
    try:
        template = spec["artifacts"][artifact]
    except KeyError:
        raise KeyError(
            f"Stage {stage!r} has no artifact {artifact!r}. "
            f"Known artifacts: {sorted(spec['artifacts'])}."
        )
    return template.format(**(format_vars or {}))


def stage_dir(
    root: PathLike,
    stage: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
) -> Path:
    """Return the directory that holds a stage's artifact, before the filename is appended.

    ``per_well`` mode requires ``well_id`` and yields ``{family}/{exp}/per_well/{well_id}``;
    ``experiment``/``merged`` modes yield ``{family}/{exp}``.
    """
    spec = _spec(stage)
    base = Path(root) / spec["family"] / str(experiment_id)
    if path_mode == PATH_MODE_PER_WELL:
        if not well_id:
            raise ValueError(
                f"path_mode='per_well' for stage {stage!r} requires well_id (the GLOBAL "
                f"{{experiment_id}}_{{well_index}} key)."
            )
        return base / "per_well" / str(well_id)
    if path_mode in (PATH_MODE_EXPERIMENT, PATH_MODE_MERGED):
        return base
    raise ValueError(
        f"Unknown path_mode {path_mode!r}; expected one of "
        f"{(PATH_MODE_EXPERIMENT, PATH_MODE_PER_WELL, PATH_MODE_MERGED)}."
    )


def artifact_path(
    root: PathLike,
    stage: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Resolve the absolute path to one stage artifact.

    Examples (see target/front_end_naming_and_flow.md for the full set)::

        artifact_path(ROOT, "ingest_scope_metadata", "raw", "20250912",
                      format_vars={"scope": "yx1"})
        #   -> {ROOT}/experiment_metadata/20250912/scope_metadata__yx1.csv

        artifact_path(ROOT, "frame_inventory_well", "inventory", "20250912",
                      path_mode="per_well", well_id="20250912_B01")
        #   -> {ROOT}/experiment_metadata/20250912/per_well/20250912_B01/20250912_B01_frame_inventory.csv
    """
    directory = stage_dir(
        root, stage, experiment_id, path_mode=path_mode, well_id=well_id
    )
    # well_id is a first-class filename token (e.g. "{well_id}_frame_inventory.csv"). In
    # per_well mode it is the well's global id; in merged mode the concatenated experiment view
    # is named with the experiment id, so the {well_id} slot resolves to experiment_id there
    # (matches target/front_end_naming_and_flow.md: merged -> {exp}_frame_inventory.csv).
    fmt = dict(format_vars or {})
    if path_mode == PATH_MODE_MERGED:
        fmt.setdefault("well_id", str(experiment_id))
    elif well_id is not None:
        fmt.setdefault("well_id", str(well_id))
    return directory / _filename(stage, artifact, fmt)


def validated_path(
    root: PathLike,
    stage: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.validated`` sentinel beside an artifact (trailing-suffix form).

    ⚠️ Today ``apply_series_mapping`` writes a LEADING-dot form (``.scope_metadata_mapped.validated``)
    while this helper assumes the TARGET trailing form (``scope_metadata_mapped.csv.validated``).
    That mismatch is an open audit item (front_end doc, "Sentinel-suffix audit still open") to be
    normalized on-disk when the stage is reconstructed — this helper deliberately emits the target
    convention.
    """
    base = artifact_path(
        root, stage, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".validated")


def provenance_path(
    root: PathLike,
    stage: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.provenance.json`` sidecar beside an artifact."""
    base = artifact_path(
        root, stage, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".provenance.json")
