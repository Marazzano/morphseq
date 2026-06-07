"""Pipeline-wide artifact-path registry (the ORCHESTRATION kingdom).

WHY THIS FILE EXISTS — it answers one question: *where does each file the pipeline produces
land?* Both the Snakefile (at parse time) and the Python entrypoints (at run time) import from
here, so rule-output paths and code-output paths are guaranteed identical.

THE VOCABULARY (used consistently across the whole refactor — do not blur these):

    stage     a PHASE of the pipeline, and the top-level FOLDER under output_root.
              e.g. ``experiment_metadata``, ``computed_features``, ``quality_control``.
              This is "where am I in the pipeline" from a user's seat. MANY steps share a stage.

    step      one unit of work WITHIN a stage — a rule / function / tasks.py verb's output slot.
              e.g. ``ingest_scope_metadata``, ``frame_inventory_well``. These are the REGISTRY
              KEYS (``PIPELINE_STEPS`` below). A step is named by a NOUN (the output slot), not a
              rule verb, because SEVERAL rules can act on one step — ``build_frame_inventory_well``
              and ``validate_frame_inventory_well`` both reference the single ``frame_inventory_well``
              step. (step != rule: one step, possibly several rules.)

    artifact  one FILE a step produces. A step can produce several, so each step's ``artifacts``
              is a dict (e.g. ``mapping`` -> ``series_well_mapping.csv``). The ``.validated``
              sentinel and ``.provenance.json`` sidecar are DERIVED from an artifact (helpers
              below), not separate artifacts.

So the layout is::

    output_root / {stage} / {experiment} / [per_well/{well_id}/] {artifact}
       │             │           │                  │                 │
      root         STAGE      experiment      per-well shape       a step's
    (env.yaml)  (folder)     (caller)        (step's fanout)        FILE

THREE THINGS KEPT DELIBERATELY SEPARATE (see
docs/refactors/streamline-snakemake/well_id_throughline_refactor_plan.md and
target/front_end_naming_and_flow.md):

    Root      output_root            env.yaml (machine-specific)    -> passed in here as `root`
    Layout    {stage}/{exp}/...      THIS FILE (a code contract)    -> PIPELINE_STEPS + helpers
    Selection which wells/features   config.yaml (a science choice) -> NOT here

This file knows ``output_root``, stages, grain, and the ``per_well/{well_id}`` shape. It does
NOT mint identifiers — that is the IDENTITY kingdom (``shared/identifiers/``). Identity flows
*into* orchestration, never the other way.

⚠️ FORWARD DECLARATION (2026-06-06). This registry names the **TARGET** artifacts
(``frame_inventory``, ``discovered_wells.txt``, the ``ingest_*``/``join_*`` step keys). Several
of these names do not exist on disk yet — they are produced once Scope 2 (the data-semantics
migration) and Scope 5 (per-well back half) reconstruct each step. The registry is the
**contract those scopes converge onto**; it is intentionally NOT yet wired into the live
Snakefile (which still emits the legacy ``frame_contract.csv`` / ``wells.txt`` names). Wire each
step to this registry as it is reconstructed.

Coverage: **front-end steps only** for now (the two metadata ingest lineages, well discovery,
and the post-fan frame-inventory tail). Back-half steps (segmentation, snips, aux, features,
QC, analysis_ready) get rows when Scope 5 specifies their stage/grain. Spec:
target/front_end_naming_and_flow.md (§ PATHS.PY REGISTRY ROWS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# Grain of a step's outputs (its "fanout"):
#   "experiment"          -> one artifact per experiment (pre-fan, or an off-spine merged view)
#   "per_well_then_merge" -> a per-well shard at per_well/{well_id}/, concatenated to an
#                            experiment-level merged view
EXPERIMENT = "experiment"
PER_WELL_THEN_MERGE = "per_well_then_merge"

# How to resolve a path for a given call (independent of the step's declared grain):
#   "experiment" -> {stage}/{exp}/{file}                      (the experiment-grain artifact)
#   "per_well"   -> {stage}/{exp}/per_well/{well_id}/{file}   (one well's shard)
#   "merged"     -> {stage}/{exp}/{file}                      (the concatenated experiment view)
PATH_MODE_EXPERIMENT = "experiment"
PATH_MODE_PER_WELL = "per_well"
PATH_MODE_MERGED = "merged"


# ──────────────────────────────────────────────────────────────────────────────────────────
# THE REGISTRY (front-end steps — TARGET names)
# ──────────────────────────────────────────────────────────────────────────────────────────
# *** THIS REGISTRY IS THE CONTRACT. ***
# Every artifact path in the pipeline is computed from this one dict — the Snakefile reads it at
# parse time, the Python entrypoints read it at run time. If a filename or stage is wrong HERE,
# it is wrong EVERYWHERE, consistently; if two places ever disagree about where a file lives, the
# bug is that one of them bypassed this registry. NEVER hardcode an artifact path elsewhere — add
# or edit a row here and call the helper functions below. Adding a step = adding one row.
#
# Per row (keyed by STEP):
#   stage     : the pipeline phase / top-level folder under output_root (many steps share one)
#   fanout    : the step's grain (EXPERIMENT | PER_WELL_THEN_MERGE)
#   artifacts : {artifact_key: filename_template}; templates may reference {well_id} and any
#               key passed via `format_vars` (e.g. {scope}). Sidecars (.validated,
#               .provenance.json) are DERIVED by the helpers below, not listed as artifacts.
PIPELINE_STEPS: dict[str, dict] = {
    # ── PLATE LINEAGE (Excel — authored design + plate geometry) ──────────────
    "ingest_plate_metadata": {
        "stage": "experiment_metadata",
        "fanout": EXPERIMENT,
        "artifacts": {"csv": "plate_metadata.csv"},
    },

    # ── SCOPE LINEAGE (raw microscope file — acquisition facts) ───────────────
    "ingest_scope_metadata": {
        "stage": "experiment_metadata",
        "fanout": EXPERIMENT,
        # {scope} -> format_vars={"scope": "yx1" | "keyence"}; the ONLY raw read.
        "artifacts": {"raw": "scope_metadata__{scope}.csv"},
    },
    "map_series_to_wells": {
        "stage": "experiment_metadata",
        "fanout": EXPERIMENT,
        # .provenance.json via provenance_path().
        "artifacts": {"mapping": "series_well_mapping.csv"},
    },
    "join_series_mapping_to_scope_metadata": {  # CONVERGENCE LINE; well_id minted here
        "stage": "experiment_metadata",
        "fanout": EXPERIMENT,
        # .validated via validated_path().
        "artifacts": {"mapped": "scope_metadata_mapped.csv"},
    },

    # ── FAN POINT (well discovery — checkpoint) ───────────────────────────────
    "discover_wells": {
        "stage": "experiment_metadata",
        "fanout": EXPERIMENT,
        "artifacts": {"wells": "discovered_wells.txt"},  # one well_id per line
    },

    # ── POST-FAN (per well_id) — frame inventory joins the spine ──────────────
    # Key is the NOUN (the output slot), not a rule verb: this one step is written by TWO rules —
    # build_frame_inventory_well (writes the shard) + validate_frame_inventory_well (writes only
    # the .validated sentinel + report; one-file+sentinel model, see stitched_handoff_contract.md).
    # A neutral noun key reads right from both. Do NOT rename to build_*.
    "frame_inventory_well": {
        "stage": "experiment_metadata",  # OPEN (findings #5 leans a dedicated frame_inventory/
                                         # stage); placeholder until that is decided.
        "fanout": PER_WELL_THEN_MERGE,
        "artifacts": {"inventory": "{well_id}_frame_inventory.csv"},  # well_id IN the filename
    },
}

# ── HELPERS: step name -> artifact path ──────────────────────────────────────────────────────
# Two private lookups turn registry keys into concrete values, failing loudly on a typo:
#   _lookup_step("discover_wells")         -> {"stage": ..., "fanout": ..., "artifacts": ...}
#   _resolve_filename("ingest_scope_metadata", "raw", {"scope": "yx1"})
#                                          -> "scope_metadata__yx1.csv"
#
# Four public functions build paths from them (root + experiment come from the caller):
#   step_dir        -> the FOLDER, no filename
#       step_dir(ROOT, "discover_wells", "20250912")
#           -> {ROOT}/experiment_metadata/20250912
#       step_dir(ROOT, "frame_inventory_well", "20250912", path_mode="per_well",
#                well_id="20250912_B01")
#           -> {ROOT}/experiment_metadata/20250912/per_well/20250912_B01
#
#   artifact_path   -> FOLDER + filename  (the one you'll call most)
#       artifact_path(ROOT, "discover_wells", "wells", "20250912")
#           -> {ROOT}/experiment_metadata/20250912/discovered_wells.txt
#       artifact_path(ROOT, "frame_inventory_well", "inventory", "20250912",
#                     path_mode="per_well", well_id="20250912_B01")
#           -> {ROOT}/.../per_well/20250912_B01/20250912_B01_frame_inventory.csv
#
#   validated_path  -> artifact_path + ".validated"        (the sentinel beside it)
#       validated_path(ROOT, "join_series_mapping_to_scope_metadata", "mapped", "20250912")
#           -> {ROOT}/experiment_metadata/20250912/scope_metadata_mapped.csv.validated
#
#   provenance_path -> artifact_path + ".provenance.json"  (the sidecar beside it)
#       provenance_path(ROOT, "map_series_to_wells", "mapping", "20250912")
#           -> {ROOT}/experiment_metadata/20250912/series_well_mapping.csv.provenance.json


def _lookup_step(step: str) -> dict:
    """Return a step's registry row, or raise KeyError naming the known steps."""
    try:
        return PIPELINE_STEPS[step]
    except KeyError:
        raise KeyError(
            f"Unknown step {step!r}. Known front-end steps: {sorted(PIPELINE_STEPS)}. "
            f"Back-half steps are not in the registry yet (added in Scope 5)."
        )


def _resolve_filename(step: str, artifact: str, format_vars: Optional[dict] = None) -> str:
    """Fill in an artifact's filename template (e.g. ``scope_metadata__{scope}.csv``).

    Raises KeyError naming the step's known artifacts if ``artifact`` is not one of them.
    """
    spec = _lookup_step(step)
    try:
        template = spec["artifacts"][artifact]
    except KeyError:
        raise KeyError(
            f"Step {step!r} has no artifact {artifact!r}. "
            f"Known artifacts: {sorted(spec['artifacts'])}."
        )
    return template.format(**(format_vars or {}))


def step_dir(
    root: PathLike,
    step: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
) -> Path:
    """Return the directory that holds a step's artifact, before the filename is appended.

    The directory lives under the step's ``stage`` folder. ``per_well`` mode requires ``well_id``
    and yields ``{stage}/{exp}/per_well/{well_id}``; ``experiment``/``merged`` modes yield
    ``{stage}/{exp}``.
    """
    spec = _lookup_step(step)
    base = Path(root) / spec["stage"] / str(experiment_id)
    if path_mode == PATH_MODE_PER_WELL:
        if not well_id:
            raise ValueError(
                f"path_mode='per_well' for step {step!r} requires well_id (the GLOBAL "
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
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Resolve the absolute path to one of a step's artifacts.

    Examples (see target/front_end_naming_and_flow.md for the full set)::

        artifact_path(ROOT, "ingest_scope_metadata", "raw", "20250912",
                      format_vars={"scope": "yx1"})
        #   -> {ROOT}/experiment_metadata/20250912/scope_metadata__yx1.csv

        artifact_path(ROOT, "frame_inventory_well", "inventory", "20250912",
                      path_mode="per_well", well_id="20250912_B01")
        #   -> {ROOT}/experiment_metadata/20250912/per_well/20250912_B01/20250912_B01_frame_inventory.csv
    """
    directory = step_dir(
        root, step, experiment_id, path_mode=path_mode, well_id=well_id
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
    return directory / _resolve_filename(step, artifact, fmt)


def validated_path(
    root: PathLike,
    step: str,
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
    normalized on-disk when the step is reconstructed — this helper deliberately emits the target
    convention.
    """
    base = artifact_path(
        root, step, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".validated")


def provenance_path(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: str = PATH_MODE_EXPERIMENT,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.provenance.json`` sidecar beside an artifact."""
    base = artifact_path(
        root, step, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".provenance.json")
