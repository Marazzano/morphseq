"""Pipeline-wide artifact-path registry (the ORCHESTRATION kingdom).

WHY THIS FILE EXISTS — it answers one question: *where does each file the pipeline produces
land?* Both the Snakefile (at parse time) and the Python entrypoints (at run time) import from
here, so rule-output paths and code-output paths are guaranteed identical.

THE VOCABULARY (used consistently across the whole refactor — do not blur these):

    stage     a PHASE of the pipeline, and the top-level FOLDER under output_root.
              e.g. ``experiment_metadata``, ``computed_features``, ``quality_control``.
              This is "where am I in the pipeline" from a user's seat. MANY steps share a stage.

    step      one unit of work WITHIN a stage — a function / ``tasks.py`` verb's OUTPUT SLOT.
              e.g. ``ingest_scope_metadata``, ``frame_inventory``. These are the
              REGISTRY KEYS (``PIPELINE_STEPS`` below).
              A step key names a logical output slot, not a Snakemake action.
              Where a step is touched by multiple rules, use a neutral noun-like name rather than
              a producer verb — e.g. ``materialize_well`` (the live per-well producer) and
              ``validate_frame_inventory_for_well`` both reference the single ``frame_inventory``
              step. Some front-end keys remain verb-shaped because they are locked by
              front_end_naming_and_flow.md.

    artifact  one FILE a step produces. A step can produce several, so each step's ``artifacts``
              is a dict (e.g. ``mapping`` -> ``position_well_mapping.csv``). The ``.validated``
              sentinel and ``.provenance.json`` sidecar are DERIVED from an artifact (helpers
              below), not separate artifacts.

So the layout is::

    output_root / {stage} / {experiment} / [per_well/{well_id}/] {artifact}
       │             │           │                  │                 │
      root         STAGE      experiment      per-well shape       a step's
    (env.yaml)  (folder)     (caller)        (step's fanout)        FILE

⛔ NO-LEAKAGE BOUNDARY (the hard rule for this file). ``paths.py`` only ever **SUBSTITUTES**
caller-provided identity tokens (``experiment_id``, ``well_id``) into filename templates. It
**NEVER MINTS OR DERIVES** an identity value — no ``f"{exp}_{well}"``, no splitting a well_id, no
stuffing one token into another's slot. Minting/parsing identity is the IDENTITY kingdom's job
(``shared/identifiers/``); anything in here that needs to *build* an id must import from there.
Identity flows *into* orchestration, never the other way.

THREE THINGS KEPT DELIBERATELY SEPARATE (see
docs/refactors/streamline-snakemake/well_id_throughline_refactor_plan.md and
target/front_end_naming_and_flow.md):

    Root      output_root            env.yaml (machine-specific)    -> passed in here as `root`
    Layout    {stage}/{exp}/...      THIS FILE (a code contract)    -> PIPELINE_STEPS + helpers
    Selection which wells/features   config.yaml (a science choice) -> NOT here

⚠️ FORWARD DECLARATION (2026-06-06, updated 2026-06-07). This registry names the **TARGET**
artifacts (``frame_inventory``, ``discovered_wells.txt``, the ``ingest_*``/``join_*`` step keys).
The live root Snakefile now uses these registry rows for the front-end metadata/discovery flow,
while the back half still emits the legacy ``frame_contract.csv`` family until Scope 5 lands.
The registry remains the **contract those scopes converge onto**; wire each remaining step to this
registry as it is reconstructed.

Coverage: **front-end steps only** for now (the two metadata ingest lineages, well discovery,
and the post-fan frame-inventory tail). Back-half steps (segmentation, snips, aux, features,
QC, analysis_ready) get rows when Scope 5 specifies their stage/grain. Spec:
target/front_end_naming_and_flow.md (§ PATHS.PY REGISTRY ROWS).
Audit: target/frame_inventory_well_runner_audit.md (review of the registry + the frame_inventory
adapter + well-runner that consume it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# The fixed sub-directory name for one well's shard. Part of the path contract, not a magic
# string scattered in the helpers.
PER_WELL_DIRNAME = "per_well"

# Grain of a step's outputs (its "fanout") — and which path_modes each grain permits:
#   "experiment"          -> one artifact per experiment. Allowed path_mode: experiment.
#   "per_well_then_merge" -> a per-well shard at per_well/{well_id}/, concatenated to an
#                            experiment-level merged view. Allowed path_mode: per_well | merged.
EXPERIMENT = "experiment"
PER_WELL_THEN_MERGE = "per_well_then_merge"

# How to resolve a path for a given call. Each step's `fanout` constrains which of these is legal
# (enforced by _normalize_path_mode):
#   "experiment" -> {stage}/{exp}/{file}                      (the experiment-grain artifact)
#   "per_well"   -> {stage}/{exp}/per_well/{well_id}/{file}   (one well's shard)
#   "merged"     -> {stage}/{exp}/{file}                      (the concatenated experiment view)
# Note: "experiment" and "merged" land in the same directory, but they are kept distinct so a
# per_well_then_merge step's merged view is requested as "merged" (explicit intent), never
# "experiment".
PATH_MODE_EXPERIMENT = "experiment"
PATH_MODE_PER_WELL = "per_well"
PATH_MODE_MERGED = "merged"

# Which path_modes each fanout allows.
_ALLOWED_PATH_MODES: dict[str, tuple[str, ...]] = {
    EXPERIMENT: (PATH_MODE_EXPERIMENT,),
    PER_WELL_THEN_MERGE: (PATH_MODE_PER_WELL, PATH_MODE_MERGED),
}


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
#   artifacts : {artifact_key: template}. A template is EITHER a single string (same filename in
#               every legal path_mode) OR a {path_mode: string} dict when the filename differs by
#               mode (e.g. a per-well shard names the well, the merged view names the experiment).
#               Templates substitute caller-provided tokens only ({experiment_id}, {well_id}, and
#               any format_vars like {scope}) — see the NO-LEAKAGE BOUNDARY above. Sidecars
#               (.validated, .provenance.json) are DERIVED by the helpers, not listed here.
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
        # acquisition_inventory: the maximal per-coordinate record emitted from that one read
        # (YX1: record-only, scope-shaped — see target/acquisition_inventory_flow.md).
        "artifacts": {
            "raw": "scope_metadata__{scope}.csv",
            "acquisition_inventory": "acquisition_inventory__{scope}.csv",
        },
    },
    "map_positions_to_wells": {
        "stage": "experiment_metadata",
        "fanout": EXPERIMENT,
        # .provenance.json via provenance_path().
        "artifacts": {"mapping": "position_well_mapping.csv"},
    },
    "apply_position_to_well_mapping": {  # CONVERGENCE LINE; well_id comes from the position map.
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

    # ── PER-WELL MATERIALIZATION (the scope backend; emits images + the inventory shard) ──
    # materialize_well[well_id] is the live promotion of the accepted candidate (Step 6). It runs
    # ONE well at a time (fanned over discovered_wells.txt), writes pixel files through
    # materialized_image_paths.py (candidate/ vs live owned THERE, not the registry), and emits the
    # per-well frame-inventory shard that validate_frame_inventory_for_well consumes. The done
    # sentinel means "the configured image-product set for this well finished."
    "materialize_well": {
        "stage": "built_image_data",
        "fanout": PER_WELL_THEN_MERGE,
        "artifacts": {
            "inventory": {
                PATH_MODE_PER_WELL: "{well_id}_frame_inventory.csv",
                PATH_MODE_MERGED: "{experiment_id}_frame_inventory.csv",
            },
            "done": {
                PATH_MODE_PER_WELL: "{well_id}.materialize_well.done",
            },
        },
    },

    # ── POST-FAN — frame inventory joins the spine ────────────────────────────
    # "frame_inventory" is the logical product (a noun), kept as ONE step. Snakemake may use
    # several rules around it: materialize_well writes the per-well shards,
    # validate_frame_inventory_for_well writes validation sentinels/reports, and a merge writes the
    # experiment-level view (one-file+sentinel model, see stitched_handoff_contract.md). Those are
    # actions (verbs); the registry keeps one noun-like step for the product they all touch. The
    # per_well shard names the WELL; the merged view names the EXPERIMENT — two honest templates,
    # so paths.py never has to fabricate a {well_id} value for the merged file.
    "frame_inventory": {
        "stage": "experiment_metadata",  # OPEN (findings #5 leans a dedicated frame_inventory/
                                         # stage); placeholder until that is decided.
        "fanout": PER_WELL_THEN_MERGE,
        "artifacts": {
            "inventory": {
                PATH_MODE_PER_WELL: "{well_id}_frame_inventory.csv",
                PATH_MODE_MERGED: "{experiment_id}_frame_inventory.csv",
            },
        },
    },
}

# ── HELPERS: step name -> artifact path ──────────────────────────────────────────────────────
# Two private lookups turn registry keys into concrete values, failing loudly on a typo:
#   _lookup_step("discover_wells")         -> {"stage": ..., "fanout": ..., "artifacts": ...}
#   _resolve_filename("ingest_scope_metadata", "raw", "experiment", {"scope": "yx1"})
#                                          -> "scope_metadata__yx1.csv"
#
# Four public functions build paths from them (root + experiment come from the caller):
#   step_dir        -> the FOLDER, no filename
#       step_dir(ROOT, "discover_wells", "20250912")
#           -> {ROOT}/experiment_metadata/20250912
#       step_dir(ROOT, "frame_inventory", "20250912", path_mode="per_well",
#                well_id="20250912_B01")
#           -> {ROOT}/experiment_metadata/20250912/per_well/20250912_B01
#
#   artifact_path   -> FOLDER + filename  (the one you'll call most)
#       artifact_path(ROOT, "discover_wells", "wells", "20250912")
#           -> {ROOT}/experiment_metadata/20250912/discovered_wells.txt
#       artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
#                     path_mode="per_well", well_id="20250912_B01")
#           -> {ROOT}/.../per_well/20250912_B01/20250912_B01_frame_inventory.csv
#
#   validated_path  -> artifact_path + ".validated"        (the sentinel beside it)
#       validated_path(ROOT, "apply_position_to_well_mapping", "mapped", "20250912")
#           -> {ROOT}/experiment_metadata/20250912/scope_metadata_mapped.csv.validated
#
#   provenance_path -> artifact_path + ".provenance.json"  (the sidecar beside it)
#       provenance_path(ROOT, "map_positions_to_wells", "mapping", "20250912")
#           -> {ROOT}/experiment_metadata/20250912/position_well_mapping.csv.provenance.json


def known_steps() -> tuple[str, ...]:
    """Return all registered step names (sorted) — handy for tests and docs generation."""
    return tuple(sorted(PIPELINE_STEPS))


def known_artifacts(step: str) -> tuple[str, ...]:
    """Return a step's registered artifact keys (sorted)."""
    return tuple(sorted(_lookup_step(step)["artifacts"]))


def _lookup_step(step: str) -> dict:
    """Return a step's registry row, or raise KeyError naming the known steps."""
    try:
        return PIPELINE_STEPS[step]
    except KeyError:
        raise KeyError(
            f"Unknown step {step!r}. Known front-end steps: {sorted(PIPELINE_STEPS)}. "
            f"Back-half steps are not in the registry yet (added in Scope 5)."
        ) from None


def _normalize_path_mode(step: str, path_mode: Optional[str]) -> str:
    """Resolve and validate ``path_mode`` against the step's fanout.

    ``path_mode=None`` means "use the obvious one": an EXPERIMENT step has exactly one legal mode
    (``experiment``), so None resolves to it — callers of experiment-grain steps need not pass
    path_mode. A PER_WELL_THEN_MERGE step has TWO legal modes (``per_well`` vs ``merged``) with no
    neutral default, so the caller MUST choose; None (or an illegal value) raises a message that
    spells out the choice. This makes ``fanout`` executable, not decorative.
    """
    fanout = _lookup_step(step)["fanout"]
    allowed = _ALLOWED_PATH_MODES.get(fanout)
    if allowed is None:
        raise ValueError(f"Step {step!r} has unknown fanout {fanout!r}.")

    if path_mode is None:
        if fanout == EXPERIMENT:
            return PATH_MODE_EXPERIMENT
        # PER_WELL_THEN_MERGE: two legal modes, no neutral default — make the user pick, say why.
        raise ValueError(
            f"Step {step!r} has fanout={PER_WELL_THEN_MERGE!r}. "
            f"Pass path_mode={PATH_MODE_PER_WELL!r} or path_mode={PATH_MODE_MERGED!r} explicitly."
        )

    if path_mode not in allowed:
        if fanout == PER_WELL_THEN_MERGE:
            raise ValueError(
                f"Step {step!r} has fanout={PER_WELL_THEN_MERGE!r}, so the caller must choose "
                f"path_mode={PATH_MODE_PER_WELL!r} for a well shard or path_mode={PATH_MODE_MERGED!r} "
                f"for the experiment-level file. Got path_mode={path_mode!r}."
            )
        raise ValueError(
            f"Step {step!r} has fanout={fanout!r}, so path_mode must be {allowed[0]!r}; "
            f"got {path_mode!r}."
        )
    return path_mode


def _resolve_filename(
    step: str,
    artifact: str,
    path_mode: str,
    format_vars: Optional[dict] = None,
) -> str:
    """Fill in an artifact's filename template for the given ``path_mode``.

    The template is either a single string (same filename in every mode) or a
    ``{path_mode: string}`` dict (filename differs by mode). Raises KeyError naming the step's
    known artifacts if ``artifact`` is unknown, and ValueError (naming the missing token + the
    template) if a ``{token}`` in the template was not supplied.
    """
    spec = _lookup_step(step)
    try:
        template = spec["artifacts"][artifact]
    except KeyError:
        raise KeyError(
            f"Step {step!r} has no artifact {artifact!r}. "
            f"Known artifacts: {sorted(spec['artifacts'])}."
        ) from None
    if isinstance(template, dict):
        try:
            template = template[path_mode]
        except KeyError:
            raise ValueError(
                f"Artifact {artifact!r} of step {step!r} has no filename for path_mode "
                f"{path_mode!r}; defined for {sorted(template)}."
            ) from None
    try:
        return template.format(**(format_vars or {}))
    except KeyError as missing:
        missing_token = missing.args[0]
        raise ValueError(
            f"Filename template {template!r} for {step}/{artifact} needs token {missing_token!r} "
            f"which was not supplied (got format_vars={format_vars!r}). Pass it via format_vars, "
            f"or well_id=/experiment_id= as appropriate."
        ) from None


# Three directory concepts in the layout, each built in exactly ONE place so they cannot drift:
#   _experiment_step_dir   {stage}/{exp}              the experiment base
#   _per_well_step_dir      {stage}/{exp}/per_well    where ALL a step's per-well shards live
#   step_dir(..., well_id)  {stage}/{exp}/per_well/{well_id}   one well's shard dir
# The public composers (step_dir, per_well_step_dir) build downward from these bricks — no caller
# ever has to strip a level off with ``.parent`` to name the per-well directory.


def _experiment_step_dir(root: PathLike, step: str, experiment_id: str) -> Path:
    """The experiment-level directory for a step: ``{stage}/{exp}``. The ONE place this is built."""
    spec = _lookup_step(step)
    return Path(root) / spec["stage"] / str(experiment_id)


def _per_well_step_dir(root: PathLike, step: str, experiment_id: str) -> Path:
    """The directory holding ALL of a step's per-well shards: ``{stage}/{exp}/per_well``.

    The ONE place ``PER_WELL_DIRNAME`` is joined to the experiment base. Asserts the step
    actually fans out per-well (``_normalize_path_mode`` fails loud on an experiment-grain step,
    which has no per-well directory). A specific shard dir is this plus ``{well_id}`` (step_dir);
    the merge lists this directly.
    """
    _normalize_path_mode(step, PATH_MODE_PER_WELL)  # guard: per_well_then_merge steps only
    return _experiment_step_dir(root, step, experiment_id) / PER_WELL_DIRNAME


def per_well_step_dir(root: PathLike, step: str, experiment_id: str) -> Path:
    """Return the directory holding ALL per-well shards for a per_well_then_merge step.

    The named path concept ``{stage}/{exp}/per_well`` — agnostic to who reads it (the merge lists
    it to discover which wells have shards; a cleanup might walk it; a report might count it). It
    is the per-well sibling of ``step_dir``'s experiment view, NOT a specific well's shard
    (that needs ``well_id`` via ``step_dir``). Fails loud on an experiment-grain step.
    """
    return _per_well_step_dir(root, step, experiment_id)


def step_dir(
    root: PathLike,
    step: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
) -> Path:
    """Return the directory that holds a step's artifact, before the filename is appended.

    The directory lives under the step's ``stage`` folder. ``per_well`` mode requires ``well_id``
    and yields ``{stage}/{exp}/per_well/{well_id}``; ``experiment``/``merged`` modes yield
    ``{stage}/{exp}``. ``path_mode=None`` resolves to the step's only legal mode for an
    experiment-grain step, but a per_well_then_merge step must be told ``per_well`` or ``merged``
    (see _normalize_path_mode). Composes from the directory bricks above so the per-well shard dir
    is always ``per_well_step_dir(...) / {well_id}`` — the two never drift.
    """
    mode = _normalize_path_mode(step, path_mode)
    if mode == PATH_MODE_PER_WELL:
        if not well_id:
            raise ValueError(
                f"path_mode='per_well' for step {step!r} requires well_id (the GLOBAL "
                f"{{experiment_id}}_{{well_index}} key)."
            )
        return _per_well_step_dir(root, step, experiment_id) / str(well_id)
    # experiment | merged -> the experiment base (the per_well_then_merge merged view lands beside
    # an experiment-grain artifact); the distinction is intent, enforced upstream by fanout.
    return _experiment_step_dir(root, step, experiment_id)


def artifact_path(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Resolve the absolute path to one of a step's artifacts.

    Examples (see target/front_end_naming_and_flow.md for the full set)::

        artifact_path(ROOT, "ingest_scope_metadata", "raw", "20250912",
                      format_vars={"scope": "yx1"})
        #   -> {ROOT}/experiment_metadata/20250912/scope_metadata__yx1.csv

        artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
                      path_mode="per_well", well_id="20250912_B01")
        #   -> {ROOT}/experiment_metadata/20250912/per_well/20250912_B01/20250912_B01_frame_inventory.csv

        artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
                      path_mode="merged")
        #   -> {ROOT}/experiment_metadata/20250912/20250912_frame_inventory.csv
    """
    mode = _normalize_path_mode(step, path_mode)
    directory = step_dir(root, step, experiment_id, path_mode=mode, well_id=well_id)
    # Build the template namespace from SUPPLIED values only — never fabricate identity (see the
    # NO-LEAKAGE BOUNDARY). experiment_id is always available; well_id only if the caller passed
    # it. The merged template names the experiment (it uses {experiment_id}); the per_well
    # template names the well (it uses {well_id}). A template that references a token we did not
    # supply raises a clear ValueError in _resolve_filename.
    fmt = dict(format_vars or {})
    reserved = {"experiment_id", "well_id"}
    bad = reserved & set(fmt)
    if bad:
        raise ValueError(
            f"Do not pass identity tokens via format_vars: {sorted(bad)}. "
            f"Use experiment_id= and well_id= instead."
        )
    fmt["experiment_id"] = str(experiment_id)
    if well_id is not None:
        fmt["well_id"] = str(well_id)
    return directory / _resolve_filename(step, artifact, mode, fmt)


def validated_path(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.validated`` sentinel beside an artifact (trailing-suffix form).

    ⚠️ Today ``apply_position_to_well_mapping`` writes a LEADING-dot form (``.scope_metadata_mapped.validated``)
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
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.provenance.json`` sidecar beside an artifact."""
    base = artifact_path(
        root, step, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".provenance.json")
