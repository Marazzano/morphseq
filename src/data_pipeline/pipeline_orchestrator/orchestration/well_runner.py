"""Well-runner orchestration: the well-level flow around per-well stages.

This module owns the well-level flow around per-well stages. Three jobs, and the MERGE has TWO
forms that differ by WHEN they run (DAG-planning vs rule-execution) — keep them distinct:

    1. COMPUTE side — "which discovered wells should I run?"
       run_well_ids_for_experiment() = discovered ∩ config target_wells. The RUN set.
       Config can narrow this (a smoke run computes one well).

    2a. MERGE, DAG form — "declare which run wells the merge depends on."
        run_well_shard_paths() resolves the RUN wells' shard paths and returns them. It runs at
        DAG-PLANNING time (inside a Snakefile input function, post-checkpoint), so it does NO disk
        check — the files may not be built yet. RETURNING a path IS the dependency: Snakemake then
        builds+validates it or halts. Pure path resolution (run set -> registry paths).

    2b. MERGE, disk form — "assemble whatever validated shards exist right now."
        collect_well_shard_paths() scans the per_well/ dir for every VALIDATED shard. It runs at
        RULE-EXECUTION time (inside a running rule), so it DOES read disk + check sentinels (skip
        mid-flight, raise on corrupt). Config-independent — a file-refresh merge over all present.

    3. concat_well_shards_to_file() row-concatenates a given list of shard files into one
       experiment-level table. Dumb + NOT well-aware (takes files, writes a file); it lives here
       only because the well merges are its sole callers today. Both 2a and 2b feed it.

The 2a/2b split is the load-bearing distinction: a DAG merge declares deps on the RUN set and
never touches disk truth (Snakemake owns existence); a disk-refresh merge reads disk truth after
things are built. ``.validated`` sentinels carry CONTENT-correctness into the DAG (Snakemake only
knows existence, so a validate rule turns "is it correct?" into "does {shard}.validated exist?").

Two hard rules this module obeys:
  - NO-LEAKAGE: it never mints or parses an id with an inline f-string or ``.split("_")``. To
    promote a local ``well_index`` to a global ``well_id`` it CALLS ``build_well_id`` from
    ``shared.identifiers`` (identity flows *into* orchestration, never the other way).
  - OUTPUT_ROOT IS A PARAMETER: paths are resolved through ``orchestration.paths`` with the
    caller-supplied ``output_root``; this module never derives it from ``PROJECT_ROOT``.

LENIENT BOUNDARY, STRICT INTERNAL (the well-selection design). Config ``target_wells`` is a
human-typed boundary: a user may write a local slug (``B01``) or a global ``well_id``
(``20250912_B01``). We normalize that to a global ``well_id`` exactly ONCE, at the seam. From
``run_well_ids_for_experiment``'s return value onward, every value is a global ``well_id`` —
no bare slug ever travels internally.

⚠️ FORWARD DECLARATION. Like ``paths.py``, this reads the TARGET artifacts via the registry
(``discovered_wells.txt``, per-well shard paths), global ``well_id``s only. It does NOT tolerate
the legacy ``wells.txt`` / local-id form and is wired into the live Snakefile's
post-checkpoint well expansion path.

Spec: docs/refactors/streamline-snakemake/well_id_throughline_refactor_plan.md (Scope 4) and
target/front_end_naming_and_flow.md (Decision 8).
Audit: target/frame_inventory_well_runner_audit.md (concat_well_shards_to_file is the canonical
merge primitive; finding #4 flags frame_inventory.merge_frame_inventory_shards for re-duplicating it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from data_pipeline.shared.identifiers import build_well_id, validate_well_id

from data_pipeline.pipeline_orchestrator.orchestration import paths

PathLike = Union[str, Path]

# The config keys that name a per-experiment well subset. Today's config uses ``experiment_wells``;
# the TARGET name (front_end doc) is ``target_wells``. Accept either so the runner works before and
# after the rename — both mean "the config-only filter over discovered wells."
_TARGET_WELLS_CONFIG_KEYS = ("target_wells", "experiment_wells")


# ──────────────────────────────────────────────────────────────────────────────────────────
# COMPUTE side — which discovered wells should run for an experiment?
# ──────────────────────────────────────────────────────────────────────────────────────────


def run_well_ids_for_experiment(
    experiment_id: str,
    config: dict,
    *,
    output_root: PathLike,
) -> list[str]:
    """Return the RUN wells for one experiment (global ``well_id``s): ``discovered ∩ target_wells``.

    "Run wells" is the noun: the wells THIS run computes. (Named ``run_*`` not ``selected_*`` —
    selecting and the selected set are different things; this returns the set.)

    The single definition of "which wells compute" (replaces the three scattered copies — config,
    ``targets.py``, and the ad-hoc Snakefile helper). The checkpoint is the source of truth for
    *which wells exist*; config only *filters*:

    - **No config wells** for this experiment -> return ALL discovered wells (no filter).
    - **Config wells present** -> normalize each to a global ``well_id`` (a local ``B01`` or a
      global ``20250912_B01`` both accepted), then return ``discovered ∩ target`` in DISCOVERED
      order. **FAIL LOUD** if config names a well not in ``discovered_wells.txt`` — a typo cannot
      silently shrink the run set.

    The return value is global ``well_id``s only — no bare slug escapes this function.

    Raises:
        FileNotFoundError: if the discover_wells checkpoint has not produced its output.
        ValueError: if config names a target well not present in discovered_wells.txt.
    """
    discovered_well_ids = _read_discovered_well_ids(experiment_id, output_root)

    config_entries = _config_target_wells(experiment_id, config)
    if not config_entries:
        # No config filter -> the data decides: run everything discovered.
        return discovered_well_ids

    target_well_ids = {
        _normalize_to_well_id(experiment_id, entry) for entry in config_entries
    }

    discovered_set = set(discovered_well_ids)
    missing_from_discovery = target_well_ids - discovered_set
    if missing_from_discovery:
        raise ValueError(
            f"Config target wells {sorted(missing_from_discovery)} for experiment "
            f"{experiment_id!r} are not in discovered_wells.txt (discovered: "
            f"{sorted(discovered_set)}). Config can only filter the discovered set, never name a "
            "well that does not physically exist."
        )

    # Intersect in DISCOVERED order — the checkpoint (data) decides existence and order; config
    # only narrows.
    return [well_id for well_id in discovered_well_ids if well_id in target_well_ids]


def _read_discovered_well_ids(experiment_id: str, output_root: PathLike) -> list[str]:
    """Read every discovered global ``well_id`` from the checkpoint's ``discovered_wells.txt``.

    Path resolved through the registry so the runner and the rule agree on the location;
    ``output_root`` is supplied by the caller (never derived). One global ``well_id`` per line,
    blanks ignored, order preserved (the data decides existence AND order). Each entry is
    validated as a GLOBAL well_id — a bare local label in this file is stale/corrupt and must not
    travel downstream (matching the contract, not just describing it).
    """
    wells_file = paths.artifact_path(output_root, "discover_wells", "wells", experiment_id)
    if not wells_file.exists():
        raise FileNotFoundError(
            f"discovered_wells.txt not found for experiment {experiment_id!r} at {wells_file}. "
            "The discover_wells checkpoint must run before wells can be selected."
        )
    discovered: list[str] = []
    for line in wells_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            discovered.append(validate_well_id(text))
        except ValueError as exc:
            raise ValueError(
                f"Invalid well_id {text!r} in discovered wells file {wells_file}: a discovered "
                "well must be a global well_id, not a bare local label."
            ) from exc
    return discovered


def _config_target_wells(experiment_id: str, config: dict) -> list[str]:
    """Return the raw (un-normalized) config well entries for one experiment, or ``[]``.

    Reads the config-only filter (``target_wells``/``experiment_wells``) — a
    ``{experiment_id: [entry, ...]}`` map. Entries are returned verbatim (local slug OR global
    well_id); normalization happens in ``_normalize_to_well_id``. An absent/empty entry yields
    ``[]``, which the caller reads as "no filter — run all discovered."
    """
    for config_key in _TARGET_WELLS_CONFIG_KEYS:
        section = config.get(config_key)
        if not section:
            continue
        value = _get_experiment_section_value(section, experiment_id)
        wells = _as_well_list(value, config_key=config_key, experiment_id=experiment_id)
        if wells:
            return wells
    return []


def _get_experiment_section_value(section: dict, experiment_id: str):
    """Return ``section[experiment_id]``, tolerating YAML int-vs-str keys (20250912 may parse int)."""
    for key, value in section.items():
        if str(key) == str(experiment_id):
            return value
    return None


def _as_well_list(value, *, config_key: str, experiment_id: str) -> list[str]:
    """Coerce a config value into a list of well-entry strings (one job: shape coercion).

    ``None`` -> ``[]`` (no filter). A bare string -> a one-element list (a user may write a single
    well, not a list). A sequence -> stringified elements. Anything else fails loud naming the
    offending config location and type.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(well) for well in value]
    except TypeError as exc:
        raise ValueError(
            f"Config {config_key}[{experiment_id!r}] must be a list of wells or a single well "
            f"string; got {type(value).__name__}."
        ) from exc


def _normalize_to_well_id(experiment_id: str, entry: str) -> str:
    """Promote one config entry to a global ``well_id`` (the lenient-boundary normalization).

    A config entry may be a LOCAL slug (``B01``) or an ALREADY-GLOBAL well_id (``20250912_B01``).
    We decide which by asking the IDENTITY kingdom — ``validate_well_id`` passes on a global id and
    raises on a bare local label — never by a string heuristic here. A local entry is promoted via
    ``build_well_id`` (the one sanctioned join point); a global entry is kept as-is.
    """
    text = str(entry).strip()
    try:
        return validate_well_id(text)
    except ValueError:
        # Not already a global well_id -> treat as a local well_index and promote it.
        return build_well_id(experiment_id, text)


# ──────────────────────────────────────────────────────────────────────────────────────────
# MERGE side — which per-well shards feed the concat?
# Two collectors (see module docstring 2a/2b): run_well_shard_paths (DAG/planning, pure) and
# collect_well_shard_paths (disk scan, runtime). Both feed concat_well_shards_to_file.
# ──────────────────────────────────────────────────────────────────────────────────────────


def run_well_shard_paths(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    well_ids: Sequence[str],
) -> list[Path]:
    """Resolve the per-well shard paths for a GIVEN set of run wells (the DAG-merge collector).

    This is the DAG form of merge collection (module docstring 2a). It runs at DAG-PLANNING time —
    inside a Snakefile input function, after the discover_wells checkpoint — to declare which run
    wells the merge depends on. It does **NO disk check**: the shards may not be built yet, and
    RETURNING a shard's path IS how the merge declares its dependency (Snakemake then builds +
    validates it, or halts the DAG). Checking existence here would fight Snakemake by raising
    during planning for files that are about to be produced.

    Each ``well_id`` is validated as a global id (a bare local label is a caller bug, caught here —
    that is pure shape-checking, not a disk read). Paths come from the registry, so the input
    function and the producing rules agree on locations.

    Returns the ARTIFACT (CSV) paths — what ``concat_well_shards_to_file`` reads. Making the merge
    rule WAIT on each shard's ``.validated`` sentinel is a separate Snakefile concern (the rule
    lists ``validated_path(...)`` as its input trigger): the content-validation gate is wired in the
    rule, not baked into this pure resolver. (Deferred to Scope 5 when rules are reconstructed.)

    Contrast ``collect_well_shard_paths`` (2b), which DOES read disk because it runs inside an
    executing rule, after the shards exist.
    """
    return [
        paths.artifact_path(
            root, step, artifact, experiment_id,
            path_mode=paths.PATH_MODE_PER_WELL, well_id=validate_well_id(well_id),
        )
        for well_id in well_ids
    ]


def collect_well_shard_paths(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
) -> list[Path]:
    """Return the paths of VALIDATED per-well artifact shards present for one experiment.

    Merge-side selection is filesystem-driven, NOT config-driven: list the step's per-well
    directory (``paths.per_well_step_dir``), treat each child directory name as a global
    ``well_id``, and resolve that well's artifact + ``.validated`` sentinel through the registry.
    Config is never consulted. The caller feeds the returned paths straight to
    ``concat_well_shards_to_file`` — no intermediate object, because the only thing the merge needs
    is the files. (If a future consumer needs per-shard metadata — a merge manifest, "merged wells:
    …" logging — promote this to return a small value object then.)

    Fail-loud taxonomy (the per-well dir is a CONTRACT directory, not scratch):
      - a child dir whose name is not a valid global ``well_id``  -> raise (pipeline dirt / bug);
      - a ``.validated`` sentinel with no artifact beside it       -> raise (corrupt shard);
      - an artifact with no sentinel                               -> SKIP (legitimately mid-flight,
                                                                      not validated yet);
      - both present                                               -> keep.

    Returns ``[]`` when the per-well dir exists but holds no validated shards yet — the caller
    (``merge_well_shards``) decides whether empty input is fatal. Raises ``FileNotFoundError`` if
    the per-well dir itself does not exist (the step has not fanned out at all).
    """
    shard_root = paths.per_well_step_dir(root, step, experiment_id)
    if not shard_root.exists():
        raise FileNotFoundError(
            f"Per-well shard directory not found for {step}/{artifact}, experiment "
            f"{experiment_id!r}: {shard_root}. The per-well compute step must run before its "
            "shards can be collected."
        )

    shard_paths: list[Path] = []
    # Collection order is by well_id directory name (deterministic merge input order).
    for well_dir in sorted(
        (p for p in shard_root.iterdir() if p.is_dir()), key=lambda p: p.name
    ):
        # A bad directory name fails loud here (validate_well_id raises on a non-global name).
        try:
            well_id = validate_well_id(well_dir.name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid well_id directory under {shard_root}: {well_dir.name!r} is not a global "
                "well_id. The per-well directory is a contract directory, not scratch."
            ) from exc
        shard_path = paths.artifact_path(
            root, step, artifact, experiment_id,
            path_mode=paths.PATH_MODE_PER_WELL, well_id=well_id,
        )
        sentinel_path = paths.validated_path(
            root, step, artifact, experiment_id,
            path_mode=paths.PATH_MODE_PER_WELL, well_id=well_id,
        )
        if sentinel_path.exists() and not shard_path.exists():
            raise FileNotFoundError(
                f"Corrupt shard for well {well_id!r}: validation sentinel {sentinel_path} exists "
                f"but its artifact {shard_path} is missing."
            )
        if shard_path.exists() and sentinel_path.exists():
            shard_paths.append(shard_path)
        # artifact without sentinel -> not validated yet; skip silently (mid-flight is normal).
    return shard_paths


# ──────────────────────────────────────────────────────────────────────────────────────────
# CONCAT — row-stack the collected shard files into one experiment-level table, to a file
# ──────────────────────────────────────────────────────────────────────────────────────────


def concat_well_shards_to_file(
    well_shard_paths: Sequence[PathLike],
    output_path: PathLike,
    *,
    required_columns: Optional[Sequence[str]] = None,
    sort_columns: Optional[Sequence[str]] = None,
) -> None:
    """Row-concatenate per-well shard tables into one experiment-level table, written to a file.

    Reads each path in ``well_shard_paths`` (CSV or parquet by suffix), checks every
    ``required_columns`` is present IN EACH SHARD (fail loud naming the offending file — a shard
    missing a contract column is a bug, not a silent drop), row-stacks them, optionally sorts by
    ``sort_columns`` (only those actually present, for a deterministic table), and writes
    ``output_path`` in the format implied by its suffix.

    The per-shard column check is deliberate: ``pd.concat`` UNIONS columns, so checking only the
    merged frame would pass even when one shard lacks the column (it would just be filled with
    NaN). The contract is "every shard carries the column," so each shard is checked before concat.

    This is intentionally NOT well-aware: it takes a list of files and writes one file. The CALLER
    (a stage's merge rule) collects the shards via ``collect_well_shard_paths`` and passes its own
    contract columns. Stage-specific behavior (e.g. symlink browse-views) stays in the stage module.

    Raises ``ValueError`` on empty input (nothing to concat) or a shard missing a required column.
    """
    paths_in = [Path(p) for p in well_shard_paths]
    if not paths_in:
        raise ValueError(
            "concat_well_shards_to_file: no shards to concatenate (well_shard_paths is empty)."
        )

    frames = []
    for path in paths_in:
        frame = _read_table(path)
        if required_columns:
            missing = [c for c in required_columns if c not in frame.columns]
            if missing:
                raise ValueError(
                    f"concat_well_shards_to_file: shard {path} is missing required columns "
                    f"{missing}. Present columns: {list(frame.columns)}."
                )
        frames.append(frame)
    merged = pd.concat(frames, axis=0, ignore_index=True)

    if sort_columns:
        present = [c for c in sort_columns if c in merged.columns]
        if present:
            merged = merged.sort_values(present).reset_index(drop=True)

    _write_table(merged, Path(output_path))


def _read_table(path: Path) -> pd.DataFrame:
    """Read a shard table by suffix (.parquet -> parquet, else CSV)."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a merged table by suffix (.parquet -> parquet, else CSV), making parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
