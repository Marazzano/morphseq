# Source Tree ↔ Output Stage Alignment — target `src/data_pipeline/` layout

**Status:** PLAN (2026-07-01). Nothing moved yet. This is the destination spec + migration
ordering to review before any file moves. Feeds the Snakemake streamline refactor.

---

## The idea (and why it "looks so nice")

The **output** tree (`data_pipeline_output/`) is organized by pipeline *regime* — a clean river:

```
acquisition → object_extraction → feature_extraction → quality_control → analysis_ready
```

The **source** tree (`src/data_pipeline/`) is organized by *implementation module* and has drifted:
duplicate homes for the same step, live-but-tangled legacy subsystems, thin shims and empty stubs
left behind by earlier moves. The output reads well because it was designed against a stage
vocabulary. The source predates that vocabulary.

**Goal:** make the source tree mirror the output regimes, so the folder layout, the on-disk output,
and the `PIPELINE_STEPS` registry in `orchestration/paths.py` all tell the *same* story:

```
stage (source folder)  ==  stage (output top-level folder)  ==  "stage" string in PIPELINE_STEPS
```

This spec is the destination. It does **not** authorize file moves — it authorizes the review that
precedes them.

---

## The mapping already exists as a code contract

The source→output mapping is not new; it already lives in
`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` as `PIPELINE_STEPS`. Its docstring:

> `stage` — a PHASE of the pipeline, and the top-level FOLDER under `output_root`. MANY steps share a stage.
> `step`  — one unit of work WITHIN a stage — a `tasks.py` verb's OUTPUT SLOT.

So every step *already* declares which stage it belongs to. The reorg makes the **source directory
layout** agree with what the registry already says — it does not invent a new taxonomy. Cross-checks:
[[project_pipeline_path_vocabulary]] (stage/step/artifact), [[project_two_tree_reconciliation]]
(morphseq-docs is the canonical tree), [[project_front_end_recompose]].

---

## Two vocabulary snags — both RESOLVED

1. **`feature_extraction` vs `features`. → RESOLVED 2026-07-01: `feature_extraction`.**
   `output_tree_doctrine.md` states the regime river as `... → features → ...`, but `paths.py`
   writes `"stage": "feature_extraction"` (comment at `paths.py:418`: "the on-disk stage matches
   it"). **Decision:** standardize on **`feature_extraction`** everywhere — it is the on-disk truth
   AND it mirrors `object_extraction`, so the river reads as parallel `*_extraction` phases. Phase C
   fixes the doctrine's river prose (`features` → `feature_extraction`). Do not leave both.

2. **`analysis_ready` is a real 5th stage.** It appears in the doctrine river, has a source module
   (`analysis_ready/`), and has a `PIPELINE_STEPS` row. The target is therefore **5 stages, not 4.**

---

## Target source layout

```
src/data_pipeline/
  acquisition/
    metadata_ingest/          # (from metadata_ingest/)  scope+plate ingest, mapping, frame_inventory
      scope/
        yx1/mappings.py       #   dialect map: YX1_CHANNEL_MAP        (already here — keep)
        keyence/mappings.py   #   dialect map: KEYENCE_CHANNEL_INDEX_MAP (already here — keep)
        shared/
          canonical_mapper.py #   applies a dialect map, fails loud    (already here — keep)
          channel_validation.py  # validate_channel_id() — self-checking (FROM schemas/, SEE #vocab)
    image_building/           # (from image_building/)   yx1/, keyence/, scope/, frame tiler
    image_materialization/    # (from image_materialization/) write policy, materializers
    # NOTE: no identifiers/ here. ID *grammar* is cross-cutting → stays in shared/ (decision #2).

  object_extraction/
    detection/                # (from detection/)         grounding-dino etc.
    segmentation/             # (from segmentation/)      grounded_sam2, sam2_video, unet, registry
    auxiliary_masks/          # (from auxiliary_masks/)   via/yolk/focus/bubble UNet, snip-grain
    snip_processing/          # (from snip_processing/)   snip inventory + aux mask snips
    #   physical_embryo_registry = ID MINTING step → lives here (decision #2)

  feature_extraction/         # (from feature_extraction/, ABSORBING features/ + embeddings/)
    latent_embeddings/ mask_geometry/ curvature_metrics/ pose_kinematics/
    stage_predictions/ fraction_alive/

  quality_control/            # (from quality_control/ — ALREADY stage-named, minimal change)
    surface_area_qc/ mask_quality_qc/ focus_qc/ motion_blur_qc/ death_detection/ snip_qc/

  analysis_ready/             # (from analysis_ready/ — ALREADY stage-named)

  # cross-cutting (NOT a stage; deliberately flat — imported BY stages, never the reverse):
  shared/
    identifiers/              # ID GRAMMAR: constructors/parsers/validators (build_snip_id, etc.)
    channel_vocabulary.py     # VALID_CHANNEL_NAMES — canonical channel tokens (FROM schemas/, SEE #vocab)
    path_contracts.py
  io/  utils/  viz/  models/
  pipeline_orchestrator/      # Snakemake rules, orchestration/paths.py, tasks.py — the DRIVER, not a stage

  # DELETED (see retirement table): config/  schemas/  identifiers/ (top-level stub)
```

**Rule of thumb:** a folder is a *stage* only if it is a phase of the regime river. `shared/`, `io/`,
`utils/`, `viz/`, `models/`, and `pipeline_orchestrator/` are cross-cutting/driver and stay at the
root — they are imported *by* stages, they are not stages. Mirroring them into a stage would be wrong.

---

## Cross-cutting folder audit — do we need all of them?

Verified 2026-07-01 (py-file counts + importers).

| Folder | Reality | Verdict |
|---|---|---|
| `shared/` | id grammar spine + path_contracts (6 files) | **Keep** — real cross-cutting home. Gains `channel_vocabulary.py`. |
| `io/` | loaders/savers/validators, 8 importers | **Keep** — real |
| `models/` | unet/sam2/groundingdino/paths, 5 files | **Keep** — real (model-weight loaders, cross-stage) |
| `viz/` | render_snip/well/overlay, 5 files | **Keep** — real |
| `utils/` | **only** `cuda_diagnostics.py` (7 importers) | **Keep** (open item — see below) |
| `config/` | **empty 0-byte `__init__.py`**; "config" importers point at other packages' config, not this one | **DELETE** — a package wrapping nothing (same class as the empty `identifiers/` stub) |
| `schemas/` | **one real file** (`channel_normalization.py`); its own `__init__` docstring says the shared-schema bucket was already retired | **DELETE** after splitting its one file (see #vocab) |
| `pipeline_orchestrator/` | the Snakemake driver | **Keep** — but it is the DRIVER, not cross-cutting; its own category |

---

## <a name="vocab"></a>Channel normalization — the four tiers (RESOLVED 2026-07-01)

`schemas/channel_normalization.py` (~40 lines) is the last occupant of `schemas/`. It contains two
*kinds* of thing, which belong in two different homes. The per-scope dialect maps it references
already live correctly in the front-end — they do **not** move, and they do **not** go in `config/`.

| Tier | Symbol / file | Nature | Home (target) |
|---|---|---|---|
| **Dialect** (per-scope) | `YX1_CHANNEL_MAP`, `KEYENCE_CHANNEL_INDEX_MAP` | scope strings → canonical; a **sticky per-scope contract** (feeds `channel_id`→`image_id`) | `scope/{yx1,keyence}/channel_map.py` — **rename from `mappings.py`; make self-validating (see below)** |
| **Mapper** | `apply_canonical_mapping` | applies a dialect map, fails loud on unmapped | `acquisition/metadata_ingest/scope/shared/canonical_mapper.py` — **already here, keep** |
| **Validator** | `validate_channel_id()` | self-checking (snip_qc-style) guard | → `acquisition/metadata_ingest/scope/shared/channel_validation.py` (co-locate w/ mapper) |
| **Vocabulary** | `VALID_CHANNEL_NAMES`, `BRIGHTFIELD_CHANNELS` | canonical tokens everything maps *into*; pure data | → `shared/channel_vocabulary.py` (pure, cross-cutting) |

**Why the vocabulary tokens go to `shared/` and NOT the front-end with the maps:** three call-sites
need the tokens — the dialect maps (front-end), the validator (front-end), and
`shared/identifiers/parsers.py`. If the tokens lived in the front-end, `parsers.py` (a `shared/`
module) would import *up into a stage* — inverting the cross-cutting dependency direction. Keeping the
**tokens** pure in `shared/` makes every edge point down: dialect→vocab ✓, validator→vocab ✓,
parsers→vocab ✓. The **validator** (the self-checking logic the user wanted near the metadata
engines) co-locates with the mapper and dialect maps in the front-end, where its only consumers are.

> User's original question answered: the scope-specific channel maps do **not** go in `config/`
> (that empty package is being deleted). They stay co-located with their scope.

### The dialect map is a sticky per-scope contract — make it self-validating (RESOLVED 2026-07-01)

User objection (well-founded): `mappings.py` is a **vague name** for something *sticky* — the map
produces `channel_id`, which is embedded in `image_id`. And the map's integrity check
(`test_scope_channel_maps_target_valid_channel_ids`) currently lives in a **test file**, bolted on
from outside — nothing guards the map if it's loaded outside pytest. This should work like `snip_qc`:
the contract *owns* its consistency check.

**Decision — Python module + module-owned validator (NOT a YAML/config file):**

- **Rename** `scope/<scope>/mappings.py` → `scope/<scope>/channel_map.py` (name states what it is).
- **Move the integrity check out of the test and into the module** — validate at
  import/construction that every target is a canonical token, via a real fail-loud (scope + raw value
  + invalid target in the message), NOT a bare `assert` (asserts strip under `python -O`; a sticky
  identity token deserves the strong guarantee). The test then *verifies the guard exists* rather than
  *being* the guard — the snip_qc parallel.
- **Preferred shape:** a small `ScopeChannelMap` wrapper (validates on construction, `.to_canonical()`),
  so the raw dict can't be imported and used unguarded, and `apply_canonical_mapping` collapses into a
  method. See the reference implementation below. Acceptable lighter shape: plain module + an
  import-time validating call. Either beats the current external-test-only guard.

### Reference implementation — `ScopeChannelMap` (readability over succinctness)

Written to be read top-to-bottom by someone new to the codebase: descriptive names, one dict entry
per line, and error messages that say what broke *and how to fix it*. Deliberately verbose — this is
an identity contract, so being obvious matters more than being short.

`acquisition/metadata_ingest/scope/shared/channel_map_contract.py`:

```python
"""The self-validating contract for a scope's channel map.

A "channel map" answers one question for one microscope: given the raw channel
name that scope writes to disk (its "dialect"), what is our canonical channel_id?

This matters because channel_id is STICKY: it gets embedded inside image_id, which
identifies every frame for the rest of the pipeline. A wrong channel_id here becomes
a wrong image_id everywhere downstream. So the map checks itself the moment it is
built — a bad map fails loudly at construction, not silently three stages later.
"""

from __future__ import annotations

from data_pipeline.shared.channel_vocabulary import VALID_CHANNEL_NAMES


class ScopeChannelMap:
    """One microscope's raw-channel-name -> canonical-channel_id lookup, self-checked.

    Build one like this (one entry per line, reads like a dictionary):

        YX1_CHANNEL_MAP = ScopeChannelMap(
            scope_name="YX1",
            raw_name_to_channel_id={
                "Empty":       "BF",
                "EYES - Dia":  "BF",
                "EYES - GFP":  "GFP",
                "EYES - RFP":  "RFP",
            },
        )

    Then translate a raw name with:

        canonical_channel_id = YX1_CHANNEL_MAP.to_canonical("EYES - Dia")   # -> "BF"
    """

    def __init__(self, scope_name: str, raw_name_to_channel_id: dict[str, str]) -> None:
        # Keep the scope name so error messages can say WHICH microscope is misconfigured.
        self.scope_name = scope_name

        # Store our own copy so nobody can change the map after we have checked it.
        self.raw_name_to_channel_id = dict(raw_name_to_channel_id)

        # Check the whole map right now, at construction time. If anything is wrong,
        # we raise here and the program stops — better than handing back a broken map.
        self._check_every_target_is_a_canonical_channel_id()

    def _check_every_target_is_a_canonical_channel_id(self) -> None:
        """Every value the map points AT must be a real canonical channel_id.

        Example of a mistake this catches: mapping "EYES - Cy5" -> "Cy5" when "Cy5"
        is not (yet) a canonical channel. That would mint a channel_id nothing else
        in the pipeline understands.
        """
        canonical_channel_ids = set(VALID_CHANNEL_NAMES)

        for raw_name, channel_id in self.raw_name_to_channel_id.items():
            if channel_id not in canonical_channel_ids:
                raise ValueError(
                    f"[{self.scope_name} channel map] The raw channel {raw_name!r} is "
                    f"mapped to {channel_id!r}, but {channel_id!r} is not a canonical "
                    f"channel_id.\n"
                    f"  Canonical channel_ids are: {sorted(canonical_channel_ids)}\n"
                    f"  To fix: either map {raw_name!r} to one of those, or — if "
                    f"{channel_id!r} is a genuinely new channel — add it to "
                    f"VALID_CHANNEL_NAMES in shared/channel_vocabulary.py first.\n"
                    f"  Do NOT map to a non-canonical token: channel_id is embedded in "
                    f"image_id, so a wrong value here mislabels every downstream frame."
                )

    def to_canonical(self, raw_channel_name: str) -> str:
        """Translate one raw scope channel name into its canonical channel_id.

        Fails loudly if this scope has never declared how to handle that raw name —
        guessing would risk silently mislabeling a real channel.
        """
        if raw_channel_name not in self.raw_name_to_channel_id:
            known_raw_names = sorted(self.raw_name_to_channel_id.keys())
            raise ValueError(
                f"[{self.scope_name} channel map] No mapping for raw channel "
                f"{raw_channel_name!r}.\n"
                f"  Raw channels this scope knows how to translate: {known_raw_names}\n"
                f"  To fix: add {raw_channel_name!r} -> <canonical channel_id> to this "
                f"scope's channel_map.py. Never guess a default — an unmapped channel "
                f"is a real gap, not a brightfield frame."
            )
        return self.raw_name_to_channel_id[raw_channel_name]
```

`acquisition/metadata_ingest/scope/yx1/channel_map.py` (the per-scope data, reads like a dict):

```python
"""YX1's channel map: raw ND2 channel names -> canonical channel_id. Self-checked on import."""

from data_pipeline.metadata_ingest.scope.shared.channel_map_contract import ScopeChannelMap

YX1_CHANNEL_MAP = ScopeChannelMap(
    scope_name="YX1",
    raw_name_to_channel_id={
        "Empty":       "BF",
        "EYES - Dia":  "BF",
        "EYES - GFP":  "GFP",
        "EYES - RFP":  "RFP",
    },
)
```

Call site — `apply_canonical_mapping(...)` collapses into a readable method call:

```python
# before:  apply_canonical_mapping("EYES - Dia", YX1_CHANNEL_MAP, vocabulary=..., field=..., scope_name="YX1")
# after:   YX1_CHANNEL_MAP.to_canonical("EYES - Dia")
```

The test then *verifies the guard exists* instead of *being* the guard (the snip_qc parallel):

```python
def test_channel_map_rejects_a_non_canonical_target():
    with pytest.raises(ValueError, match="not a canonical channel_id"):
        ScopeChannelMap("YX1", {"EYES - Cy5": "Cy5"})   # the MAP rejects it — not the test
```

**Why NOT a per-scope YAML/config + loader:** (1) it revives the `config/` dir we are deleting;
(2) it trades compile-time safety (Python literal, canonical tokens, CI-checked) for runtime-only
loader validation — the *wrong* direction for an identity token; (3) the docstring's
"config-loaded source could replace this later" is a hypothetical that has never paid off — these
maps change ~once per new channel (a 1-line reviewable diff), which is a code-review problem Python
already solves. Config-as-data here is the *illusion* of robustness; static + self-validating is the
real thing.

---

## Full step → stage → source-today mapping

| Output stage | Step (PIPELINE_STEPS key) | Lives today in `src/data_pipeline/` |
|---|---|---|
| acquisition | ingest_plate_metadata, ingest_scope_metadata, map_positions_to_wells, apply_position_to_well_mapping, discover_wells, frame_inventory*, keyence_stitch_map | `metadata_ingest/` |
| acquisition | materialize_well | `image_building/`, `image_materialization/` |
| object_extraction | frame_detections | `detection/` |
| object_extraction | frame_masks, physical_embryo_registry | `segmentation/` |
| object_extraction | snip_inventory, snip_auxiliary_masks | `snip_processing/`, `auxiliary_masks/` |
| feature_extraction | latent_embeddings | `feature_extraction/`, `embeddings/` (shim), `features/` (shim) |
| feature_extraction | mask_geometry, curvature_metrics, pose_kinematics, stage_predictions, fraction_alive | `feature_extraction/` |
| quality_control | surface_area_qc, mask_quality_qc, focus_qc, motion_blur_qc, death_detection_qc, death_event, snip_qc | `quality_control/` |
| analysis_ready | (analysis_ready export) | `analysis_ready/` |

---

## Duplicates, stubs & tangles to retire (Phase A — do this FIRST)

Verified 2026-07-01. Not uniform — some are dead stubs, one is a live subsystem:

| Module | State | Action |
|---|---|---|
| `identifiers/` (top-level) | **Empty stub** — single 0-byte `__init__.py`, **zero importers.** Leftover from the Scope-1 split. | `git rm` the dir. Real grammar lives in `shared/identifiers/` (42 importers), stays put. |
| `config/` | **Empty 0-byte `__init__.py`**, wraps no code. | `git rm`. |
| `schemas/` | one real file → split per #vocab, then empty. | `git rm` after the split. |
| `features/` | **1 file**, a shim. Importers in `analyze/` + `results/` (outside the pipeline). | Collapse into `feature_extraction/`; update downstream importers. Cheap. |
| `embeddings/` | **1 file**, a shim. Importers in `results/`. | Collapse into `feature_extraction/latent_embeddings/`; update downstream. Cheap. |
| `segmentation_and_tracking/` | **LIVE — 27 files.** Imported by `detection/backends`, `snip_processing/ops.py`, several `feature_extraction/` metrics, `models/paths.py`, and a test. | Real untangle, NOT a `git rm`. Fold into `segmentation/` (+ `detection/`) and rewire importers. Highest-effort item — its own sub-task. |
| `feature_extraction/legacy_embeddings/`, `feature_extraction/core/` (if present), `schemas/_archive/` | legacy | Confirm dead, then delete (matches the recent island/facade retirements). |

**Why A before B:** don't move modules you're about to delete; don't rewire imports twice.

---

## Migration ordering (strict — do not reverse)

```
Phase A (retire)  →  Phase B (regroup under stages)  →  Phase C (reconcile vocabulary)
```

- **Phase A — Retire.** Delete empty stubs (`identifiers/`, `config/`); split `schemas/`'s one file
  per #vocab then delete it; collapse `features/` + `embeddings/` shims; untangle
  `segmentation_and_tracking/`; delete confirmed-dead legacy. Result: **one home per step.** Highest
  value, most independent, same *kind* of work as the recent schemas/frame_contract deletions.

- **Phase B — Regroup under the 5 stage folders.** `git mv` each step-module under
  `acquisition/ object_extraction/ feature_extraction/ quality_control/ analysis_ready/`. Mechanical
  but **wide import churn** — every `from data_pipeline.metadata_ingest...` becomes
  `from data_pipeline.acquisition.metadata_ingest...`. Do it stage-by-stage, run tests per stage
  (`PYTHONPATH=src pytest --import-mode=importlib`, per [[feedback_pytest_import_mode_importlib]]).

- **Phase C — Reconcile vocabulary.** Make source folder names *equal* the `paths.py` stage strings;
  update `output_tree_doctrine.md` river prose (`features` → `feature_extraction`) + the `paths.py`
  docstring so all three agree. Land the `PIPELINE_STEPS` stage strings and the source layout in the
  same commit so they can never drift again.

---

## Open decisions

1. ~~**`feature_extraction` vs `features`**~~ — RESOLVED: `feature_extraction` (on-disk truth +
   mirrors `object_extraction`). See snag #1.
2. ~~**identifier code home**~~ — RESOLVED. Verified: top-level `identifiers/` is an empty 0-byte
   stub, zero importers; `shared/identifiers/` is the real grammar (42 importers). Split by nature:
   - **ID grammar** (build/parse/validate id strings) — pure, cross-cutting → **`shared/identifiers/`**,
     not a stage. ([[project_identity_spine_doctrine]])
   - **ID minting** (`physical_embryo_registry`) — a step, writes an artifact → travels with
     **`object_extraction/`**. It *uses* the grammar; it is not the grammar.
     ([[project_physical_embryo_registry_complete]])
   - Empty top-level `identifiers/` stub → delete in Phase A.
3. ~~**channel_normalization home**~~ — RESOLVED. Split four tiers (see #vocab): dialect maps + mapper
   stay in front-end (already there); validator → front-end `scope/shared/`; vocabulary tokens →
   `shared/channel_vocabulary.py`; `schemas/` deleted. Scope maps do NOT go in `config/`.
4. **`utils/` (one file, `cuda_diagnostics.py`)** — fold into `shared/` (no bare junk-drawer) or keep
   as-is (conventional name, avoids churning 7 importers)? Low stakes; still open.
5. ~~**Transition style**~~ — RESOLVED 2026-07-01: **hard cutover**, stage-by-stage, tests gating each
   stage. No re-export shims at old import paths. Each stage's `git mv` + import sweep lands as one
   commit that leaves the tree green (`PYTHONPATH=src pytest --import-mode=importlib`).

6. **Channel dialect map shape** — RESOLVED 2026-07-01 (see #vocab): rename `mappings.py` →
   `channel_map.py`, module-owned self-validation (fail-loud, not `assert`), Python not YAML.
   Preferred `ScopeChannelMap` wrapper; open sub-choice: full wrapper vs plain module + import-time
   check. Low stakes — decide at implementation.

---

## What this spec does NOT do

- Does not move, rename, or delete any file (yet).
- Does not merge the cross-cutting modules (`shared/`, `io/`, `utils/`, `viz/`, `models/`,
  `pipeline_orchestrator/`) into stages — they are deliberately flat.
- Does not change `PIPELINE_STEPS` step keys — only (Phase C) reconciles stage strings.
