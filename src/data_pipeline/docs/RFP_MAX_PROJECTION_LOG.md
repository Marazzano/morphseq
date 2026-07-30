# Log — per-channel max projection (RFP) on the front end

Running log for wiring `RFP × projection × max` through materialization. Newest status at the top of
each section. Companion to the approved plan; this file tracks what actually happened, including
decisions taken without user sign-off (flagged **[UNILATERAL]**).

Branch: `feat/experiment-collections` (worktree `.coll_wt`). Base commit: `3737569a`.

---

## Status

| Step | State |
|---|---|
| Baseline test capture | DONE — 12 failed / 436 passed |
| RFP test subject identified | DONE — pbx 3-source YX1 collection |
| 1. `write_index_map` plan field + schema v2 | pending |
| 2. Polarity base default flip | pending |
| 3. Resolver gates opened | pending |
| 4. Executor max branch + de-hardcode `"BF"` | pending |
| 5. Exact-pixel uint16 test | pending |
| 6. `index_map_path` rename + `write_index_map` column | pending |
| 7. Channel-absent diagnostic failure | pending |
| 8. Per-run RFP config + real pbx validation | pending |

---

## Pre-work findings (2026-07-29)

### Test baseline BEFORE any edit
`tests/data_pipeline/acquisition/` at `3737569a`: **12 failed / 436 passed** in ~27s. Full list saved
to `.rfp_work/baseline_failures.txt`. Every failure is stale-fixture drift with the same root cause:

```
Missing required columns in frame_inventory: ['flip_polarity']
```

Affected: `test_build_keyence_stitch_map.py` (6), `test_materialize_well_keyence.py` (2),
`test_product_shard_assembly.py` (1), `test_product_shard_merge_integration.py` (3).

**These are inherited, not caused by this work.** Any new failure must be diffed against that file.
Note the count is 12, not the ~15 quoted in `PLAN_zslice_stitch_resolution.md` — that doc predates
intervening fixture work, so 12 is the number to hold this change to.

### The RFP test subject exists: the pbx fluorescence collection
The plan's one blocking unknown ("does a real YX1 RFP experiment exist?") is **resolved**. The YX1
3-source collection is `20260624/25/26_pbx_flouresence_bf_pilot_plate01_t{33,52,77}hpf`.

Verified from the real per-source scope metadata under
`.collection_test_scratch/e2e_yx1/_per_source/`: all three sources carry canonical channels
**`['BF', 'RFP']`**. So RFP is already ingested and canonicalized end-to-end today; only
materialization refuses it.

`COLLECTION_STEP_BY_STEP.md` (L255) documents this same pilot as `(P, Z, C, Y, X)` with `n_c=2`,
second channel **tdTomato** — i.e. *the very file whose channel axis was masquerading as the Z stack*
(old slice returned `(2, 2304, 2304)`; correct is `(9, 2304, 2304)`). The fix and the test subject are
the same data, which makes the shape assertion a direct regression test for that bug.

Related raw dir also present: `raw_image_data/YX1/20260624_2x_td_bf_pbx_coll`.

### Channel map already covers it — no map edit needed
`scope/yx1/channel_map.py` already maps **both** `"tdtomato" → "RFP"` and `"EYES - RFP" → "RFP"`.
The plan's caveat ("raw name may be an unmapped tdTomato variant") is closed.

**Latent issue noted, not fixed:** `ScopeChannelMap.to_canonical()` is an **exact-match** dict lookup
with no case normalization, while the map key is lowercase `"tdtomato"`. An ND2 recording
`"tdTomato"` (the conventional capitalization) would raise rather than resolve. Not hit here — the pbx
metadata already shows canonical `RFP`, so its raw name matched something in the map. Out of scope;
worth a follow-up since the failure mode is a confusing hard stop on a channel that "is" mapped.

---

## Decisions taken WITHOUT user input

**[UNILATERAL] Baseline count of record is 12, not ~15.** The plan cites ~15 pre-existing failures
from `PLAN_zslice_stitch_resolution.md`. Measured reality at `3737569a` is 12. Using the measured
number as the regression gate.

**[UNILATERAL] Did not fix the channel-map case-sensitivity gap.** See above. It is a real latent
bug but outside this change's scope, and fixing it silently would widen the diff. Recorded here
instead.

---

## Decisions made WITH the user (for traceability)

These were settled in conversation before implementation; recorded so the reasoning survives.

1. **Method token is `max`**, not `max_projection` — the vocabulary and `PIPELINE_OVERVIEW.md` §C1.3
   already committed to it; stale docstrings get corrected to match.
2. **`write_index_map` is a product-plan option**, not a method-specific accident. Separates
   *projection method* from *whether to preserve which Z plane fed each pixel*.
3. **No channel×method policy in the resolver.** An earlier draft made `focus_stack` brightfield-only
   on the theory that LoG focus scoring is noise-driven on sparse fluorescence. That is an
   *unvalidated inference*; the product plan is the declared place to choose a method, so a resolver
   that overrides it on a guess is incoherent. `RFP × focus_stack` is permitted. If output looks
   wrong, that is evidence, and only then does a rule get written with a reason behind it.
   Consequence: `BRIGHTFIELD_CHANNELS` is deliberately **not** imported by the resolver.
4. **Polarity: flip the base default, don't make the policy channel-aware.** `_BASE_DEFAULT` goes to
   `flip_polarity: False` so the safe case is the default; inverting is a brightfield special case and
   must be opted into. The write policy keeps no notion of what a channel is. This is one **atomic**
   edit — flipping the default while missing either BF entry would invert brightfield and break
   snip_processing (which assumes dark background).
5. **Native resolution + uint16 for RFP max.** Area-downsampling averages punctate signal with dark
   surroundings (a 2×2 of `[[0,0],[0,100]]` → `25`), so peak intensity is not preserved; uint8 would
   compress 65,536 levels to 256. Therefore `downsample_factor: 1`, `pixel_dtype: uint16`, `png`,
   `jpeg_quality: None` (required: `_validate_policy` rejects non-null for non-jpg), **no** target
   µm/px.
6. **RFP request is per-run, never global.** A universal RFP entry plus fatal-on-missing would fail
   every BF-only experiment — individually sound decisions, jointly fatal. Base `config.yaml` stays
   BF-only; RFP goes in a dedicated runtime config.
7. **Absent requested channel = hard failure, with a diagnostic message** that separates
   "not imaged" / "imaged but unmapped" / "mapped but no frames for this well" and names the fix for
   each. Reporting the **raw** channel names is what makes those distinguishable.
8. **Keep the legacy `focus_index_map/` directory name** for focus_stack via a shim rather than
   renaming uniformly — orphaning already-materialized BF sidecars is worse than slight naming
   inconsistency.

---

## Design correction found during pre-work

**The frame_inventory validator cannot see the resolved plan — VERIFIED.**
`_validate_focus_index_map_provenance(df, *, image_root, scope_label)`
(`frame_inventory_validation_rules.py:333`) receives only the shard rows. Its own docstring is
explicit: *"Context available here is the shard rows + the .npz files (NOT the acquisition
inventory)"* — which is exactly why the inventory-aware `z_indices` check was split into the separate
`validate_focus_index_map_against_inventory`.

So the plan's original "key the L4 rule off the plan" is **unreachable** without breaking that
deliberate context isolation. Hence `write_index_map` becomes a frame_inventory **column** and the
rule stays row-local. This reverses an earlier "no new inventory column" position, which had assumed
validator access that does not exist.
