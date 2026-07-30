# Log — per-channel max projection (RFP) on the front end

Running log for wiring `RFP × projection × max` through materialization. Newest status at the top of
each section. Companion to the approved plan; this file tracks what actually happened, including
decisions taken without user sign-off (flagged **[UNILATERAL]**).

Branch: `feat/experiment-collections` (worktree `.coll_wt`). Base commit: `3737569a`.

---

## Status

| Step | State | Commit |
|---|---|---|
| Baseline test capture | DONE — 12 failed / 436 passed | — |
| RFP test subject identified | DONE — pbx 3-source YX1 collection | — |
| 1. `write_index_map` plan field + schema v2 | **DONE** | `9ad37f19` |
| 2. Polarity base default flip + RFP write policy | **DONE** | `9e9d04ca` |
| 5. Exact-pixel uint16 test | **DONE** (landed with #2) | `9e9d04ca` |
| 3. Resolver gates opened | **DONE** | `52777158` |
| 4. Executor max branch + de-hardcode `"BF"` | **DONE** | `55074b1e` |
| 7. Channel-absent diagnostic failure | **DONE — was already built** | — |
| 6. `index_map_path` rename + `write_index_map` column | **DONE** | `2068be8f` |
| 8. Per-run RFP config + real pbx validation | **DONE** | `66facf69` |

**ALL STEPS COMPLETE.** Final suite: **12 failed / 466 passed** — the 12 are byte-identical to the
pre-change baseline throughout; every commit was regression-checked with
`comm -13 .rfp_work/baseline_failures.txt <current>`.

## VERIFIED ON REAL DATA (2026-07-29)

Materialized well A01 of the pbx collection (source `20260624_..._t33hpf.nd2`) and compared against
the ND2 itself:

```
ND2 axes: T=1 P=96 Z=9 C=2 (array order ('P','Z','C','Y','X'),
          channel_id RFP -> channel_index 1)
RFP source stack: (9, 2304, 2304)     <- NOT (2, ...), the channel-as-Z bug stays fixed
```

| check | result |
|---|---|
| written PNG == `np.max(RFP_stack, axis=0)` | **byte-exact** |
| dtype / range | uint16, max 12505 (no uint8 rescale) |
| is it really RFP? | differs from BF's max; RFP mean 815 vs BF mean 18193 |
| polarity | 34% of pixels well ABOVE median vs 0.7% below → signal-on-dark, **not inverted** |
| layout | `RFP/projection/max/{image_id}.png`, `index_map_path` NA, `write_index_map` False |
| visual | bright tdTomato embryo on a dark well, well rim visible — looks like a real max projection |

Preview (display-stretched 422–1911 for viewing only; the stored file is untouched uint16):
`.rfp_work/rfp_max_A01_preview.png`.

### Known interaction, NOT a regression
The executor is per-SOURCE, so handing it all 3 sources of a collection at once trips the existing
`Ambiguous source_nd2_path` guard. Isolating one source is correct usage and is how this was
validated. Wiring per-source fanout for collection materialization is separate work.

### Step 7 needed NO work — it already existed

`resolve_channel_index()` in `metadata_ingest/scope/shared/acquisition_channels.py` already raises
exactly the diagnostic error the plan specified: it names the absent channel, lists the channels that
ARE present, and points at both fixes ("the requested product names a channel this file does not
contain, or the scope's `channel_map.py` is missing a raw-name mapping"). It also covers the third
case (a channel mapping to multiple indices → "fix the producer"). Already tested in
`tests/.../scope/shared/test_acquisition_channels.py`, whose fixture is literally named `PBX_TRIPLE`
and is BF+RFP.

Confirmed reachable from the executor: it fired correctly (unprompted) during step-4 testing when a
BF-only inventory was handed an RFP product. Nothing to add — the plan's §4b was already satisfied by
the channel-facts layer the collections branch built.

### Notable choices inside the completed steps

- **`write_index_map` is NOT part of the product key.** The key indexes `_PRODUCT_DEFAULTS`, path
  directories, and every `.validated` artifact on disk; a provenance flag is not a different product.
- **Schema bumped to v2 and the loader *requires* the key** (`payload["write_index_map"]`, not
  `.get`). A stale v1 plan must fail loud, not load with a defaulted flag.
- **Keyence keeps a narrowed executor guard.** `max` is genuinely unwired for Keyence (it would need
  a per-tile max reduce feeding `stitch_frame_tiles`), so the guard stays — but relabelled as an
  executor limit rather than a scope rule.
- **Three resolver tests were DELETED, not adapted** (`test_non_bf_channel_rejected`,
  `test_non_focus_stack_rejected`, `test_non_bf_z_stack_rejected`). They asserted the retired BF-only
  rule; adapting them would have preserved a fiction. Replacements derive their parametrize cases from
  `SUPPORTED_CHANNELS` / `_IMPLEMENTED_PROJECTION_METHODS` / `BRIGHTFIELD_CHANNELS` per the
  import-contracts-don't-mint-in-place rule, so new channels or methods extend coverage automatically.

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

## Deferred: per-channel display colors (analysis layer, NOT this change)

User asked what color RFP should render as for snip/raw display. **Out of scope for
`data_pipeline`** — the write policy deliberately knows nothing about what a channel *means*, and
colors belong in `src/analyze/viz/styling/color_mapping_config.py` (existing home of
`GENOTYPE_SUFFIX_COLORS` / `B9D2_PHENOTYPE_COLORS`). No `CHANNEL_COLORS` exists there today.

Proposed, keyed by fluorophore color so the mapping states a physical fact rather than a naming
convention. **[UNILATERAL — provisional, user said "choose one now but I can come back to it"]**
RFP = `#E63946` (imperial red): unmistakably red, distinct from the genotype crimson `#B2182B`,
legible on light and dark.

```python
CHANNEL_COLORS = {
    'BF':  '#4D4D4D',   # neutral gray — brightfield is not a fluorophore
    'RFP': '#E63946',
    'GFP': '#2CA02C',
    'BFP': '#3B76D9',
    'CFP': '#17BECF',
    'YFP': '#E8B92E',   # darkened from pure yellow for white-bg legibility
}
```

Alternatives considered for RFP: `#D62728` (matplotlib default, muted/safe), `#FF3B30` (most
fluorescent-looking, closest to real tdTomato, can vibrate on white), `#C1121F` (print-friendly but
competes with the genotype crimson).

Open concerns when this lands: YFP's amber collides somewhat with genotype `heterozygous` `#F7B267`;
and CFP/BFP/GFP sit close in colorblind space, so plotting three fluorescent channels together needs
a re-check. Should key off `VALID_CHANNEL_NAMES` so the two vocabularies cannot drift. Ship as its own
commit, separate from the materialization work.

---

## Decisions taken WITHOUT user input

**[UNILATERAL] Baseline count of record is 12, not ~15.** The plan cites ~15 pre-existing failures
from `PLAN_zslice_stitch_resolution.md`. Measured reality at `3737569a` is 12. Using the measured
number as the regression gate.

**[UNILATERAL] Did not fix the channel-map case-sensitivity gap.** See above. It is a real latent
bug but outside this change's scope, and fixing it silently would widen the diff. Recorded here
instead. (Moot for pbx: its raw name is lowercase `tdtomato`, which matches the map key exactly.)

**[UNILATERAL] Deleted three resolver tests rather than adapting them.**
`test_non_bf_channel_rejected`, `test_non_focus_stack_rejected`, `test_non_bf_z_stack_rejected`
asserted the BF-only rule that this change removes on purpose. Adapting them would have preserved a
fiction; replacements pin the new contract and derive their cases from the vocabularies.

**[UNILATERAL] Keyence keeps `write_index_map: True` hardcoded on its projection rows.** The Keyence
backend always writes its canvas focus_index_map, so the row states the truth — but it does not yet
honor a `write_index_map: False` request. Threading the plan flag through the Keyence executor is
follow-up work; a comment marks the spot.

**[UNILATERAL] Left `BF__z_stack`'s `downsample_factor: 4` config override alone.** It predates this
work and, per the mutual-exclusion rule, beats the 6.5 µm/px product default — so BF z-slices are
still on the blind-factor path that `PLAN_zslice_stitch_resolution.md` P1 wanted to replace. Out of
scope here, but worth knowing it is still live.

**[UNILATERAL] Legacy fallback in `_row_requested_index_map`.** When the `write_index_map` column is
absent the rule falls back to the old "focus_stack carries one" behavior, so the 232 already-written
inventories keep validating. This softens the hard rename the user approved; flagged because it is a
back-compat branch that should eventually be removed once those inventories regenerate.

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
