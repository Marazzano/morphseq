# Core-model status

Generated `2026-08-27T20:34:25+00:00` by `scripts/status.py`; do not edit by hand.

## Git

- Branch: `core-model-refactor`
- HEAD: `19061cbf5ea760b4e4514a81a9f2a14012dee105`
- Dirty tree: **yes** (22 entries)
- Upstream: `origin/core-model-refactor`
- Unpushed commits: **196**

```text
19061cbf Merge remote-tracking branch 'origin/main' into core-model-refactor
77f25bba Audit of the audit
7b028916 Merge branch 'feat/fp-snip-products'
09f79707 CORRECTION: the trailing-blank H label IS repaired, and my rewrite was wrong
71b4e741 Merge branch 'feat/fp-snip-products'
8b755251 Re-express main's ND2 indexing tests against the surviving API
5976f8d2 Merge pull request #31 from nlammers371/feat/fp-snip-products
48fdbd92 Merge main (SeaHub + 9 upstream commits) into feat/fp-snip-products
a37a3441 Merge remote-tracking branch 'origin/main'
a0dd9ca5 Merge main into feat/fp-snip-products
13ae07ee Resolve the five stale test modules instead of suppressing them
71ca8b69 Record stitch-map provenance; fix two tests that outlived their contract
70287f48 Fix legacy_embeddings to emit z_mu_b_*/z_mu_n_* for disentangled VAEs
6d8c695e Pause marker: record where both threads stand
e3c45a72 Temporal consistency as ground truth; gate belongs at TRACK grain
dd48c3cf Calibrate the QC gate against biological ground truth, not a derived floor
7508c764 Fix: effective_states was bit-depth dependent
8234b00b Design a channel_intensity_qc feature module and gate
771320fa FULL PLATE: the law confirmed; the null test degraded; threshold recalibrated
860395c2 Confirm the entropy-scale law at 3x the sample
3b2867ea Formal argument: raw entropy differences ARE the scale term
af6b6995 Normalization preserves information, measured against a matched-brightness null
2cb0000e 52 wells: the carriers are ONE continuous mode, not a 2x split
7b1131e5 Three-part intensity QC gate; separate pattern use from dosage use
f75556f1 ANSWER on the matched pair: normalization DOES preserve the information
7f0d3b72 CORRECTION: A09 vs G09 is the matched pair, and it reads 4-7x not 2x
03c02368 Integrated intensity is NOT the better cell-count proxy; the mean is
d49b1088 Clear a stale Snakemake lock before running
4b843582 Per-timepoint p90 intensity histograms; symlink results onto main
75218a57 26 wells: a ~2x split appears, and it is stable across timepoints
7ff249d0 Move run policy into a shared Snakemake profile
dc931eca Add --keep-going: one bad well must not stop the other 95
1cd86f30 An unmeasurable well is a warning, not a dead DAG
32b6d7e8 Estimate the background null per (well, TIMEPOINT), not per well
01c06a90 Test dosage within the tdtomato wells; fix the qsub env.yaml backup
af705fd4 Run the full plate: remove the 4-well smoke restriction
b9100197 CORRECTION: class 0 is the non-transgenic control, and the ladder is not dosage
2346c2e8 Move the dosage analysis into results/, and answer class 0 vs class 1
e190d555 ANSWER: the information does NOT survive normalization
694e5e3f Cluster dosage WITHIN each timepoint; check class stability across them
d1d8774a Remove the unwired _archive segmentation code and its tests
6b66a968 Fix 18 pre-existing test failures: hand-rolled contracts that drifted
d4b155dd Move the embryo mask to snip_geometry: one depositor, one place
5264d412 Test the pooling entrypoint's I/O behaviour
151960f8 Wire the pooling half: well-level null + corrected intensity
e0ed570d Prove the last exposure hop without a full pipeline re-run
bf8f0a42 Record the completed exposure chain and what now blocks dosage
b459ba23 Copy exposure onto the intensity measurement row
66b5047d Add the channel_intensity row contract; satisfy both repo-wide rule guards
ffa70a1c Put channel_intensity in the DAG
7a335bc4 Add the channel-intensity CLI verb
f2f7b434 Register channel_intensity in the paths registry
9a96ee15 Drop the 4-slot PE request from the test runner
199c6083 Record how B02's recovery revises the dosage read
da9016b3 Reject frame-sized "neighbours"; recovers B02's background estimate
d60ae7fd Add a compute-node test runner, and record two ways it silently lied
5a57786e Apply exposure normalization in the analysis, and say where exposure came from
28ff8306 Verify exposure capture through the real ingest entrypoint
30e78828 Fix: illumination columns are OPTIONAL, not required
9adfe318 Carry exposure/illumination onto acquisition_inventory and frame_inventory
7b1c6b64 Read per-channel exposure and illumination from ND2 text metadata
09747ad0 ROOT CAUSE: fluorescence exposure was 600ms at t0, 300ms at t1/t2
b99a5094 RESULT: dosage is not readable from this data, and why
d26ca4c6 Skip invalid masks as intensity TARGETS; record the annulus-erasure finding
7d59cf4a Wire channel_intensity to disk: the I/O half of the measurement path
20fbb387 Milestone: two-product snip fanout verified end to end
483e8b28 Merge snip inventories from the resolved shard list
bd6104ac RFP snips render on real data with dosage intact
6a96fa33 Derive the validation-scope CLI choices from the validator
e931e8a9 Validate frame inventories at product-shard grain
65a073e4 Support multi-source YX1 collection materialization
4c82d791 Join collection acquisition rows on source_ordinal, not position alone
e368b057 Rename the render rule to snip_materialization_per_well
ba5ce7cf Fan snip rendering out over well x snip_product_key
34049081 Cover the orchestration seam: argv -> parser -> entrypoint
7771d85c Make each product row self-describing about its own raster
f89c43d3 Move resolved-chain ownership to the product row; reject flip_x=True
c7162a75 Wire the geometry gate into Snakemake
4fb96f03 Add Gene14 cilia module analysis
50d97b34 Assert the geometry gate structurally, not by convention
7ffaee6e Promote the mask freeze from a row condition to a preflight contract
eb010b81 Derive snip geometry once per well, behind a gate the renderer cannot bypass
4a19fe0c Merge the pixel-center coordinate convention repair
fd00365e Render a second snip product from its own source frames
4e28693a Stamp the coordinate convention so v1 caches fail structurally
1d87415b Record what the equivalence suite cannot see, and anchor it
bfaedd02 Bind canonical.py to the one shared pixel-center correction
bf6fb232 Split the canonical downscale out of the affine so it can anti-alias
6bf6d5a9 Map canonical placement affines through pixel centers, not pixel corners
8afce454 Map placement affines through pixel centers in image_geometry
7ef5a5d9 Pin the pixel-center affine convention from first principles
e1dd2fb7 Dispatch snip photometry through the recipe registry
54f15020 Move snip pixels under {physical_embryo_id}/{snip_product_key}/
400d16bc Make snip inventory product-aware
46725cd5 Add the snip product key grammar and recipe registry
69e21a1a Refuse non-uint8 sources instead of autoscaling them to uint8
13da7562 Dispatch transform steps through one executor registry
ccab8ff6 Reject anisotropic product pixels instead of silently squaring them
f5404577 Pin portability for BOTH centering modes, not just continuous
940069be Resolve the crop center against the product's own calibration
6b5190b2 Pool a well's annuli into a background null and correct against it
1e0197b4 Extract per-embryo fluorescence evidence on the native raster
d6d0a3bb Reconstruct the canonical transform from a table row
c2fc31a2 Make the canonical snip transform mask-free and grid-explicit
8d5bb39c Merge the DRY consolidation of raster/geometry primitives
bbcbd8ab Consolidate duplicated raster/geometry primitives into image_geometry
f9cf6730 Merge the shared embryo-orientation engine
ae360565 Persist the snip transform as a per-well table
d0758824 Render snips through the transform seam (kernel + direct-render)
cbf01a4b Add the frame-independent AP rule as a fourth orientation policy
dd592436 Extract one shared embryo-orientation engine, reporting its evidence
01b373e3 Add the snip transform seam (not yet wired)
25fd1fbd Move image_geometry tests to tests/image_geometry
f46f99f6 Finish the typed-step migration: kind executes, name describes
97ede9cb Group Gene14 DACT fits by timepoint
a9c88852 Pin the resize coordinate convention before depending on it
67164a44 Add typed raster step kinds: Resize / Affine / CropPad
b6393fc8 Promote generic coordinate primitives to root image_geometry
124fa4f8 Pin snip pixel bytes; repair stale registry fixture
e0d8ff05 one source of truth for implemented methods; document max-projection ties
0e25a2cd docs: record the channel-color follow-up and the well-rim caveat
d5236157 channel colors live with the vocabulary; add a two-channel display composite
e2715cf6 docs: RFP max projection log — all steps done, real-data results recorded
66facf69 config: RFP product is per-run, never global; VERIFIED on real pbx ND2
2068be8f frame_inventory: index_map_path is generic; the ROW says if one was requested
556d1486 Add phenotype DACT contrasts and cluster submission
f68409cd Simplify Gene14 analysis workflow
55074b1e yx1: wire the max projection; channel identity is read, never assumed
52777158 resolver: gate on capability, not on channel x method taste
9e9d04ca write policy: polarity default fails SAFE; add quantitative RFP max product
9ad37f19 plan: write_index_map becomes a product-plan option (schema v2)
3737569a strangle the legacy stitch path (a 5th ND2 reader with both bugs)
8c5889ae docs: Step 7 + channel convergence are DONE (statuses were stale)
823c9e4f Add MorphSeq to sequencing embryo map
83c05521 registry: read n_sources from provenance; frame_inventory carries no merge column
55ba13dc docs: Steps 4-6 DONE — SAM2 question answered (BRIDGED), two back-half bugs fixed
641ef9ca fix: carry n_sources onto frame_inventory (unblocks the merge policy)
d876e58f registry: pass frame_inventory to the build rule; stage_predictions joins source_ordinal
895e6181 stage_predictions: key start_age_hpf on source_ordinal (fixes timelapse sources)
e7ffbf92 channel: converge the laggard `channel` column to canonical `channel_id`
b945068e docs: record the A-D vocabulary/layering sequence in the worklist
76dc3617 refactor: rename collection classification to provenance (DAG-visible)
85307b3b refactor: centralize collection discovery; ONE filesystem interpretation
07d93f2e refactor: rename collection children to plate sources
b9b6747d fix: make collection provenance AUTHORITATIVE (acquisition ingest stops re-globbing)
e7de65c0 docs: record the channel_index-is-a-recorded-fact finding + its layering
35fc3838 yx1: channel_index is READ from the minted triple, not re-derived by name
02bb5223 collection: dry-run config uses experiment_wells (the key the Snakefile reads)
4966c2f3 docs: collection worklist — Steps 1-3 DONE, plus the two cross-cutting findings
5a030f9b yx1: name-based axis handling on the PIXEL path too (one reader, no drift)
b18221f8 yx1: extractor reads axes by name; ND2 file may be named directly
8b62d270 yx1: read ND2 dimensions BY NAME; separate row grain from frame address
e0b2ef1f collection: declared age alone orders sources; duplicate age RAISES
65ef519d collection: make source ordering TOTAL (two sources can share a declared age)
60a2393f collection: cross-artifact consistency guard
8ec28134 collection: key the position mapping on the SOURCE, not the frame
26cd215f collection: per-scope plate re-key via one router, one shared rebuild
743142aa collection: both unions share ONE time-axis remapper
f9bffbf8 collection: name the source ordinal, retire the misleading age-map key
39a79b43 collection: ONE shared source->merged time_index remapper
5d17afb7 identifiers: anchor plate/age tokens by shape, not position
011c5ff6 docs: make explicit — union HOLDS per-source metadata, Step 2 re-splits by time_index
6f80e88c docs: Step 1 correction — scope metadata is PER-SOURCE (acquisition facts differ)
58896432 gene14: validate morphseq sequencing identity map
728c51e4 gene14: rt_block sheet in plate Excels; rekey crosswalk on parsed embryo_ID
f74e1fc3 docs: time_index is THE join key in Step 1 too (source_id/source_path ride along)
d2cd1865 docs: clarify bare 'channel' is a misnamed channel_id (not a distinct concept)
5d208975 docs: Step 1 (ingest scope metadata) design LOCKED — the source-key keystone
b4c2ccd6 collection: provenance records in classify artifact + Step 2 map design locked
5bb231b0 docs: step-by-step collection implementation checklist
afff5b6f docs: keystone — ONE experiment_id + ONE provenance artifact (no per-file ids)
deaa6f28 orchestration: wire collection through the DAG (dry-run plans 36 jobs end-to-end)
ff74a606 test: fix two pre-existing feature/keyence fixture drifts
957d7c18 collection: classify-once artifact + per-timepoint age in stage predictions
a1cbe63a docs: age rides in the classify artifact (no separate age product)
0cdeec92 docs: age = plate_age_by_timepoint companion (Y2), not in frame_inventory
1c43ff9d docs: rewrite spec to ONE coherent current truth (remove superseded trail)
d0eb82d6 docs: FINAL collection design — classify once, age as separate product
505d61c5 docs: CORRECT model — pool per-well at materialization, not union at acquisition
c7699c94 docs: sharpen DAG plan — collection SKIPS scope/map/apply (topology branch)
805ca5b8 identifiers: is_collection_plate_id + parse_collection_name_from_plate_id
29d0184a acquisition: collection emits position_well_mapping for the native path
90a5701b docs: record DAG-wiring status + remaining scope/acquisition-inventory seam
add098d4 acquisition: collection acquisition ingest (DAG-callable union wiring)
a7d07441 docs: flag SAM2 track_id-collision open question for FRACTURE keying
833683f6 acquisition: offset time_index_claimed atom in lockstep with time_index
40eb185f orchestration: resolve-experiment-ids CLI verb + overview one-liner
fa706d69 registry: EmbryoMergePolicy driven by per-well n_sources
203a5170 acquisition: union snapshot sources into one experiment; add n_sources
9b4e5c3d identifiers: collection grammar + resolve_experiment_ids expander
636374b1 docs: lock n_sources merge policy (supersedes source_group draft)
9e25e931 docs: experiment collection/plate model spec (MVP)
2b012fa2 Merge pull request #20 from Marazzano/time-index-rename-clean
c4dfdd65 Merge pull request #19 from Marazzano/fix/frame-inventory-fixture-writepolicy-drift
6833f6a5 test: fix frame/acquisition inventory fixture drift (required columns)
be9fc84e Rename time_int -> time_index across the live data_pipeline
```

Dirty entries:

```text
 M AGENTS.md
D  DECISIONS.md
 D docs/refactors/core-model/.DS_Store
D  docs/refactors/core-model/AGENTS.md
M  docs/refactors/core-model/DECISIONS.md
 M docs/refactors/core-model/PLAN.md
RM docs/refactors/core-model/audits/CORE_REFACTOR_PHASE0_AUDIT.md -> docs/refactors/core-model/_archive/CORE_REFACTOR_PHASE0_AUDIT.md
RM docs/refactors/core-model/audits/NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md -> docs/refactors/core-model/_archive/NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md
RM docs/refactors/core-model/reports/REVIEW_2026-08-26.md -> docs/refactors/core-model/_archive/REVIEW_2026-08-26.md
RM docs/refactors/core-model/THREAD_BRIEFS.md -> docs/refactors/core-model/_archive/THREAD_BRIEFS.md
RM docs/refactors/core-model/evidence/TRAINING_READINESS_REMAINDER.md -> docs/refactors/core-model/_archive/TRAINING_READINESS_REMAINDER.md
RM docs/refactors/core-model/evidence/OUTSTANDING_PIPELINE_ISSUES.md -> docs/refactors/core-model/pipeline/OUTSTANDING_PIPELINE_ISSUES.md
RM docs/refactors/core-model/evidence/SNIP_IMAGE_REGRESSION_STATUS.md -> docs/refactors/core-model/pipeline/SNIP_IMAGE_REGRESSION_STATUS.md
RM docs/refactors/core-model/evidence/UPSTREAM_PIPELINE_STATE.md -> docs/refactors/core-model/pipeline/UPSTREAM_PIPELINE_STATE.md
R  docs/refactors/core-model/evidence/AGENT_BRIEFS_PHASE1.md -> docs/refactors/core-model/plans/AGENT_BRIEFS_PHASE1.md
D  docs/refactors/core-model/plans/THREAD_BRIEFS.md
 M scripts/status.py
?? docs/refactors/core-model/README.md
?? docs/refactors/core-model/_archive/README.md
?? docs/refactors/core-model/plans/salvage/README.md
?? docs/refactors/core-model/plans/salvage/pipeline_contracts.py
?? docs/refactors/core-model/reports/AUDIT_regeneration_scope.md
```

| Commit of interest | Revision | Provenance | Ancestor of HEAD | Ancestor of `origin/main` |
|---|---|---|---:|---:|
| Rendering defaults restored | `37aeb639` | Added by Codex for task (ii)2; evidence: `docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md:639-653`; added 2026-08-27 | yes | yes |

## Tests

- Command: `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python -m pytest tests/core -q --color=no --disable-warnings --tb=short -p scripts.status --status-json <status-json>`
- Duration: **137.12s** (limit 600.0s)
- Result: **PASS** (pytest exit 0)
- Counts: **5 passed · 0 failed · 0 skipped · 0 xfailed · 0 xpassed · 0 not run**

## Decisions

- Ledger: `docs/refactors/core-model/DECISIONS.md`

> **UNVERIFIED DECISIONS: D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25**

**23 decisions · 0 with tests (0 passing) · unverified: [D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25]**

| Decision | Marked tests | Verification |
|---|---:|---|
| D1 | 0 | unverified |
| D2 | 0 | unverified |
| D4 | 0 | unverified |
| D5 | 0 | unverified |
| D6 | 0 | unverified |
| D7 | 0 | unverified |
| D8 | 0 | unverified |
| D9 | 0 | unverified |
| D10 | 0 | unverified |
| D11 | 0 | unverified |
| D12 | 0 | unverified |
| D13 | 0 | unverified |
| D15 | 0 | unverified |
| D16 | 0 | unverified |
| D17 | 0 | unverified |
| D18 | 0 | unverified |
| D19 | 0 | unverified |
| D20 | 0 | unverified |
| D21 | 0 | unverified |
| D22 | 0 | unverified |
| D23 | 0 | unverified |
| D24 | 0 | unverified |
| D25 | 0 | unverified |

## Environment

- Python: `3.10.16` at `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python`
- `pyarrow` imports: **no (ModuleNotFoundError: No module named 'pyarrow')**

| Package | Version |
|---|---|
| `pytest` | `9.0.2` |
| `torch` | `2.5.1` |
| `torchvision` | `0.20.1` |
| `pandas` | `2.2.3` |
| `numpy` | `1.26.4` |
| `pyarrow` | `not installed` |
| `pytorch-lightning` | `2.5.1` |
| `hydra-core` | `1.3.2` |
| `omegaconf` | `2.3.0` |

## Data

Skipped (`--with-data` was not supplied).

## Generator

- Total no-data runtime: **144.61s**
