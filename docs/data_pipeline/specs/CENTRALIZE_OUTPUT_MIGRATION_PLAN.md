# Centralize Pipeline Output/Input Migration Plan

**Goal:** Move the real bytes of the data-pipeline output onto shared `nlammers` storage so both
mdcolon and nlammers run the pipeline into ONE canonical tree, while mdcolon keeps a repo-local
**symlink** to investigate it exactly as today. Formalize inputs (plate metadata) and pull the YX1
reference grid into the committed repo.

**Date:** 2026-07-12

---

## Final architecture

Two independent questions, answered separately. **Do not mix them.**
- **(A) PHYSICAL** — where the actual bytes sit on disk. Exactly one real place per item.
- **(B) ACCESS** — the string each person uses to reach those bytes. Pointers only; zero duplication.

### (A) PHYSICAL LOCATION — where the real bytes live

| Item | Physical location (the real bytes) | Size | Note |
|---|---|---|---|
| Raw images | `nlammers/.../morphseq/raw_image_data/{YX1,Keyence}/` | large | **Already here. NEVER moves.** nlammers' existing archive. |
| Pipeline output | `nlammers/.../morphseq/data_pipeline_output/` | 246 G | **Moves here** from mdcolon-local during migration. |
| Plate metadata (raw .xlsx) | `nlammers/.../morphseq/data_pipeline_output/inputs/plate_metadata/` | tiny | Rides in with the output tree. |
| Models (weights) | `nlammers/.../morphseq/data_pipeline_output/models/` | small | Rides in with the output tree. |
| YX1 reference CSV | **in git repo:** `<repo>/metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv` | 3 KB | **NOT on shared storage.** Versioned with code. |

Everything shared lives under the ONE nlammers path `.../morphseq/data_pipeline_output/`, sitting
**beside** the existing `.../morphseq/raw_image_data/` — same volume (`gs2`, 20 T free). The reference
CSV is the sole exception: it lives in git, not on shared storage.

The tree on shared storage:
```
nlammers/.../morphseq/
  raw_image_data/{YX1,Keyence}/{experiment}/...          ← real raw bytes (pre-existing, unmoved)
  data_pipeline_output/                                   ← 246 G moved in
    inputs/
      plate_metadata/{experiment}_well_metadata.xlsx     ← raw plate sheets (INPUT), git-tracked
      raw_image_data/{YX1,Keyence}  → ../../raw_image_data/{YX1,Keyence}   ← SYMLINK (see B, two-hop)
    models/segmentation/...
    acquisition/{experiment}/ingest_metadata/plate_metadata.csv   ← DERIVED validated plate metadata
    acquisition/{experiment}/{frame_inventory,materialized_images,well_identities}/...
    object_extraction/ feature_extraction/ quality_control/ analysis_ready/
```

### (B) ACCESS PATH — how each person reaches those exact bytes

Nobody's bytes are copied. These are pointers into (A).

| Who / what | Access path | Resolves to (A) |
|---|---|---|
| mdcolon browsing (editor/shell) | `<repo>/data_pipeline_output` (**symlink, browsing ONLY**) | real `nlammers/.../data_pipeline_output/` |
| mdcolon running pipeline | `env.yaml` roots = **real nlammers path** (NOT the symlink) | same bytes |
| nlammers running pipeline | *their* `env.yaml` roots = **real nlammers path** | same bytes |
| Pipeline reading raw images | `input_root/raw_image_data/{YX1,Keyence}` (symlink, two-hop) | real `raw_image_data/{YX1,Keyence}/` |

**DECISION (locked):** `env.yaml` roots point at the **real nlammers path** for BOTH users. Running never
depends on the repo symlink existing (nlammers has no such symlink). The repo symlink is purely
mdcolon's browsing convenience.
```yaml
# env.yaml (mdcolon AND nlammers — same strings):
paths:
  input_root:  /net/.../nlammers/.../morphseq/data_pipeline_output/inputs
  output_root: /net/.../nlammers/.../morphseq/data_pipeline_output
  models_root: /net/.../nlammers/.../morphseq/data_pipeline_output/models
```

**The raw_image_data "two-hop"** (the caught inconsistency + its fix):
- `inputs/` is NOT self-contained: raw images physically live at `.../morphseq/raw_image_data/{YX1,Keyence}`
  (A, row 1), which is a SIBLING of `data_pipeline_output/`, not inside it.
- The pipeline reaches them at `input_root/raw_image_data/{YX1,Keyence}` — bridged by **symlinks**. This
  bridge is the intended, simplest design; keep it.
- **FIX (decision):** today the bridge is an ABSOLUTE, cross-user link
  (`inputs/raw_image_data/YX1 → /net/.../nlammers/.../raw_image_data/YX1`). After migration BOTH ends live
  under `.../morphseq/`, so re-point it **RELATIVE**: `inputs/raw_image_data/YX1 → ../../raw_image_data/YX1`.
  Relative survives the whole `morphseq/` tree moving/renaming and stops hardcoding a user's home inside a
  now-shared tree.
- `rsync -a` (NOT `-L`) preserves links as links (never duplicates raw bytes); the relative re-point is a
  separate explicit step AFTER the copy.

### Committed in git (travels with code, not on shared storage)
```
<repo>/metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv   ← YX1 grid reference (fixed hardware fact)
<repo>/src/... , config.yaml (science) , env.example.yaml (template)
```
env.yaml itself is GITIGNORED (per-machine environment).

### The funnel (how paths resolve — verified in code)
`env.yaml` supplies exactly **3 roots**; the Snakefile binds them and the entire layout beneath is a
**code contract**, not config:
```
env.yaml paths.output_root  →  Snakefile:127  DATA_ROOT
env.yaml paths.input_root   →  Snakefile:128  INPUTS_DIR
env.yaml paths.models_root  →  Snakefile:129  MODELS_DIR
  INPUTS_DIR/plate_metadata/{experiment}_well_metadata.xlsx     (Snakefile:169; rule ingest_plate_metadata:378)
  INPUTS_DIR/raw_image_data/{MICROSCOPE}/{experiment}           (Snakefile:170,416)
  DATA_ROOT/{acquisition,object_extraction,feature_extraction,quality_control,analysis_ready}  (173–177)
```
**⇒ Moving data needs ZERO code changes.** Only bytes move, `env.yaml` repoints (gitignored), and one
symlink is created. Both users write to the same tree; layout stays identical.

---

## Key findings (verified, drive the decisions)

1. **Decoupling already exists** — the earlier env.yaml refactor made roots environment and layout code.
   Migration = move bytes + repoint 3 lines + symlink.
2. **Local output = 246 G** real dir; **nlammers already has a partial `pipeline_output/`** (4.3 G,
   4 hotchem experiments mdcolon lacks + full 96-well versions of the 2 overlaps). Local unique:
   `20250612_...`, `20250912`. → **Combine = NLAMMERS BASE + local's non-overlapping uniques** (NOT
   "local wins": local's copies of the 2 overlap experiments are PARTIAL — 87/74 wells vs nlammers' 96).
3. **`experiment_metadata/` is a DEAD directory** — no code writes it (zero refs), mtime Jun 10, last
   commit "Phase 1". Superseded by `acquisition/{exp}/ingest_metadata/`. → **Retire, do not migrate.**
4. **Two "plate metadata" things are NOT a collision:** `inputs/plate_metadata/*.xlsx` = raw INPUT you
   author; `acquisition/{exp}/ingest_metadata/plate_metadata.csv` = derived OUTPUT. Different roots,
   different roles, both correctly named. No rename needed.
5. **YX1 reference CSV** (`morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv`, 3 KB,
   untracked) is a fixed hardware fact, not per-run data → **commit into repo** at
   `metadata/reference/yx1/`, drop the `YX1_` filename prefix (path says it).
6. **`inputs/raw_image_data/{YX1,Keyence}` are already symlinks** into nlammers raw_image_data. Raw
   input is NOT part of the 246 G. Use `rsync -a` (never `-L`) to keep them as links.
7. **19 T free** on the volume; 246 G copy is comfortable.

---

## Decisions (locked)
| Question | Decision |
|---|---|
| Reconcile local 246 G vs nlammers 4.3 G | **nlammers base + local's non-overlapping uniques** (nlammers has complete 96-well overlaps; local's are partial) |
| Target dir name at nlammers | **`data_pipeline_output/`** |
| YX1 reference CSV home | **repo `metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv`** (committed) |
| Plate metadata `.xlsx` | **Keep git-tracked** AND on shared `inputs/plate_metadata/` |
| `experiment_metadata/` dead dir | **Retire — don't migrate** |

---

## Steps

### 0. Preflight
- [ ] No Snakemake run in flight — check `data_pipeline_output/_snakemake_runtime/` for a lock.
- [ ] `git status` clean (it is). Record `du -sh data_pipeline_output` (≈246 G) for post-move verify.

### 1. Commit the YX1 reference into the repo
- [ ] `mkdir -p metadata/reference/yx1`
- [ ] `cp morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv`
- [ ] `git add metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv`
- [ ] Repoint `config.yaml` `scope_metadata.yx1.ref_xy_csv` → repo-relative `metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv`
      (confirm how tasks.py resolves relative vs abs; make absolute-from-repo-root if needed).
- [ ] Update `generate_xy_reference.py` `OUTPUT_PATH` (currently writes into playground) to the new repo path.

### 2. REVISED (2026-07-12): SKELETON + REGENERATE — do NOT copy the 246 G output
**Pivot:** user will regenerate all pipeline OUTPUT into the central location, so copying 244 G of
disposable `acquisition/` output is wasted work. The add-layer rsync was cut off at ~49 G and the
partial target was WIPED. The centralized tree now starts as a SKELETON of only the non-regenerable
inputs; the pipeline repopulates everything else.

**DONE (verified):**
- Target wiped clean, recreated.
- `inputs/models/` (443 M, 5 seg model dirs) copied — checksum OK.
- `inputs/plate_metadata/` = **9 sheets**: local's 3 (20240418, 20250612_ctrl_atf6, 20250912) + **nlammers'
  6 hotchem** (24/30/36hpf ×2). The 6 hotchem sheets were MISSING from local inputs (hotchem is Keyence,
  their sheets live at `nlammers/.../morphseq/plate_metadata/`). All 6 checksummed OK. **This was the
  critical catch — regenerating hotchem would fail on missing plate metadata without them.**
- `inputs/raw_image_data/{YX1,Keyence}` symlinks preserved (rsync -a, `link: 2`).

**NOT copied (regenerate):** all `acquisition/` and downstream output, incl. nlammers' base hotchem.

**Superseded original plan (kept for reference): NLAMMERS BASE + local's non-overlapping uniques**
**NOT "local wins."** Per-experiment evidence (verified) shows nlammers has the COMPLETE plates for the
2 overlapping experiments; local's copies are PARTIAL. So nlammers wins the overlap; local contributes
only its unique experiments.

| Experiment | LOCAL wells | NLAMMERS wells | Winner |
|---|---|---|---|
| 20260702_hotchem_36hpf_plate01 | 87 (partial) | **96 (full)** | nlammers |
| 20260702_hotchem_36hpf_plate02 | 74 (partial) | **96 (full)** | nlammers |
| 20260702_hotchem_{24,30}hpf_plate{01,02} (×4) | — | present | nlammers (unique) |
| 20250612_24hpf_ctrl_atf6, 20250912 | present | — | local (unique) |

Local also uniquely has: `inputs/`, `models/`, `scoped_runs/` (wells-B01 smoke). Neither side has any
stage beyond `acquisition/` yet. Local's newer mtime on the overlaps = a partial re-run, NOT more complete.

- [ ] `mkdir -p /net/.../nlammers/.../morphseq/data_pipeline_output`   (TARGET; does not exist yet)
- [ ] **Base layer (nlammers, incl. the complete 96-well overlaps):**
      `rsync -a --info=progress2 nlammers/.../pipeline_output/ TARGET/`
- [ ] **Add layer (local uniques ONLY — exclude the 2 overlaps + dead dir):**
      ```
      rsync -a --info=progress2 \
        --exclude='acquisition/20260702_hotchem_36hpf_plate01' \
        --exclude='acquisition/20260702_hotchem_36hpf_plate02' \
        --exclude='experiment_metadata' \
        <repo>/data_pipeline_output/ TARGET/
      ```
      - `-a` keeps symlinks as symlinks (raw_image_data — do NOT `-L`)
      - the two `--exclude` lines are the SAFETY: they stop local's partial 87/74-well plates from
        overwriting nlammers' complete 96-well plates. Without them → Frankenstein experiment.
      - carries local's `20250612_...`, `20250912`, `inputs/`, `models/`, `scoped_runs/` — zero collisions.
- [ ] Verify post-combine: overlap experiments show **96** wells; `ls acquisition/` shows all 8 experiments.

### 4. Re-point the raw_image_data bridge symlinks (absolute→relative)
The raw images stay physically at `.../morphseq/raw_image_data/` (never moved). After the output tree
lands under the same `morphseq/`, convert the bridge from absolute cross-user to relative sibling:
- [ ] In `nlammers/.../data_pipeline_output/inputs/raw_image_data/`:
      `ln -sfn ../../raw_image_data/YX1 YX1` and `ln -sfn ../../raw_image_data/Keyence Keyence`
- [ ] Verify each resolves: `readlink -f inputs/raw_image_data/YX1` → the real archive; `ls` shows experiments.

### 5. Swap repo dir for a browsing-only symlink
- [ ] `mv data_pipeline_output data_pipeline_output.PREMIGRATE`  (keep, don't delete yet)
- [ ] `ln -s /net/.../nlammers/.../morphseq/data_pipeline_output <repo>/data_pipeline_output`
- [ ] Verify: `ls data_pipeline_output/acquisition` shows the full merged experiment set through the link.

### 6. Repoint env.yaml (gitignored — free to edit)
```yaml
paths:
  input_root:  /net/.../nlammers/.../morphseq/data_pipeline_output/inputs
  output_root: /net/.../nlammers/.../morphseq/data_pipeline_output
  models_root: /net/.../nlammers/.../morphseq/data_pipeline_output/models
```
- [ ] Update `env.example.yaml` prose to describe the centralized shared layout (committed template).
- [ ] Coordinate: nlammers sets THEIR env.yaml to the same 3 paths so both write one tree.

### 7. Git bookkeeping
- [ ] `git rm --cached data_pipeline_output/experiment_metadata/20250912/plate_metadata.csv \
        data_pipeline_output/experiment_metadata/20250912/series_number_map.csv`  (dead dir, untrack)
- [ ] Plate `.xlsx` under `inputs/plate_metadata/`: **keep tracked** (user choice). Note: `.gitignore`
      has `data_pipeline_output/` — since that path is now a symlink, confirm git still tracks the xlsx
      through it, OR (cleaner) track the xlsx from a committed repo location and symlink/copy into inputs.
      **Decide during execution** — flagged as the one git subtlety introduced by the symlink.
- [ ] The two `inputs/raw_image_data/{YX1,Keyence}` symlinks were tracked — re-evaluate under new layout.

### 8. Cutover verification (do NOT delete `.PREMIGRATE` until green)
- [ ] Orchestrator config load resolves the 3 roots to the nlammers path; finds inputs + models + ref CSV.
- [ ] Cheap smoke (front-half config, one well) end-to-end; artifacts land under nlammers tree, visible
      through the repo symlink.
- [ ] YX1 `map_positions_to_wells` finds the committed reference CSV.

### 9. Cleanup (irreversible — separate confirmation each)
- [ ] After clean smoke + a real run: `rm -rf data_pipeline_output.PREMIGRATE` (reclaim 246 G, incl. dead
      experiment_metadata).
- [ ] Once fully subsumed, retire nlammers' old `pipeline_output/`.
- [ ] Optionally remove the loose playground reference CSV once the committed one is confirmed in use.

---

## Risks / open items
- **rsync `-a` not `-L`** — else nested raw_image_data symlinks deref into duplicated raw images.
- **Git-through-symlink (step 6)** — the one genuinely new wrinkle: `data_pipeline_output/` becomes a
  symlink but you still want plate `.xlsx` tracked. Cleanest resolution is likely to keep the *authored
  source* of the xlsx in a committed repo path and have `inputs/plate_metadata/` reference it, rather
  than tracking files through the symlink. Resolve during execution.
- **Group perms** — nlammers tree is `trapnelllab` setgid; new files inherit group. Confirm umask lets
  labmates read.
- **Two users, one tree** — both env.yaml files must point at the same 3 paths (coordinate with nlammers).
- **Relative vs absolute ref path** — confirm `tasks.py` resolves `ref_xy_csv` correctly when made
  repo-relative (step 1); use absolute-from-repo-root if relative isn't supported.
