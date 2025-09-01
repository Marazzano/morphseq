# Refactor-010: Standardize Morphological Embedding Generation (Build06)

**Created**: 2025-09-01  
**Status**: Proposal  
**Depends On**: Refactor-009 SAM2 Pipeline Validation

## **Executive Summary**
- Goal: Centralize how we generate and publish morphological embeddings and provide a canonical pipeline artifact (df03) that merges df02 with z_mu_*.
- Decision: Adopt the legacy latent store as the source of truth for per‑experiment embeddings and add a Build06 step to standardize generation + merging.
- Model: Use legacy model `models/legacy/20241107_ds_sweep01_optimum` (AutoModel‑compatible).
- Outputs: 
  - Pipeline artifact: `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv` (df02 + z_mu_* by snip_id)
  - Per‑experiment copies for analysis users under central data root: `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/df03_{experiment}.csv` (non‑destructive)
  - Training‑run join (optional): `<root>/training_data/<train_name>/embryo_metadata_with_embeddings.csv`

## **Background & Problem**
- Today embeddings are generated via:
  - Ad‑hoc assessment scripts (`assess_image_set.py`, `assess_vae_results.py`) using `final_model/` folders
  - Batch analysis path (`analysis_utils.calculate_morph_embeddings`) writing per‑experiment `morph_latents_{exp}.csv` under `analysis/latent_embeddings/legacy/<model_name>/`
- There is no standardized pipeline step to merge embeddings into df02. Downstream users rely on notebooks or ad‑hoc merges.

## **Scope (This Refactor)**
- Add a Build06 step that:
  - Ensures latents exist for experiments present in df02 (optionally generates missing via legacy batch path)
  - Aggregates per‑experiment latents and merges them into df02 to produce a canonical df03
  - Optionally writes a per‑experiment df03 under the central data root for analysis users
  - Provides clear CLI, safety checks, and non‑destructive defaults

## **Source of Truth & Paths**
- Legacy model (for embedding):
  - `<DATA_ROOT>/models/legacy/20241107_ds_sweep01_optimum`
- Per‑experiment latent tables (reference and/or generation target):
  - `<DATA_ROOT>/analysis/latent_embeddings/legacy/20241107_ds_sweep01_optimum/morph_latents_{experiment}.csv` 
- Pipeline inputs/outputs:
  - df02 (input): `<root>/metadata/combined_metadata_files/embryo_metadata_df02.csv`
  - df03 (output): `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv`
  - Training metadata (optional join): `<root>/training_data/<train_name>/embryo_metadata_df_train.csv`
- Analysis copies (optional, non‑destructive):
  - `<DATA_ROOT>/metadata/metadata_n_embeddings/20241107_ds_sweep01_optimum/df03_{experiment}.csv`

## **Design**
- Inputs:
  - `--root`: pipeline root containing df02
  - `--data-root`: central data root (default via `MORPHSEQ_DATA_ROOT`)
  - `--model-name`: default `20241107_ds_sweep01_optimum`
  - `--experiments`: optional explicit list; otherwise inferred from df02 `experiment_date`
  - `--generate-missing-latents`: ensure missing `morph_latents_{exp}.csv` are created using legacy batch path
  - `--train-name`: optional; if present, write joined train metadata with embeddings
  - `--dry-run`: list planned actions without writing
  - `--overwrite`: allow overwriting df03 or analysis copies; default is non‑destructive
- Algorithm:
  1) Resolve df02 and experiment list
  2) For each experiment, check `<DATA_ROOT>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_{exp}.csv`
     - If missing and `--generate-missing-latents`, call `calculate_morph_embeddings(data_root, model_name, model_class="legacy", experiments=[exp])`
  3) Load all per‑experiment latents; select columns `[snip_id] + z_mu_*` (and `z_mu_b_*`/`z_mu_n_*` if present)
  4) Normalize `snip_id` stems to match Build05 naming
  5) Merge df02 with combined latents on `snip_id` → df03
  6) Write df03 to `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv` (respect `--overwrite`)
  7) If `--train-name` provided and train metadata exists, also write `<root>/training_data/<train_name>/embryo_metadata_with_embeddings.csv`
  8) If `--export-analysis-copies`, write per‑experiment df03 copies to `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/df03_{exp}.csv` (respect `--overwrite`)

## **CLI Sketch**
- Build06 entry (proposed):
  - `python -m src.run_morphseq_pipeline.cli build06 \
      --root <root> \
      --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
      --model-name 20241107_ds_sweep01_optimum \
      --generate-missing-latents \
      --train-name <optional> \
      --export-analysis-copies \
      [--experiments 20250612_30hpf_ctrl_atf6 20240626] \
      [--dry-run] [--overwrite]`

## **Safety & Validation**
- Non‑destructive by default; writing df03 or analysis copies requires `--overwrite` if files exist
- Coverage report: % of df02 rows with a matched embedding by `snip_id` (warn if < 95%)
- Hygiene: verify z columns are finite (no NaN/Inf)
- Determinism: legacy embedding path runs models in eval mode; matching to existing `morph_latents_{exp}.csv` ensures stable results
- Clear logs for: missing latents, generated latents, skipped writes, and path summaries

## **Testing Strategy**
- Unit (simulate wiring): create tmp df02 + tiny latent CSV; assert df03 columns and coverage
- Integration (reference compare): load an existing latent file, run Build06 with `--dry-run` then real run (non‑overwrite), assert outputs and schema
- Negative: missing latent for an experiment without `--generate-missing-latents` → graceful warning and partial merge

## **Notes on Current State & Improvements**
- Current generation is scattered across assessment scripts and analysis utilities; this refactor centralizes merging and clearly documents the latent source of truth (legacy store)
- Improvements:
  - Optional: cache a manifest of available `morph_latents_{exp}.csv`
  - Optional: emit a metadata sidecar (JSON) describing model, commit hash, and run parameters for reproducibility

## **Acceptance Criteria**
- Build06 produces `embryo_metadata_df03.csv` with z columns and >=95% join coverage on representative datasets
- Running with `--export-analysis-copies` creates per‑experiment df03 files under `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/` without overwrites unless `--overwrite` is set
- Clear CLI UX and logs; dry‑run mode lists planned actions

## **Example Commands**
- Minimal merge (no latent generation):
  - `python -m src.run_morphseq_pipeline.cli build06 --root <root> --data-root <DATA_ROOT> --model-name 20241107_ds_sweep01_optimum`
- Generate missing latents then merge:
  - `python -m src.run_morphseq_pipeline.cli build06 --root <root> --data-root <DATA_ROOT> --model-name 20241107_ds_sweep01_optimum --generate-missing-latents`
- Export per‑experiment df03 copies (safe):
  - `python -m src.run_morphseq_pipeline.cli build06 --root <root> --data-root <DATA_ROOT> --model-name 20241107_ds_sweep01_optimum --export-analysis-copies`

---

## **Implementation Plan: gen_embeddings.py (Pipeline‑First Service)**

Module: `src/run_morphseq_pipeline/services/gen_embeddings.py`

Purpose: Centralize embedding ingestion/generation and df02 merge under clear, testable functions used by Build06. Keeps existing assessment scripts intact.

Proposed API (clean names):
- `resolve_model_dir(data_root, model_name) -> Path`
  - Resolves `<DATA_ROOT>/models/legacy/<model_name>`; accepts direct or `final_model/` layout; validates `model_config.json`.
- `ensure_latents_for_experiments(data_root, model_name, experiments, generate_missing=False, logger=None) -> dict[exp, Path]`
  - Ensures `<DATA_ROOT>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_{exp}.csv` exists for each experiment; optionally generates via `analysis_utils.calculate_morph_embeddings`.
- `load_latents(latent_paths: dict[str, Path]) -> pd.DataFrame`
  - Reads per‑experiment latent CSVs, keeps `snip_id` + all `z_mu_*` columns, validates finite values, de‑dups by `snip_id`.
- `merge_df02_with_embeddings(root, latents_df, overwrite=False, out_name="embryo_metadata_df03.csv") -> Path`
  - Joins df02 with embeddings on `snip_id`; writes df03 alongside df02; reports join coverage.
- `merge_train_with_embeddings(root, train_name, latents_df, overwrite=False) -> Optional[Path]`
  - Joins `training_data/<train_name>/embryo_metadata_df_train.csv` with embeddings; writes `embryo_metadata_with_embeddings.csv` if train metadata exists.
- `export_df03_copies_by_experiment(df03, data_root, model_name, overwrite=False) -> None`
  - Writes per‑experiment `df03_{experiment}.csv` under `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/` (non‑destructive by default).
- `build_df03_with_embeddings(root, data_root, model_name, experiments=None, generate_missing=False, export_analysis=False, train_name=None, overwrite=False, dry_run=False) -> Path`
  - One‑shot orchestrator used by Build06; returns df03 path.

Schema policy (simplified):
- **MVP Approach**: Grab all columns starting with `z_mu_` - let downstream consumers filter as needed
- Avoids complex schema detection logic and multiple modes
- Future enhancement: add filtering options if specific downstream needs emerge

Join key:
- `snip_id` — exact filename stem of Build05 snips (no extension). We will normalize stems as needed to match Build05 naming.

Safety & UX:
- Non‑destructive writes (`--overwrite` required to replace existing files).
- `--dry-run` prints actions (experiments found, files to read/write, missing latents) and exits.
- Coverage report printed after merge; hygiene check enforces finite numeric z’s.

Build06 integration:
- Build06 invokes `build_df03_with_embeddings(...)` and includes all `z_mu_*` columns found in latent files.

Testing (unit‑first):
- Simulated latent CSVs + tiny df02: assert df03 join coverage, z schema presence, and path handling; negative case for missing latents without generation.

---

## **Analysis & Rationale (for audit)**

- Current embedding generation is split across ad‑hoc assessment scripts and analysis utilities. The pipeline lacks a canonical df02+z merge.
- We standardize around the legacy per‑experiment latent store as source of truth and expose a pipeline‑first service (`gen_embeddings.py`) so Build06 can be thin and deterministic.
- **Simplified approach**: Include all `z_mu_*` columns found in latent files, letting downstream consumers filter as needed rather than complex schema detection.
- This reduces duplication, keeps legacy scripts for research workflows, and provides a single, safe, documented path for pipeline outputs (df03).

---

## **Strategic Validation & Analysis (2025-09-01)**

### **External Validation Summary**
**Gemini Pro Strategic Analysis:** The implementation plan is "excellent and well-structured" with industry-standard best practices for scientific data pipelines. The conservative gradual rollout approach is optimal for preventing data corruption.

### **Key Strategic Confirmations** ✅
1. **Risk Mitigation Approach**: Conservative gradual rollout is the correct strategy for scientific pipelines where data integrity is paramount
2. **Parallel Validation**: Running old and new pipelines in parallel during transition is industry best-practice
3. **Non-Destructive Defaults**: `--overwrite` flags and safety checks appropriately protect existing data
4. **Source of Truth**: Using legacy latent store as canonical source is architecturally sound

### **Critical Timeline Adjustment** ⚠️
**Recommended Timeline**: **3-week phased rollout** (NOT 4-7 day compression)
- **Week 1**: Thorough validation & testing with quantitative output comparison
- **Week 2**: Backwards compatibility implementation with proper adapter pattern
- **Week 3**: Safe production deployment with monitoring

**Rationale**: The breaking `image_ids` change (list→dict) is a core data structure modification. Rushing implementation risks introducing subtle bugs that could corrupt scientific data.

### **Architectural Improvements Required**

#### **1. Adapter Pattern Implementation**
Instead of scattered if/else checks for `image_ids` format, implement a single compatibility layer:
```python
def normalize_metadata_format(metadata):
    """Adapter: Always returns new dictionary format regardless of input"""
    if isinstance(metadata.get('image_ids'), list):
        # Convert legacy list format to new dict format
        return convert_legacy_to_enhanced(metadata)
    return metadata
```

#### **2. Explicit Data Versioning**
Add version field to metadata instead of inferring from data types:
```python
metadata = {
    "metadata_version": "2.0",  # Explicit versioning
    "image_ids": {...},         # New dict format
    # ... other fields
}
```

#### **3. Configuration-Driven Rollout**
Use environment variables for pipeline version control:
```bash
METADATA_VERSION="enhanced"  # vs "legacy" or "parallel"
```

### **Enhanced Validation Requirements**

#### **Quantitative Data Integrity Validation**
- Write automated comparison scripts for old vs new pipeline outputs
- Require bit-for-bit identical results for numerical data
- Any discrepancy must be investigated and justified

#### **Automated Testing Requirements**
- Unit tests for all pipeline scripts consuming `image_ids`
- Integration tests simulating both list and dict input formats
- Edge case testing: zero images, missing fields, large experiments

#### **Formal Schema Validation**
- Implement Pydantic models or dataclasses for metadata schema
- Automatic validation of structure, data types, and constraints
- Replace manual CSV inspection with programmatic validation

---

## **morphseq_playground Integration Opportunities**

### **Proven Assets Available** ✅
The `morphseq_playground` environment contains working implementations that should be leveraged:

#### **Working Embedding Generation**
- **Location**: `morphseq_playground/training_data/sam2_test_20250831_1121/`
- **Format**: 16-dimensional z_mu embeddings (`z_mu_00` to `z_mu_15`)
- **Status**: Successfully generated and validated

#### **Proven Joining Logic**
- **Script**: `morphseq_playground/join_embeddings_metadata.py`  
- **Function**: Joins z_mu embeddings with biological metadata on `snip_id`
- **Output**: `embryo_metadata_with_embeddings.csv` (76 columns total)

#### **Complete Workflow Validation**
- **Pipeline**: Build03 → Build04 → Build05 → embed → join
- **Test Data**: 2 embryos (atf6, inj-ctrl) with distinct embedding profiles
- **Schema**: Compatible with proposed df03 format

### **Implementation Strategy Using Playground**

#### **Phase 1: Reference Implementation**
1. Use `morphseq_playground/join_embeddings_metadata.py` as template for `gen_embeddings.py`
2. Validate schema compatibility between playground output and proposed df03 format
3. Ensure `snip_id` joining logic matches existing patterns

#### **Phase 2: Integration Testing**
1. Use playground as validation environment for Build06
2. Test against proven embedding files (`embeddings.csv`, `embryo_metadata_with_embeddings.csv`)
3. Verify schema consistency and joining accuracy

#### **Phase 3: Production Scaling**
1. Extend playground patterns to larger datasets
2. Implement legacy model integration using playground's working embedding generation
3. Scale from 2-embryo proof-of-concept to full experiment processing

### **Schema Compatibility Requirements**
- Maintain compatibility with playground's z_mu_00-15 format
- Support both flat (`z_mu_XX`) and structured (`z_mu_b_XX`/`z_mu_n_XX`) schemas
- Preserve `snip_id` as primary joining key matching Build05 output

---

## **Enhanced Implementation Requirements**

### **gen_embeddings.py Service Enhancements**

#### **Additional API Functions Required**
```python
# Compatibility and validation functions
def normalize_metadata_format(metadata: dict) -> dict:
    """Adapter pattern for legacy/enhanced format compatibility"""

def validate_schema(df: pd.DataFrame, schema_version: str = "auto") -> bool:
    """Formal schema validation using Pydantic models"""

def compare_pipeline_outputs(old_path: Path, new_path: Path) -> Dict[str, Any]:
    """Quantitative comparison for validation"""

def get_metadata_version(metadata: dict) -> str:
    """Determine metadata version from structure or explicit field"""
```

#### **Configuration Management**
- Environment variable support: `MORPHSEQ_METADATA_VERSION`
- Config file integration for rollout phases
- Runtime switching between legacy/enhanced/parallel modes

#### **Enhanced Error Handling**
- Graceful degradation for missing latents
- Clear error messages for schema mismatches
- Validation checkpoints with detailed reporting

### **Testing Infrastructure Requirements**

#### **Playground-Based Integration Tests**
```python
def test_playground_compatibility():
    """Ensure Build06 works with morphseq_playground outputs"""
    
def test_schema_equivalence():
    """Verify playground and production schemas match"""
    
def test_embedding_generation_consistency():
    """Compare playground patterns with legacy model outputs"""
```

#### **Quantitative Validation Suite**
- Automated comparison of old vs new pipeline outputs
- Statistical validation of embedding consistency
- Performance benchmarking for large datasets

---

## **Open Questions (.insight)**

- .insight: Should Build06 always prefer existing legacy latents, and only generate when `--generate-missing-latents` is set (default yes)?
- .insight: Do we also want an option to embed directly from a provided `--model-dir` (bypassing the legacy store) as a future enhancement?
- .insight: **[RESOLVED]** For z schema - simplified to include all `z_mu_*` columns, avoiding complex detection modes
- .insight: Do we want a metadata sidecar (JSON) alongside df03 capturing `model_name`, data_root, timestamp, git commit, and coverage stats for reproducibility?
- .insight: Should per‑experiment df03 copies be enabled by default, or remain opt‑in via `--export-analysis-copies`?
- .insight: **[NEW]** How should the adapter pattern handle mixed environments where some experiments use legacy format and others use enhanced?
- .insight: **[NEW]** Should morphseq_playground patterns be formalized into reusable utilities, or kept as reference implementations?
- .insight: **[NEW]** What's the migration strategy for existing `morph_latents_{exp}.csv` files to the new versioned schema?

---

## **MVP Rollout Strategy: Simplified 3-Stage Framework**

### **Core Philosophy: Validate Technical Risk First**
Instead of building pipeline infrastructure around potentially broken foundations, validate that embedding generation works at all before investing in CLI, merging logic, or operational features.

### **Stage 1: Prove Embedding Generation Works (~2 days)** 
**Goal**: Can we generate embeddings at all?  
**Scope**: Standalone technical validation - zero pipeline dependencies

**Critical Question**: Does the legacy model actually produce usable embeddings?

**Implementation**:
```python
# standalone script: test_embedding_generation.py
def test_legacy_model():
    """Validate core technical risk: can we generate embeddings?"""
    model_path = Path("models/legacy/20241107_ds_sweep01_optimum")
    
    # Test with actual experiment snip images
    test_images = [
        "path/to/20250612_30hpf_ctrl_atf6_C12_e01_t0000.png",
        "path/to/20250612_30hpf_ctrl_atf6_E06_e01_t0000.png"
    ]
    
    embeddings = generate_embeddings(model_path, test_images)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Validate against playground reference
    assert embeddings.shape[1] == 16, "Should be 16-dimensional z_mu"
    assert embeddings.columns.tolist() == [f"z_mu_{i:02d}" for i in range(16)]
    
    # Save to test location for next stage
    embeddings.to_csv("test_embeddings.csv", index=False)
    return embeddings

if __name__ == "__main__":
    embeddings = test_legacy_model()
    print("✅ SUCCESS: Embedding generation works!")
```

**Success Criteria**:
- Generate embeddings from legacy model 
- Verify they're 16-dimensional (`z_mu_00` to `z_mu_15`)
- Compare format with playground reference (`morphseq_playground/training_data/sam2_test_20250831_1121/embeddings.csv`)
- Save to CSV for Stage 2 consumption

**Risk Mitigation**: If this fails, stop here - no point building pipeline infrastructure around broken embedding generation.

---

### **Stage 2: Pipeline Integration (~2 days)**
**Goal**: Build06 MVP that assumes embeddings exist  
**Scope**: Simple merge logic with fail-fast if latents missing

**Implementation**:
```python
# src/run_morphseq_pipeline/services/gen_embeddings.py (MVP)
def build06_mvp(root: Path) -> Path:
    """Build06 MVP: merge existing latents with df02 (fail if missing)"""
    
    # Load df02
    df02_path = root / "metadata/combined_metadata_files/embryo_metadata_df02.csv"
    df02 = pd.read_csv(df02_path)
    experiments = df02['experiment_date'].unique()
    
    # Load existing latents - FAIL if any missing
    latents_list = []
    for exp in experiments:
        latent_path = find_latent_path(exp)  # Look in standard locations
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latents for {exp} at {latent_path}")
        latents_list.append(pd.read_csv(latent_path))
    
    # Combine latents - keep snip_id + all z_mu_* columns
    all_latents = pd.concat(latents_list, ignore_index=True)
    z_cols = [col for col in all_latents.columns if col.startswith('z_mu_')]
    all_latents = all_latents[['snip_id'] + z_cols]  # Simplified schema
    
    # Merge with df02 on snip_id
    df03 = pd.merge(df02, all_latents, on='snip_id', how='inner')
    
    # Write df03
    df03_path = root / "metadata/combined_metadata_files/embryo_metadata_df03.csv"
    df03.to_csv(df03_path, index=False)
    
    print(f"✅ df03 created: {len(df03)} rows, {len(df03.columns)} columns")
    print(f"📊 Join coverage: {len(df03)/len(df02)*100:.1f}%")
    
    return df03_path

# CLI integration
def build06_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Pipeline root")
    args = parser.parse_args()
    
    try:
        df03_path = build06_mvp(Path(args.root))
        print(f"SUCCESS: {df03_path}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Generate latents first, then retry")
        sys.exit(1)
```

**Success Criteria**:
- df03 created from pre-existing latents
- Proper snip_id joining (>95% coverage expected)
- Clear error message when latents missing
- Compatible with playground schema format

---

### **Stage 3: Smart Detection (~2-3 days)**
**Goal**: Add `--generate-missing` flag for operational convenience  
**Scope**: Production-ready Build06 with intelligent latent management

**Implementation**:
```python
def build06_smart(root: Path, generate_missing: bool = False) -> Path:
    """Build06 with smart latent detection and optional generation"""
    
    df02_path = root / "metadata/combined_metadata_files/embryo_metadata_df02.csv"
    df02 = pd.read_csv(df02_path)
    experiments = df02['experiment_date'].unique()
    
    # Find missing latents
    existing_latents = {}
    missing_experiments = []
    
    for exp in experiments:
        latent_path = find_latent_path(exp)
        if latent_path.exists():
            existing_latents[exp] = latent_path
        else:
            missing_experiments.append(exp)
    
    # Handle missing latents
    if missing_experiments:
        if not generate_missing:
            print(f"❌ Missing latents for: {missing_experiments}")
            print("Options:")
            print("1. Run with --generate-missing to auto-generate")
            print("2. Generate latents separately using legacy tools")
            print("3. Remove experiments from df02")
            sys.exit(1)
        else:
            print(f"🔧 Generating missing latents for: {missing_experiments}")
            for exp in missing_experiments:
                latent_path = generate_latents_for_experiment(exp)
                existing_latents[exp] = latent_path
                print(f"✅ Generated: {latent_path}")
    
    # Now proceed with merge (same logic as Stage 2)
    latents_list = [pd.read_csv(path) for path in existing_latents.values()]
    all_latents = pd.concat(latents_list, ignore_index=True)
    z_cols = [col for col in all_latents.columns if col.startswith('z_mu_')]
    all_latents = all_latents[['snip_id'] + z_cols]  # Simplified schema
    
    df03 = pd.merge(df02, all_latents, on='snip_id', how='inner')
    df03_path = root / "metadata/combined_metadata_files/embryo_metadata_df03.csv"
    df03.to_csv(df03_path, index=False)
    
    print(f"✅ df03 created: {len(df03)} rows from {len(experiments)} experiments")
    return df03_path

# Enhanced CLI
def build06_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--generate-missing", action="store_true", 
                       help="Generate missing latents automatically")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done")
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN: Would check for missing latents and show plan")
        return
        
    df03_path = build06_smart(Path(args.root), args.generate_missing)
    print(f"SUCCESS: {df03_path}")
```

**Success Criteria**:
- Handles both scenarios cleanly: fail with helpful message OR generate missing latents
- Production-ready error handling and user guidance
- Maintains backward compatibility with Stage 2 behavior (fail-fast default)
- Clear operational logging

---

### **Simplified Timeline & Risk Management**

**Total Time**: 6-7 days (vs original 11-17 days)
- **Stage 1**: 2 days - Technical validation
- **Stage 2**: 2 days - Basic pipeline integration  
- **Stage 3**: 2-3 days - Production features

**Risk Distribution**:
- **Stage 1**: Validates THE critical risk - can we generate embeddings?
- **Stage 2**: Proves pipeline integration works with known-good latents
- **Stage 3**: Adds operational convenience without breaking core functionality

**Key Insight**: Each stage delivers independent value and can be tested in isolation. Stage 1 is make-or-break - if embedding generation doesn't work, we know immediately rather than discovering after building pipeline infrastructure.
