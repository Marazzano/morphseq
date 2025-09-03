# Refactor-011: SAM2 CLI Integration & Pipeline Orchestration

**Created**: 2025-09-03  
**Status**: Partial Implementation  
**Depends On**: Refactor-010 Standardize Embeddings, Refactor-010-B Complete

## **Executive Summary**

Integrate SAM2 segmentation directly into the CLI pipeline with proper orchestration between Build02 QC masks and SAM2 embryo masks. This refactor adds a `sam2` subcommand that runs the segmentation_sandbox pipeline and ensures Build03 can leverage both legacy QC masks and superior SAM2 embryo segmentation.

**Key Goals:**
- Add `sam2` subcommand to CLI for direct SAM2 pipeline execution
- Establish proper data flow between Build02 QC masks and SAM2 embryo masks
- Enable Build03 to use best-of-both-worlds mask combinations
- Provide clean Python-based SAM2 orchestration (no bash script dependencies)
- Auto-discovery of SAM2 outputs by Build03

## **Background & Problem**

**Current State:**
- SAM2 segmentation exists in `segmentation_sandbox/` but requires manual execution
- Build02 generates critical QC masks (embryo, yolk, focus, bubble, viability) needed for quality control
- Build03 has partial SAM2 support but relies on external hardcoded paths
- No integrated workflow for combining Build02 QC masks with SAM2 embryo masks
- Pipeline lacks batch processing capability for multiple experiments

**Issues:**
- Manual SAM2 execution breaks pipeline automation
- QC functionality degraded when using SAM2-only workflow
- Hardcoded paths in Build03 reduce portability
- No clear data flow documentation between segmentation approaches

## **Scope (This Refactor)**

### **In Scope**
1. **CLI SAM2 Integration**: Add `sam2` subcommand with direct Python orchestration
2. **Pipeline Orchestration**: Define clear Build01→Build02→SAM2→Build03 flow
3. **Hybrid Mask Support**: Enable Build03 to use SAM2 embryo + Build02 QC masks
4. **Path Standardization**: Remove hardcoded external paths from Build03
5. **Auto-Discovery**: Build03 automatically finds SAM2 outputs
6. **E2E Enhancement**: Update e2e workflow to include SAM2 option
7. **Validation**: Pre-flight checks for required inputs/outputs

### **Out of Scope**
- Modifications to segmentation_sandbox pipeline logic
- Changes to SAM2 model parameters or training
- Build02 algorithm improvements
- Performance optimizations of individual segmentation steps

## **Architecture & Data Flow**

### **Pipeline Sequence**
```
Build01 (Stitched Images) 
    ↓
Build02 (Legacy QC Masks) 
    ↓
SAM2 (Embryo Masks + Metadata)
    ↓  
Build03 (Hybrid: SAM2 Embryo + Build02 QC)
    ↓
Build04 → Build05 → Build06
```

### **Data Flow Specification**

#### **Build01: Image Preparation**
- **Input**: Raw microscope data
- **Output**: 
  - `built_image_data/stitched_FF_images/<exp>/<well>_t####.jpg`
  - `metadata/built_metadata_files/<exp>_metadata.csv`

#### **Build02: Complete QC Mask Generation**  
- **Input**: Stitched images
- **Models**: Runs all 5 UNet models in single command:
  - `mask_v0_0100` (embryo masks)
  - `yolk_v1_0050` (yolk masks)
  - `focus_v0_0100` (focus masks)
  - `bubble_v0_0100` (bubble masks)
  - `via_v1_0100` (viability/alive-dead masks)
- **Output**:
  - `segmentation/mask_v0_0100_predictions/<exp>/` (embryo masks)
  - `segmentation/yolk_v1_0050_predictions/<exp>/` (yolk masks)
  - `segmentation/focus_v0_0100_predictions/<exp>/` (focus masks)
  - `segmentation/bubble_v0_0100_predictions/<exp>/` (bubble masks)
  - `segmentation/via_v1_0100_predictions/<exp>/` (viability masks)
- **Purpose**: Provides complete QC mask suite for fraction_alive calculation, dead_flag, and all quality control flags

#### **SAM2: Batch Embryo Segmentation**
- **Input**: Auto-detects stitched images in `built_image_data/stitched_FF_images/`
- **Processing**: Batch mode - processes all experiments found automatically
- **Pipeline Steps**:
  1. `01_prepare_videos.py` → Video preparation for all experiments
  2. `03_gdino_detection.py` → Object detection  
  3. `04_sam2_video_processing.py` → SAM2 segmentation
  4. `05_sam2_qc_analysis.py` → Quality control annotation
  5. `06_export_masks.py` → Mask export
  6. `export_sam2_metadata_to_csv.py` → Per-experiment CSV export
- **Data Structure**:
  ```
  <data_root>/sam2_pipeline_files/
  ├── detections/gdino_detections.json
  ├── exported_masks/<exp>/masks/*.png
  ├── raw_data_organized/<exp>/
  ├── embryo_metadata/grounded_sam_segmentations.json
  └── sam2_expr_files/
      └── sam2_metadata_<exp>.csv
  ```
- **Environment**: `MORPHSEQ_SANDBOX_MASKS_DIR=<root>/sam2_pipeline_files/exported_masks`

#### **Build03: Hybrid Mask Processing**
- **Input**: 
  - Stitched images (pixel data)
  - SAM2 metadata CSV (embryo masks, positions, bboxes)
  - Build02 masks (yolk, focus, bubble, viability for QC)
- **Auto-Discovery**: Finds `sam2_pipeline_files/sam2_expr_files/sam2_metadata_<exp>.csv`
- **Logic**:
  - **Primary segmentation**: Use SAM2 embryo masks for superior accuracy
  - **QC analysis**: Use Build02 yolk/focus/bubble/viability masks for complete quality flags
  - **Fraction alive**: `compute_fraction_alive(sam2_embryo_mask, build02_via_mask)`
  - **Snip extraction**: Combine both mask types for optimal training data
- **Output**:
  - `metadata/combined_metadata_files/embryo_metadata_df01.csv`
  - `training_data/bf_embryo_snips/<exp>/*.jpg` (+ mask variants)

## **Implementation Plan**

### **Stage 1: Core SAM2 Integration (2-3 days)**

#### **1.1 Create run_sam2.py Wrapper** [❌ NOT IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/steps/run_sam2.py`
- **Function**: `run_sam2(root: str, exp: str, **kwargs)`
- **Implementation**:
  - Direct Python invocation of sandbox scripts (no bash dependency)
  - Set `MORPHSEQ_SANDBOX_MASKS_DIR` environment variable
  - Execute sandbox pipeline steps in sequence with error handling
  - Export SAM2 metadata CSV to root directory
- **Parameters**:
  - `root`: Pipeline root directory
  - `exp`: Experiment name 
  - `confidence_threshold`: GroundingDINO confidence (default: 0.45)
  - `iou_threshold`: GroundingDINO IoU threshold (default: 0.5)
  - `target_prompt`: SAM2 prompt (default: "individual embryo")
  - `workers`: Parallel workers (default: 8)

#### **1.2 Add CLI sam2 Subcommand** [❌ NOT IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/cli.py`
- **Implementation**:
  ```python
  # Add sam2 subparser
  p_sam2 = sub.add_parser("sam2", help="Run SAM2 segmentation pipeline")
  p_sam2.add_argument("--data-root", required=True)
  p_sam2.add_argument("--exp", required=True) 
  p_sam2.add_argument("--confidence-threshold", type=float, default=0.45)
  p_sam2.add_argument("--iou-threshold", type=float, default=0.5)
  p_sam2.add_argument("--target-prompt", default="individual embryo")
  p_sam2.add_argument("--workers", type=int, default=8)
  ```

#### **1.3 Validation Framework** [❌ NOT IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/validation.py`
- **Functions**:
  - `validate_stitched_images(root, exp)`: Check Build01 outputs exist
  - `validate_build02_masks(root, exp, model_name)`: Check QC masks exist
  - `validate_sam2_outputs(root, exp)`: Check SAM2 CSV and masks exist
- **Integration**: Called by each step's pre-flight checks

### **Stage 2: Build02 Enhancement & Build03 Updates (2-3 days)**

#### **2.1 Enhanced Build02 - All UNet Models** [⚠️ PARTIALLY IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/steps/run_build02.py`
- **Current Status**: Basic legacy mode exists ✅
- **Enhancement**: Run all 5 UNet models in single command:
  ```python
  models = [
      ("mask_v0_0100", 2),      # embryo masks
      ("yolk_v1_0050", 1),      # yolk masks
      ("focus_v0_0100", 1),     # focus masks
      ("bubble_v0_0100", 1),    # bubble masks
      ("via_v1_0100", 1)        # viability masks
  ]
  for model_name, n_classes in models:
      apply_unet(root, model_name, n_classes, overwrite)
  ```

#### **2.2 Remove Hardcoded Paths** [✅ COMPLETE]
- **File**: `src/build/build03A_process_images.py`  
- **Status**: ✅ Complete - Build03 uses `_load_build02_masks_for_row()` with `<root>/segmentation/<model>_*/<date>/` pattern

#### **2.3 Auto-Discovery Logic** [⚠️ PARTIALLY IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/steps/run_build03.py`
- **Current Status**: Build03 accepts `--sam2-csv` parameter ✅
- **Enhancement**:
  ```python
  def run_build03(root, exp, sam2_csv=None, **kwargs):
      if sam2_csv is None:
          # Auto-discover SAM2 CSV in organized structure
          auto_csv = Path(root) / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp}.csv" 
          if auto_csv.exists():
              sam2_csv = str(auto_csv)
              print(f"🔍 Auto-discovered SAM2 CSV: {sam2_csv}")
      # Continue with existing logic...
  ```

#### **2.4 Enhanced Hybrid Mask Loading** [✅ COMPLETE]
- **File**: `src/build/build03A_process_images.py`
- **Status**: ✅ Complete via Refactor-010-B
- **Logic**:
  - **Primary**: Load SAM2 embryo masks from CSV-specified paths ✅
  - **Secondary**: Load Build02 QC masks from all 5 model outputs ✅:
    - Yolk masks from `yolk_v1_0050_predictions/`
    - Focus masks from `focus_v0_0100_predictions/`
    - Bubble masks from `bubble_v0_0100_predictions/`
    - **Viability masks from `via_v1_0100_predictions/`**
  - **Validation**: Warn if expected QC masks missing but continue ✅
  - **Fraction alive**: `compute_fraction_alive(sam2_mask, build02_via_mask)` ✅
  - **Fallback**: Use dummy masks only if both sources unavailable ✅

### **Stage 3: E2E Orchestration (1-2 days)**

#### **3.1 Update E2E Command** [❌ NOT IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/cli.py`
- **Changes**:
  ```python
  pe2e.add_argument("--run-sam2", action="store_true", 
                    help="Include SAM2 segmentation step")
  pe2e.add_argument("--sam2-confidence", type=float, default=0.45)
  pe2e.add_argument("--sam2-workers", type=int, default=8)
  ```

#### **3.2 E2E Execution Logic** [❌ NOT IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/cli.py`
- **Flow**:
  ```python
  if not args.skip_build01:
      run_build01(...)
  if not args.skip_build02:  
      run_build02(...)  # Standard pipeline step
  if args.run_sam2:
      run_sam2(...)
  if not args.skip_build03:
      run_build03(...)  # Will auto-discover SAM2 CSV if available
  # Continue with build04, build05...
  ```

#### **3.3 Documentation Updates** [❌ NOT IMPLEMENTED]
- **File**: `src/run_morphseq_pipeline/README.md`
- **Content**:
  - Updated pipeline flow diagram
  - SAM2 subcommand usage examples  
  - Data flow and artifact locations
  - Troubleshooting guide for mask path issues

## **CLI Usage Examples**

### **Individual Steps**
```bash
# Complete pipeline with SAM2 (using test data)
conda activate mseq_data_pipeline_env

# Step 1: Build01 - Stitch images
python -m src.run_morphseq_pipeline.cli build01 --data-root morphseq_playground --exp 20250529_24hpf_ctrl_atf6 --microscope keyence

# Step 2: Build02 - Generate all 5 QC mask types
python -m src.run_morphseq_pipeline.cli build02 --data-root morphseq_playground --mode legacy
# Runs: embryo, yolk, focus, bubble, viability UNets

# Step 3: SAM2 - Batch process (auto-detects experiments)
python -m src.run_morphseq_pipeline.cli sam2 --data-root morphseq_playground
# Outputs: sam2_pipeline_files/sam2_expr_files/sam2_metadata_20250529_24hpf_ctrl_atf6.csv

# Step 4: Build03 - Hybrid masks (auto-discovers SAM2 CSV)
python -m src.run_morphseq_pipeline.cli build03 --data-root morphseq_playground --exp 20250529_24hpf_ctrl_atf6
# Uses: SAM2 embryo masks + Build02 yolk/focus/bubble/viability masks
```

### **E2E with SAM2**
```bash
conda activate mseq_data_pipeline_env

python -m src.run_morphseq_pipeline.cli e2e \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --run-sam2 \
  --train-name test_sam2_20250903

# Pipeline: Build01 → Build02(5 UNets) → SAM2(batch) → Build03(hybrid) → Build04 → Build05
```

### **Legacy E2E (no SAM2)**
```bash
conda activate mseq_data_pipeline_env

python -m src.run_morphseq_pipeline.cli e2e \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --train-name legacy_test_20250903
# Uses Build02 embryo masks for all segmentation (still runs all 5 UNets for QC)
```

## **Benefits & Impact**

### **Operational Benefits**
- **Automated SAM2**: No more manual segmentation_sandbox execution
- **Best-of-Both**: Superior SAM2 embryo masks + Build02 QC capabilities
- **Pipeline Integration**: SAM2 becomes a first-class pipeline citizen
- **Validation**: Pre-flight checks prevent runtime failures

### **Development Benefits** 
- **Clean Architecture**: Clear separation between segmentation approaches
- **Maintainable Paths**: No hardcoded external dependencies
- **Extensible**: Easy to add new segmentation approaches
- **Testable**: Validation framework enables robust testing

### **User Benefits**
- **Auto-Discovery**: Build03 finds SAM2 outputs automatically
- **Flexible Workflows**: Can use SAM2, legacy, or hybrid approaches
- **Clear Documentation**: Understand data flow and requirements
- **Error Messages**: Actionable feedback when inputs missing

## **Risk Assessment & Mitigation**

### **Technical Risks**
- **Runtime Performance**: Running both Build02 + SAM2 increases processing time
  - *Mitigation*: Parallel execution where possible, clear user expectations
- **Storage Requirements**: Additional mask files increase disk usage  
  - *Mitigation*: Document storage requirements, provide cleanup utilities
- **Path Dependencies**: Changes to Build03 mask loading could break existing workflows
  - *Mitigation*: Thorough testing, backward compatibility validation

### **Integration Risks**
- **Segmentation_sandbox Dependencies**: Python environment compatibility
  - *Mitigation*: Environment validation, clear setup documentation
- **Mask Format Compatibility**: Ensuring Build02 and SAM2 masks work together
  - *Mitigation*: Format validation, conversion utilities if needed

## **Testing Strategy**

### **Real Data Validation**
- **Test Dataset**: 26GB `20250529_24hpf_ctrl_atf6` (96 wells, Keyence)
- **Environment**: `mseq_data_pipeline_env` conda environment
- **Location**: `morphseq_playground/` for safe testing

### **Unit Tests**
- `test_run_sam2.py`: SAM2 wrapper functionality
- `test_build02_all_unets.py`: Verify all 5 UNet models run
- `test_validation.py`: Pre-flight check logic  
- `test_build03_auto_discovery.py`: SAM2 CSV discovery
- `test_viability_masks.py`: Viability mask loading and fraction_alive calculation

### **Integration Tests**
- `test_e2e_with_sam2.py`: Full pipeline with SAM2 integration
- `test_hybrid_masks.py`: Build03 with mixed mask sources including viability
- `test_path_standardization.py`: Verify no hardcoded paths
- `test_batch_processing.py`: SAM2 auto-detection of multiple experiments

### **End-to-End Tests**
- Complete pipeline with real 96-well dataset
- Validation of all QC flags (including dead_flag from viability masks)
- Performance benchmarking: Build02 (5 UNets) + SAM2 vs legacy
- Storage requirements analysis (Build02 + SAM2 outputs)

## **Updated Acceptance Criteria**

### **Functional Requirements**
- [ ] Build02 runs all 5 UNet models (embryo, yolk, focus, bubble, viability) in single command
- [ ] `sam2` CLI subcommand executes segmentation_sandbox pipeline in batch mode
- [ ] Build03 auto-discovers `sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`
- [x] ✅ Build03 loads all Build02 QC masks including viability masks (no hardcoded externals)
- [x] ✅ `fraction_alive` calculated using SAM2 embryo masks + Build02 viability masks  
- [ ] E2E pipeline with `--run-sam2` produces df01 with complete QC flags including dead_flag
- [ ] Validation functions provide clear error messages for missing inputs
- [x] ✅ All operations use `mseq_data_pipeline_env` conda environment

### **Quality Requirements**  
- [x] ✅ All QC flags functional (yolk, focus, bubble, frame, dead, no_yolk flags)
- [x] ✅ Viability masks properly integrated for accurate fraction_alive and dead_flag calculations
- [x] ✅ SAM2 embryo masks preferred over Build02 embryo masks in hybrid mode
- [ ] Build02 runs efficiently with all 5 UNet models in sequence
- [ ] SAM2 batch processing auto-detects and processes multiple experiments
- [x] ✅ No regression in existing legacy pipeline functionality  
- [ ] Documentation covers all new CLI options and data flow

### **Performance Requirements**
- [ ] SAM2 integration adds <20% overhead to e2e pipeline time
- [ ] Validation checks complete in <10 seconds
- [x] ✅ Auto-discovery logic adds <1 second to Build03 startup (Build03 accepts --sam2-csv directly)

## **Timeline & Dependencies**

**Total Estimated Time**: 5 days

- **Day 1** (Data Setup & Build01): Copy 26GB test data, validate Build01
  - Depends on: Available disk space and data transfer time
- **Day 2** (Enhanced Build02): Update to run all 5 UNet models, validate outputs
  - Depends on: UNet model availability and GPU resources
- **Day 3-4** (SAM2 Integration): Batch processing, CSV organization, Build03 hybrid masks
  - Depends on: segmentation_sandbox pipeline stability
- **Day 5** (E2E Validation): Full pipeline testing, performance analysis
  - Depends on: All previous stages completion

**Parallel Work Opportunities**:
- Documentation can be drafted during implementation
- Unit tests can be written alongside core functionality
- Validation framework can be developed independently

## **Future Enhancements**

### **Short Term (Next Refactor)**
- Performance optimization: parallel Build02/SAM2 execution
- Advanced mask fusion: weighted combination of Build02/SAM2 embryo masks
- Configuration management: YAML-based pipeline configuration

### **Long Term**
- SAM2 model fine-tuning integration
- Real-time segmentation quality monitoring  
- Automated mask quality assessment and selection

---

## **Appendix: File Changes Summary**

### **New Files**
- `src/run_morphseq_pipeline/steps/run_sam2.py`

### **Modified Files**
- `src/run_morphseq_pipeline/cli.py` - Add sam2 subcommand, enhance e2e
- `src/run_morphseq_pipeline/steps/run_build03.py` - Auto-discovery logic
- `src/build/build03A_process_images.py` - Remove hardcoded paths, hybrid masks
- `src/run_morphseq_pipeline/validation.py` - Add validation functions
- `src/run_morphseq_pipeline/README.md` - Document new workflow

### **Configuration Changes**
- Environment: `MORPHSEQ_SANDBOX_MASKS_DIR` configuration
- Paths: Standardize all mask loading to use `<root>/segmentation/...`