# Refactor-008: SAM2-Legacy Pipeline Integration Validation & Documentation

**Created**: 2025-08-30  
**Status**: PLANNING  
**Previous**: [Refactor-007 SAM2 MVP Integration](./refactor-007-mvp-sam2-embryo-mask-integration.md)

## 📋 **EXECUTIVE SUMMARY**

**Objective**: Validate and document the complete SAM2-Legacy pipeline integration, create minimal testing framework, and provide production deployment guidance.

**Context**: Refactor-007 successfully implemented SAM2 embryo mask integration with legacy yolk masks and created a working runner script. However, the execution models and integration points remain opaque, and the system needs validation with minimal test datasets before full production deployment.

**Scope**: Documentation, validation testing with 5-10 sample subsets, and production readiness assessment.

---

## 🎯 **CURRENT STATUS ANALYSIS**

### **✅ Refactor-007 Achievements (COMPLETE)**
- ✅ **Core SAM2 Integration**: Embryo masks loading from `segmentation_sandbox/data/exported_masks/`
- ✅ **Yolk Mask Enhancement**: Legacy masks from `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/segmentation/yolk_v1_0050_predictions/` with automatic dimension matching
- ✅ **Dual Image Path System**: Prefers high-quality stitched images over JPEG copies
- ✅ **Production Runner**: `results/2024/20250830/run_build03_sam2.py` processes complete 92-sample dataset
- ✅ **Legacy Compatibility Bridge**: Metadata output compatible with Build04/05 pipeline

### **❌ Current Gaps (TO BE ADDRESSED)**
- ❌ **Execution Models**: Unclear how normal vs SAM2-enhanced pipelines are executed
- ❌ **Integration Timing**: Opaque when/how SAM2 pipeline runs relative to legacy steps
- ❌ **Subset Testing**: No framework for testing with minimal samples (currently experiment-at-a-time only)
- ❌ **Validation**: No systematic testing of Build03A → Build04 → Build05 integration
- ❌ **Documentation**: Missing production deployment guidance and troubleshooting

---

## 📚 **PIPELINE EXECUTION MODELS ANALYSIS**

### **1. Legacy Build Pipeline Architecture**

**Normal Execution Flow**:
```
Build01A/01B → Build02B → Build03A → Build04 → Build05
(Raw Images) → (Segmentation) → (Snip Export) → (QC) → (Training Data)
```

**Execution Pattern**:
- **Scripts**: `src/build/buildXXX.py` (core functionality)
- **Runners**: `results/YYYY/MMDD/run_buildXX.py` (experiment-specific execution)
- **Data Processing**: Experiment-at-a-time (e.g., `20250612_30hpf_ctrl_atf6`)
- **Dependencies**: Sequential - each step depends on previous outputs

**Example Legacy Execution**:
```bash
# From results/2024/20241015/ pattern
python run_build03.py  # Extract embryo snips
python run_build04.py  # QC + stage inference  
python run_build05.py  # Make training snips
```

**Key Pipeline Steps**:
- **Build01A**: Keyence raw → stitched FF images + metadata
- **Build01B**: YX1 ND2 → stitched FF images + metadata
- **Build02B**: Trained UNet/FPN models → segmentation predictions
- **Build03A**: Masks + metadata → cropped embryo snips + QC
- **Build04**: Rule-based QC + developmental stage inference
- **Build05**: Curated snips → training folder structure

### **2. Segmentation Sandbox Pipeline Architecture**

**SAM2 Pipeline Flow**:
```bash
# Orchestrated by segmentation_sandbox/scripts/pipelines/run_pipeline.sh
01_prepare_videos.py     # Raw stitched → organized videos + metadata
03_gdino_detection.py    # GroundingDINO object detection → bounding boxes
04_sam2_video_processing.py  # SAM2 segmentation → masks + tracking
05_sam2_qc_analysis.py   # Quality control analysis → QC flags
06_export_masks.py       # Export integer-labeled masks → PNG files
```

**Integration Timing**: 
- **Prerequisite**: `built_image_data/stitched_FF_images/` must exist (from Build01A/01B)
- **Parallel Execution**: SAM2 pipeline runs independently after stitched images available
- **Output Integration**: Generates `sam2_metadata_*.csv` for SAM2-enhanced Build03A

**Key Outputs for Integration**:
- **Masks**: `segmentation_sandbox/data/exported_masks/{experiment}/masks/*.png` (integer-labeled)
- **Metadata**: Root directory `sam2_metadata_{experiment}.csv` (39-column format)
- **QC Data**: Confidence scores, tracking quality, temporal consistency

### **3. SAM2-Enhanced Integration Model**

**Hybrid Execution Flow**:
```
[Legacy] Build01A/01B → built_image_data/stitched_FF_images/
    ↓
[SAM2 Sandbox] run_pipeline.sh → sam2_metadata_*.csv + exported_masks/
    ↓  
[SAM2-Enhanced] run_build03_sam2.py → training_data/ + embryo_metadata/
    ↓
[Legacy] Build04 → Build05 (unchanged)
```

**Technical Integration Points**:
1. **Mask Loading**: `resolve_sandbox_embryo_mask()` loads integer-labeled masks
2. **Yolk Integration**: Legacy yolk masks resized to match SAM2 dimensions
3. **Metadata Bridge**: SAM2 CSV → legacy DataFrame format via `segment_wells_sam2_csv()`
4. **Compatibility Layer**: Output metadata format matches Build04 expectations

---

## 🧪 **VALIDATION REQUIREMENTS ANALYSIS**

### **Current Testing Limitations**

**Experiment-Level Processing Only**:
- Current runner scripts process entire experiments (90+ embryos)
- SAM2 snip extraction: ~20 seconds per embryo = 30+ minutes for full dataset
- No framework for subset testing during development/validation

**Missing Validation Points**:
1. **Format Compatibility**: Does SAM2-generated metadata work with Build04/05?
2. **Quality Validation**: Are SAM2 embryo orientations better than legacy?
3. **Performance Benchmarking**: Resource usage and processing times?
4. **Error Handling**: What happens with missing masks, corrupted data?

### **Proposed Testing Framework**

**Subset Testing Infrastructure**:
```python
# Add --max-samples parameter to all build scripts
def compile_embryo_stats(root, tracked_df, max_samples=None):
    if max_samples:
        tracked_df = tracked_df.head(max_samples)
    # ... rest of function
```

**Integration Test Suite**:
1. **Test 1: SAM2 → Build03A** (5 samples)
   - Input: `sam2_metadata_20250612_30hpf_ctrl_atf6.csv` 
   - Process: First 5 rows only
   - Output: `training_data/bf_embryo_snips/` + metadata
   - Validation: 5 snip files created, metadata format correct

2. **Test 2: Build03A → Build04** (compatibility)
   - Input: SAM2-generated `embryo_metadata_df01.csv`
   - Process: QC analysis and stage inference
   - Output: `embryo_metadata_df02.csv`
   - Validation: No format errors, QC flags applied correctly

3. **Test 3: Build04 → Build05** (training data)
   - Input: QC'd metadata from SAM2 pipeline
   - Process: Training snip export
   - Output: `training_data/{train_name}/`
   - Validation: Folder structure correct, images accessible

---

## 📋 **DETAILED IMPLEMENTATION PLAN**

### **Phase 1: Documentation & Analysis (60 minutes)**

**1.1 Pipeline Execution Documentation**
- [ ] Document normal execution model with runner script patterns
- [ ] Explain SAM2 integration timing and dependencies  
- [ ] Create execution flow diagrams for both pipelines
- [ ] Document data format requirements and compatibility points

**1.2 Integration Point Analysis**
- [ ] Document `resolve_sandbox_embryo_mask()` function and path resolution
- [ ] Explain yolk mask dimension matching implementation
- [ ] Document `segment_wells_sam2_csv()` metadata transformation
- [ ] Analyze Build04 input requirements and compatibility bridge

### **Phase 2: Testing Framework Development (45 minutes)**

**2.1 Subset Processing Infrastructure**
- [ ] Add `--max-samples` parameter to `compile_embryo_stats()`
- [ ] Add `--max-samples` parameter to `extract_embryo_snips()`
- [ ] Create `src/build/test_utils.py` with subset utilities
- [ ] Update runner scripts to support subset testing

**2.2 Integration Test Scripts**
- [ ] Create `results/2024/20250830/run_sam2_integration_test.py`
- [ ] Implement 5-sample SAM2 → Build03A test
- [ ] Implement Build03A → Build04 compatibility test
- [ ] Implement Build04 → Build05 training data test

### **Phase 3: Validation Testing (30 minutes)**

**3.1 Execute Integration Tests**
- [ ] Run 5-sample SAM2-enhanced Build03A
- [ ] Validate metadata output format and content
- [ ] Test Build04 processing of SAM2 metadata
- [ ] Test Build05 training snip generation

**3.2 Performance Analysis**
- [ ] Benchmark processing times for each stage
- [ ] Document resource usage (GPU memory, storage)
- [ ] Compare SAM2-enhanced vs legacy processing speed
- [ ] Identify bottlenecks and optimization opportunities

### **Phase 4: Production Documentation (15 minutes)**

**4.1 Deployment Guide**
- [ ] Document complete pipeline execution commands
- [ ] Create troubleshooting guide for common issues
- [ ] Document scaling considerations for different experiment sizes
- [ ] Provide rollback procedures for legacy pipeline

**4.2 Future Integration Planning**
- [ ] Identify additional experiments ready for SAM2 processing
- [ ] Document known limitations and workarounds
- [ ] Plan next-generation improvements and optimizations

---

## 🎯 **SUCCESS CRITERIA & VALIDATION METRICS**

### **Documentation Quality**
- ✅ Complete execution model documentation for both pipelines
- ✅ Clear integration timing and dependency mapping
- ✅ Comprehensive troubleshooting and deployment guide

### **Testing Framework**
- ✅ Working subset processing (5-10 samples) for all build stages
- ✅ Automated integration test suite covering SAM2 → Build03A → Build04 → Build05
- ✅ Performance benchmarks and resource documentation

### **Integration Validation**
- ✅ Successful 5-sample end-to-end pipeline execution
- ✅ Format compatibility confirmed between all pipeline stages
- ✅ Quality assessment: SAM2 orientation vs legacy comparison

### **Production Readiness**
- ✅ Clear deployment procedures for new experiments
- ✅ Resource requirements and scaling guidance documented
- ✅ Rollback procedures and error recovery documented

---

## ⚠️ **RISK ASSESSMENT & MITIGATION**

### **Technical Risks**

**Format Incompatibilities**
- **Risk**: SAM2 metadata format may not match Build04/05 expectations
- **Mitigation**: Validate with small subsets before full processing
- **Fallback**: Implement format conversion utilities if needed

**Performance Bottlenecks**
- **Risk**: SAM2 pipeline may be too slow for large experiments
- **Mitigation**: Benchmark with subset testing first
- **Optimization**: Identify parallelization opportunities

**Data Dependencies**
- **Risk**: Missing yolk masks or SAM2 outputs could break pipeline
- **Mitigation**: Implement robust error handling and fallback mechanisms
- **Validation**: Test with incomplete datasets

### **Process Risks**

**Scope Creep**
- **Risk**: Validation could expand into new feature development
- **Mitigation**: Strict focus on testing existing integration only
- **Timeline**: Limit to 2.5 hours total implementation time

**Integration Complexity**
- **Risk**: Multiple pipeline integration points could introduce bugs
- **Mitigation**: Systematic testing of each integration point separately
- **Documentation**: Clear separation of SAM2 vs legacy components

---

## 🚀 **EXPECTED OUTCOMES**

### **Immediate Deliverables**
1. **`refactor-008-sam2-pipeline-validation.md`** - This comprehensive documentation
2. **`results/2024/20250830/run_sam2_integration_test.py`** - Minimal testing framework
3. **`src/build/test_utils.py`** - Subset processing utilities
4. **Validation Report** - Integration test results and performance benchmarks

### **Long-term Impact**
- **Production Readiness**: Clear deployment path for additional experiments
- **Quality Assurance**: Validated integration with performance metrics
- **Development Efficiency**: Subset testing framework for future modifications
- **Documentation Standard**: Template for future pipeline integration projects

### **Next Steps After Refactor-008**
- **Experiment Scaling**: Process additional experiments with validated pipeline
- **Performance Optimization**: Address identified bottlenecks
- **Feature Enhancement**: Add advanced QC metrics and analysis capabilities
- **Automation**: Implement automated pipeline orchestration

---

## 📝 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Documentation** ⏱️ 60min
- [ ] Pipeline execution models documented
- [ ] Integration timing and dependencies mapped
- [ ] Data format compatibility analyzed
- [ ] Troubleshooting scenarios identified

### **Phase 2: Testing Framework** ⏱️ 45min  
- [ ] Subset processing infrastructure implemented
- [ ] Integration test scripts created
- [ ] Test data prepared (5-10 sample subset)
- [ ] Automated validation checks implemented

### **Phase 3: Validation** ⏱️ 30min
- [ ] Integration tests executed successfully
- [ ] Performance benchmarks collected
- [ ] Compatibility issues identified and resolved
- [ ] Quality comparisons documented

### **Phase 4: Production Guide** ⏱️ 15min
- [ ] Deployment procedures documented
- [ ] Resource requirements specified
- [ ] Scaling considerations documented
- [ ] Future roadmap outlined

**Total Estimated Time**: 2.5 hours
**Priority**: High (blocks additional experiment processing)
**Dependencies**: Refactor-007 completion, access to test datasets

---

*This document will be updated as implementation progresses and validation results become available.*