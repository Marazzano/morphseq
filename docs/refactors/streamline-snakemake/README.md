# Streamline-Snakemake Refactor Documentation

**Organization Date:** 2025-11-06
**Status:** Clean, organized, ready for implementation

---

## 📋 Core Documentation (CURRENT - READ THESE)

### **1. processing_files_pipeline_structure_and_plan.md** (44 KB)
**THE BIBLE** - Complete architectural specification for the refactored pipeline
- Detailed directory structure
- Module descriptions for all 8 phases
- Key design decisions
- ID conventions and naming standards

### **2. snakemake_rules_data_flow.md** (36 KB)
**IMPLEMENTATION SPEC** - Detailed Snakemake rules definitions
- Rule-by-rule breakdown for Phases 1-8
- Input/output specifications for each rule
- Module mappings to code locations
- Data flow diagrams

### **3. data_ouput_strcutre.md** (9.1 KB)
**OUTPUT SPEC** - Complete data pipeline output directory structure
- All consolidated output locations
- ID hierarchy and naming conventions
- Schema validation markers
- Cross-references to other docs

### **4. DATA_INGESTION_AND_TESTING_STRATEGY.md** (20 KB)
**CURRENT & CRITICAL** - Created 2025-11-06
- Answers all 5 implementation questions
- Symlink-based data management strategy
- Replacement for poorly-named `morphseq_playground`
- Test data extraction instructions
- Snakefile creation checklist

---

## 📁 Subdirectories

### **logs/**
- Implementation progress logs (by date)
- Code review reports
- Testing records
- Status updates

### **_Archive/**
- Outdated planning documents (kept for historical reference)
- Previous versions of specs
- Deprecated reviews
- Early draft reviews
- **Do not use for current work**

### **supplementary_files_(maybenot_uptodate)/**
- Detailed QC file organization analysis
- Snip ID and tracking table design
- Analysis table design
- Status: **May not reflect final implementation**

---

## 🚀 Getting Started

### For Implementation
1. Read **processing_files_pipeline_structure_and_plan.md** (understand architecture)
2. Read **DATA_INGESTION_AND_TESTING_STRATEGY.md** (understand data setup)
3. Read **snakemake_rules_data_flow.md** (understand rules)

### For Testing
1. Follow **DATA_INGESTION_AND_TESTING_STRATEGY.md**
2. Create `data_pipeline_output/` with symlinks
3. Extract test data (10 frames per well)
4. Create Snakefile using rule patterns from **snakemake_rules_data_flow.md**
5. Run phases incrementally (1-2, 3-5, 6-8)

### For Code Review
1. Reference **processing_files_pipeline_structure_and_plan.md** for design intent
2. Use **snakemake_rules_data_flow.md** for rule validation
3. Check implementation against **data_ouput_strcutre.md** for schema compliance

---

## 📊 Document Matrix

| Document | Purpose | Last Updated | Status |
|----------|---------|--------------|--------|
| processing_files_pipeline_structure_and_plan.md | Architecture spec | 2025-11-05 | ✅ Current |
| snakemake_rules_data_flow.md | Rules specification | 2025-11-05 | ✅ Current |
| data_ouput_strcutre.md | Output directory structure | 2025-11-05 | ✅ Current |
| DATA_INGESTION_AND_TESTING_STRATEGY.md | Data setup & testing | 2025-11-06 | ✅ Current |

---

## 🗂️ Cleanup Summary

**Date:** 2025-11-06
**Action:** Archived outdated planning/execution documents to `_Archive/`

**Moved to Archive:**
- `codex_review(2025-10-13).md` - Superseded by code review in commit 0dd2857
- `data_validation_plan.md` - Schema logic now in `src/data_pipeline/schemas/`
- `snakemake_rules_data_flow.md.bak` - Backup file no longer needed
- `IMPLEMENTATION_PLAN.md` - Historical execution log (work completed)
- `PARALLEL_EXECUTION_PLAN.md` - Historical execution strategy (work completed)
- `AGREED_CODE_CHANGES.md` - Decisions already implemented in code

**Kept in Main:**
- 4 core implementation documents (see above)
- `supplementary_files_(maybenot_uptodate)/` - Detailed design docs (reference only)
- `logs/` - Implementation progress tracking

---

## ✅ Next Steps

1. **Create Snakefile** using patterns from `snakemake_rules_data_flow.md`
2. **Set up data_pipeline_output/** following `DATA_INGESTION_AND_TESTING_STRATEGY.md`
3. **Extract test data** (YX1: 20240418, Keyence: 20240509_24hpf)
4. **Validate Phase 1-2** outputs
5. **Run full pipeline** on test data
6. **Archive morphseq_playground** once new pipeline validated

---

**For questions:** Reference the core documentation matrix above. Each document has a clear purpose and intended audience.
