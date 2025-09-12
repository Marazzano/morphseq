# Tech Debt: Microscope Metadata Missing

**Issue**: Microscope field is empty in SAM2 CSV and experiment metadata.

**Current State**: 
- SAM2 CSV has `microscope` column but all values are empty
- Experiment metadata JSON lacks microscope information
- Build03 correctly preserves empty values from upstream

**Root Cause**: Microscope info not extracted during SAM2 pipeline generation.

**Impact**: Low priority - doesn't affect QC or morphological analysis.

**Future Fix**: Propagate microscope metadata from ND2 files or experimental setup into SAM2 CSV generation phase.
