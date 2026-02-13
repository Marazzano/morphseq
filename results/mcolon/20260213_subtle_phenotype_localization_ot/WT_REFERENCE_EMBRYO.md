# WT Reference Embryo for Phenotype Localization

**Date**: 2026-02-13

## Background

For the pilot subtle-phenotype localization analysis, we use a **single high-quality WT reference embryo** at 48 hpf. All mutant embryos are mapped onto this reference using unbalanced optimal transport (UOT).

---

## Reference Embryo Source

**From existing GPU OT development work** (`results/mcolon/20260121_uot-mvp/`):

### Candidate embryos tested in cross-embryo comparisons:
- **`20251113_A05_e01`** - Used in `run_cross_embryo_comparison.py`
- **`20251113_E04_e01`** - Used in `run_cross_embryo_comparison.py`

**Data source**: `/results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`

**Developmental stage**: ~48 hpf (target stage with tolerance ±1 hour)

---

## Selection Criteria

The reference embryo should:
1. **High-quality mask** - Clean segmentation, no artifacts
2. **Typical WT morphology** - Representative of wildtype phenotype
3. **48 hpf stage** - Mature phenotype, good WT vs mutant differentiation
4. **Well-tested** - Already validated in GPU OT development work

---

## Implementation Notes

### Step 1: Load reference embryo (Script 00)

```python
# In scripts/00_select_reference_and_load_data.py

import pandas as pd
from src.analyze.optimal_transport_morphometrics.uot_masks import load_mask_from_csv

# Load embryo data
csv_path = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
df = pd.read_csv(csv_path)

# Find frame for reference embryo at ~48 hpf
# (Use existing helper from run_cross_embryo_comparison.py)
embryo_id = "20251113_A05_e01"  # Or E04, whichever looks better
target_hpf = 48.0
tolerance = 1.0

subset = df[
    (df["embryo_id"] == embryo_id) &
    (df["predicted_stage_hpf"] >= target_hpf - tolerance) &
    (df["predicted_stage_hpf"] <= target_hpf + tolerance)
]

# Get closest frame
subset["dist"] = (subset["predicted_stage_hpf"] - target_hpf).abs()
closest = subset.loc[subset["dist"].idxmin()]
frame_idx = int(closest["frame_index"])

# Load mask
ref_mask = load_mask_from_csv(csv_path, embryo_id, frame_idx)
```

### Step 2: Visual QC

Before committing to this reference, visualize:
- Mask overlay on canonical grid
- Centerline spline fit
- Coverage (should span most of canonical grid)

If mask has issues, try the other candidate embryo.

---

## Future Scaling

**Note for future**: If we need more WT reference embryos for statistical power:

1. **Find similar WTs**:
   ```python
   # Filter for cep290_wildtype at 48 hpf
   wt_candidates = df[
       (df["genotype"] == "cep290_wildtype") &
       (df["predicted_stage_hpf"].between(47.0, 49.0))
   ]
   ```

2. **Match shape/size to original reference**:
   - Compute mask area, aspect ratio, curvature
   - Select WTs with similar morphology to original reference
   - Ensures canonical grid alignment works consistently

3. **Test OT cost**:
   - Run UOT between candidate WTs and original reference
   - Low cost = similar morphology = good match
   - This validates that new WTs are morphologically similar

---

## Genotype Labels

From `20251229_cep290_phenotype_extraction` dataset:
- **WT**: `genotype == "cep290_wildtype"`
- **Mutant**: `genotype == "cep290_homozygous"`

Ensure correct genotype filtering when loading embryos.

---

**Status**: Reference embryo candidates identified. Will select and validate in Script 00.
