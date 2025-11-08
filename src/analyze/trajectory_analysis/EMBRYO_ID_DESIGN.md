# Embryo ID Design - Option 1 Implementation

**Date:** 2025-11-07  
**Status:** Approved for Implementation  
**Design Decision:** Required string-based embryo_ids in all data structures

---

## Overview

All data structures in the trajectory_analysis package include explicit embryo ID tracking via a `embryo_ids` key. This ensures data is self-documenting, prevents index confusion, and enables cross-experiment comparisons.

---

## Rationale

### Why embryo_ids are Required (Not Optional)

1. **Prevents Silent Bugs**: Optional IDs with fallback to indices [0,1,2,...] just re-introduces the original bug.
2. **Self-Documenting**: `embryo_ids = ['cep290_wt_run1_emb_05']` is instantly clear; index 0 is ambiguous.
3. **Cross-Experiment**: Combine results from multiple runs: `['cep290_wt_run1_emb_05', 'cep290_het_run2_emb_03']`
4. **Better Errors**: If array sizes don't match embryo_ids, it's caught immediately.

### Why Strings (Not Integers)

- Integers only work within a single experiment
- Strings can encode: genotype, run number, experiment ID, embryo number
- Example: `'cep290_wt_run1_embryo_05'` vs `5` (useless)

---

## Design Specification

### All Return Dictionaries Include `embryo_ids`

```python
bootstrap_results = {
    'embryo_ids': list of str,     # NEW: Required
    'reference_labels': np.ndarray,
    'bootstrap_results': [...],
    ...
}

posterior_analysis = {
    'embryo_ids': list of str,     # NEW: Required
    'p_matrix': np.ndarray,
    'max_p': np.ndarray,
    ...
}

classification = {
    'embryo_ids': list of str,     # NEW: Required
    'category': np.ndarray,
    'cluster': np.ndarray,
    ...
}
```

### All Functions Accept `embryo_ids` Parameter

```python
# Bootstrap
run_bootstrap_hierarchical(
    D: np.ndarray,
    k: int,
    embryo_ids: List[str],  # REQUIRED
    ...
) -> Dict[str, Any]

# Posteriors (gets embryo_ids from bootstrap_results)
analyze_bootstrap_results(
    bootstrap_results_dict: Dict[str, Any]
    # embryo_ids extracted from input
) -> Dict[str, Any]

# Classification (accepts embryo_ids as optional parameter)
classify_membership_2d(
    max_p: np.ndarray,
    log_odds_gap: np.ndarray,
    modal_cluster: np.ndarray,
    embryo_ids: Optional[List[str]] = None,  # Optional here
    ...
) -> Dict[str, Any]
```

### Consistent Ordering

All arrays are ordered identically:
```python
# Array index i always corresponds to embryo_ids[i]
embryo_ids = ['emb_01', 'emb_02', 'emb_03']
max_p      = [0.95,     0.82,     0.45]     # max_p[0] is for 'emb_01'
entropy    = [0.15,     0.42,     0.88]     # entropy[0] is for 'emb_01'
```

---

## Usage Patterns

### Pattern 1: Lookup by Embryo ID

```python
# Find index for specific embryo
idx = posterior_analysis['embryo_ids'].index('emb_02')

# Access all metrics for that embryo
max_p = posterior_analysis['max_p'][idx]
entropy = posterior_analysis['entropy'][idx]
category = classification['category'][idx]
```

### Pattern 2: Filter by Category

```python
# Get all core embryos
core_mask = classification['category'] == 'core'
core_embryo_ids = [
    eid for eid, is_core in 
    zip(classification['embryo_ids'], core_mask) 
    if is_core
]
```

### Pattern 3: Iterate Through Results

```python
# Process each embryo
for i, embryo_id in enumerate(classification['embryo_ids']):
    cat = classification['category'][i]
    prob = classification['max_p'][i]
    print(f"{embryo_id}: {cat} (p={prob:.2f})")
```

### Pattern 4: Cross-Experiment Merge

```python
# Results from two experiments
results_exp1 = run_analysis(...)  # embryo_ids = ['exp1_emb_01', 'exp1_emb_02', ...]
results_exp2 = run_analysis(...)  # embryo_ids = ['exp2_emb_01', 'exp2_emb_02', ...]

# Combine results - embryo_ids encode which experiment
all_embryo_ids = results_exp1['embryo_ids'] + results_exp2['embryo_ids']
all_categories = np.concatenate([
    results_exp1['classification']['category'],
    results_exp2['classification']['category']
])

# Now 'exp1_emb_01' vs 'exp2_emb_01' are distinguished
```

---

## Implementation Checklist

### Data Structures
- [x] Add `embryo_ids: List[str]` to bootstrap_results_dict
- [x] Add `embryo_ids: List[str]` to posterior_analysis
- [x] Add `embryo_ids: List[str]` to classification

### Functions
- [x] Update `run_bootstrap_hierarchical()` to accept and return embryo_ids
- [x] Update `analyze_bootstrap_results()` to preserve embryo_ids
- [x] Update `classify_membership_2d()` to accept optional embryo_ids
- [x] Update `classify_membership_adaptive()` to accept optional embryo_ids

### Documentation
- [x] Update README.md with embryo_ids in data structures section
- [x] Add embryo_ids to example usage code
- [x] Document lookup patterns
- [x] Explain design rationale

### Testing
- [ ] Test that arrays and embryo_ids have same length
- [ ] Test lookup by embryo_id (both existing and non-existing)
- [ ] Test filtering operations preserve ID-array correspondence
- [ ] Test cross-experiment merging

---

## Backward Compatibility

Old code that doesn't use embryo_ids:
```python
# OLD (doesn't work anymore)
bootstrap_results = run_bootstrap_hierarchical(D, k=3)
# Returns dict without embryo_ids

# NEW (required)
embryo_ids = ['emb_01', 'emb_02', 'emb_03']  # MUST provide
bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids)
# Returns dict with embryo_ids key
```

**Migration path:** Any existing code will need to provide embryo_ids. This is intentional to prevent silent bugs.

---

## References

- **Design Decision:** Option 1 (Dict with embryo_ids)
- **ID Type:** Strings (not integers)
- **Required vs Optional:** Required in bootstrap/posteriors, optional in classification
- **Reason for Change:** Prevents index-based bugs, enables cross-experiment analysis
