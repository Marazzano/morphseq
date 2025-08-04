# Entity ID Parsing Conventions

## Overview
Entity IDs in the pipeline follow a hierarchical structure. The key principle is to **parse backwards** from the end of the ID string because experiment IDs can contain arbitrary underscores and be complex.

## ID Structure Hierarchy

```
experiment_id
    └── video_id = {experiment_id}_{WELL}
            ├── image_id = {video_id}_{FRAME}
            └── embryo_id = {video_id}_e{NN}
                    └── snip_id = {embryo_id}_{FRAME} or {embryo_id}_s{FRAME}
```

## Parsing Rules (MUST Parse Backwards!)

### 1. video_id
- **Format**: `{experiment_id}_{WELL}`
- **WELL Pattern**: `[A-H][0-9]{2}` (always at the END)
- **Examples**:
  - `20240411_A01` → experiment_id: `20240411`, well: `A01`
  - `20250529_36hpf_ctrl_atf6_A01` → experiment_id: `20250529_36hpf_ctrl_atf6`, well: `A01`
  - `20250624_chem02_28C_T00_1356_H01` → experiment_id: `20250624_chem02_28C_T00_1356`, well: `H01`

### 2. image_id
- **Format**: `{video_id}_t{FRAME}` (NEW: 't' prefix for future-proofing)
- **FRAME Pattern**: `_t[0-9]{3,4}` (t + 3-4 digits at the END)
- **Examples**:
  - `20240411_A01_t0042` → video_id: `20240411_A01`, frame: `0042`
  - `20250529_36hpf_ctrl_atf6_A01_t0042` → video_id: `20250529_36hpf_ctrl_atf6_A01`, frame: `0042`
  
**IMPORTANT TRANSITION NOTE**: We are transitioning from `{video_id}_{FRAME}` to `{video_id}_t{FRAME}` format. The 't' prefix makes parsing significantly easier by disambiguating image IDs from other entity types that might end with numbers. This prevents conflicts with complex experiment IDs that may contain trailing numbers.

### 3. embryo_id
- **Format**: `{video_id}_e{NN}`
- **Embryo Pattern**: `_e[0-9]+` (at the END)
- **Examples**:
  - `20240411_A01_e01` → video_id: `20240411_A01`, embryo_number: `01`
  - `20250529_36hpf_ctrl_atf6_A01_e03` → video_id: `20250529_36hpf_ctrl_atf6_A01`, embryo_number: `03`

### 4. snip_id
- **Format**: `{embryo_id}_{FRAME}` OR `{embryo_id}_s{FRAME}`
- **FRAME Pattern**: `_s?[0-9]{3,4}` (optional 's' prefix, at the END)
- **Examples**:
  - `20240411_A01_e01_s0042` → embryo_id: `20240411_A01_e01`, frame: `0042`
  - `20240411_A01_e01_0042` → embryo_id: `20240411_A01_e01`, frame: `0042`
  - `20250529_36hpf_ctrl_atf6_A01_e01_s0042` → embryo_id: `20250529_36hpf_ctrl_atf6_A01_e01`, frame: `0042`

## Parsing Strategy

Always parse from the END of the string:

1. **For video_id**: Look for `_[A-H][0-9]{2}$` at the end
2. **For image_id**: Look for `_[0-9]{3,4}$` at the end
3. **For embryo_id**: Look for `_e[0-9]+$` at the end
4. **For snip_id**: Look for `_s?[0-9]{3,4}$` at the end

## Common Pitfalls to Avoid

❌ **DON'T** try to parse experiment_id first by looking for patterns
❌ **DON'T** assume experiment_id follows any specific format
❌ **DON'T** split on underscores and assume positions

✅ **DO** always work backwards from the end
✅ **DO** use the specific end patterns for each entity type
✅ **DO** handle complex experiment IDs like `20250624_chem02_28C_T00_1356`

## Examples of Complex Parsing

```python
# Complex experiment ID with multiple underscores
snip_id = "20250624_chem02_28C_T00_1356_H01_e01_s034"

# Parse backwards:
# 1. Find frame: _s034 → frame = "034", remainder = "20250624_chem02_28C_T00_1356_H01_e01"
# 2. Find embryo: _e01 → embryo_number = "01", remainder = "20250624_chem02_28C_T00_1356_H01"
# 3. Find well: _H01 → well = "H01", remainder = "20250624_chem02_28C_T00_1356"
# 4. What's left is experiment_id: "20250624_chem02_28C_T00_1356"

# Result:
{
    'experiment_id': '20250624_chem02_28C_T00_1356',
    'well_id': 'H01',
    'video_id': '20250624_chem02_28C_T00_1356_H01',
    'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01',
    'embryo_number': '01',
    'frame_number': '034',
    'snip_id': '20250624_chem02_28C_T00_1356_H01_e01_s034'
}
```