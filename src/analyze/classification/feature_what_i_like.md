# Classification API — Agreed Design Spec

Status: approved front-end spec for the next implementation phase.

This document is the canonical reference for the new `classify()` API and its comparison
resolution contract. It supersedes the previous conversation dump.

---

## Guiding principles

1. **DataFrame-first.** Input is always a plain `pd.DataFrame`. No wrapper object required.
2. **Multi-feature is the native unit of thought.** `features={}` dict is always named; the
   results table always has a `feature_set` column.
3. **One entry point.** `classify()` covers all modes. No `run_ovr()`, `run_pairwise()` siblings.
4. **Layered signature.** Data contract → comparison spec → features → model/binning → output.
5. **Fail fast.** All validation happens before any computation.
6. **Symmetric pooling.** Pooled positives and pooled negatives work identically at the
   binary-label level. `tuple` = pooled, `list` = enumerate, `str` = single.

---

## Public entry point

```python
def classify(
    # ── Layer 1: data contract (required, keyword-only after df) ──────────
    df: pd.DataFrame,
    *,
    class_col: str,                  # column holding class labels
    id_col: str,                     # embryo/unit identity column
    time_col: str,                   # continuous time column for binning

    # ── Layer 2: comparison spec ──────────────────────────────────────────
    # Simple path — explicit positive / negative
    positive: UserComparisonSpec | None = None,
    negative: UserComparisonSpec | None = None,

    # Scheme / advanced path — overrides simple path
    # str scheme accepts positive= as a class-scope LIST filter only
    comparisons: Literal["all_vs_rest", "all_pairs"] | pd.DataFrame | None = None,

    # ── Layer 3: features (always named, always a dict) ───────────────────
    features: dict[str, str | list[str]],   # {"name": "prefix_or_col_list"}

    # ── Layer 4: model / binning ──────────────────────────────────────────
    bin_width: float = 4.0,
    n_permutations: int = 100,
    n_splits: int = 5,
    min_samples_per_class: int = 3,
    n_jobs: int = 1,
    random_state: int = 42,

    # ── Layer 5: output / control ─────────────────────────────────────────
    verbose: bool = True,
    save_null: bool = True,
    allow_overlap: bool = False,
) -> ClassificationAnalysis:
```

### Comparison mode rules (mutually exclusive, enforced at call time)

| `positive` | `negative` | `comparisons` | Mode |
|---|---|---|---|
| omitted | omitted | omitted | all-vs-rest across all classes in `class_col` |
| scalar or list | scalar, list, or tuple | omitted | explicit cartesian product |
| list (scope filter) | omitted | `"all_vs_rest"` | all-vs-rest within scope |
| list (scope filter) | omitted | `"all_pairs"` | all unordered pairs within scope |
| omitted | omitted | `pd.DataFrame` | explicit design table |

**Hard mutual-exclusion errors:**
- `comparisons=DataFrame` + any `positive` or `negative` → `ValueError`
- `comparisons=str_scheme` + `negative` → `ValueError`
- `comparisons=str_scheme` + `positive` as scalar → `ValueError`
  (scalar positive with a scheme is ambiguous; use a 1-element list)
- `negative` set but `positive` omitted, `comparisons` omitted → `ValueError`
  (negative without positive is almost always a user mistake)

### Three standard usage patterns

```python
# 1. Discovery — all classes, all-vs-rest, one or more feature sets
results = classify(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    features={"embedding": "z_mu_b"},
)

# 2. Targeted — explicit positive/negative, multiple named feature sets
results = classify(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    positive=["homo", "het"],
    negative="wildtype",
    features={
        "embedding": "z_mu_b",
        "shape": ["total_length_um", "baseline_deviation_normalized"],
    },
    bin_width=4.0,
    n_permutations=100,
)

# 3. Advanced — explicit design table
design = pd.DataFrame({
    "positive": ["homo", "homo", "het"],
    "negative": ["wildtype", "het", "wildtype"],
})
results = classify(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    comparisons=design,
    features={"embedding": "z_mu_b"},
)
```

---

## Comparison vocabulary

| Term | Meaning | How to invoke |
|---|---|---|
| one-vs-one | One binary comparison between exactly two classes | `positive="homo", negative="wildtype"` |
| one-vs-rest | One class against all others pooled | `positive="homo"` (comparisons omitted) |
| all-vs-rest | Every class gets its own one-vs-rest | all omitted, or `comparisons="all_vs_rest"` |
| all-pairs / pairwise | All unordered one-vs-one pairs | `comparisons="all_pairs"` |
| pooled side | Multiple classes merged into one binary side | `tuple`: `("wildtype","het")` |
| enumerated | Multiple comparisons, one per element | `list`: `["wildtype","het"]` |

**Do not use** "all-vs-all" — it is ambiguous (ordered pairs? multiclass model? full confusion?).

---

## Type definitions

```python
ClassLabel     = str
PooledGroup    = tuple[str, ...]          # ≥ 2 elements, sorted+deduped at ingest
ComparisonSide = ClassLabel | PooledGroup

# What the user may pass to positive= or negative=
UserComparisonSpec = ClassLabel | PooledGroup | list[ClassLabel | PooledGroup]

# What comparisons= accepts
ComparisonScheme = Literal["all_vs_rest", "all_pairs"] | pd.DataFrame | None
```

### Encoding convention

| User value | Meaning |
|---|---|
| `"homo"` | single class |
| `("wildtype", "het")` | pooled: wildtype + het merged into one binary side |
| `["homo", "het"]` | enumerate: two separate comparisons |
| `[("wildtype","het"), "crispant"]` | two comparisons: once vs pooled {wt+het}, once vs crispant |

**Symmetry:** `tuple`-as-pool works identically for `positive` and `negative`. There is no
mechanical difference between a pooled positive and a pooled negative — both collapse to a
binary `_y` vector. The biological interpretation differs, but the code does not.

---

## Output type: `ResolvedComparison`

The backend receives a list of these. Everything downstream is unaware of pooling.

```python
@dataclass(frozen=True)
class ResolvedComparison:
    positive_members: tuple[str, ...]    # ≥ 1 label; sorted; maps to _y = 1
    negative_members: tuple[str, ...]    # ≥ 1 label; sorted; maps to _y = 0
    positive_label: str                  # human-readable: "homo" or "homo+crispant"
    negative_label: str                  # human-readable: "wildtype" or "wildtype+het"
    comparison_id: str                   # filesystem-safe: "homo__vs__wildtype_het"

    @property
    def is_pooled_positive(self) -> bool:
        return len(self.positive_members) > 1

    @property
    def is_pooled_negative(self) -> bool:
        return len(self.negative_members) > 1

    @property
    def all_members(self) -> frozenset[str]:
        return frozenset(self.positive_members) | frozenset(self.negative_members)
```

**Invariants:**
- `positive_members` and `negative_members` are disjoint sorted tuples (enforced before construction)
- `comparison_id` contains only `[A-Za-z0-9._-]` (spaces, `+`, `/` etc. → `_`)
- `positive_label` / `negative_label` are human-readable and unsanitized

---

## Comparison resolution pipeline

```
user input (positive, negative, comparisons, available_labels)
    ↓  Step 1  — validate types
    ↓  Step 2  — determine mode + normalize to sides
    ↓  Step 3  — canonicalize pooled tuples (sort + dedupe, both sides)
    ↓  Step 4  — expand to raw pairs (mode-dependent)
    ↓  Step 5  — overlap check (positive_members ∩ negative_members = ∅)
    ↓  Step 6  — label existence check (pure — no DataFrame access)
    ↓  Step 7  — rest expansion (rest-mode only)
    ↓  Step 8  — deduplicate (on final pairs, after rest expansion)
    ↓  Step 9  — _to_resolved (→ list[ResolvedComparison])
    ↓  Step 10 — min-sample check + per-member warnings
backend receives: list[ResolvedComparison]
```

### Public entry point (pure function)

```python
def resolve_comparisons(
    positive: UserComparisonSpec | None,
    negative: UserComparisonSpec | None,
    comparisons: ComparisonScheme,
    available_labels: set[str],    # = set(df[class_col].unique()), computed by caller
    class_col: str,                # used only in error messages
    allow_overlap: bool = False,
) -> list[ResolvedComparison]:
    """
    Pure function. No DataFrame access.
    available_labels must be computed by the caller before calling this.
    """
```

### Step 1 — Type validation

```python
def _validate_comparison_side(val, param_name: str) -> None:
    if val is None:
        return
    if isinstance(val, str):
        return
    if isinstance(val, tuple):
        if not all(isinstance(v, str) for v in val):
            raise TypeError(f"{param_name}: tuple elements must all be strings. Got {val!r}")
        if len(val) < 2:
            raise ValueError(
                f"{param_name}: pooled tuple must have ≥ 2 elements. "
                f"For a single class use a string, not a 1-tuple."
            )
        return
    if isinstance(val, list):
        for i, item in enumerate(val):
            _validate_comparison_side(item, f"{param_name}[{i}]")
        return
    raise TypeError(
        f"{param_name} must be str, tuple[str,...], or list thereof. Got {type(val)}"
    )

_validate_comparison_side(positive, "positive")
_validate_comparison_side(negative, "negative")
```

### Step 2 — Determine mode and normalize to sides

```python
def _to_sides(val: UserComparisonSpec, all_labels: list[str]) -> list[ComparisonSide]:
    if isinstance(val, (str, tuple)):
        return [val]
    return list(val)

# Mode detection (after mutual-exclusion checks in classify())
if isinstance(comparisons, pd.DataFrame):
    mode = "design_table"

elif comparisons == "all_pairs":
    mode = "all_pairs"
    # positive= accepted as a scope filter (list only; scalar already blocked)
    class_scope: list[str] = (
        list(positive) if positive is not None else sorted(available_labels)
    )
    # Tuples forbidden in scope for all_pairs
    for i, s in enumerate(class_scope):
        if isinstance(s, tuple):
            raise ValueError(
                f"comparisons='all_pairs': positive[{i}] is a tuple. "
                f"Scope entries must be single class labels (strings)."
            )

elif comparisons in ("all_vs_rest", None) and negative is None:
    mode = "rest"
    pos_sides = _to_sides(positive, sorted(available_labels)) if positive is not None \
                else list(sorted(available_labels))

else:  # comparisons is None, negative is not None
    mode = "explicit"
    pos_sides = _to_sides(positive, sorted(available_labels)) if positive is not None \
                else list(sorted(available_labels))
    neg_sides = _to_sides(negative, sorted(available_labels))
```

### Step 3 — Canonicalize pooled tuples

```python
def _canonicalize_side(side: ComparisonSide) -> ComparisonSide:
    if isinstance(side, tuple):
        result = tuple(sorted(set(side)))
        if len(result) < 2:
            raise ValueError(
                f"Pooled tuple collapsed to < 2 unique elements after deduplication: {side!r}"
            )
        return result
    return side

if mode == "explicit":
    pos_sides = [_canonicalize_side(s) for s in pos_sides]
    neg_sides = [_canonicalize_side(s) for s in neg_sides]
elif mode == "rest":
    pos_sides = [_canonicalize_side(s) for s in pos_sides]
# all_pairs and design_table: sides are strings, canonicalization is a no-op
```

### Step 4 — Expand to raw pairs (mode-dependent)

```python
from itertools import product, combinations

if mode == "explicit":
    # Cartesian product: every positive × every negative
    # "list enumerates" rule: [A, B] × [C, D] → (A,C), (A,D), (B,C), (B,D)
    raw_pairs = list(product(pos_sides, neg_sides))

elif mode == "rest":
    # Defer negative — computed per-positive in Step 7
    raw_pairs = [(p, None) for p in pos_sides]

elif mode == "all_pairs":
    # Unordered combinations; deterministic direction = alphabetical (a < b → a is positive)
    labels = sorted(class_scope)
    raw_pairs = [(a, b) for a, b in combinations(labels, 2)]

elif mode == "design_table":
    _validate_design_table(comparisons)
    raw_pairs = [
        (row["positive"], row["negative"])
        for _, row in comparisons.iterrows()
    ]
```

### Step 5 — Overlap check

```python
def _members(side: ComparisonSide) -> set[str]:
    return {side} if isinstance(side, str) else set(side)

for pos_side, neg_side in raw_pairs:
    if neg_side is None:
        continue  # rest-mode: skip until Step 7
    overlap = _members(pos_side) & _members(neg_side)
    if overlap and not allow_overlap:
        raise ValueError(
            f"Comparison ({pos_side!r} vs {neg_side!r}) has overlapping class labels: "
            f"{sorted(overlap)}. The same label cannot appear on both sides. "
            f"Pass allow_overlap=True to permit this explicitly."
        )
```

### Step 6 — Label existence check (pure)

```python
def _check_labels_exist(
    raw_pairs: list[tuple],
    available_labels: set[str],
    class_col: str,
) -> None:
    referenced = set()
    for pos_side, neg_side in raw_pairs:
        referenced |= _members(pos_side)
        if neg_side is not None:
            referenced |= _members(neg_side)
    unknown = referenced - available_labels
    if unknown:
        raise ValueError(
            f"Class labels not found in {class_col!r}: {sorted(unknown)}. "
            f"Available: {sorted(available_labels)}"
        )

_check_labels_exist(raw_pairs, available_labels, class_col)
```

### Step 7 — Rest expansion (rest-mode only)

```python
if mode == "rest":
    expanded = []
    for pos_side, _ in raw_pairs:
        rest = tuple(sorted(available_labels - _members(pos_side)))
        if len(rest) == 0:
            raise ValueError(
                f"No remaining classes to form 'rest' for positive={pos_side!r}. "
                f"All available labels are in the positive side."
            )
        expanded.append((pos_side, rest))
    raw_pairs = expanded
```

### Step 8 — Deduplicate (on final pairs)

```python
seen: set[tuple] = set()
deduped: list[tuple] = []
for pair in raw_pairs:
    # Both elements are str or sorted tuple → hashable
    if pair not in seen:
        seen.add(pair)
        deduped.append(pair)
raw_pairs = deduped
```

### Step 9 — Convert to `ResolvedComparison`

```python
def _sanitize_id(label: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", label)

def _to_resolved(pos_side: ComparisonSide, neg_side: ComparisonSide) -> ResolvedComparison:
    pos_members = (pos_side,) if isinstance(pos_side, str) else pos_side
    neg_members = (neg_side,) if isinstance(neg_side, str) else neg_side
    positive_label = "+".join(pos_members)
    negative_label = "+".join(neg_members)
    comparison_id = _sanitize_id(positive_label) + "__vs__" + _sanitize_id(negative_label)
    return ResolvedComparison(
        positive_members=pos_members,
        negative_members=neg_members,
        positive_label=positive_label,
        negative_label=negative_label,
        comparison_id=comparison_id,
    )

resolved: list[ResolvedComparison] = [_to_resolved(p, n) for p, n in raw_pairs]
```

### Step 10 — Min-sample check and per-member warnings

`label_counts` must count **unique `id_col` units (embryos)**, not rows.
Row counts are inflated by time points and must not be used here.

```python
def _check_min_samples(
    resolved: list[ResolvedComparison],
    label_counts: dict[str, int],    # {class_label: n_unique_embryos}
    min_samples: int,
    warn_threshold: int = 5,
) -> None:
    for rc in resolved:
        for members, side_label, side_name in [
            (rc.positive_members, rc.positive_label, "positive"),
            (rc.negative_members, rc.negative_label, "negative"),
        ]:
            union_n = sum(label_counts.get(m, 0) for m in members)
            if union_n < min_samples:
                raise ValueError(
                    f"Comparison {rc.comparison_id!r}: {side_name} '{side_label}' "
                    f"has only {union_n} embryos (min={min_samples})."
                )
            for m in members:
                n = label_counts.get(m, 0)
                if n < warn_threshold:
                    warnings.warn(
                        f"Comparison {rc.comparison_id!r}: member '{m}' in {side_name} "
                        f"pool '{side_label}' has only {n} embryos. "
                        f"Consider whether pooling is appropriate.",
                        UserWarning, stacklevel=3,
                    )
```

---

## Design table validation

```python
def _validate_design_table(df: pd.DataFrame) -> None:
    required = {"positive", "negative"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"comparisons DataFrame missing columns: {sorted(missing)}")
    for col in required:
        if df[col].isnull().any():
            raise ValueError(f"comparisons DataFrame column {col!r} contains nulls.")
        if not df[col].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f"comparisons DataFrame column {col!r} must contain strings only.")
    if df.duplicated(subset=["positive", "negative"]).any():
        raise ValueError("comparisons DataFrame contains duplicate (positive, negative) rows.")
```

---

## The one backend function that knows about pooling

Everything downstream of this function (CV, AUROC, permutation) sees only `_y` and is
unaware that pooling occurred.

```python
def build_binary_labels(
    df: pd.DataFrame,
    class_col: str,
    comparison: ResolvedComparison,
) -> pd.DataFrame:
    """
    Filter df to rows belonging to either side of the comparison and assign
    binary labels: _y = 1 for positive_members, _y = 0 for negative_members.
    Rows from classes not in either side are dropped.

    The inference target is separability of the pooled groups, not of each
    constituent class individually. Permutation tests shuffle this binary _y
    vector within time strata — pooled positives and pooled negatives are
    handled identically.
    """
    subset = df[df[class_col].isin(comparison.positive_members | set(comparison.negative_members))].copy()
    subset["_y"] = subset[class_col].isin(comparison.positive_members).astype(int)
    return subset
```

**Note:** `_y` is derived from the subset's class labels directly (not from outer mask
indexing) to avoid index-alignment footguns.

---

## Permutation / null testing

Because the null permutation shuffles `_y` within time-bin strata:

- Pooling only changes which rows get `_y=1` vs `_y=0`.
- Once `_y` exists, permutation is identical regardless of how many members are
  on each side: shuffle `_y` within bin, recompute AUROC, collect null distribution.
- The null corresponds to the binary task "positive_members vs negative_members."

**The hypothesis being tested is separability of the pooled groups, not of each
constituent class individually.**

---

## Worked example (end-to-end trace)

**Inputs:**
```python
available_labels = {"wildtype", "het", "homo", "crispant"}
positive = ("homo", "crispant")     # pooled positive
negative = ("wildtype", "het")      # pooled negative
comparisons = None
```

**Step 1:** Both sides are valid tuples with ≥ 2 strings. ✓

**Step 2:** `comparisons=None`, `negative` is not None → `mode = "explicit"`.
- `pos_sides = [("homo", "crispant")]`
- `neg_sides = [("wildtype", "het")]`

**Step 3:** Canonicalize:
- `("homo", "crispant")` → `("crispant", "homo")` (sorted)
- `("wildtype", "het")` → `("het", "wildtype")` (sorted)

**Step 4:** Cartesian product → `[(("crispant","homo"), ("het","wildtype"))]`

**Step 5:** `{"crispant","homo"} ∩ {"het","wildtype"} = ∅` ✓

**Step 6:** All four labels in `available_labels`. ✓

**Step 7:** Not rest-mode. Skip.

**Step 8:** One pair, no duplicates.

**Step 9:**
```python
ResolvedComparison(
    positive_members = ("crispant", "homo"),
    negative_members = ("het", "wildtype"),
    positive_label   = "crispant+homo",
    negative_label   = "het+wildtype",
    comparison_id    = "crispant_homo__vs__het_wildtype",
)
```

**Step 10:** Check union embryo counts for each side against `min_samples`.

**Backend:**
```python
subset = df[df["genotype"].isin({"crispant","homo","het","wildtype"})].copy()
subset["_y"] = subset["genotype"].isin({"crispant","homo"}).astype(int)
# downstream CV / AUROC / permutation operates on _y only
```
# Backend & Object Spec — Locked

---

## Internal factory line

```
classify(df, ...)
    │
    ├─ 1. _resolve_feature_columns()   → dict[str, list[str]]  (feature_set → col list)
    ├─ 2. resolve_comparisons()        → list[ResolvedComparison]
    ├─ 3. _build_binary_labels()       → filtered df with _y column  (per comparison)
    ├─ 4. _bin_and_aggregate()         → df binned by (embryo_id, time_bin)
    ├─ 5. _run_classification_loop()   → raw per-bin result dicts
    │       ├─ cross_val_predict()     → probabilities
    │       ├─ roc_auc_score()         → auroc_obs
    │       └─ _permutation_test_ovr() → null_aurocs → pval, null_mean, null_std
    ├─ 6. _collect_scores()            → scores DataFrame  ← THE canonical table
    ├─ 7. _collect_predictions()       → predictions DataFrame  (optional)
    └─ 8. _collect_confusion()         → confusion DataFrame  (optional)
```

Each step runs once per `(feature_set, comparison)` pair. Results are concatenated into
the canonical tables after all pairs complete.

---

## Step 5 — inner loop output (raw per-bin dict)

Produce these keys inside the loop. No renaming at boundaries.

```python
{
    # time
    "time_bin":        int(t),
    "time_bin_center": float(t) + bin_width / 2.0,
    "bin_width":       float(bin_width),
    # results — canonical names, set here, never renamed later
    "auroc_obs":       float,        # was auroc_observed — renamed HERE not at boundary
    "pval":            float,
    "n_positive":      int,
    "n_negative":      int,
    # null summary
    "auroc_null_mean": float,
    "auroc_null_std":  float,
    "n_permutations":  int,
    # raw null array — collected separately, NOT a dict key
    "_null_array":     np.ndarray,   # shape (P,), removed before scores assembly
}
# DROPPED from inner loop: positive_class, negative_class, negative_mode, groupby
```

---

## Step 6 — `_collect_scores(bin_results, comparison, feature_set) -> list[dict]`

The only place that assembles identity keys + results into a canonical scores row.

```python
def _collect_scores(
    bin_results: list[dict],
    comparison: ResolvedComparison,
    feature_set: str,
) -> list[dict]:
    rows = []
    for r in bin_results:
        rows.append({
            # identity (always present)
            "feature_set":     feature_set,
            "comparison_id":   comparison.comparison_id,
            "positive_label":  comparison.positive_label,
            "negative_label":  comparison.negative_label,
            # time
            "time_bin_center": r["time_bin_center"],
            "time_bin":        r["time_bin"],
            "bin_width":       r["bin_width"],
            # results
            "auroc_obs":       r["auroc_obs"],
            "pval":            r["pval"],
            "n_pos":           r["n_positive"],
            "n_neg":           r["n_negative"],
            # null summary
            "auroc_null_mean": r.get("auroc_null_mean"),
            "auroc_null_std":  r.get("auroc_null_std"),
            "n_permutations":  r.get("n_permutations"),
        })
    return rows
```

---

## Step 7 — `_collect_predictions()` — binary per comparison

Works for every mode (all-vs-rest, pairwise, all-pairs). One row per
(embryo_id, time_bin_center, comparison_id, feature_set).

```python
{
    "feature_set":     str,
    "comparison_id":   str,
    "embryo_id":       str,
    "time_bin_center": float,
    "y_true":          int,    # 1 = positive side, 0 = negative side
    "p_pos":           float,  # probability of positive class
    "y_pred":          int,    # hard call
    "is_correct":      bool,
}
```

The multiclass `pred_proba_{class}` wide format is NOT part of the primary predictions
contract. If needed, expose as a separate optional "multiclass_predictions" layer.

---

## Step 8 — `_collect_confusion()`

Wraps `extract_temporal_confusion_profile`. Add `feature_set`, `comparison_id`, and
use `time_bin_center` as the canonical time key.

```python
{
    "feature_set":     str,
    "comparison_id":   str,    # "all_vs_rest" for multiclass runs
    "time_bin_center": float,
    "true_class":      str,
    "predicted_class": str,
    "proportion":      float,
    "count":           int,
    "is_correct":      bool,
}
```

Only emitted for multiclass (all-vs-rest) runs. Skip for pairwise comparisons — the
2×2 confusion adds nothing over AUROC.

---

## Canonical `scores` table — column contract

```
REQUIRED (always present, never null):
  feature_set      str     name from features={} dict
  comparison_id    str     "homo__vs__wildtype_het"  (filesystem-safe)
  positive_label   str     "homo" or "homo+crispant"
  negative_label   str     "wildtype" or "wildtype+het"
  time_bin_center  float   canonical x-axis for all plots
  auroc_obs        float   observed AUROC

STANDARD (present unless n_permutations=0):
  pval             float
  auroc_null_mean  float
  auroc_null_std   float
  n_permutations   int
  n_pos            int
  n_neg            int

OPTIONAL:
  time_bin         int     bin start (join key, not for plotting)
  bin_width        float
```

**Unique key:** `(feature_set, comparison_id, time_bin_center)` — enforced by
`_validate_scores()`.

**No JSON blobs, no list-in-cell, no alias columns.**

Comparison membership lives in `uns["comparisons"]` keyed by `comparison_id`:

```json
{
  "comparisons": {
    "homo__vs__wildtype_het": {
      "positive_members": ["homo"],
      "negative_members": ["wildtype", "het"],
      "positive_label": "homo",
      "negative_label": "wildtype+het"
    }
  }
}
```

---

## `ClassificationAnalysis` object

```python
@dataclass
class ClassificationAnalysis:
    scores: pd.DataFrame      # required — always eager
    uns:    dict              # required — always eager; treat as read-only
    layers: _LazyLayers       # optional artifacts — lazy from disk

    def __post_init__(self):
        _validate_scores(self.scores)

    # Properties
    @property
    def feature_sets(self) -> list[str]:
        return sorted(self.scores["feature_set"].unique())

    @property
    def comparison_ids(self) -> list[str]:
        return sorted(self.scores["comparison_id"].unique())

    # Subsetting — forks layer cache so subset and parent don't share state
    def subset(self, feature_set=None, comparison_id=None,
               positive_label=None, time_range=None) -> "ClassificationAnalysis":
        s = self.scores
        if feature_set   is not None: s = s[s["feature_set"].isin(_listify(feature_set))]
        if comparison_id is not None: s = s[s["comparison_id"].isin(_listify(comparison_id))]
        if positive_label is not None: s = s[s["positive_label"].isin(_listify(positive_label))]
        if time_range    is not None: s = s[s["time_bin_center"].between(*time_range)]
        return ClassificationAnalysis(scores=s.copy(), uns=self.uns,
                                      layers=self.layers._fork())

    # Stacking — scores only; layers are NOT merged
    def stack(self, other, on_conflict="error") -> "ClassificationAnalysis":
        """Merge scores tables. Layers are not merged; stacked object is in-memory only."""
        new_keys = set(zip(other.scores["feature_set"], other.scores["comparison_id"]))
        existing = set(zip(self.scores["feature_set"], self.scores["comparison_id"]))
        overlap  = new_keys & existing
        if overlap and on_conflict == "error":
            raise ValueError(f"Overlapping (feature_set, comparison_id) pairs: {overlap}")
        scores = pd.concat([self.scores, other.scores], ignore_index=True)
        if on_conflict == "overwrite" and overlap:
            scores = scores.drop_duplicates(
                subset=["feature_set", "comparison_id", "time_bin_center"], keep="last")
        return ClassificationAnalysis(scores=scores, uns={**self.uns, **other.uns},
                                      layers=_LazyLayers(None))

    # Plotting
    def plot_aurocs(self, facet_col="feature_set", **kwargs):
        from .viz.auroc_over_time import plot_aurocs_over_time
        return plot_aurocs_over_time(self.scores, curve_col="positive_label",
                                     facet_col=facet_col, **kwargs)

    # Persistence
    def save(self, path, overwrite=False) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        _write_parquet(self.scores, path / "scores.parquet", overwrite)
        _write_json(self.uns, path / "metadata.json", overwrite)
        self.layers._save_to_dir(path, overwrite)
        return path

    @classmethod
    def load(cls, path) -> "ClassificationAnalysis":
        path = Path(path)
        scores = pd.read_parquet(path / "scores.parquet")
        with open(path / "metadata.json") as f:
            uns = json.load(f)
        return cls(scores=scores, uns=uns, layers=_LazyLayers(path))

    @classmethod
    def from_legacy(cls, path) -> "ClassificationAnalysis":
        from .legacy import load_legacy_ovr_results
        return load_legacy_ovr_results(path)
```

---

## `_LazyLayers` — complete implementation

```python
class _LazyLayers:
    """
    Lazy-loading dict-like interface for optional artifacts.

    Layers
    ------
    "predictions"  pd.DataFrame    predictions.parquet
    "confusion"    pd.DataFrame    confusion.parquet
    "null_full"    NullDistributions  null_distributions.npz
    """

    _REGISTRY: dict[str, tuple[str, str]] = {
        "predictions": ("predictions.parquet", "parquet"),
        "confusion":   ("confusion.parquet",   "parquet"),
        "null_full":   ("null_distributions.npz", "nulls"),
    }

    def __init__(self, base_dir: Path | None) -> None:
        self._dir   = base_dir
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown layer '{key}'. Known: {sorted(self._REGISTRY)}")
        if key in self._cache:
            return self._cache[key]
        if self._dir is None:
            raise KeyError(f"Layer '{key}' not in cache and no backing directory.")
        fname, kind = self._REGISTRY[key]
        path = self._dir / fname
        if not path.exists():
            raise KeyError(f"Layer '{key}' not found at {path}")
        data = self._load(kind, path)
        self._cache[key] = data
        return data

    def get(self, key: str, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        """Pure existence check — never triggers a disk load."""
        if key in self._cache:
            return True
        if self._dir is None:
            return False
        fname, _ = self._REGISTRY.get(key, (None, None))
        return fname is not None and (self._dir / fname).exists()

    def available(self) -> list[str]:
        """Keys that exist on disk, sorted."""
        if self._dir is None:
            return []
        return sorted(
            k for k, (fname, _) in self._REGISTRY.items()
            if (self._dir / fname).exists()
        )

    def cached(self) -> list[str]:
        """Keys currently in memory."""
        return sorted(self._cache)

    def store(self, key: str, data: Any) -> None:
        """Cache an in-memory artifact (written to disk on analysis.save())."""
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown layer '{key}'. Known: {sorted(self._REGISTRY)}")
        self._cache[key] = data

    def _fork(self) -> "_LazyLayers":
        """New _LazyLayers with same backing dir but empty cache. Used by subset()."""
        return _LazyLayers(self._dir)

    # ── load/save internals ───────────────────────────────────────────────

    @staticmethod
    def _load(kind: str, path: Path) -> Any:
        if kind == "parquet":
            return pd.read_parquet(path)
        if kind == "nulls":
            return NullDistributions.load(path)
        raise ValueError(f"Unknown kind: {kind}")

    def _save_to_dir(self, path: Path, overwrite: bool) -> None:
        for key, data in self._cache.items():
            fname, kind = self._REGISTRY[key]
            fpath = path / fname
            if fpath.exists() and not overwrite:
                raise FileExistsError(f"{fpath} exists. Pass overwrite=True.")
            if kind == "parquet":
                data.to_parquet(fpath, index=False)
            elif kind == "nulls":
                data.save(fpath)
            else:
                raise TypeError(f"Don't know how to save layer '{key}'")
        self._dir = path
```

---

## `NullDistributions` — array-indexed null handle

Avoids delimiter hell by storing a parallel index rather than encoding keys into strings.

```python
@dataclass
class NullDistributions:
    """
    Handle for raw per-permutation AUROC null distributions.

    Storage layout (null_distributions.npz):
      null_auc         float32  shape (N, P)   N = cells, P = permutations
      feature_set      U64 str  shape (N,)
      comparison_id    U64 str  shape (N,)
      time_bin_center  float64  shape (N,)

    Access:
      nd.get("embedding", "homo__vs__wildtype", 26.0)  → np.ndarray shape (P,)
      nd.index_df                                       → DataFrame with N rows
    """
    null_auc:        np.ndarray     # (N, P)  float32
    feature_set:     np.ndarray     # (N,)    str
    comparison_id:   np.ndarray     # (N,)    str
    time_bin_center: np.ndarray     # (N,)    float64
    _index: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        N = len(self.feature_set)
        assert self.null_auc.shape[0] == N
        assert len(self.comparison_id) == N
        assert len(self.time_bin_center) == N

    @property
    def index_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_set":     self.feature_set,
            "comparison_id":   self.comparison_id,
            "time_bin_center": self.time_bin_center,
        })

    def _build_index(self) -> dict:
        return {
            (str(fs), str(cid), float(tbc)): i
            for i, (fs, cid, tbc) in enumerate(
                zip(self.feature_set, self.comparison_id, self.time_bin_center)
            )
        }

    def get(self, feature_set: str, comparison_id: str,
            time_bin_center: float) -> np.ndarray:
        if self._index is None:
            object.__setattr__(self, "_index", self._build_index())
        key = (feature_set, comparison_id, float(time_bin_center))
        if key not in self._index:
            raise KeyError(f"No null distribution for {key}")
        return self.null_auc[self._index[key]]

    @classmethod
    def load(cls, path: Path) -> "NullDistributions":
        npz = np.load(path, allow_pickle=False)
        return cls(
            null_auc        = npz["null_auc"],
            feature_set     = npz["feature_set"],
            comparison_id   = npz["comparison_id"],
            time_bin_center = npz["time_bin_center"],
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            null_auc        = self.null_auc.astype(np.float32),
            feature_set     = self.feature_set,
            comparison_id   = self.comparison_id,
            time_bin_center = self.time_bin_center,
        )
```

**Building a `NullDistributions` during classification:**

```python
# Accumulate per-bin nulls during the loop
null_rows: list[dict] = []
for (fs, cid, tbc), null_array in per_bin_nulls:
    null_rows.append({
        "feature_set":     fs,
        "comparison_id":   cid,
        "time_bin_center": tbc,
        "nulls":           null_array,
    })

# At end of run
nd = NullDistributions(
    null_auc        = np.array([r["nulls"] for r in null_rows], dtype=np.float32),
    feature_set     = np.array([r["feature_set"] for r in null_rows]),
    comparison_id   = np.array([r["comparison_id"] for r in null_rows]),
    time_bin_center = np.array([r["time_bin_center"] for r in null_rows]),
)
analysis.layers.store("null_full", nd)
```

---

## `_validate_scores` — minimum only

```python
_SCORES_REQUIRED = frozenset({
    "feature_set", "comparison_id", "positive_label", "negative_label",
    "time_bin_center", "auroc_obs",
})

def _validate_scores(df: pd.DataFrame) -> None:
    missing = _SCORES_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")
    dupes = df.duplicated(subset=["feature_set", "comparison_id", "time_bin_center"])
    if dupes.any():
        raise ValueError(
            f"scores has {dupes.sum()} duplicate "
            f"(feature_set, comparison_id, time_bin_center) rows."
        )
```

---

## On-disk layout

```
my_run/
  scores.parquet            ← always
  metadata.json             ← always (uns dict)
  predictions.parquet       ← optional (save_predictions=True)
  confusion.parquet         ← optional (multiclass runs only)
  null_distributions.npz    ← optional (save_nulls="full")
```

---

## Artifact tiers

| Artifact | Storage | Default | When |
|---|---|---|---|
| Null stats (mean/std/n) | columns in `scores` | always | free, always useful |
| Raw null arrays | `null_distributions.npz` via `NullDistributions` | off (`save_nulls="full"`) | diagnostic only |
| Confusion profile | `confusion.parquet` | on for multiclass, skip for pairwise | cheap + meaningful |
| Predictions | `predictions.parquet` | off (`save_predictions=True`) | grows with scale |

---

## `uns` structure

```python
uns = {
    # provenance
    "schema_version": "classification_v1",
    "created_at":     "2026-03-23T...",
    "git_commit":     "abc123",

    # run config
    "class_col":  "genotype",
    "id_col":     "embryo_id",
    "time_col":   "predicted_stage_hpf",
    "bin_width":  4.0,
    "n_permutations": 300,
    "feature_sets": {
        "embedding": {
            "spec":    "z_mu_b",                      # original user input
            "columns": ["z_mu_b_0", "z_mu_b_1", ...], # resolved columns used
        },
        "shape": {
            "spec":    ["total_length_um", "baseline_deviation_normalized"],
            "columns": ["total_length_um", "baseline_deviation_normalized"],
        },
    },

    # comparison membership (replaces negative_members JSON blob in table)
    "comparisons": {
        "homo__vs__wildtype_het": {
            "positive_members": ["homo"],
            "negative_members": ["wildtype", "het"],
            "positive_label":   "homo",
            "negative_label":   "wildtype+het",
        },
    },
}
```

---

## Helper

```python
def _listify(val: str | list[str]) -> list[str]:
    return [val] if isinstance(val, str) else list(val)
```

  ---
  The internal factory line, named clearly

  classify(df, ...)
      │
      ├─ 1. _resolve_feature_columns()        → list[str] per feature set
      ├─ 2. resolve_comparisons()             → list[ResolvedComparison]
      ├─ 3. _build_binary_labels()            → filtered df with _y column
      ├─ 4. _bin_and_aggregate()              → df binned by (embryo_id, time_bin)
      ├─ 5. _run_classification_loop()        → raw per-bin results dict
      │       ├─ cross_val_predict()          → probabilities
      │       ├─ roc_auc_score()              → auroc_obs
      │       └─ _permutation_test_ovr()      → null_aurocs → pval, null_mean, null_std
      ├─ 6. _collect_scores()                 → scores DataFrame (THE canonical table)
      ├─ 7. _collect_predictions()            → predictions DataFrame (optional)
      └─ 8. _collect_confusion()              → confusion_profile DataFrame (optional)

  Each step is currently tangled inside _run_multiclass_classification and run_classification_test. The names above are what
  they should be called after cleanup.

  ---
  What each step produces and what's wrong with it today

  Step 5 output — raw per-bin dict (inside _run_multiclass_classification)

  Currently produces two different column name sets depending on the path:

  # Inner loop output (lines 369-387)
  {
      "auroc_observed": ...,      # ← wrong name, gets renamed at boundary
      "positive_class": ...,      # ← duplicate of positive added later
      "negative_class": "Rest",   # ← only in all-vs-rest path
      "time_bin": int(t),
      "time_bin_center": ...,
      ...
  }

  Fix: rename auroc_observed → auroc_obs here, inside the loop. Drop positive_class and negative_class entirely — they're not
  needed; positive_label and negative_label come from the ResolvedComparison.

  Step 6 — _collect_scores() (currently lines 725–742, 806–822)

  This is the boundary where per-comparison results get merged with ResolvedComparison metadata. Currently it does column
  renaming at this boundary (auroc_observed → auroc_obs), adds positive/negative from the spec, and serializes negative_members
  as a JSON string.

  The clean version:

  def _collect_scores(
      bin_results: list[dict],          # raw rows from step 5
      comparison: ResolvedComparison,
      feature_set: str,
  ) -> list[dict]:
      rows = []
      for r in bin_results:
          rows.append({
              # identity keys
              "feature_set":     feature_set,
              "comparison_id":   comparison.comparison_id,
              "positive_label":  comparison.positive_label,
              "negative_label":  comparison.negative_label,
              # time
              "time_bin_center": r["time_bin_center"],
              "time_bin":        r["time_bin"],
              "bin_width":       r["bin_width"],
              # results
              "auroc_obs":       r["auroc_obs"],          # renamed in step 5
              "pval":            r["pval"],
              "n_pos":           r["n_positive"],
              "n_neg":           r["n_negative"],
              # null summary (optional but standard)
              "auroc_null_mean": r.get("auroc_null_mean"),
              "auroc_null_std":  r.get("auroc_null_std"),
              "n_permutations":  r.get("auroc_n_permutations"),
          })
      return rows

  What gets dropped: positive_class, negative_class, negative_mode, groupby (repeated every row — move to metadata),
  negative_members JSON blob (move to metadata keyed by comparison_id).

  Step 7 — _collect_predictions() (currently lines 402–429)

  Currently wide-format with pred_proba_{class} columns for every class — only works for the multiclass all-vs-rest path. For
  pairwise comparisons it's never populated.

  The clean version — binary per comparison:

  # One row per (embryo_id, time_bin_center, comparison_id, feature_set)
  {
      "feature_set":     feature_set,
      "comparison_id":   comparison.comparison_id,
      "embryo_id":       eid,
      "time_bin_center": ...,
      "y_true":          1 or 0,     # binary label for THIS comparison
      "p_pos":           float,      # probability of positive side
      "y_pred":          1 or 0,     # hard call
      "is_correct":      bool,
  }

  This works for every mode — all-vs-rest, pairwise, all-pairs. The multiclass pred_proba_{class} columns can be kept as a
  separate optional multiclass_predictions table if needed, but they're not the primary predictions contract.

  Step 8 — _collect_confusion() (currently extract_temporal_confusion_profile)

  This is fine — just needs feature_set and comparison_id added, and time_bin_center as the canonical time key instead of
  time_bin.

  ---
  The canonical scores table — exact column contract

  REQUIRED (always present, never null):
    feature_set      str     name from features={} dict
    comparison_id    str     "homo__vs__wildtype_het"
    positive_label   str     "homo" or "homo+crispant"
    negative_label   str     "wildtype" or "wildtype+het"
    time_bin_center  float   canonical x-axis for all plots
    auroc_obs        float   observed AUROC

  STANDARD (present unless n_permutations=0):
    pval             float
    auroc_null_mean  float
    auroc_null_std   float
    n_permutations   int
    n_pos            int
    n_neg            int

  OPTIONAL:
    time_bin         int     bin start (useful as join key, not for plotting)
    bin_width        float

  Key invariants:
  - (feature_set, comparison_id, time_bin_center) is the unique key — no duplicates
  - No JSON blobs, no list-in-cell
  - No positive_class / negative_class alias columns
  - auroc_obs is the one AUROC column name — no alias fallback needed anywhere

  Where negative_members / positive_members live:
  In metadata["comparisons"] keyed by comparison_id:

  {
    "comparisons": {
      "homo__vs__wildtype_het": {
        "positive_members": ["homo"],
        "negative_members": ["wildtype", "het"],
        "positive_label": "homo",
        "negative_label": "wildtype+het"
      }
    }
  }

  ---
  What the plotter needs from scores

  plot_aurocs_over_time currently checks for auroc_obs/auroc_observed via _auroc_col() and time_bin_center/time_bin via
  _time_col(). Under the new contract those alias helpers are deleted — the columns are always auroc_obs and time_bin_center, no
   fallback needed.

  The plotter already uses curve_col="positive" and facet_col="feature_set" as optional parameters. Under the new contract:

  # multi-feature, single comparison → facet by feature_set
  plot_aurocs_over_time(results.scores, curve_col="positive_label", facet_col="feature_set")

  # single feature, multi-comparison → curves by positive_label
  plot_aurocs_over_time(results.scores, curve_col="positive_label")

  No alias logic, no dict-of-DataFrames, no intermediate reshaping.

  ---
  The null_summary table — verdict

  Delete it. The null stats (auroc_null_mean, auroc_null_std, n_permutations) are already in scores. The only reason to have a
  separate null_summary table is if you store raw per-permutation arrays, which isn't implemented. Until that's needed,
  null_summary is pure redundancy. Remove the slot from ClassificationAnalysis and remove the separate parquet file from save().

  ---
  Summary: what changes where

  ┌──────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │           Location           │                                 Change                                  │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Inner loop (step 5)          │ Rename auroc_observed→auroc_obs, drop positive_class/negative_class     │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ _collect_scores() (new)      │ Canonical column assembly, feature_set added, no JSON blobs             │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ _collect_predictions() (new) │ Binary per comparison, works for all modes                              │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ results.py                   │ Rename comparisons→scores, drop null_summary slot                       │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ save()                       │ Remove null_summary parquet, add comparison_membership to metadata.json │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Plotters                     │ Delete _auroc_col() / _time_col() alias helpers                         │
  └──────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

---

# Plotting Spec — Locked

---

## Design principles

- Plotters take DataFrames, not result objects.
- `plot_aurocs_over_time(scores, ...)` is the one canonical plotter.
- `results.plot_aurocs(...)` is sugar that passes `results.scores` and nothing else.
- All defaults are inferred from the data.
- Overlays are disabled with a warning if their required columns are absent.
- No alias helpers (`_auroc_col`, `_time_col`) — columns are always `auroc_obs` and
  `time_bin_center`.

---

## Return types

```python
from matplotlib.figure import Figure as MplFigure
from plotly.graph_objects import Figure as PlotlyFigure
```

Use `@overload` so type checkers get precise return types:

```python
@overload
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: Literal["plotly"] = "plotly", **kw
) -> PlotlyFigure: ...

@overload
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: Literal["matplotlib"], **kw
) -> MplFigure: ...

@overload
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: Literal["both"], **kw
) -> tuple[PlotlyFigure, MplFigure]: ...
```

Implementation uses the union internally:

```python
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: str = "plotly", **kw
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
    ...
```

Always return `Figure`, not `(fig, ax)`. Caller can access axes via `fig.axes` if needed.
Returning `Figure` is correct for multi-facet grids where there are many axes.

---

## `infer_curve_col(scores)` — smart default, lives in plotter module

```python
def infer_curve_col(scores: pd.DataFrame) -> str:
    """
    Infer the best default curve_col for plot_aurocs_over_time.

    Rules
    -----
    Requires columns: positive_label, negative_label, comparison_id.

    If each positive_label maps to exactly one negative_label:
        → "positive_label"   (unambiguous, clean labels)
    If any positive_label appears with multiple negative_labels:
        → "comparison_id"    (avoids silently merging distinct comparisons)
        → emits UserWarning naming the ambiguous labels

    Notes
    -----
    - Does not handle the mirror case (one negative, multiple positives) because
      that is unambiguous for curve_col purposes. If you want curves per negative,
      pass curve_col="negative_label" explicitly.
    - inference only runs when curve_col=None is passed to the plotter.
    """
    _require_columns(scores, {"positive_label", "negative_label", "comparison_id"})
    pairs = scores[["positive_label", "negative_label"]].drop_duplicates()
    neg_per_pos = pairs.groupby("positive_label")["negative_label"].nunique()
    if (neg_per_pos > 1).any():
        ambiguous = sorted(neg_per_pos[neg_per_pos > 1].index.tolist())
        warnings.warn(
            f"positive_label is ambiguous for {ambiguous} "
            f"(each appears with multiple negative_labels). "
            f"Defaulting to curve_col='comparison_id'. "
            f"Pass curve_col='positive_label' explicitly to override, or "
            f"use facet_col='negative_label' to separate comparisons.",
            UserWarning, stacklevel=3,
        )
        return "comparison_id"
    return "positive_label"
```

---

## `_require_columns` — plotter-internal

```python
def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"scores missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )
```

---

## `plot_aurocs_over_time` — full locked signature

```python
def plot_aurocs_over_time(
    scores: pd.DataFrame,
    *,
    # ── what to plot ──────────────────────────────────────────────────────
    curve_col: str | None = None,
    # None → infer_curve_col(scores); requires positive_label + negative_label + comparison_id
    # explicit str → used directly; only that column is required to exist

    facet_row: str | None = None,
    facet_col: str | None = None,
    # None → "feature_set" if scores["feature_set"].nunique() > 1, else None

    # ── overlays — disabled with UserWarning if columns absent ────────────
    show_null_band: bool = False,
    # requires: auroc_null_mean, auroc_null_std

    show_significance: bool = True,
    # requires: pval

    sig_threshold: float = 0.01,
    show_chance_line: bool = True,

    # ── styling ───────────────────────────────────────────────────────────
    color_lookup: dict[str, str] | None = None,
    ylim: tuple[float, float] = (0.3, 1.05),
    xlim: tuple[float, float] | None = None,
    title: str = "AUROC over time",
    x_label: str = "Hours Post Fertilization (hpf)",
    y_label: str = "AUROC",
    style: StyleSpec | None = None,

    # ── output ────────────────────────────────────────────────────────────
    backend: Literal["plotly", "matplotlib", "both"] = "plotly",
    output_path: str | Path | None = None,
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
```

### Validation block (runs first, before any computation)

```python
    # 1. Always-required columns
    _require_columns(scores, {"time_bin_center", "auroc_obs"})

    # 2. Infer curve_col (requires pos/neg/comparison_id if curve_col is None)
    if curve_col is None:
        curve_col = infer_curve_col(scores)   # emits warning if ambiguous
    else:
        _require_columns(scores, {curve_col})

    # 3. Infer facet_col
    if facet_col is None:
        facet_col = "feature_set" if scores["feature_set"].nunique() > 1 else None

    # 4. Conditional overlays — warn once per call site, then disable
    if show_null_band:
        missing = {"auroc_null_mean", "auroc_null_std"} - set(scores.columns)
        if missing:
            warnings.warn(
                f"show_null_band=True requires {sorted(missing)}. "
                f"Disabling null band.",
                UserWarning, stacklevel=2,
            )
            show_null_band = False

    if show_significance:
        if "pval" not in scores.columns:
            warnings.warn(
                "show_significance=True requires 'pval'. "
                "Disabling significance markers.",
                UserWarning, stacklevel=2,
            )
            show_significance = False
```

Warning spam in loops: use `warnings.warn(..., stacklevel=2)` — the default Python
warning filter deduplicates by (message, category, module, lineno), so identical calls
in a loop produce one warning. No extra machinery needed.

---

## Object method — no logic, just sugar

```python
def plot_aurocs(
    self,
    *,
    curve_col: str | None = None,
    facet_col: str | None = None,
    **kwargs,
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
    from .viz.auroc_over_time import plot_aurocs_over_time
    return plot_aurocs_over_time(
        self.scores,
        curve_col=curve_col,
        facet_col=facet_col,
        **kwargs,
    )
```

No logic. All inference happens in `plot_aurocs_over_time`.

---

## Confusion plotter

```python
def plot_confusion(
    scores: pd.DataFrame,       # for time axis reference
    confusion: pd.DataFrame,    # from layers["confusion"]
    *,
    feature_set: str | None = None,
    time_range: tuple[float, float] | None = None,
    backend: Literal["plotly", "matplotlib", "both"] = "plotly",
    output_path: str | Path | None = None,
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
    """
    Required confusion columns: feature_set, comparison_id, time_bin_center,
    true_class, predicted_class, proportion.
    """

# Object convenience
def plot_confusion(self, **kwargs):
    conf = self.layers.get("confusion")
    if conf is None:
        raise KeyError(
            "No confusion layer available. "
            "Re-run classify() — confusion is saved automatically for multiclass runs."
        )
    from .viz.confusion import plot_confusion
    return plot_confusion(self.scores, conf, **kwargs)
```

---

## What gets deleted from `classification/viz/`

| File / symbol | Fate |
|---|---|
| `plot_feature_comparison_grid` | deleted — replaced by `facet_col="feature_set"` |
| `plot_multiclass_ovr_aurocs` | deleted — was a wrapper around `MulticlassOVRResults` |
| `plot_multiple_aurocs` | deleted — dict-of-DataFrames pattern is gone |
| `plot_auroc_with_null` | **kept** — useful low-level primitive for custom figures |
| `_auroc_col()` helper | deleted |
| `_time_col()` helper | deleted |
| `classification.py` | becomes a shim with `FutureWarning` on all exports |

`misclassification.py` and `trajectory.py` are unaffected — they take
`embryo_predictions` directly and never touch `scores`.

---

## Compatibility shim — `classification.py`

```python
# classification.py — DEPRECATED, will be removed in a future release

import warnings
from .auroc_over_time import plot_aurocs_over_time

def plot_feature_comparison_grid(*args, **kwargs):
    warnings.warn(
        "plot_feature_comparison_grid is deprecated. "
        "Use plot_aurocs_over_time(scores, facet_col='feature_set') instead.",
        FutureWarning, stacklevel=2,
    )
    raise NotImplementedError("Use plot_aurocs_over_time with facet_col='feature_set'.")

def plot_multiclass_ovr_aurocs(*args, **kwargs):
    warnings.warn(
        "plot_multiclass_ovr_aurocs is deprecated. "
        "Use plot_aurocs_over_time(results.scores) instead.",
        FutureWarning, stacklevel=2,
    )
    raise NotImplementedError("Use plot_aurocs_over_time(results.scores).")

def plot_multiple_aurocs(*args, **kwargs):
    warnings.warn(
        "plot_multiple_aurocs is deprecated. "
        "Use plot_aurocs_over_time(scores, curve_col=...) instead.",
        FutureWarning, stacklevel=2,
    )
    raise NotImplementedError("Use plot_aurocs_over_time.")
```

Temporary `MulticlassOVRResults` input shim inside `plot_aurocs_over_time`:

```python
# At top of plot_aurocs_over_time, before validation
if hasattr(scores, "comparisons") and isinstance(scores, MulticlassOVRResults):
    warnings.warn(
        "Passing MulticlassOVRResults to plot_aurocs_over_time is deprecated. "
        "Pass result.scores (a DataFrame) instead.",
        FutureWarning, stacklevel=2,
    )
    scores = scores.comparisons.rename(columns={
        "positive":       "positive_label",
        "negative":       "negative_label",
        "auroc_observed": "auroc_obs",
    })
    if "feature_set" not in scores.columns:
        scores = scores.copy()
        scores["feature_set"] = "default"
```

Use `isinstance(scores, MulticlassOVRResults)` not `hasattr` — avoids false matches.

---

## Final viz module layout

```
classification/viz/
  __init__.py           re-exports canonical + legacy shimmed symbols
  auroc_over_time.py    plot_aurocs_over_time + infer_curve_col + _require_columns
  confusion.py          plot_confusion (new)
  classification.py     FutureWarning shims only
  misclassification.py  unchanged (takes embryo_predictions df directly)
  trajectory.py         unchanged
```

---

## User experience summary

```python
# All modes — defaults just work
results.plot_aurocs()
# all-vs-rest + multi-feature → curve=positive_label, facet=feature_set
# pairwise + multi-feature    → warns, curve=comparison_id, facet=feature_set
# single feature              → curve=positive_label or comparison_id, no facet

# Override curve grouping
results.plot_aurocs(curve_col="comparison_id")
results.plot_aurocs(curve_col="negative_label")   # curves per reference

# Override faceting
results.plot_aurocs(facet_col=None)               # all curves on one panel
results.plot_aurocs(facet_row="negative_label", facet_col="feature_set")

# Overlays
results.plot_aurocs(show_null_band=True, show_significance=True, sig_threshold=0.05)

# Colors
results.plot_aurocs(color_lookup={"homo": "#B2182B", "het": "#F7B267"})

# Backend
results.plot_aurocs(backend="matplotlib", output_path="figures/auroc.png")
results.plot_aurocs(backend="both")   # → (plotly_fig, mpl_fig)

# Standalone — same function, takes DataFrame directly
from analyze.classification.viz import plot_aurocs_over_time
plot_aurocs_over_time(results.scores, facet_col="feature_set")

# Confusion
results.plot_confusion()
results.plot_confusion(feature_set="embedding", backend="matplotlib")
```

---

# Module Organisation — Locked

---

## File layout

```
classification/
  __init__.py                     ← public API surface (see below)

  # ── new implementation files ─────────────────────────────────────────────
  _classify.py                    classify() entry point
  _comparison_resolution.py       resolve_comparisons(), ResolvedComparison,
                                  all validators (_validate_comparison_side,
                                  _canonicalize_side, _check_labels_exist, etc.)
  _loop.py                        _run_classification_loop(), _bin_and_aggregate(),
                                  _build_binary_labels(), _collect_scores(),
                                  _collect_predictions(), _collect_confusion()
  _null.py                        NullDistributions dataclass + save/load
  _analysis.py                    ClassificationAnalysis, _LazyLayers,
                                  _validate_scores, _listify

  # ── legacy files (shimmed, not deleted) ──────────────────────────────────
  classification_test.py          FutureWarning shims:
                                    run_classification_test
                                    run_multiclass_classification_test
                                    extract_temporal_confusion_profile
  results.py                      FutureWarning shims:
                                    MulticlassOVRResults, ComparisonSpec
  classification_results.py       FutureWarning shim: ClassificationResults
  permutation_utils.py            unchanged (internal, not public)

  # ── viz ───────────────────────────────────────────────────────────────────
  viz/
    __init__.py                   exports canonical + shimmed legacy symbols
    auroc_over_time.py            plot_aurocs_over_time, infer_curve_col,
                                  _require_columns  (updated)
    confusion.py                  plot_confusion  (new)
    classification.py             FutureWarning shims only
    misclassification.py          unchanged
    trajectory.py                 unchanged

  # ── misclassification submodule ───────────────────────────────────────────
  misclassification/              unchanged internally
    __init__.py
    pipeline.py
    flagging.py
    io.py
    metrics.py
    null.py
    trajectory.py

  # ── tests ─────────────────────────────────────────────────────────────────
  tests/
    test_classify.py              new: classify() + ClassificationAnalysis
    test_comparison_resolution.py new: resolve_comparisons()
    test_null_distributions.py    new: NullDistributions save/load roundtrip
    test_classification_test.py   existing (keep until migration complete)
    test_classification_results.py existing (keep until migration complete)
    test_misclassification_*.py   unchanged
```

Leading underscore on implementation files (`_classify.py` etc.) signals
"internal — import from the package, not from these files directly."

---

## `__init__.py` — complete proposed surface

```python
"""
analyze.classification
======================

Public API for time-binned AUROC classification with permutation testing.

Primary interface
-----------------
    classify(df, ...)                 → ClassificationAnalysis
    ClassificationAnalysis.load(path) → ClassificationAnalysis

Legacy (deprecated, will be removed)
-------------------------------------
    run_classification_test           → use classify()
    MulticlassOVRResults              → use ClassificationAnalysis
    ClassificationResults             → use ClassificationAnalysis
"""

# ── Primary ───────────────────────────────────────────────────────────────────
from ._classify import classify
from ._analysis import ClassificationAnalysis

# ── Submodules ────────────────────────────────────────────────────────────────
from . import viz
from . import misclassification

# ── Misclassification pipeline (unchanged, stays public) ──────────────────────
from .misclassification import run_misclassification_pipeline, run_stage_geometry

# ── Legacy (FutureWarning fires on call, not on import) ───────────────────────
from .classification_test import (
    run_classification_test,
    run_multiclass_classification_test,
    extract_temporal_confusion_profile,
)
from .results import MulticlassOVRResults, ComparisonSpec
from .classification_results import ClassificationResults

__all__ = [
    # Primary
    "classify",
    "ClassificationAnalysis",
    # Submodules
    "viz",
    "misclassification",
    # Misclassification pipeline
    "run_misclassification_pipeline",
    "run_stage_geometry",
    # Legacy
    "run_classification_test",
    "run_multiclass_classification_test",
    "extract_temporal_confusion_profile",
    "MulticlassOVRResults",
    "ComparisonSpec",
    "ClassificationResults",
]
```

---

## Public surface — three tiers

| Tier | Symbols | Status |
|---|---|---|
| **Primary** | `classify`, `ClassificationAnalysis` | new, canonical |
| **Submodules** | `viz`, `misclassification` | unchanged |
| **Pipeline** | `run_misclassification_pipeline`, `run_stage_geometry` | unchanged, stays public |
| **Legacy** | `run_classification_test`, `MulticlassOVRResults`, `ClassificationResults`, `ComparisonSpec` | importable, `FutureWarning` on call |

---

## FutureWarning shim pattern

Warning fires on **call**, not on import — existing import lines don't break.

```python
# classification_test.py — legacy shim
import warnings

def run_classification_test(df, groupby, groups="all", reference="rest",
                             features="z_mu_b", **kwargs):
    warnings.warn(
        "run_classification_test() is deprecated and will be removed in a future release. "
        "Use classify() instead:\n"
        "  from analyze.classification import classify\n"
        "  results = classify(df, class_col=groupby, id_col=..., time_col=...,\n"
        "                     positive=groups, negative=reference, features={...})",
        FutureWarning, stacklevel=2,
    )
    from ._classify import classify as _classify
    # translate old kwargs → new kwargs and delegate
    ...
```

---

## User import surface

```python
# New — canonical
from analyze.classification import classify, ClassificationAnalysis

# Old — still works, warns on call
from analyze.classification import run_classification_test, MulticlassOVRResults

# Viz
from analyze.classification.viz import plot_aurocs_over_time   # standalone
results.plot_aurocs()                                           # object sugar

# Misclassification (unchanged)
from analyze.classification import run_misclassification_pipeline, run_stage_geometry
from analyze.classification.viz import plot_confusion_profile

# difference_detection shim (unchanged, adds classify + ClassificationAnalysis)
from analyze.difference_detection import classify   # FutureWarning on module import
```

---

## `analyze.difference_detection` update

```python
# analyze/difference_detection/__init__.py
import warnings
warnings.warn(
    "analyze.difference_detection is deprecated. "
    "Use analyze.classification instead.",
    FutureWarning, stacklevel=2,
)
from analyze.classification import (
    classify,
    ClassificationAnalysis,
    run_classification_test,
    MulticlassOVRResults,
    ClassificationResults,
    viz,
    misclassification,
    run_misclassification_pipeline,
    run_stage_geometry,
)
```
