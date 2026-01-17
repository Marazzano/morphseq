# Next Steps for Trajectory Analysis Reorganization

**Branch**: `feat/traj-reorg`
**Working Directory**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/trajectory_analysis`
**Last Completed**: Phase 4 - Clustering Subpackage ✓

## Quick Start for Next Session

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
git checkout feat/traj-reorg
cd src/analyze/trajectory_analysis
```

## What's Been Done ✓

- ✅ **Phase 0**: Directory structure created
- ✅ **Phase 1**: Config consolidation (merged plot_config.py → config.py)
- ✅ **Phase 2**: Distance/Utilities/IO subpackages created
- ✅ **Phase 3**: QC subpackage created (outlier detection + filtering)
- ✅ **Phase 4**: Clustering subpackage created (6 modules, 23 functions)

**Recent Commits**:
- `1340e1e8` - Phase 4: Create clustering subpackage with 6 modules
- `cab05ae1` - Phase 4 follow-up: Address backward-compat and import issues

## Phase 5: Viz Restructure (START HERE)

This is the **current priority**. Phase 5 reorganizes visualization code.

### Files to Move

#### To `viz/` (top level):
1. **dendrogram.py** → `viz/dendrogram.py`
   - Currently imports: `.config`, `.plotting`, `.genotype_styling`
   - Will need: `..config`, `.plotting.core`, `.styling`

2. **genotype_styling.py** → `viz/styling.py`
   - Currently imports: `.config`
   - Will need: `..config`

#### To `viz/plotting/` (with renames):
3. **plotting.py** → `viz/plotting/core.py`
   - Main plotting functions
   - Currently imports: `.utilities`, `.genotype_styling`
   - Will need: `...utilities`, `..styling`

4. **facetted_plotting.py** → `viz/plotting/faceted.py`
   - Faceted trajectory plots
   - Currently imports: `.config`, `.utilities.trajectory_utils`, `.genotype_styling`
   - Will need: `...config`, `...utilities.trajectory_utils`, `..styling`

5. **plotting_3d.py** → `viz/plotting/plotting_3d.py`
   - Keep name (3D plotting)
   - Currently imports: `.config`
   - Will need: `...config`

6. **pair_analysis/** → Keep in place, but update its imports

### Step-by-Step Tasks for Phase 5

1. **Move files using git mv**:
   ```bash
   git mv dendrogram.py viz/
   git mv genotype_styling.py viz/styling.py
   git mv plotting.py viz/plotting/core.py
   git mv facetted_plotting.py viz/plotting/faceted.py
   git mv plotting_3d.py viz/plotting/
   ```

2. **Update relative imports** in each moved file:
   - In `viz/dendrogram.py`: `.config` → `..config`, etc.
   - In `viz/styling.py`: `.config` → `..config`
   - In `viz/plotting/core.py`: `.utilities` → `...utilities`, `.genotype_styling` → `..styling`
   - In `viz/plotting/faceted.py`: `.config` → `...config`, `.utilities` → `...utilities`, `.genotype_styling` → `..styling`
   - In `viz/plotting/plotting_3d.py`: `.config` → `...config`

3. **Create `viz/__init__.py`** - Export key visualization functions:
   ```python
   # Dendrogram
   from .dendrogram import generate_dendrograms, plot_dendrogram_with_categories, ...

   # Styling
   from .styling import get_color_for_genotype, build_genotype_style_config, ...

   # Plotting (re-export from submodules)
   from .plotting import plot_cluster_trajectories_df, plot_multimetric_trajectories, ...
   ```

4. **Create `viz/plotting/__init__.py`** - Export from all plotting modules:
   ```python
   from .core import plot_cluster_trajectories_df, plot_membership_trajectories_df, ...
   from .faceted import plot_trajectories_faceted, plot_multimetric_trajectories
   from .plotting_3d import plot_3d_scatter
   ```

5. **Update main `__init__.py`**:
   - Change `from .plotting import ...` → `from .viz.plotting import ...`
   - Change `from .genotype_styling import ...` → `from .viz.styling import ...`
   - Change `from .dendrogram import ...` → `from .viz import ...`
   - Change `from .facetted_plotting import ...` → `from .viz.plotting import ...`

6. **Update `pair_analysis/__init__.py`**:
   - Change `from ..genotype_styling import ...` → `from ..viz.styling import ...`
   - Change `from ..facetted_plotting import ...` → `from ..viz.plotting.faceted import ...`

7. **Create backward-compat shims** at old locations:
   - `dendrogram.py` → deprecation wrapper
   - `genotype_styling.py` → deprecation wrapper
   - `plotting.py` → deprecation wrapper
   - `facetted_plotting.py` → deprecation wrapper
   - `plotting_3d.py` → deprecation wrapper

8. **Create `tests/test_phase5.py`**:
   - Test viz subpackage imports
   - Test viz.plotting imports
   - Test backward compatibility shims
   - Test pair_analysis imports
   - Test main __init__.py imports

9. **Run tests**:
   ```bash
   MORPHSEQ_REPO_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq \
       python src/analyze/trajectory_analysis/tests/test_phase5.py
   ```

10. **Update `docs/PROGRESS.md`** with Phase 5 results

11. **Commit with test results**:
    ```bash
    git commit -m "Phase 5: Create viz subpackage with plotting and styling modules"
    ```

### Important Notes

- The `viz/` and `viz/plotting/` directories already exist with empty `__init__.py` files
- Use the Phase 3 and Phase 4 test files as templates for creating `test_phase5.py`
- Follow the same pattern: move → update imports → create __init__ → update main → shims → test
- Many files import from genotype_styling and facetted_plotting, so be thorough with updates

### Key Files to Read First

1. `docs/PROGRESS.md` - Full progress tracker
2. `docs/IMPLEMENTATION_PLAN.md` - Detailed plan for all phases
3. `tests/test_phase3.py` and `tests/test_phase4.py` - Test pattern examples
4. `genotype_styling.py` - Check current imports (line 12)
5. `facetted_plotting.py` - Check current imports (lines 26, 31)
6. `pair_analysis/__init__.py` - Will need updates

## After Phase 5

**Phase 6**: Documentation (create WORKFLOW.md, EXAMPLES.md)
**Phase 7**: Final __init__.py polish and version bump to 0.3.0
**Phase 8**: Comprehensive testing with test_all_imports.py

## Testing Commands

```bash
# Set repo root
export MORPHSEQ_REPO_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq

# Run Phase 5 tests
python src/analyze/trajectory_analysis/tests/test_phase5.py

# Check all imports still work
cd results/mcolon/20251219_b9d2_phenotype_extraction/
python -c "from src.analyze.trajectory_analysis import *; print('All imports OK')"
```

## Git Strategy

Continue using `git mv` to preserve history. Create one commit for Phase 5 with comprehensive test results in the commit message, following the pattern from Phase 3 and Phase 4.
