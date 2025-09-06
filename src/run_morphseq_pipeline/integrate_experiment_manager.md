# Integrating ExperimentManager Into MorphSeq CLI (Codex Plnan)

This document outlines a focused rollout to make `src/build/pipeline_objects.py` the first‑class orchestrator behind the unified CLI, enabling robust end‑to‑end processing of one or many experiments.

## Objectives
- Unify orchestration: Drive Build01→SAM2→Build03→Build04→Build05→Build06 via a single CLI entry.
- Centralize state: Use `Experiment` flags/timestamps for resumability and cohort runs.
- Standardize IDs: Use canonical snip_id `embryo_id_t####` with parsing_utils as the single source of truth.
- Keep steps pluggable: Thin adapters call existing step functions; no hard rewrites.

## Scope
- Add a `pipeline` subcommand to `src/run_morphseq_pipeline/cli.py` that delegates to `ExperimentManager`.
- Extend `Experiment` with thin wrappers for modern steps (SAM2 + Build03–06).
- Extend `ExperimentManager` with façade methods for each step and `e2e`.
- Add `needs_*` detection for SAM2/Build03/04/05/06 artifacts.

## Architecture
- Source of truth for snip_ids: `segmentation_sandbox/scripts/utils/parsing_utils.py`
  - Canonical builder: `build_snip_id(embryo_id, frame_number) → embryo_id_t####`
  - Validators: `is_snip_t_style(...)`, `validate_snip_t_style(...)`
- Orchestration mapping:
  - `Experiment.run_sam2()` → `src/run_morphseq_pipeline/steps/run_sam2.py`
  - `Experiment.run_build03()` → `src/build/build03A_process_images.py`
  - `Experiment.run_build04()` → `src/build/build04_perform_embryo_qc.py`
  - `Experiment.run_build05()` → `src/build/build05_prepare_training_data.py` (existing entry)
  - `Experiment.run_build06()` → `src/run_morphseq_pipeline/services/gen_embeddings.build_df03_with_embeddings`

## CLI Changes
- New subcommand: `pipeline`
  - Global flags: `--data-root`, `--experiments exp1,exp2 | --later-than YYYYMMDD [--earlier-than YYYYMMDD]`, `--force-update`, `--workers`.
  - Actions (pick one): `report | export | stitch | sam2 | build03 | build04 | build05 | build06 | e2e`.
  - Build06 options: `--model-name`, `--enable-env-switch`, `--export-analysis-copies`, `--overwrite`.
  - Build05 options: `--train-name`, `--overwrite`.

Examples:
- `python -m src.run_morphseq_pipeline.cli pipeline --data-root <root> report`
- `python -m src.run_morphseq_pipeline.cli pipeline --data-root <root> sam2 --experiments 20250529_30hpf_ctrl_atf6`
- `python -m src.run_morphseq_pipeline.cli pipeline --data-root <root> e2e --later-than 20250101 --workers 8`

## Experiment Methods (new)
- `run_sam2(self, workers: int = 8, **kwargs)`
  - Calls SAM2 wrapper; writes CSV/masks under `sam2_pipeline_files/...` in `data_root`.
- `run_build03(self)`
  - Calls Build03A; SAM2 aware paths already supported.
- `run_build04(self)`
  - Calls QC + stage inference to produce df02.
- `run_build05(self, train_name: str, overwrite: bool)`
  - Prepares training snips; writes under `training_data/bf_embryo_snips/<date>`.
- `run_build06(self, model_name: str, export_analysis: bool, overwrite: bool, enable_env_switch: bool)`
  - Calls `build_df03_with_embeddings(...)` and enforces Python 3.9 (env switch opt‑in via `MSEQ_ENABLE_ENV_SWITCH`).

Each method is decorated with `@record("step")` to update flags/timestamps and persist state.

## `needs_*` Detection (artifacts)
- `needs_sam2`: missing `sam2_pipeline_files/sam2_expr_files/sam2_metadata_<date>.csv`.
- `needs_build03`: missing `metadata/combined_metadata_files/embryo_metadata_df01.csv`.
- `needs_build04`: missing `metadata/combined_metadata_files/embryo_metadata_df02.csv`.
- `needs_build05`: missing `training_data/bf_embryo_snips/<date>/`.
- `needs_build06`: missing `metadata/combined_metadata_files/embryo_metadata_df03.csv` (or model‑tagged copies under data_root).

## ExperimentManager Façades
- `sam2_experiments(...)` → `_run_step("run_sam2", "needs_sam2", extra_filter=…)`
- `build03_experiments(...)` → `_run_step("run_build03", "needs_build03")`
- `build04_experiments(...)` → `_run_step("run_build04", "needs_build04")`
- `build05_experiments(...)` → `_run_step("run_build05", "needs_build05")`
- `build06_experiments(...)` → `_run_step("run_build06", "needs_build06")`
- `e2e_experiments(...)`: orchestrates all in order using the above (respect `needs_*` and `--force-update`).

## Environment & Performance
- Python 3.9 enforced for Build06: default env `mseq_pipeline_py3.9`. Opt‑in auto switch via `--enable-env-switch` (sets `MSEQ_ENABLE_ENV_SWITCH=1`).
- Workers: pass `Experiment.num_cpu_workers` or CLI `--workers` to steps that support parallelism.
- GPU: `Experiment.has_gpu/gpu_names` available for step‑specific branching (optional).

## Data Paths (canonical)
- Raw: `<root>/raw_image_data/<Keyence|YX1>/<date>/`
- SAM2 masks/CSV: `<root>/sam2_pipeline_files/.../sam2_metadata_<date>.csv`
- Build03 df01: `<root>/metadata/combined_metadata_files/embryo_metadata_df01.csv`
- Build04 df02: `<root>/metadata/combined_metadata_files/embryo_metadata_df02.csv`
- Build05 snips: `<root>/training_data/bf_embryo_snips/<date>/`
- Build06 df03: `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv`

## Validation & Testing
- Unit: add light tests for manager filters/needs detection on temp dirs.
- Functional: run `pipeline report/stitch/sam2/build03/build04/build05/build06` on a small experiment.
- E2E: exercise `pipeline e2e --experiments <date>` including Build06 env switch path.

## Rollout Plan
1) Implement CLI `pipeline` subcommand + wiring (no behavior change for existing subcommands).
2) Add `Experiment.run_*` wrappers (SAM2, Build03–06) with `@record()`.
3) Add `needs_*` detection for new steps.
4) Document in `README.md` with usage examples.
5) Smoke test on a single experiment, then a date window.
6) Optional: add CI jobs for `pipeline report` and a dry-run smoke (`--dry-run` in Build06).

## Risks & Mitigations
- Path mismatches: Use existing path helpers in Build03/04/05/06 to avoid hardcoding; log discovered paths.
- Environment drift: Hard‑fail Build06 on Python ≠ 3.9 unless `--enable-env-switch` is set.
- Long‑running steps: Surface `--workers`, print progress summaries per experiment.

## Open Questions
- Do we want `pipeline e2e` to skip heavy legacy UNets by default and rely on SAM2? (Current CLI supports SAM2; legacy runs remain available.)
- Should we add a metadata registry per experiment under `<root>/metadata/experiments/<date>.json` for richer status? (Flags/timestamps already persisted; registry could add provenance.)

---

Implementation is intentionally thin and composable: `Experiment` shells call the step functions you already use in the CLI, while `ExperimentManager` provides cohort selection and resumability. This keeps refactors safe and local while delivering a "true pipeline" operator experience.

---

# Claude Sonnet's Plan - Comprehensive ExperimentManager Integration
**Transform CLI from individual commands to true pipeline orchestration**

**Note**: This is a more comprehensive alternative approach focusing on enterprise-grade orchestration features.

## Executive Summary

The current CLI system executes individual pipeline commands but lacks **true orchestration**. The existing `pipeline_objects.py` contains a sophisticated `ExperimentManager` system for batch operations and dependency tracking that is currently unused by the CLI.

**Goal**: Integrate `ExperimentManager` to provide:
- **Automatic experiment discovery** from data structure
- **Dependency tracking** and state persistence
- **Batch processing** across multiple experiments  
- **Resource optimization** with automatic GPU/CPU detection
- **True pipeline orchestration** with resume capabilities

## Current Architecture Problems

### CLI Command Issues
1. **Manual experiment specification**: Users must know which experiments exist
2. **No dependency tracking**: Steps run regardless of input freshness  
3. **No state persistence**: Cannot resume interrupted pipelines
4. **Individual execution**: Must run each experiment separately
5. **Resource blindness**: No automatic detection of available resources

### Missing Orchestration Features
- **No batch discovery**: Cannot find all experiments automatically
- **No status reporting**: Cannot see pipeline state across experiments
- **No intelligent resumption**: Cannot skip completed steps
- **No resource management**: Manual CPU/GPU worker specification
- **No progress visibility**: No unified view of pipeline progress

## Solution: Advanced ExperimentManager Integration

### Integration Plan

#### **Phase 1: Basic Integration (2-3 days)**

**1.1 Enhanced CLI Commands**
Add orchestration mode to existing commands:

```python
# New CLI arguments for all commands
parser.add_argument("--use-orchestration", action="store_true",
                   help="Use ExperimentManager for dependency tracking")
parser.add_argument("--discover-experiments", action="store_true", 
                   help="Auto-discover experiments from data structure")
parser.add_argument("--batch-mode", action="store_true",
                   help="Process multiple experiments in batch")
```

**1.2 ExperimentManager CLI Wrapper**
```python
# New file: src/run_morphseq_pipeline/orchestration.py
class CLIOrchestrator:
    def __init__(self, data_root: Path):
        self.manager = ExperimentManager(data_root)
        
    def run_with_orchestration(self, command: str, **kwargs):
        """Execute CLI command with ExperimentManager orchestration"""
        if kwargs.get('discover_experiments'):
            experiments = self.manager.discover_experiments()
        else:
            experiments = kwargs.get('experiments', [])
            
        for exp in experiments:
            if self.should_run_step(exp, command):
                self.execute_step(exp, command, **kwargs)
                exp.record_step(command)
```

**1.3 Extend Experiment Class for SAM2/Build06**
```python
# Extend pipeline_objects.py Experiment class
class Experiment:
    # Add new pipeline steps
    @property
    def needs_sam2(self) -> bool:
        last_run = self.timestamps.get("sam2", 0)
        newest = newest_mtime(self.stitch_ff_path, PATTERNS["stitch"])
        return newest >= last_run
    
    @property  
    def needs_build06(self) -> bool:
        last_run = self.timestamps.get("build06", 0)
        newest = newest_mtime(self.meta_path_embryo, PATTERNS["meta"])
        return newest >= last_run
    
    # Add new pipeline methods
    @record("sam2")
    def run_sam2_pipeline(self, **kwargs):
        from ..run_morphseq_pipeline.steps.run_sam2 import run_sam2
        return run_sam2(root=self.data_root, exp=self.date, **kwargs)
        
    @record("build06")  
    def run_build06_pipeline(self, **kwargs):
        from ..run_morphseq_pipeline.steps.run_build06 import run_build06
        return run_build06(data_root=self.data_root, experiments=[self.date], **kwargs)
```

#### **Phase 2: New Orchestration Commands (2-3 days)**

**2.1 Status Command**
```bash
# Show pipeline state across all experiments
python -m src.run_morphseq_pipeline.cli status --data-root /data

# Output:
# Experiment Status Report
# ========================
# 20250529_24hpf_ctrl_atf6: [✅ build01] [✅ build02] [❌ sam2] [❌ build03] [❌ build04] [❌ build06] 
# 20250529_30hpf_ctrl_atf6: [✅ build01] [✅ build02] [✅ sam2] [✅ build03] [❌ build04] [❌ build06]
# 20250612_48hpf_heat_atf6: [✅ build01] [❌ build02] [❌ sam2] [❌ build03] [❌ build04] [❌ build06]
# 
# Summary: 3 experiments discovered, 1 ready for build04
```

**2.2 Discover Command** 
```bash
# Auto-discover and register new experiments
python -m src.run_morphseq_pipeline.cli discover --data-root /data

# Output:  
# Scanning raw_image_data/ for experiments...
# Found: 20250529_24hpf_ctrl_atf6 (Keyence, 2.1GB)
# Found: 20250529_30hpf_ctrl_atf6 (Keyence, 1.8GB) 
# Found: 20250612_48hpf_heat_atf6 (YX1, 3.2GB)
# 
# Registered 3 experiments in metadata/experiments/
```

**2.3 Orchestrate Command - The Power Feature**
```bash
# Run complete pipeline on all discovered experiments
python -m src.run_morphseq_pipeline.cli orchestrate \
  --data-root /data \
  --run-sam2 \
  --train-name batch_run_20250906

# Run pipeline on specific experiments only  
python -m src.run_morphseq_pipeline.cli orchestrate \
  --data-root /data \
  --experiments exp1,exp2,exp3 \
  --run-sam2 \
  --skip-build01

# Resume interrupted pipeline (skips completed steps)
python -m src.run_morphseq_pipeline.cli orchestrate \
  --data-root /data \
  --resume \
  --run-sam2
```

**Orchestrate Logic**:
```python
def orchestrate_pipeline(experiments, **options):
    for exp in experiments:
        print(f"\n🧪 Processing {exp.date}")
        
        # Build01 (if needed and not skipped)
        if not options['skip_build01'] and exp.needs_export:
            exp.export_images()
            
        # Build02 (if needed and not skipped)  
        if not options['skip_build02'] and exp.needs_segment:
            exp.segment_images()
            
        # SAM2 (if requested and build01 complete)
        if options['run_sam2'] and exp.flags['stitch'] and exp.needs_sam2:
            exp.run_sam2_pipeline(**sam2_options)
            
        # Build03 (if build02 or sam2 complete)
        if (exp.flags['segment'] or exp.flags['sam2']) and exp.needs_build03:
            exp.run_build03_pipeline(**build03_options)
            
        # Continue with build04, build06...
```

#### **Phase 3: Advanced Features (1-2 days)**

**3.1 Resource-Aware Scheduling**
```python
class ResourceAwareOrchestrator(CLIOrchestrator):
    def optimize_workers(self, step: str) -> dict:
        """Auto-select optimal worker counts based on resources"""
        if step in ["build02", "sam2"] and self.manager.has_gpu:
            return {"num_workers": self.manager.num_cpu_workers // 2}  # Share CPU with GPU
        else:
            return {"num_workers": self.manager.num_cpu_workers}
            
    def schedule_experiments(self, experiments: List[Experiment]) -> List[Experiment]:
        """Sort experiments by resource requirements and data size"""
        return sorted(experiments, key=lambda e: e.data_size, reverse=True)
```

**3.2 Progress Reporting**
```python  
def run_orchestrated_pipeline():
    with ProgressReporter() as reporter:
        for i, exp in enumerate(experiments):
            reporter.update_experiment(i, len(experiments), exp.date)
            
            for step in pipeline_steps:
                if exp.should_run_step(step):
                    reporter.update_step(step, "running")
                    exp.run_step(step)  
                    reporter.update_step(step, "complete")
                else:
                    reporter.update_step(step, "skipped")
```

**3.3 Pipeline Validation**
```python
def validate_pipeline_readiness(experiments: List[Experiment]) -> Dict[str, List[str]]:
    """Check if experiments are ready for pipeline execution"""
    issues = {"missing_inputs": [], "disk_space": [], "dependencies": []}
    
    for exp in experiments:
        if not exp.raw_path or not exp.raw_path.exists():
            issues["missing_inputs"].append(f"{exp.date}: Raw data not found")
            
        if not exp.meta_path or not exp.meta_path.exists():
            issues["missing_inputs"].append(f"{exp.date}: Metadata not found") 
            
        # Check disk space requirements
        estimated_size = exp.estimate_pipeline_output_size()
        if get_free_space(exp.data_root) < estimated_size:
            issues["disk_space"].append(f"{exp.date}: Need {estimated_size}GB free space")
    
    return issues
```

## Usage Examples

### Current vs. Future Workflow

**Current (Manual)**:
```bash
# User must manually track experiments and run individually
python -m src.run_morphseq_pipeline.cli build01 --exp exp1 --microscope keyence --data-root /data
python -m src.run_morphseq_pipeline.cli build02 --data-root /data  
python -m src.run_morphseq_pipeline.cli sam2 --exp exp1 --data-root /data
python -m src.run_morphseq_pipeline.cli build03 --exp exp1 --data-root /data
python -m src.run_morphseq_pipeline.cli build04 --data-root /data
python -m src.run_morphseq_pipeline.cli build06 --morphseq-repo-root /repo --data-root /data --experiments exp1

# Repeat for each experiment...
```

**Future (Orchestrated)**:
```bash
# Automatic discovery and batch processing
python -m src.run_morphseq_pipeline.cli orchestrate --data-root /data --run-sam2 --train-name batch_20250906

# Or resume an interrupted pipeline
python -m src.run_morphseq_pipeline.cli orchestrate --data-root /data --resume

# Or check status across all experiments  
python -m src.run_morphseq_pipeline.cli status --data-root /data
```

## Benefits

### **For Users**
- **Zero experiment tracking**: Automatic discovery and management
- **Intelligent execution**: Only run steps when inputs have changed  
- **Batch processing**: Process dozens of experiments with one command
- **Resume capabilities**: Never lose progress from interruptions
- **Resource optimization**: Automatic CPU/GPU worker selection

### **For Development**  
- **Clean architecture**: Separation of orchestration from individual steps
- **Extensible design**: Easy to add new pipeline steps
- **State management**: Robust tracking of pipeline progress
- **Error resilience**: Graceful handling of partial failures
- **Testing friendly**: Mock orchestration for unit tests

### **For Operations**
- **Progress visibility**: Clear status reporting across experiments
- **Resource management**: Optimal utilization of compute resources  
- **Error diagnosis**: Clear identification of failed steps and reasons
- **Scalability**: Handle large experiment batches efficiently

## Implementation Timeline

### Week 1: Core Integration
- **Day 1-2**: Extend `Experiment` class with SAM2/Build06 methods
- **Day 3-4**: Create `CLIOrchestrator` wrapper class
- **Day 5**: Add `--use-orchestration` mode to existing commands

### Week 2: New Commands  
- **Day 1-2**: Implement `status` and `discover` commands
- **Day 3-5**: Implement powerful `orchestrate` command with dependency logic

### Week 3: Advanced Features
- **Day 1-2**: Resource-aware scheduling and progress reporting
- **Day 3**: Pipeline validation and error handling
- **Day 4-5**: Testing and documentation

## Key Differences from Focused Plan

1. **More comprehensive orchestration**: Full `status`, `discover`, and `orchestrate` commands vs focused `pipeline` subcommand
2. **Advanced features**: Resource-aware scheduling, progress reporting, pipeline validation
3. **Enterprise focus**: Designed for large-scale batch processing with dozens of experiments
4. **Gradual migration**: Opt-in orchestration features vs integrated approach
5. **Extensive validation**: Pre-flight checks and error diagnosis capabilities

This comprehensive approach provides enterprise-grade pipeline orchestration capabilities while maintaining backward compatibility with existing workflows.
