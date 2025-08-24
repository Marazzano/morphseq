# MVP-First Reorganized Plan (UOT → Clustering → ODE)

**Goal:** stand up a _working_ end‑to‑end pipeline using **`bias_mode='none'`** first, then layer in biasing.  
This makes debugging easier and gives you a baseline to compare subsequent improvements against.
Enhance the existing neural ODE in src/flux/ that models developmental dynamics by integrating Unbalanced Optimal Transport (UOT) with intelligent minibatching and trajectory-aware biasing.
Embryo Data (130k images, sparse tracking)
    ↓
Time-Weighted Spectral Clustering [Task 02]
    ↓
Smart Minibatch Sampling (adjacent time clusters)
    ↓
Trajectory-Biased UOT [Tasks 01 + 03]
    ↓
Teacher Velocities (v̂ᵢ)
    ↓
Neural ODE Training (ż = f_θ(z,t))

---

## Phase 0: Codebase Discovery (CRITICAL - Do This First!)
Agent Instructions: Before implementing ANYTHING, spend time understanding the existing system:
A. Find and Document the ODE Implementation
bash# Search in src/flux/ for ODE-related files
find src/flux -name "*.py" | xargs grep -l "ode\|ODE\|neural_ode\|dynamics"
find src/flux -name "*ode*.py"
find src/flux -name "*train*.py"
Document:

 Location of ODE model class (likely models/ode.py or similar)
 Current loss function implementation
 Training script entry point
 Data loading pipeline (how embeddings + timestamps are loaded)
 Current hyperparameter handling (config files? argparse? hydra?)

B. Understand Data Format and Hierarchy
bash# Look for data loading and parsing utilities
find src/flux -name "*utils*.py" | xargs grep -l "parse\|embryo\|snip"
find . -name "*.csv" | head -5  # Find example data files
Critical: Look at parsing utils to understand:

 Data hierarchy: experiment_id → embryo_id → snip_id → image_id
 How embryo trajectories are represented
 CSV schema (columns for embeddings, time, IDs)
 Which columns contain VAE latents (z_) vs biological components (bc_)

C. Analyze Example Data
python# Agent should run this to understand data structure
import pandas as pd
import glob

# Find an embryo data CSV
csv_files = glob.glob("**/embryo*.csv", recursive=True)
if csv_files:
    df = pd.read_csv(csv_files[0], nrows=100)
    print("Columns:", df.columns.tolist())
    print("\nData types:", df.dtypes)
    print("\nTime column values:", df['predicted_stage_hpf'].describe() if 'predi

## Phase 1 — Core Pipeline (MVP)

### Task 01 — Implement a Basic UOT Function
- **Target:** `src/flux/transport/minibatch_uot.py`
- **Function:** `compute_minibatch_uot(...)`
- **Constraint:** start with **no biasing** (`bias_mode='none'` implicit).
- **Inputs (min):** `x_t, x_tp1, t, tp1, cost_fn, eps, reg, rng`
- **Outputs:** `(P, v_teacher)` where:
  - `P`: transport plan (minibatch × minibatch)
  - `v_teacher`: teacher velocities aligned to `x_t`
- **Success:** runs without errors; returns valid shapes; non‑NaN; numerically stable on a tiny smoke test.

### Task 02 — Time‑Weighted Spectral Clustering
- **Target:** `src/flux/sampling/spectral.py`
- **Function:** `time_weighted_spectral_clustering(...)`
- **Objective:** produce clusters that are **morphologically similar** _and_ **temporally adjacent**.
- **Scope:** implement core logic; defer large hyper‑parameter sweeps.
- **Success:** qualitative plots show tight clusters along time; sampler returns balanced batches.

### Task 03 — Integrate and Train the ODE
- **Targets:**
  - `src/flux/sampling/clustered_minibatch_sampler.py` (new)
  - ODE training loop (e.g., `train_uot_ode.py` or `src/flux/train/ode.py`)
- **Steps:**
  1. Build `ClusteredMinibatchSampler` from spectral results.
  2. In the training loop, draw a minibatch → call **basic** `compute_minibatch_uot`.
  3. Use extracted `v_teacher` to train the Neural ODE.
- **Success:** end‑to‑end run completes; loss decreases over epochs on a small subset.

---

## Phase 2 — Biasing Modes (after MVP is green)

### Task 04 — Implement `lower_bound` Biasing
- **Modify:** `compute_minibatch_uot(..., bias_mode='lower_bound', bias_kwargs=...)`
- **Logic:** apply self‑edge lower bounds using `m_hat` and `s_hat` (see `self_edge_bounds.md`).
- **Success:** training runs; compare convergence + smoothness vs MVP baseline.

### Task 05 — Add Additional Biasing Modes (e.g., `z_hard`)
- Ship one at a time with **small** diffs.
- Keep inference identical to MVP unless a flag is on.
- **Success:** each mode is numerically stable and improves a target metric _or_ is easy to revert.

---

## Deliverables (this pass)
- This file + four module plans:
  1. `01_minibatch_uot_MVP.md`
  2. `02_time_weighted_spectral_clustering.md`
  3. `03_training_loop_integration.md`
  4. `04_biasing_modes_lower_bound.md`
  5. `05_biasing_modes_additional.md`
  6. `99_checklist_and_milestones.md`

**Notes**
- Keep all deltas traceable; prefer pure‑function style; add deterministic seeds for regression tests.
- Use tiny synthetic fixtures for smoke testing each layer.