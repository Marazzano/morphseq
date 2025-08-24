# Task 03 — Integration & Neural ODE Training (MVP)

**New:** `src/flux/sampling/clustered_minibatch_sampler.py`  
**Train:** `src/flux/train/ode.py` (or `train_uot_ode.py` script)

## `ClusteredMinibatchSampler` (proposed)
```python
class ClusteredMinibatchSampler:
    def __init__(self, X, times, labels, batch_size, rng=None):
        self.X = X; self.times = times; self.labels = labels
        self.batch_size = batch_size; self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        # sample a cluster → choose consecutive time windows where possible
        # return indices for x_t and x_tp1 (aligned temporally)
        ...
```

## Training Loop (skeleton)
```python
for step, (idx_t, idx_tp1) in enumerate(sampler):
    x_t, x_tp1 = X[idx_t], X[idx_tp1]
    P, v_teacher = compute_minibatch_uot(
        x_t, x_tp1, t=times[idx_t], tp1=times[idx_tp1],
        cost_fn=pairwise_cost, eps=eps, reg=reg, bias_mode="none"
    )
    # Teacher → student (Neural ODE) loss
    loss = ode_model.loss(x_t, v_teacher)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

## Metrics
- Train loss curve (should decrease).
- Velocity alignment (cosine between model v and `v_teacher`).
- Optional validation on held‑out sequences.

## Reference / Current Code to Preserve
```python
# Agent should run this to understand data structure
import pandas as pd
import glob

# Find an embryo data CSV
csv_files = glob.glob("**/embryo*.csv", recursive=True)
if csv_files:
    df = pd.read_csv(csv_files[0], nrows=100)
    print("Columns:", df.columns.tolist())
    print("\nData types:", df.dtypes)
    print("\nTime column values:", df['predicted_stage_hpf'].describe() if 'predicted_stage_hpf' in df else "No time column found")
```