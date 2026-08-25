# Tech Debt: No General Pattern for Routing Models Into the Pipeline

**Status:** known gap, mdcolon 2026-06-26. Recorded as an honest tech debt entry — not a blocker
for current work, but named so the design problem is understood before we wire the second and third
model. Deferred deliberately; mdcolon will not solve the general case now.

---

## The gap

The pipeline runs **model inference** at several stages (detection/GroundingDINO, segmentation/SAM2,
the legacy VAE encode, and any future model — stage prediction, classifiers, label-transfer,
phenotype models). There is **no single, principled pattern for how a model is resolved, configured,
and invoked**. Each model that has been wired so far was wired ad hoc:

- the legacy VAE crosses a Python-3.9 interpreter boundary (`MODEL_RUN`, `model_python_executable` /
  `model_python_env` in `env.yaml`) because its pickles do not survive 3.10;
- detection/segmentation run under the normal 3.10 `RUN` env;
- weights live under `models_root` (`env.yaml`), but the *layout beneath that root* and the
  *config a given model needs* are model-specific and currently improvised per model.

**Every model works differently.** Different weight formats, different interpreter/env requirements,
different config shapes, different device assumptions, different ways a model resolves which weights
to load (some depend on experiment/microscope/channel/run context, not a fixed path). There is no
contract that says "here is how you declare a new model, here is how the pipeline finds and runs it."

---

## Why it is hard (the real problem)

A model is not a single artifact path the way a CSV is. Resolving "which model, which weights, which
config, run it how" can depend on context the path registry does not capture:

- **Resolution may be conditional** — the right weights/config can vary by experiment, microscope,
  channel, product, or run intent. One static path under `models_root` cannot express that.
- **Configs differ per model** — there is no single schema; each model needs its own knobs.
- **Execution differs per model** — interpreter/env, device, batch vs per-well, load-once vs
  reload, even whether it crosses a process boundary (the VAE 3.9 case).

mdcolon's framing: "I can do my best, but I can't handle every single path." The pipeline cannot
hard-code a resolver for every model, but it also cannot leave each model a bespoke snowflake. We
need a **middle line**: a small, fixed routing contract that gives each model enough flexibility to
declare its own resolver + config + execution, without every model reinventing the seam.

---

## The interim posture (what we do FOR NOW)

Assume a **single `models_root` "model route"**: every model lives nicely under that root, addressed
by a model key, with weights/config beneath it. This is the minimum that lets the VAE encode and the
detection/segmentation models work today. It is explicitly a placeholder — it does **not** solve
conditional resolution or per-model config schemas.

> Today's reality: `MODELS_DIR = Path(env["paths"]["models_root"])`, plus the VAE's
> `model_python_executable`/`model_python_env` interpreter escape hatch. That is the whole "pattern"
> right now — one root + one interpreter override. Good enough for the models wired so far; not a
> general contract.

---

## What the eventual pattern needs (sketch, NOT a decision)

When we do tackle this, the target is a **per-model routing contract** — the model analogue of
"add a stage = one PIPELINE_STEPS row + one compute fn + one verb + one templated rule." Each model
declares:

- **a resolver** — how to find its weights/config given run context (so resolution can be
  conditional, not a fixed path). Different models get different **resolver types**; the pipeline
  routes to the right one by model kind.
- **its own config** — a model-scoped config block, not forced into one shared schema.
- **its execution profile** — interpreter/env, device, batch/per-well, load-once vs reload, process
  boundary.

The pipeline owns the **routing seam** (a registry of models → resolver kind → config → execution);
each model owns the **specifics behind its resolver**. That keeps the "two kingdoms" discipline: the
orchestration knows *how to route to a model*, the model package knows *how it resolves itself* —
neither hard-codes the other.

This must conform to `pipeline_file_philosophy.md` when built: no raw paths (resolve through a
registry), fail-loud-at-the-boundary when a model is misconfigured (message names the fix), one
authoritative resolver per model, and `env.yaml` owns machine-specific roots/interpreters while
`config.yaml` owns model *intent*.

---

## Why it was not fixed now

- Only a handful of models are wired, and each currently works under the interim single-root +
  interpreter-override posture.
- Designing the general routing contract before we have ≥2 genuinely different new models to route
  risks over-fitting to the ones we have (the detection/segmentation/VAE trio is not a
  representative sample of "every model").
- It is a real design effort with cross-cutting reach (env.yaml, paths.py-adjacent registry,
  tasks.py verbs, the rule template), not a quick wiring change. mdcolon is deferring it.

**Surfacing rule:** the next time a new model is wired and the interim single-route posture forces an
ad-hoc hack to resolve/config/run it, that is the signal to design this contract for real. Record the
specific friction that forced the hack here when it happens — those concrete cases are the input to
the general design.

---

## Related

- `model_input_handoff_contract.md` — the VAE encode handoff (the one model boundary that IS
  contracted today; the 3.9 interpreter escape hatch lives here).
- `pipeline_file_philosophy.md` — the conventions the eventual routing contract must honor.
- `output_tree_doctrine.md` — confirms model weights/configs live OUTSIDE the output tree
  (`models_root` in `env.yaml`); the pipeline reads them, never writes them.
