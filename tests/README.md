# Running the tests

## Always pass `--import-mode=importlib`

```bash
PYTHONPATH=src conda run -n segmentation_grounded_sam --no-capture-output \
  python -m pytest tests/ --import-mode=importlib
```

### Why this is required (a known, persistent quirk)

The test tree mirrors the package tree: tests live under `tests/data_pipeline/<subpackage>/`,
and some of those directories contain an `__init__.py` while `tests/` and `tests/data_pipeline/`
do **not**. Under pytest's default `prepend` import mode this makes Python assemble a *second*
`data_pipeline` namespace package rooted in `tests/`, which then **shadows the real package in
`src/`**. The symptom is confusing: importing the package (`data_pipeline.snip_processing`)
succeeds and even resolves to the correct `src/...__init__.py`, but importing a **submodule**
(`data_pipeline.snip_processing.entrypoints`, `...snip_frame_shape`, etc.) fails with
`ModuleNotFoundError` — because the shadowing namespace has no such submodule.

`--import-mode=importlib` imports each test module under its own fully-qualified name without
prepending test directories to `sys.path`, so the real `src/` package wins and submodule imports
resolve. This is not specific to any one test file — it bites **any** test that imports a
`data_pipeline` submodule, including pre-existing ones (e.g.
`tests/data_pipeline/snip_processing/test_run_snip_processing.py`). If you see a
`No module named 'data_pipeline.<something>'` collection error, you almost certainly dropped the
flag, not introduced a real import bug.

A permanent fix would be to either add a `pytest.ini`/`pyproject.toml` that pins
`addopts = --import-mode=importlib`, or make the `tests/` tree a consistent package (add the
missing `__init__.py` files, or remove all of them). Until then, pass the flag.
