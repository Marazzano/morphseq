# SAM3 Model Backend

This directory owns the SAM3 runtime integration. It does not own pipeline contracts or adapters.

Command responsibilities:

```txt
install-backend sam3  -> runtime/code
                         env deps are supplied by pixi, native SAM3 checkout is cloned/pinned,
                         importability and vendor manifest are checked

install-model sam3    -> weights/artifacts
                         gated SAM3.1 checkpoint is fetched into the MorphSeq model cache,
                         auth/cache placement and checkpoint manifest are checked

verify-backend sam3   -> truth/report
                         runtime + model + adapter boundary are tested and setup_report.json is written
```

The command names stay short for consistency with the backend-split spec. The implementation keeps
the lifecycle split explicit so changing runtime code never redownloads weights, and fixing artifact
auth never rewrites the runtime checkout.

Current SAM3 facts:

- SAM3.1 loads through native `facebookresearch/sam3`, not transformers.
- `version="sam3.1"` is the canonical CUDA path.
- `version="sam3"` is the CPU/dev path until equivalence passes.
- FA3 is a separate production performance smoke and should not be part of the equivalence harness.
