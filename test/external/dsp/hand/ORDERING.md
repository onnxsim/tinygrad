# The DSP oracle tests are order-dependent (known, pre-existing)

`test/unit/test_dsp_render.py` and `test/external/dsp/` **must not be run in the same pytest
process.** Doing so fails 22 tests that pass when either suite runs alone.

Reproduced on `hand-kernel-oracles` (#9) and on the consolidated branch, 2026-09-25:

| invocation | result |
|---|---|
| `pytest test/unit/test_dsp_render.py` | 37 passed |
| `pytest test/external/dsp/` | 47 passed, 8 skipped, 3 xfailed |
| `pytest test/unit/test_dsp_render.py test/external/dsp/` | **22 failed**, 62 passed, 8 skipped, 3 xfailed |

The failures are `hand/msda` (6) and `hand/roialign` (16), and they are float-op-order
mismatches between the hand kernel and tinygrad -- e.g. `test_msda[rtdetr_decoder_f32]`
reports `56502 of 76800 outputs differ`. The hand kernels' contracts *are* ORT's exact
float op order, which is why `hand/conftest.py` pins `FLOAT_REASSOC=0` and appends
`-ffp-contract=off` to `$CC`.

Not yet root-caused. Ruled out by experiment:

- `FLOAT_REASSOC=0` and `CC="clang-19 -ffp-contract=off"` set **before** the interpreter
  starts: no effect (the conftest already does this, via `os.environ.setdefault`).
- `HVX_ARCH=v65` set explicitly: no effect.
- `-p no:cacheprovider`: no effect.
- Any single render-test class followed by `hand/msda`: passes. So it is cumulative across
  the render classes, not one test leaking a global. `TestDSPRender`, `TestDSPQfloat`,
  `TestDSPQfMath`, `TestDSPVrmpyGemv`, `TestDSPHmx`, `TestDSPHmxI8`, `TestDSPQLinearAdd`
  each pass on their own with `hand/msda`, and `TestDSPHmx` + `TestDSPHmxI8` together do too.
- The render tests' only Python-level state is `ops_dsp.HVX_QFLOAT`, which
  `TestDSPQfloat.setUp`/`tearDown` save and restore correctly.

Most likely something in tinygrad's rewrite pipeline is being memoised for the whole
process and the render suite (which builds hundreds of HMX forms, including the float
qfloat path) leaves a rule or a cache entry that changes how a later float expression
lowers. `tinygrad.uop.symbolic`'s real-arithmetic rewrites and the `to_program` cache in
`tinygrad/codegen/__init__.py` are the places to look.

## What CI does about it

`.github/workflows/hexagon-dsp.yml` runs the two suites as **separate jobs** (tiers 1 and
2), so neither can perturb the other, and a real failure is attributable. Until the
underlying leak is found, do not "fix" a red oracle run by reordering -- check which job
produced it, and reproduce that job's exact command on its own.

To run them the way CI does:

```
pytest test/unit/test_dsp_render.py       # tier 1
pytest test/external/dsp/                 # tier 2
```
