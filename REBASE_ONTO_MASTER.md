# Rebasing dsp-consolidated onto upstream tinygrad master

`dsp-consolidated` is 484 commits behind `onnxsim/tinygrad` master, and merging it produces
**8 conflicts in core compiler files**:

```
tinygrad/codegen/__init__.py
tinygrad/codegen/late/coalesce.py
tinygrad/codegen/opt/heuristic.py
tinygrad/codegen/opt/postrange.py
tinygrad/codegen/opt/tc.py          (modify/delete: deleted upstream, modified here)
tinygrad/runtime/ops_dsp.py
tinygrad/schedule/rangeify.py
tinygrad/uop/symbolic.py
```

## Why this is not a routine merge

Upstream removed `tinygrad/codegen/opt/tc.py` in `eb5cfe903` ("fix circular imports", #17903) -
it restructured the tensor-core opt system. That file is where `tc.hexagon_hmx`,
`tc.hexagon_hmx_i8` and `tc.hexagon_v65` are declared, and the whole HMX line hangs off them:

| work | depends on |
|---|---|
| `hvx-hmx` fp16 tensor core | `tc.hexagon_hmx` |
| `hvx-hmx-qdq` int8 `:cm` core, fused requant, conv | `tc.hexagon_hmx_i8` |
| `hvx-vrmpy` GEMV TC | the TC opt system generally |
| `ops_dsp._dsp_tcs()` | all three |

So this is a rebase of the DSP stack onto a moved compiler core, not a merge, and the seven
`ops_dsp.py`/codegen conflicts are in exactly the code the DSP work modifies. Doing it blind
risks the stack that is currently green.

## Order of work

1. Read `eb5cfe903` and whatever else in those 484 commits touched the TC opt system, and work
   out where `hexagon_hmx*` should live now. `test/unit/test_dsp_render.py` (37 tests, no
   toolchain needed) is the fast check that the tensor-core declarations are still reachable.
2. Rebase `dsp-consolidated` in small, individually-pushed chunks, running
   `pytest test/unit/test_dsp_render.py` and `pytest test/external/dsp/` (as **separate**
   invocations - they are order-dependent, see `hand/ORDERING.md`) after each.
3. Re-pin `TINYGRAD_SHA` in onnxsim/onnxsim's `hexagon-tinygrad.yml` to the new head, and bump
   `fetch_hand_kernels.sh`'s default to match.
4. The fork's own `hexagon-dsp.yml` gates all of it.

## Do not do this as a blind `git merge origin/master`

The conflicts land in the HMX lowering itself. Each one needs a decision about whether the
DSP rewrite still applies to the restructured code, and that cannot be resolved by picking a
side.
