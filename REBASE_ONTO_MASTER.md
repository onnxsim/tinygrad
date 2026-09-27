# Rebasing `dsp-consolidated` onto upstream master

**Status: the tree imports and 21 of 37 render tests pass (up from 0). The `rangeify.py` conflict is
resolved; 15 render tests remain, all HMX tensor-core, from one structural cause. Details below,
with the measurements.**

## Progress

| stage | render tests |
|---|---|
| conflicts resolved, tree does not import | 0 |
| rangeify ported, tree imports | 0 (instant `AttributeError`) |
| `UOp`'s dropped `dtype` parameter fixed | 3 passed / 35 failed |
| `wmma_args` 7-tuple -> 5-tuple fixed | 16 passed / 21 failed |
| HMX + int8 fragment element order fixed | **22 passed / 15 failed** |

The 15 remaining are all `TestDSPHmx` / `TestDSPHmxI8` and all one cause: the fp16 path's **B
operand** reaches the HMX row packer lane-ordered instead of interleaved. See "The open one" below.

## What upstream actually did

The handoff said #17903 "deleted `tinygrad/codegen/opt/tc.py`" and that this was the blocker. That
is true but it understates it. #17903 (`eb5cfe903`, "fix circular imports") was a **file move** -
`codegen/opt/tc.py` -> `renderer/tc.py` - to break an import cycle. The real problem is everything
that landed *after* it.

### 1. `TensorCore` was redesigned (ported, done)

`dims` / `threads` / `elements_per_thread` / `opts` / `swizzle` / `dtype_in_b` are gone. A TC is now
`frag_a` / `frag_b` / `frag_c`, each `(lane bits, element bits)`, and `dims`/`threads` are *derived*
properties. The three Hexagon TCs were ported by transcribing the index-bit layouts their own
comments already recorded - all three construct and report the right shapes:

```
v65     dims=(32, 1, 4)    threads=1
hmx     dims=(32, 32, 32)  threads=1
hmx_i8  dims=(32, 64, 32)  threads=1
```

`hexagon_v65` lost its mixed `dtype_in_b` (uint8 x int8) entry: `dtype_in_b` no longer exists. The
narrowing is now applied in the HVX renderer instead, and `postrange.py` notes where.

Two things the new API enforced that the old one let slide, and which caught real errors in the port:
each fragment's element bits must cover *all* of its own axes (`frag_b` for HMX initially had an `m`
bit where the fifth `k` belonged), and A and B must relabel `k` identically.

### 2. `Range` gained a tuple `axis_id` and an `axis_type` property (done)

`build_range_map` is now keyed on `r.axis_id` (a tuple) and filters on `r.axis_type`, with
`AxisType` moved to `arg[-1]`. Our store-order feature was ported onto the new signature. The
`OptOps.UPCAST` / `OptOps.LOCAL` calls became `OptOps.SPLIT(axis, (size, AxisType.X))` throughout
`heuristic.py`, and the tensor-core branch was rewritten onto upstream's new `split()` helper with
the Hexagon guard preserved: on the DSP the M/N upcasts are skipped entirely, because there is no
warp to build and an extra upcast lands *inside* the 32 accumulator lanes, turning every WMMA's C
into a strided gather.

## The blocker: `tinygrad/schedule/rangeify.py` — RESOLVED

Upstream **deleted the `Ops.CONTIGUOUS`, `Ops.TUPLE` and `Ops.GETTUPLE` ops**, and with them the
`BUFFERIZE` pipeline that our DSP scheduling code sits inside. Our file references all of them:

| op | uses in our `rangeify.py` | status upstream |
|---|---|---|
| `Ops.CONTIGUOUS` | 6 | **gone** |
| `Ops.TUPLE` | 2 | **gone** |
| `Ops.GETTUPLE` | 2 | **gone** |
| `Ops.FUNCTION` | 1 | **renamed** to `Ops.STAGE` |
| `helpers.PCONTIG` | 2 | **gone** (upstream simplified that path to a plain `return None`) |

Our file is 624 lines, upstream's is 385, and neither is a superset: upstream has `pm_no_views` and
the const-indexing rules, we have `pm_fold_moved_after`, `split_reduceop`, `_mop_index` and the
PCONTIG path.

What made this tractable is that for all three deleted ops, the only surviving reference in this
fork was the **consumer** — nothing produced them:

- `CONTIGUOUS`: `get_contiguous` moved an `Opt` tuple off the op into `ctx.opts`, feeding
  `KernelInfo(opts_to_apply=...)`. Every `opts_to_apply` in the tree is `()` — "no override" — and the
  only producer of that op was already gone. Dead in our fork too, so it went with upstream's
  `KernelInfo()`.
- `TUPLE`/`GETTUPLE`: one rewrite rule, no producer. Gone.
- The other three `CONTIGUOUS` uses map onto upstream's spellings directly: the toposort gate just
  drops the guard, `ALWAYS_RUN_OPS` loses a member, the Invalid-store rule matches the CONST.

Plus `pm_no_views` grafted in, the `PCONTIG` branch removed, and `FUNCTION` -> `STAGE`.

## The open one: the fp16 HMX B operand

**Symptom.** All 15 remaining failures are HMX tensor-core tests. `_hmx_acc_rewrite` bails at
`HMX_DEBUG`'s "check 1" — `_hmx_rows` returned `None` — so `self._hmx_acc` stays `False` and
`_HMX_ACC_HELPERS` is never appended. The output is **12** `__hmx_` references where green has
**529**. No error anywhere: the path is simply off.

**Measured.** Both operands reach the packer as 1024-lane STACKs. A's lanes are `[0,0,1,1,2,2]` —
identical to green. B's are `[0,1,0,1,0,1]`. `_hmx_rows` maps `p -> (2*(p//64) + p%2, (p%64)//2)` and
needs consecutive `p` to share a value in pairs, so A passes and B fails.

**Why.** HMX gives A and B different crouton orientations: A is `IDX(m, k)` and B is `IDX(k, n)`. The
new fragment API's `relabel()` maps each fragment bit to a slot axis **positionally** — pairing
`frag_b[i]` with `base_upcast_axes()[:len(frag_b)][::-1][i]` — and `base_upcast_axes()` is a single
shared list built from `frag_c` plus the k bits. It interleaves `m` and `n`, so it cannot express the
A/B asymmetry: both operands get the same axis list. A happens to come out right because its bits
already sit in C's slot order; B's do not.

**What I tried, and why it is not landed.** Giving each operand its own element-bit list
(`TensorCore.operand_upcast_axes`, added) and building `tc_upcast_axes` per operand is the right
shape — but `expand_wmma` looks every axis up in the range map, and the per-operand lists are 10
entries where the shared one is 15, so it raised `KeyError (13,)`; padding to equal length did not
fix it either, and reversing to MSB-first changed the key to `(17,)` without resolving it. I reverted
rather than land it half-done. The session established that **all three candidate B orders and the
A/B k-consistency check** agree the current `frag_b = (n0, k0..k4, n1..n4)` is the only
self-consistent one — so the fragments are right and the fix has to happen in how the axis list is
built, not in the fragment declarations.

## What is done and worth keeping

- `renderer/tc.py` moved, with the three Hexagon TCs re-expressed in the fragment API, plus
  `dtype_in_b` re-added (upstream has no mixed-dtype TC, and HMX `:cm` is u8 x s8).
- `build_range_map` / `expand_wmma` ported to tuple `axis_id`.
- `coalesce.py`, `heuristic.py`, `postrange.py`, `symbolic.py`, `rangeify.py`, `ops_dsp.py` resolved.
- Four real bugs the port surfaced, all of which a conflict resolution hides: `UOp`'s dropped `dtype`
  parameter (6 call sites, the DType landed in `src`), `wmma_args` 7-tuple -> 5-tuple, `ParamArg.size`
  moving off a src operand, and the coalesce memory key gaining a field.

**`dsp-consolidated` itself is untouched and green** (5/5 CI). This branch is a working record of
the port, not a shippable branch.

## Recommended order

1. Finish the A/B axis list (the one open item), then re-run the render tier — it is the fast signal
   and needs no toolchain.
2. Validate against the HMX/HVX goldens and the MCC oracles before trusting the render output.
3. Then the two oracle tiers, and only then merge.

## Measurement note

Every claim above was checked against the merged tree, not inferred. Enum membership was read by
loading `tinygrad/uop/__init__.py` in isolation - a regex over the source gave false positives
(`SHR`, `STORE` and `STAGE` were all reported missing and are not), so the numbers come from
`Ops.__members__` plus an AST walk of every `Ops.X` reference in `tinygrad/`. The A/B lane patterns
were read out of the live renderer, and the fragment orders were checked with `frag_coords()` rather
than reasoned about.
