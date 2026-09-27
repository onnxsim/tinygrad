# Rebasing `dsp-consolidated` onto upstream master

**Status: the tree imports and 28 of 37 render tests pass. The `rangeify.py` conflict is resolved;
9 remain, 6 of them HMX int8. Details below, with the measurements.**

## Progress

| stage | render tests |
|---|---|
| conflicts resolved, tree does not import | 0 |
| rangeify ported, tree imports | 0 (instant `AttributeError`) |
| `UOp`'s dropped `dtype` parameter fixed | 3 passed / 35 failed |
| `wmma_args` 7-tuple -> 5-tuple fixed | 16 / 21 |
| HMX + int8 fragment element order fixed | 22 / 15 |
| B leads with `k0`, CUSTOM arg carries its dtype | 25 / 12 |
| three regex-mangled CUSTOM sites repaired | 27 / 10 |
| qadd helper check reads `arg[0]` | **28 / 9** |

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

## The open one: HMX int8 has no accumulator chain to rewrite

**Symptom.** Six of the nine remaining failures are `TestDSPHmxI8`. The `:cm` tile op is correct —
`activation.ub = mxmem(%0,%1):cm` is emitted and the tile op itself is right — but
`_hmx_acc_rewrite` bails at **check 25** and the output carries 2 `__hmx_` references where the fp16
path now carries 529. So `:deep`, `#ifdef HMX_REF` and the `__hmx_i8_begin()` bookkeeping are all
missing: the int8 tiles are emitted one at a time with no accumulator kept across the K loop.

**Isolated to the WMMA application, and the WMMA itself is not the problem.** Comparing the two
trees stage by stage on `TestDSPHmxI8.src(128, 256, 256)`:

| stage | green | this tree |
|---|---|---|
| scheduled AST (before codegen) | 3 ADDs, 1 REDUCE | 3 ADDs, 1 REDUCE |
| entering `full_rewrite_to_sink` | — | 3 ADDs, 0 WMMA |
| after `apply_opts` (first logged stage) | — | 1 WMMA, **0 `ADD(LOAD, ...)`** |
| after `full_rewrite_to_sink` | 1 WMMA, 128 LOADs, **16 `ADD(LOAD, ...)`** | 1 WMMA, 128 LOADs, **0 `ADD(LOAD, ...)`** |

The WMMA that comes out is identical on both sides — `uint8 x int8 -> int32`, shapes
`(2048,), (1024,), (2048,)`, accumulator `dtypes.int` indexed 2048 ways. So the tile is right and the
**16 accumulator `+=`s are never created**. `_hmx_rows_i8` also succeeds for both operands, so the
earlier fragment-order bug really is fixed and this is a different thing.

**Ruled out, each by measurement rather than by reading:**

- `do_stack_wmma` and `linearizer.py` are byte-identical to green.
- `_apply_tc_opt`'s src fixup and its "preserve extra reduces" tail are equivalent (the two
  divergences I did find there — `with_missing_tc_axes` dropped, and the extra-reduce guard comparing
  UOp objects instead of range ids — are fixed in the previous commit).
- The scheduled AST, the entering AST, and every `graph_rewrite` stage in `full_rewrite_to_sink`
  carry 0 `ADD(LOAD, ...)` throughout, so it is not a rewrite inside codegen removing them.
- `operand_upcast_axes` width is a symptom, not a cause: green is 11, and 10, 11 and 16 were all
  tried. 16 is what the fp16 path needs; the others regress it.

**What this leaves.** The adds have to come from something that runs between `apply_opts` and the
first logged rewrite — i.e. inside `_apply_tc_opt` or `hand_coded_optimizations` on the *green* side
that this port does not reproduce. Comparing those two functions line by line against green is the
next step; the heuristic diff showed `extra_opts` exists in green's signature area but is only
documented, never passed, so it is not that. I have not found it, and I would rather say so than land
a guess.

**Also still open**, from the same family: `test_strided_reduce_prefetches_rows_ahead` and
`test_epilogue_single_m_tile` / `test_single_k_tile_matmul` — `assertIn` failures on expected source
text, not crashes, and likely downstream of this once it lands.

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
