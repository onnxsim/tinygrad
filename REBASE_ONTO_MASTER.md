# Rebasing `dsp-consolidated` onto upstream master

**Status: 5 of the 8 conflicts resolved. The 6th (`tinygrad/schedule/rangeify.py`) is not a
merge conflict - it is a structural divergence, and the rebase cannot be finished honestly until it
is decided. Details below, with the measurements.**

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

## The blocker: `tinygrad/schedule/rangeify.py`

Upstream **deleted the `Ops.CONTIGUOUS`, `Ops.TUPLE` and `Ops.GETTUPLE` ops**, and with them the
`BUFFERIZE` pipeline that our DSP scheduling code sits inside. Our file references all of them:

| op | uses in our `rangeify.py` | status upstream |
|---|---|---|
| `Ops.CONTIGUOUS` | 6 | **gone** |
| `Ops.TUPLE` | 2 | **gone** |
| `Ops.GETTUPLE` | 2 | **gone** |
| `Ops.FUNCTION` | 1 | **renamed** to `Ops.STAGE` |
| `helpers.PCONTIG` | 2 | **gone** (upstream simplified that path to a plain `return None`) |

Our file is 624 lines, upstream's is 385. Neither is a superset of the other:

- Upstream has `pm_no_views` and the const-indexing / `UPat.cvar` rules we do not.
- We have `pm_fold_moved_after`, `split_reduceop`, `_mop_index`, the PCONTIG partial-contiguity
  path, and `Opt` handling that the DSP backends depend on.

I grafted `pm_no_views` into our file and removed the `PCONTIG` branch (both mechanically sound),
and renamed `FUNCTION` -> `STAGE`. That left exactly the three deleted ops. There is no spelling
that works: `CONTIGUOUS` is what `get_contiguous`, the `ALWAYS_RUN_OPS` set and the dead-store rule
all key on. Porting that means **re-deriving our DSP scheduling against a pipeline upstream has
replaced**, which is a piece of work in its own right - not something to do silently inside a
rebase and then declare done.

## What is done and worth keeping

The resolved conflicts are real and correct, and are worth keeping as a branch even though the
rebase cannot be finished:

- `renderer/tc.py` moved, with the three Hexagon TCs re-expressed in the fragment API.
- `build_range_map` / `expand_wmma` ported to tuple `axis_id`.
- `coalesce.py`, `heuristic.py`, `postrange.py`, `symbolic.py`, `ops_dsp.py` all resolved.
- `rangeify.py` = our file + upstream's `pm_no_views`, minus the deleted `PCONTIG` path.

**This tree does not import yet** (the three deleted ops). It is a working record of the port, not
a shippable branch. `dsp-consolidated` itself is untouched and green.

## Recommended order

1. **Decide the scheduling story first.** Either (a) port the DSP backends' scheduling onto
   upstream's replacement pipeline, or (b) pin the DSP stack to a known-good upstream SHA and take
   upstream master separately. (b) is a fraction of the work and unblocks merging the DSP stack now.
2. Then finish the rebase mechanically; the other 5 conflicts are done and are unlikely to regress.
3. Re-run the three CI tiers. The `dsp-render` tier (no toolchain, 37 tests) is the fast signal.

## Measurement note

Every claim above was checked against the merged tree, not inferred. Enum membership was read by
loading `tinygrad/uop/__init__.py` in isolation - a regex over the source gave false positives
(`SHR`, `STORE` and `STAGE` were all reported missing and are not), so the numbers come from
`Ops.__members__` plus an AST walk of every `Ops.X` reference in `tinygrad/`.
