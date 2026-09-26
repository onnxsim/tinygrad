# Reconciling the requant cost: 0.2% measured, ~48% claimed

Two numbers exist for the same quantity and they disagree by more than two orders of magnitude:

| source | figure | how it was obtained |
|---|---|---|
| this measurement | **52 us, 0.2% of the graph** | device A/B, one tree, one flag |
| the `HMX README` bisection | **~48% of the 3x3 family** | simulator, per-kernel bisect |

The hypothesis when this started: the README's number came from the simulator, which has already
misled this work twice (per-kernel shares off by up to 1.5x; runs V69 `.sf` ops as IEEE where the
hardware computes qf32). That hypothesis is **not yet confirmed** - see "what is still open" below.
What *is* confirmed is the instrument.

## The instrument: `HMX_RQ_STUB`

`HMX_RQ_STUB` is a build-time switch in `tinygrad/runtime/ops_dsp.py`, injected as
`#define HMX_RQ_STUB 0/1`. It exists so that both arms of the A/B come out of **one tree**, which is
the only way the measurement means anything (see "wrong instruments" below).

It keeps the entire epilogue:

- the bias add (`x0..x3 = a[i] + b`),
- the `|acc| > 2^24` flag,
- both packs (`vpackhub_sat` / `vpackwh_sat`),
- the four 32-byte stores,
- the `__hmx_rq_any` reduction and the out-of-line `__hmx_rq4x` exact redo.

and replaces only the ORT-exact arithmetic of the fast path: `__hmx_rqwf`'s per-lane normalization,
its 24x24 product, and the tie window.

### One thing it also drops, deliberately

`__hmx_rqwf` has a side effect: it sets `*flag` when a lane's result lands inside the tie window

```c
*flag |= (__hmx_u32x32)(d <= win) & (__hmx_u32x32)(F > 0);
```

which is a *second* trigger for the exact `__hmx_rq4x` redo, separate from the `|acc| > 2^24` one.
The stub does not reproduce that flag, so a tile that would have redone exactly no longer does.

This is intentional and it is what makes the A/B measure the right thing. `__hmx_rq4x` is
data-dependent: its event rate differs between the two arms' outputs (the stub's output is a raw
saturating pack, the real arm's is the requantized value), so leaving the window trigger in would
charge the arithmetic arm for a redo the stub arm never triggers, and the two arms would no longer
differ by exactly one thing. The `|acc| > 2^24` flag *is* kept in `rq4f`, and on ResNet-18 no tile
takes the exact redo at all, so on this graph the two arms differ by the fast-path arithmetic and
nothing else.

**The cost of that choice:** the 52 us is the price of the arithmetic *excluding* the event rate of
the exact redo. If a graph does take `__hmx_rq4x` often, the real cost is 52 us plus that event
rate, and this measurement will under-report it.

## The measurement

Phone: Xiaomi 12S, SM8475/taro = **V69**, `HMX_VTCM_KB=4096` (the phone grants 4 MB; the 256 KB
default is a 14.6% difference that briefly looked like a regression). `HAP_power_set_HMX` (`hmx 0`)
vote is not optional. ResNet-18 QDQ, both arms built from `tinygrad@6565de79` back to back:

| arm | time |
|---|---|
| real requantization | 22,328.6 us |
| arithmetic stubbed | 22,277.0 us |
| **difference** | **51.6 us = 0.2%** |

Default build verified unchanged by the same commit: 0/25088 mismatch, 23,946,105 pcycles,
`test_dsp_render.py` 37 passed.

### Independent reproduction (2026-09-27)

Re-measured on the phone from the already-deployed probe builds, to confirm the figure is not an
artefact of one build. Both arms, `HMX_VTCM_KB=4096` (the device reports `vtcm 4194304`), 5 iters:

| arm | time |
|---|---|
| `rn18-rqreal` (real requantization) | 22,862.6 us |
| `rn18-rqstub` (arithmetic stubbed) | 22,800.0 us |
| `rn18-rqstub13` (a later stub revision) | 22,703.4 us |
| **difference (rqreal - rqstub)** | **62.6 us = 0.27%** |

Both arms report `0/25088 mismatches`, so the stub is a correct-output build and not a
fast-but-wrong one. The 62.6 us sits alongside the 52 us from the single-tree A/B: two independent
build pairs, two runs, the same order of magnitude and the same sub-1% conclusion. The requant is
not a lever on this graph.

(For anyone re-running these: the skel only loads with `ADSP_LIBRARY_PATH="."` and a **relative**
`file:///tg_hmx_rpc.so?...` URI, from inside the build's own directory. Absolute paths and
`ADSP_LIBRARY_PATH=/data/local/tmp/<build>` both fail with `open failed -2147482618`, which looks
like a build problem and is not.)

## What this means

Requantization is **not** a lever on this graph. `QC_FAST` - the host-table rewrite that would trade
requantization accuracy for speed - is not worth its accuracy cost here. The epilogue's cost is the
stores and the flag reduction, which the stub deliberately keeps; the arithmetic is 0.2%.

## Wrong instruments, recorded so they are not repeated

Three earlier probes measured the wrong thing. All three are in the history.

1. **`HMX_RQ=0`** - drops the epilogue entirely, the rewrite bails, and the kernel falls back to
   scalar. Measured 24x slower. It measures the scalar fallback, not the requantization.
2. **`HMX_RQ_PROBE`** - removes the requantize *rows*. That reshapes the addq folding and the store
   chain, so it came out *slower* than the baseline. It measures a different kernel.
3. **A hand-edit reporting 2,178 us (9.8%)** - measured against a build from a *different* tree
   (`wt-tgci`) than the one edited (`tg-final`). A reminder that an edit in tree A and a build from
   tree B produce a number about neither. `HMX_RQ_STUB` exists to make that impossible.

## What is still open

The README's ~48% has not been re-derived or retracted. The two figures are not necessarily
contradictory - ~48% *of the 3x3 family* is a different denominator than 0.2% *of the graph*, and
the 3x3 family is a minority of the graph - but the two have never been placed on the same basis,
and the README's number came from the simulator, which is the instrument this project has least
confidence in.

The next step is to re-derive the 3x3-family breakdown **on the phone**. The per-kernel `G_PROF`
timing path now exists and works, so this is a measurement, not new instrumentation. Until that is
done, the README figure should be treated as unverified, and the 52 us as verified for its stated
scope.
