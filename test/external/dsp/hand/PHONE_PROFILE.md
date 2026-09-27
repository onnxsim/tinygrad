# Per-kernel phone profile (ResNet-18 QDQ, Xiaomi 12S V69)

The first **device-side** per-kernel breakdown. Every earlier per-kernel number in this project came
from hexagon-sim, which had already proved wrong twice (per-kernel shares off by up to 1.5x; V69 `.sf`
float paths run as IEEE where the hardware computes qf32). The phone has had a per-call profiler the
whole time - `HAP_perf_get_time_us` in the skel, `G_PROF(i)` after each kernel, returned in `t[1..]`
- but it had never been used, because the client takes the literal string `prof` as `argv[4]` and the
earlier invocations passed a number, so every run silently profiled nothing and reported `0.0
us/inference`.

    client 'file:///tg_hmx_rpc.so?tg_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' <case> <iters> prof

`argv[4]` is `strcmp`-ed against `"prof"`. `1` is not `prof`, and the run does not fail - it just
returns a total with an empty breakdown, which reads like a dead timer rather than a wrong argument.

## The graph

**22,791 us**, 256 calls, `0/25088` mismatches vs ORT, `HMX_VTCM_KB=4096` (the device reports
`vtcm 4194304`).

| family | calls | us | share |
|---|---|---|---|
| layer4 3x3 (`r_16_..._3_3_16`) | 3 | 7,780 | **34.1%** |
| all other 3x3 | 14 | 5,921 | 26.0% |
| stem 1x1 (calls 56-61) | 6 | 6,437 | 28.2% |
| everything else | ~233 | ~2,653 | 11.6% |

**The 3x3 family is 60.1% of the graph**, not the ~40% the simulator implied. Layer4's three calls
alone are a third of the total.

## layer4 is the target

`r_16_2_2_2_2_2_2_2_2_2_2_3_3_16` at 2,593 us each. The useful number is the density:

    2,593 us for 2*2*16*16*9*64 = 589,824 MACs  ->  4.4 ns/MAC, or ~100 cycles/MAC at ~23 MHz HVX

The stem's 1x1 (`r_2_202_..._4_4`, 4,703 us) is the *better*-performing reference: it is already
quad on the weight path, so the comparison that matters is 3x3's 4.4 ns/MAC against whatever the 1x1
achieves on the same tile shape.

The earlier diagnostics for this family stand and are not contradicted:

- `1.00x` wasted MACs - 2304 x 65,536 is the exact real conv size, so there is no redundant work to
  remove. The cost is not waste, it is density.
- The weight path is already quad (`rank 4`); the **activation pack** is not, and was priced at
  0.5 ms of the 3.4 ms. That measurement was taken against a build from a different tree, so it should
  be re-taken before acting on it.
- `rn18-quada0` on the device now measures 23,084.7 us against `rn18-base` 23,059.0 - i.e. **turning
  the activation quad off changes nothing**, so whatever `HMX_I8_QUAD_A=0` actually toggles is not
  costing 0.5 ms in the current build. That earlier figure does not reproduce and should not be
  trusted.

## The weight repack is the layer4 cost

Reading the generated kernel (`k75.c`, 629 lines) rather than guessing:

    __hmx_i8_begin();
    for (Ridx0 < 3)            // dy tap
      for (Ridx1 < 3)          // dx tap
        for (Ridx2 < 16)       // K block
          __hmx_i8_copy_a(...);                                  // 1 activation copy
          __hmx_i8_pack_b4(_b+0,   ... +0, +512, +1024, +1536);  // 9 weight packs
          ... x9
          __hmx_i8_mac(_a, _b);

So per inference: **2,304 macs, 2,304 activation copies, 20,736 weight packs, 82,944 weight loads.**
The packs outnumber the macs 9:1, and each `pack_b4` gathers four rows 512 bytes apart. At
1.133 us per mac, the arithmetic is nowhere near the cost.

The redundancy: `alu0 = Lidx4 << 5` puts the **output tile loop outermost**, so the same 36 weight
values are re-gathered for all 16 output tiles - a 16x redundancy in the gather if the weights could
be hoisted or kept in VTCM across the `Lidx4` loop.

## Confirmed on the device: the quad rank is worth 1.4 ms, and only there

`HMX_I8_QUAD=0`, same tree, same profile harness:

| call | kernel | quad | no-quad | delta |
|---|---|---|---|---|
| 129 | `r_16_..._3_3_8` | 492 us | 1,222 us | **+730** |
| 57 | `r_2_202_..._4_4` (stem) | 4,701 us | 4,806 us | +105 |
| 132/138/143 | `r_16_..._3_3_16` (layer4) | 2,533/2,610/2,602 | 2,558/2,591/2,612 | ~0 |
| | **sum of >100 us kernels** | **21,939** | **23,336** | **+1,397** |

The whole regression is one kernel. Layer4 - a third of the graph - is **already** on the quad rank
and does not change, so the pack count there is not the limiter; call 129 was the one that had not
been promoted.

`HMX_I8_QUAD_A=0` (the activation pack): **22,799.4 us against 22,808.6** - no difference, within
noise. This does not reproduce the 0.5 ms figure recorded earlier from a build of a different tree,
and PHONE_PROFILE.md and STATUS should treat that number as withdrawn.

VTCM pool, phone-granted 4 MB:

| pool | us/inference |
|---|---|
| 1 MB | 23,372.7 |
| 2 MB | 23,108.3 |
| **4 MB** | **22,808.6** |
| 8 MB | 23,110.7 |

4 MB is the optimum and the device grants exactly that, so the pool is already where it wants to be.
2 MB costs 300 us - this is the 14.6% the handoff warned about, and it is why `HMX_VTCM_KB=4096` is
mandatory for any measurement here.

## The VTCM operand cache is worth 7.8 ms - and layer4 does not get it

`HMX_I8_CACHE=0` on the device, same tree:

| call | kernel | cache | no cache | delta |
|---|---|---|---|---|
| 129 | `r_16_..._3_3_8` | 492 | 3,354 | **+2,862** |
| 57 | stem 1x1 | 4,701 | 5,461 | +760 |
| 108 | 3x3 | 304 | 889 | +585 |
| 87 | 3x3 | 441 | 1,007 | +566 |
| 132/138/143 | **layer4 3x3** | 2,533/2,610/2,602 | 2,544/2,609/2,626 | **+11/-1/+24** |
| | graph total | **22,808.6** | **30,587.2** | **+7,778.6** |

So the cache is doing enormous work *everywhere except layer4*, which is a third of the graph.
Layer4 gets 34 us out of a possible ~7.7 ms.

**And the planner thinks it is caching layer4.** A temporary `HMX_DBG_PLAN` print in `plan()` reports
for layer4 (`outer=16 inner=2 reds=[3,3,8]`, pool 1920 slots):

    A(0) = (rank 4, A 144 slots, B 144 slots, quad=True)

which is the best possible answer, and every int8 conv in the graph reports rank 4 or 3. But the
*emitted* layer4 kernel contains no `__hmx_ca(...)` or `__hmx_cb(...)` **call** - only their
definitions - and uses `__hmx_i8_mac` / `__hmx_i8_mac2` with `__hmx_i8_copy_a` + 9 `__hmx_i8_pack_b4`
per mac, i.e. the uncached path. The plan and the emission disagree.

**Root cause, found.** The int8 rewrite's cache block is gated on

    if loops is not None and e is not None and all(ro) and getenv("HMX_I8_CACHE", 1):

and `_hmx_tile_loops` returns `None` when the kernel has **fewer than two output-tile loops**
(`if len(loops) < 2: return None`). A temporary `HMX_DBG_EMIT` print on that gate reports, for the whole
ResNet-18 graph:

    3 convs:  loops=NONE   e=yes  ro=[True, True]     <- layer4
    16 convs: loops=yes    e=yes  ro=[True, True]

**Those three are exactly the layer4 triple (34.1% of the graph).** Confirmed in the emitted C:

| kernel | Lidx loops | `__hmx_ca` uses | branch |
|---|---|---|---|
| k72 (call 129) | `Lidx4<16`, `Lidx5<2` | 36 | quad, cached |
| k75 (call 132) | `Lidx4<16` | **0** | plain `__hmx_i8_mac` + 8 `pack_b4` |
| k81 (call 138) | `Lidx4<16` | 0 | plain |
| k82 (call 143) | `Lidx4<16` | 0 | plain |

Layer4's 16 output channels fit in a single `Lidx4<16` tile loop, so there is no second tile axis to
index the VTCM pool by, and the whole cached path is skipped. The other 3x3 convs have two tile loops
(`Lidx<16>` x `Lidx<2>`) and are cached.

So this is not a tuning knob and not an emission bug - it is a precondition that does not hold for
this shape. The two ways forward, in order of cost:

1. **Let the single-tile-loop case use the pool anyway**, keyed on the one tile loop it does have
   (`slot()` already handles `deps == {inner}` and `deps == {outer}`; the blocker is only that
   `outer`/`inner` come from a pair). The natural split is (reduce loop, tile loop) rather than
   (outer tile, inner tile). That reuses all the existing fill/slot logic.
2. **Tile layer4's output into two loops** so it matches its siblings, at the cost of splitting the
   accumulator and the store.

(1) is the smaller change and does not alter what the kernel computes. It is not started here: the
`plan()` arithmetic assumes `outer` and `inner` are both tile loops, so the pool sizing
(`need(r)` returning `ti*kt` or `kt`) has to be re-derived, and getting that wrong produces a kernel
that reads the wrong VTCM slot - correct-looking, wrong answer.

The useful consequence either way: the next win is not a micro-optimization, it is making layer4 use
the cache the planner already thinks it is using. That is worth up to ~7 ms on a 22.8 ms graph, and it
is a correctness-of-emission question, not a tuning one.

## FIXED: layer4 caches its activation tile - 22,808.6 -> 21,910.6 us

The fix described below, taken: pair the single tile loop with the innermost reduce loop, int8 only.

| call | kernel | before | after |
|---|---|---|---|
| 132 | `r_16_..._3_3_16` | 2,533 | **2,183** |
| 138 | `r_16_..._3_3_16` | 2,610 | **2,272** |
| 143 | `r_16_..._3_3_16` | 2,602 | **2,260** |
| | **graph** | **22,808.6** | **21,910.6** (-898.4, -3.9%) |

`0/25088` vs ORT, and `test_dsp_render` 37 passed unchanged. In the emitted C, k75/k81/k82 went from
**0** `__hmx_ca` uses to 1 each.

A second run of the same build measures **21,753.8 us**, so the honest figure is **21.75-21.91 ms
against a 22.81 ms baseline: about -1.05 ms, -4.6%**, with run-to-run spread of roughly 150 us. The
per-family comparison across the two profiles:

| family | before | after | delta |
|---|---|---|---|
| layer4 3x3 (3 calls) | 7,745 | 6,657 | **-1,088** |
| other 3x3 (14 calls) | 5,923 | 5,933 | +10 |
| stem 1x1 (6 calls) | 6,441 | 6,457 | +16 |
| graph | 22,743 | 21,693 | -1,050 |

The win is entirely layer4 and nothing else moved, which is what a targeted change should look like.

**Layer4's share falls 34.1% -> 30.7% and the stem 1x1 becomes the largest single kernel** at 4.7 ms
(21.8% of the graph, `r_2_202_2_2_2_2_2_2_2_2_2_2_2_4_4`). That is the next target, and it is a
different problem: the stem is already quad on both operands and is a 1x1 conv, so there is no
repack to hoist.

**It lands on rank 2 (A cached, B streamed), which is the right answer.** With `ti = 16` (the tile
loop) and `kt = 144` (all reduce iterations), caching B would need `ti*kt = 2,304` slots against a
**1,920**-slot pool. A - the activation crouton, re-read once per tap - is what the 898 us is. B, the
weight, is still gathered by 8 `pack_b4` per mac.

**Layer4's B operand: the pool is not the lever, and the question is closed.** Caching B needs
`ti*kt <= pool`, which at `ti = 16, kt = 144` is **2,304 against 1,920**. I checked every way to buy
those 384 slots:

| pool arrangement | slots | fits 2,304? |
|---|---|---|
| current (`HMX_VTCM_KB // 2`, ca=1792 + cb=128) | 1,920 | no |
| + the 192 KB VTCM that nothing accounts for | 2,016 | no |
| + unallocated, and hand the whole A region to B | 3,808 | yes, but then A gets nothing - rank 0 |

The 192 KB gap is real (`_HMX_AO` is 64 KB, the operand pool is `HMX_VTCM_KB // 2`, so 192 KB of a
4 MB VTCM is unallocated) and worth reclaiming on its own merits, but it is 5% of what is needed and
does not change the answer.

**There is no pool arrangement that gives layer4 both A and B.** The only lever is reducing
`ti*kt` - splitting the loop so fewer (tile x reduce) combinations are live at once. Hoisting the dy
tap out would make `kt = 48` and `need(B) = 768`, which fits easily, but it gives each tap its own
accumulator: 3 zeroings and 3 stores instead of 1 each (48 stores against 2,304 mac iterations, so
the stores are not the problem) at the cost of no longer keeping the accumulator resident across the
taps, which is the entire point of the current form. That is a real trade, not a free win, and it
needs its own measurement rather than an assumption.

So: layer4 is at rank 2 and the next gain there is a loop-split experiment, not a knob.

## The stem 1x1 is now the largest kernel, and it is already cached

`r_2_202_2_2_2_2_2_2_2_2_2_2_2_4_4`, 4,722 us, 21.8% of the graph. I went looking for the same
class of miss as layer4's and did not find one: the plan reports `reds=[4,4] outer=202 inner=2
ti=2 kt=16 rank=3`, both operands cached, and the emitted code agrees.

    void* _a = __hmx_ca(0+((Ridx0)*4+Ridx1));              if ((Lidx3)==0) { copy_a(...); }
    void* _b = __hmx_ca(16+(Lidx3)*16+((Ridx0)*4+Ridx1));   if ((Lidx4)==0) { pack_b4 x8; }

B - the weight - is filled once for the whole inference. A is filled once per `Lidx4`, and that is
**required, not redundant**: `alu20 = (Ridx0*3680) + (Ridx1<<5) + alu1` with `alu1 = Lidx4<<11`, so
each of the 202 output tiles reads a different activation crouton. 6,464 mac iterations, 3,232
`copy_a` calls moving **6.5 MB into VTCM**, 4,722 us - 0.731 us per mac iteration.

So the stem is not a caching miss. It is VTCM write bandwidth: 202 distinct 2 KB activation
croutons, each copied once, for a conv that only does 6,464 tiles. The lever is the *fill*, not the
plan - either copy A straight into the HMX `:cm` load without the intermediate VTCM round trip
(the `__hmx_i8_copy_a` + `__hmx_i8_sa` pair), or hoist A across the two `Lidx3` tiles.

**Measured, and it is a negative result: the stem is not the target.** The `__hmx_i8_copy_a` that
looks like a byte loop in the helper list is not what runs - a later, vectorized definition
(`for (int q = 0; q < 16; q++) ((__hmx_v*)d)[q] = (__hmx_v)((const __hmx_i8vu*)src)[q]`, unaligned
128-byte loads into aligned stores) overrides it, and `__hmx_i8_mac` then reads the VTCM slot
directly (`mxmem(%0,%1):cm` with `%1 = 0x7ff`) with no second staging copy.

So the arithmetic is:

    6,464 mac iterations, 4,722 us  ->  0.731 us each
    one :cm mac is 64x32 = 2,048 MACs  ->  0.36 ns/MAC  ->  ~8 cycles/MAC at 23 MHz

**8 cycles/MAC is good density** - better than the stem's 823 pcy/MAC figure of a few weeks ago, and
in the same range as the best layer4 number. The 3,232 `copy_a` calls are ~0.2% of the instruction
count; they are not what the 4,722 us is made of. The stem is simply doing 6,464 real 64x32 tiles,
and at that density there is very little left to remove.

**Do not spend more time here.** The remaining 3x3s (14 calls, 5.9 ms) are the same shape and were
not broken down individually; layer4 is 6.7 ms and already at rank 2. If more is wanted it is in the
loop split for layer4's B operand (the pool needs `ti*kt <= 1920` and it is 2,304), not in the stem.

## What this rules out

The requantization is **0.2-0.27%** (`REQUANT_RECONCILE.md`, 52 us single-tree and 62.6 us reproduced
on the phone, both `0/25088` mismatches). The profile agrees: the biggest single kernel is 20.6% of
the graph, four orders of magnitude above the requant. There is nothing to win in the epilogue
arithmetic, and `QC_FAST` is not worth its accuracy cost.

## Reproducing

The ablations already deployed on the device, all `0/25088`:

| build | us/inference | delta |
|---|---|---|
| `rn18-base` | 23,059 | - |
| `rn18-vtcm4m` | 23,086 | +0.1% (noise) |
| `rn18-quada0` | 23,085 | +0.1% (noise) |
| `rn18-noquad` | 24,537 | +6.4% |
| `rn18-nocache` | 30,863 | +33.8% |
| `rn18-norq` | 530,049 | +2200% - the 24x scalar fallback, not a measurement |
| `rn18-rqreal` | 22,862.6 | requant A/B arm |
| `rn18-rqstub` | 22,800.0 | requant A/B arm |

The skel only loads with `ADSP_LIBRARY_PATH="."` and a **relative** `file:///tg_hmx_rpc.so?...` URI,
run from inside the build's own directory. An absolute path, or
`ADSP_LIBRARY_PATH=/data/local/tmp/<build>`, gives `open failed -2147482618`, which presents as a
build failure and is not one.
