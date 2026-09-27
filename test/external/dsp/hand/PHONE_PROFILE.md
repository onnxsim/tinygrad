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
