# Hand-written DSP kernels as test oracles

onnxsim's hand-written Hexagon kernels (C, hand-scheduled HVX / HMX) live here as **reference oracles** for what tinygrad's DSP
backend generates. Each test builds the hand kernel and the tinygrad lowering of the same op for the same shape and data, runs
both on `hexagon-sim -mv69 --mhmx 1`, asserts they agree bit for bit (and, where the op has an external definition such as
ORT's QDQ formulas, that the hand kernel agrees with it), and reports cycles with the hand kernel's count as the performance
target. Tests skip when the Hexagon toolchain (`HEXAGON_TOOLS`, default `~/.cache/hexagon-oa-19/Tools`) or a Hexagon-capable
`CC` for MOCKDSP is missing.

Layout (shared with the non-HMX hand kernels, which go under their own subdirectories):

| path | what |
|---|---|
| `hexsim.py` | the harness: `tools()`, `capture_dsp()` (record MOCKDSP kernel calls instead of running them: qemu can't run HMX), `run_captured(src, bufs, work)` (one captured kernel on hexagon-sim -> output, pcycles per call), `run_hand(c_file, work, *args, includes=)` (build + run a hand driver) |
| `hmx/` | HMX kernels copied verbatim from onnxsim `scripts/android/hmx_gemm` (`SOURCE_REV` = the onnxsim commit), plus one small driver per family (`hand_<family>.c`: reads inputs from its working directory, prints `pcycles N` for one timed call after a warm-up, writes its output) |
| `test_hand_hmx_<family>.py` | one test file per kernel family |
| `rpn/`, ... | the qemu-based HVX / scalar families, below |

A driver never needs onnx/onnxruntime: tests write their own case directories (for the QDQ convs, `hmx/qc_case.h`'s format:
`meta.txt` = `M K N zx zy relu sx sy [H W k stride]` with hex floats, `x.bin`, `w.bin` (ONNX layout), `bq.bin`, `sw.bin`,
`ref.bin`), with the reference computed in numpy.

Run: `HMX=1 DEV=DSP MOCKDSP=1 TC=1 HVX_ARCH=v69 CC=clang-19 HEXAGON_TOOLS=... python -m pytest -s test/external/dsp/hand`
(`-s` shows the cycle lines). onnxsim's Hexagon tinygrad CI job runs this directory against the pinned fork.

| family | hand kernel | tinygrad | hexagon-sim (hand / tinygrad pcycles) |
|---|---|---|---|
| `gemm` | fp16 GEMM, DDR-fed, weights prepacked on the host (`hmx_gemm.h`) | HMX fp16 TensorCore, accumulator kept in HMX | 128x576x256: 35032 / 39149; 64^3: 2715 / 3976 |
| `qconv` | 1x1 QDQ conv, `QC_EXACT` (`hmx_qconv.h`) | int8 `:cm` TensorCore + fused exact requant | 256x128x128: 24710 / 64842; 128x256x64 relu: 25358 / 21345 |
| `qconv` | 3x3 QDQ conv s1/s2, `QC_EXACT` (`hmx_qconv3.h`: shifted copies / phase split, `:single` windows, stitch) | grid-form conv (`TC_OPT=1`) + fused exact requant | 16x16x64->64 s1: 21088 / 87147; 16x16x128->128 s2 relu: 28388 / 114694 |

## HVX / scalar families under qemu (`harness.py`)

The non-HMX kernels (RPN post-processing, RoiAlign, deformable-attention sampling, ...) run on **qemu-hexagon** on both
sides instead: the hand kernel as a freestanding ELF built with clang-19, tinygrad through MOCKDSP. They target
`HVX_ARCH=v65` (qemu 8.2 can't decode the v68+ qfloat ops), so `conftest.py` skips them when `HVX_ARCH` is v68 or newer
-- run them in their own pytest invocation, without the HMX families' `HMX=1 TC=1 HVX_ARCH=v69`:

```
PYTHONPATH=. CC=clang-19 HEXAGON_TOOLS=<Tools> python -m pytest test/external/dsp/hand/rpn -v
```

Needs `qemu-hexagon(-static)`, an LLVM clang with the Hexagon target plus `ld.lld` (clang-19; `HEXAGON_CLANG` to
override), and for kernels written with `Q6_*` intrinsics the toolchain's target headers (`HEXAGON_TOOLS`). Each
family directory holds verbatim copies of the hand kernels (`*_kernel.h`, provenance below), one `*_oracle.c` qemu
driver per kernel, `tg_<family>.py` with the tinygrad implementations, and `test_<family>.py`.

| file | what |
|---|---|
| `harness.py` | toolchain discovery / skip reasons, `build_hand` (freestanding qemu ELF, `-ffp-contract=off`), `run_hand` (files in, files out, `insns <label> <n>` lines back), `run_tinygrad` (outputs + summed instruction count + kernel count), `record`, bit-exact `mismatch` |
| `qemu_rt.h` | the drivers' runtime: trap0 syscalls, file load/store, QEMU's instruction counter, libc/divide helpers. A driver's argv: input files, output files, then integers; it prints `insns <label> <n>` around the kernel call |
| `exact.py` | host-side helpers that make ORT's exactness contracts expressible in tinygrad (QuantizeLinear by threshold count) |

### The metric: instructions, not cycles

Both sides are counted with QEMU's executed-instruction counter on the same inputs: the hand kernel around its call,
tinygrad as the sum over its kernels (MOCKDSP returns `inscount / 1e9` per kernel as its "time"). That is a count, not
a cycle count -- hexagon-sim `--timing` would give cycles, but tinygrad's `HEXSIM=1` runs every kernel on zero-filled
buffers, which says nothing for the data-dependent gathers most of these kernels exist for. Read the hand count as the
target and the ratio as the signal. Input copies to the device aren't counted on either side. (`hexsim.py` replays one
captured kernel on hexagon-sim with its real arguments, which suits the single-kernel HMX families; these graphs chain
up to 190 kernels and ~4 G instructions, too slow for hexagon-sim `--timing`.)

### Status

`.results.jsonl` (gitignored) gets one line per case when the tests run. Counts below: hand / tinygrad instructions
under qemu, and tinygrad's kernel count.

#### `rpn/` -- Mask R-CNN RPN post-processing (all bit-exact with ONNX Runtime)

Hand kernels: `pd_kernel.h` (proposal decode), `topk_kernel.h` (exact TopK), `nms_kernel.h` (NonMaxSuppression).
Shapes are the real graph's (800x1088 input, FPN P2..P6, k = 1000); values synthetic in the real ranges.

| case | result | hand insns | tinygrad insns | ratio | kernels |
|---|---|---:|---:|---:|---:|
| proposal decode P2 (163 200 anchors, k 1000) | exact | 343 539 | 16 852 735 | 49x | 22 |
| proposal decode P5 (2 550 anchors, k 1000) | exact | 323 074 | 6 673 454 | 21x | 18 |
| proposal decode P6 (663 anchors, k 663) | exact | 218 070 | 4 414 001 | 20x | 18 |
| TopK n 163 200, k 1000 | exact | 1 710 527 | 71 359 937 (was 3 874 019 724) | 42x (was 2265x) | 155 |
| TopK n 40 800, k 1000 | exact | 506 655 | 17 703 673 (was 1 369 516 127) | 35x (was 2703x) | 133 |
| TopK n 10 200, k 1000 | exact | 223 817 | 4 383 131 (was 725 581 225) | 20x (was 3242x) | 111 |
| TopK n 2 550, k 1000 | exact | 153 759 | 1 077 481 (was 78 335 193) | 7.0x (was 509x) | 89 |
| TopK n 1 465, k 1000 | exact | 112 446 | 512 813 | 4.6x | 78 |
| TopK n 663, k 663 | exact | 61 666 | 221 934 | 3.6x | 67 |
| TopK n 106, k 100 | exact | 17 107 | 16 664 | 0.97x | 37 |
| NMS pairwise SuppressByIOU, 1000 boxes (iou 0.7) | exact | 26 005 765 | 84 413 024 | 3.2x | 1 |
| NMS pairwise SuppressByIOU, 300 boxes (iou 0.5) | exact | 2 448 821 | 6 759 624 | 2.8x | 1 |
| NMS greedy, blocked, 1000 boxes (iou 0.7) | exact | 7 728 599 | 109 943 091 | 14x | 1168 |
| NMS greedy, blocked, 300 boxes (iou 0.5) | exact | 216 107 | 9 775 829 | 45x | 390 |

- Proposal decode: exact through `exact.py`'s QuantizeLinear-by-thresholds (tinygrad's float division is `x * (1/y)`,
  and its symbolic rewrites treat float algebra as real algebra, so `round(x / scale)` can't be written directly) and
  the hand kernel's own Cephes `expf` polynomial written as Tensor ops. The gap is the gathers: each of the 4 deltas and
  the anchor base is a separate one-hot-folded gather kernel.
- TopK: one `uint64` sort key (`tk_key(value) << 32 | ~index`) through a bitonic network is exactly ORT's order,
  ties included, without `Tensor.sort`'s O(n^2) index recovery (26.6 G mask elements at n = 163 200). Two fixes took
  it from 2265x to 42x at the real P2 size:
  - **an arange that gets padded isn't folded** by tinygrad's symbolic: `pad(f(x, arange(n)))` stayed the O(n^2)
    reduce form of arange (99% of the n = 2 550 case). The index is now built from an arange over the unpadded
    length and the *key* is padded. (A tinygrad gap worth fixing generally; not fixed here.)
  - a tournament instead of one full sort: sort chunks of next_pow2(k) keys, then merge pairs and keep each merge's
    top chunk -- n log2(k)^2 / 2 compare-exchanges instead of n log2(n)^2 / 2.
  The hand kernel is still ahead because it is select-then-sort (threshold, stream compaction of ~1.5 k survivors,
  counting sort): tinygrad has no stream compaction (data-dependent output size).
- NMS: the pairwise test is the same fp32 ops in the same order as ORT's `SuppressByIOU` (tinygrad renders
  `a * (1/b)` as a true division, `FDIV`, on the C backends). Greedy selection is the lexicographically-first
  maximal independent set -- inherently sequential, and tinygrad has no device-side data-dependent loop. The exact
  form used here is blocked: in visit order, blocks of 32 boxes are checked against the earlier blocks' kept boxes in
  one reduction, then at most 32 sweeps resolve the block itself -- O(n^2 + n * 32^2), exact for any input.
- Not covered: `rpn_fused` itself (it composes these three kernels plus the level merge; checking it needs ORT
  captures of the real rest.onnx span, as onnx-simplifier's own CI notes).

#### `roialign/` -- Mask R-CNN box/mask-head RoiAlign

Hand kernels: `roialign_kernel.h` (fp32, channels-last, per-tap HVX FMA over channels) and `roialign_u8_kernel.h`
(the shipped one: uint8 NHWC FPN map straight into the quantized head input, Q14 bilinear weights, exact int32
accumulation, `vzxt` + `vmpyacc`). Real FPN level shapes P2..P5 (C = 256), 7x7 and 14x14, sampling ratio 2, 16
synthetic RoIs per call (the real count is data-dependent). Hand vs ORT: fp32 within 1e-4, uint8 within 1 LSB of
QuantizeLinear(ORT fp32). tinygrad vs hand: bit-exact in all 16 cases.

| case | hand insns | tinygrad insns | ratio | kernels |
|---|---:|---:|---:|---:|
| fp32 P2 7x7 | 10 037 685 | 48 543 409 | 4.8x | 2 |
| fp32 P2 14x14 | 50 305 393 | 168 890 354 | 3.4x | 2 |
| fp32 P5 14x14 | 47 623 873 | 169 258 673 | 3.6x | 2 |
| uint8 P2 7x7 | 274 055 | 42 382 946 | 155x | 2 |
| uint8 P2 14x14 | 1 167 725 | 72 662 014 | 62x | 2 |
| uint8 P5 14x14 | 1 107 999 | 69 962 926 | 63x | 2 |

- The data-dependent gather (each tap reads the C channels of a pixel chosen by the RoI) is plain Tensor indexing by
  a computed int tensor; tinygrad folds the one-hot form into a direct load, so the gather itself is not the gap.
- `bin_h = roi_h / OH` divides by a constant, which tinygrad folds into `* (1/OH)` -- not IEEE division for OH = 7, 14.
  The divisors come in as a two-element device buffer instead, so the division stays a division.
- fp32 is within 5x; the uint8 kernel's gap (62-155x) is its per-tap `vzxt` + `vmpyacc` over 128 channels at once:
  tinygrad's code for the u8 x Q14 multiply-accumulate is scalar-ish 32-bit lane math.
- These compile tinygrad's kernels with `-ffp-contract=off` (conftest.py, qemu families only): with clang's default
  contraction the fp32 4-tap sum becomes FMAs, and 5% of outputs differ from the hand kernel (and ORT) by an ulp.

#### `msda/` -- multi-scale deformable attention (RT-DETR decoder, BEVFormer TSA / SCA)

Hand kernel: `msda_kernel.h`'s scalar body (built for v65; its HVX body is V68+ qf32, which neither qemu nor
hexagon-sim reproduces). Cases are onnx-simplifier's `msda_ref.py synthetic` shapes with its edge cases (points off the
map, on pixel centers/borders, a query no camera sees), float32 and uint8 value maps. tinygrad is bit-exact in all 6.

| case | hand insns | tinygrad insns | ratio | kernels |
|---|---:|---:|---:|---:|
| RT-DETR decoder, 300 queries, 3 levels, f32 | 36 383 901 | 1 016 174 045 | 28x | 99 |
| RT-DETR decoder, u8 | 52 349 853 | 1 845 601 460 | 35x | 99 |
| BEVFormer TSA, 96 queries, 2 frames, f32 | 5 501 573 | 1 299 275 214 | 236x | 67 |
| BEVFormer SCA, 6 cameras + visibility, f32 | 9 210 909 | 63 037 011 | 6.8x | 194 |

The gap is structural: the exact contract fixes the accumulation order (every map, level, point, tap in turn), so the
tinygrad version is one long add chain of gathered rows that it can't reduce in parallel. Exactness needed three
float-order fixes in tinygrad itself (below: FLOAT_REASSOC, the right-operand parens, `-x` in parens).

#### `layout/` -- FPN NCHW -> NHWC transpose (RoiAlign's input layout)

`layout_kernels.h`: HVX 32x32-block register transpose (five `vshuff` rounds). tinygrad `x.T.contiguous()`, one
kernel, bit-exact, **1.8-2.0x** the hand instructions on every level (P2: 16.1 M vs 29.0 M).

#### `mcc/` -- MCC decoder HVX steps: **xfail**, tracked target

LayerNorm / base-2 softmax / GELU in `mb_hvx.h` are V69 qfloat code: qemu can't decode it and hexagon-sim runs V69's
`.sf` ops as IEEE, so neither gives the phone's rounding. Needs golden tiles captured on the phone (test_mcc.py says how);
this pass had no phone access.

Not covered yet: Fast-BEV's `fbgather` (the tinygrad version crashed clang at the real 41 MB volume size -- needs a
look), StreamPETR / Sparse4D attention kernels, the LLM decode loop (`llm_tinygrad/hvx`, qfloat like MCC).

### Performance: before / after (tinygrad instructions / hand instructions, qemu)

All still bit-exact. "Before" is the first passing version of each oracle.

| family, case | before | after | what changed |
|---|---:|---:|---|
| RPN TopK, n 163 200 (P2) | 2265x | **42x** | index arange no longer padded (see below) + tournament of chunk sorts |
| RPN TopK, n 2 550 | 509x | 7.0x | same |
| MSDA BEVFormer TSA f32 / u8 | 236x / 252x | **1.7x / 1.2x** | tinygrad: one-hot gathers no longer split (`schedule/rangeify.py`) |
| MSDA RT-DETR decoder f32 / u8 | 28x / 35x | **1.4x / 1.2x** | same |
| MSDA BEVFormer SCA f32 / u8 | 6.8x / n.a. | 6.0x / 2.4x | same + one kernel per value map (the fused u8 kernel was 700 KB of C) |
| RPN NMS greedy, real size (1000 / 663 / 300 boxes) | xfail | **14x / 15x / 45x** | blocked greedy form: exact for any input, no data-dependent loop needed |
| RoiAlign uint8, 7x7 / 14x14 | 134-155x / 62-66x | **14-16x / 15-18x** | tinygrad DSP heuristic: HVX-width upcast of the channel axis (`codegen/opt/heuristic.py`) + weights/tap rows realized first |
| RPN proposal decode, P2..P6 | 20-49x | **4.6-4.9x** | gather split fix + QuantizeLinear threshold count as an 8-step binary search |
| RoiAlign fp32, NMS pairwise, FPN layout | 3-5x, 2-3x, 1.8-2x | unchanged | |

tinygrad changes behind these (fork-wide, not special-cased in the tests):
- **`split_reduceop` skips one-hot gathers** (`schedule/rangeify.py`): `sum(where(idx == arange, src, 0))` is folded by
  codegen into a direct load of `src[idx]`, so there is nothing to parallelize -- but when the reduce is split first
  (it is for any one-hot range >= 32768), the fold leaves one compare + guarded load per chunk: a 40 000-row gather
  became 160 guarded loads per output plus a 160-way reduce. This one fix is most of MSDA's and proposal decode's gains.
- **DSP heuristic: no masked-axis upcast when a contiguous HVX upcast is available, for reductions**
  (`codegen/opt/heuristic.py`): the full upcast of small masked axes (up to 7x7) filled the upcast budget, so
  RoiAlign's 256-channel axis stayed a scalar loop around 784 gathers. Limited to kernels with a reduce: for the
  bitonic sort's elementwise split/cat stages the masked upcasts are better (dropping them doubled TopK).
- Found, not fixed: **an arange that is later padded isn't folded** by symbolic and stays arange's O(n^2) reduce form
  (99% of TopK n = 2 550 before). The TopK oracle now pads the key instead of the arange.

What's left, per family:
- TopK (42x at P2): tinygrad sorts every element; the hand kernel thresholds and compacts to ~1.5 k survivors first.
  Stream compaction (a data-dependent output size) is the missing capability.
- NMS greedy (14-45x): the O(n^2) pairwise matrix is computed in full; the hand kernel tests each candidate only
  against the boxes kept so far and exits early (the data-dependent loop again).
- RoiAlign u8 (~15x): the u8 x Q14 multiply-accumulate is 32-bit lane math; the hand kernel's `vzxt` + `vmpyacc`
  (u16 x u16 -> u32 accumulate on the widened pair) has no tinygrad lowering yet.
- MSDA SCA f32 (6x): the per-point attention-weight scaling is 192 tiny kernels; fusing it is next.

### tinygrad DSP backend fixes these tests needed (`tinygrad/runtime/ops_dsp.py`)

Found by the oracles, all in rendering; none touch the HMX paths.

1. **Bool vectors were bit-packed.** clang's `ext_vector_type` of `_Bool` stores N bools in N *bits*, so a `_Bool8`
   store wrote one byte where tinygrad's bool buffer has eight -- a silent wrong result (a pairwise `>` kernel came
   back with 60 of 76 true entries), and a 128-lane one aborted the Hexagon backend (`Cannot select: store (s128)`),
   so even `Tensor > 0.5` didn't compile on DEV=DSP. Bool vectors now render as `unsigned char` lanes (0/1).
2. **`truncf` in a freestanding link** (MOCKDSP): floor/ceil/round decompose to TRUNC, which rendered as
   `__builtin_truncf`, a libm call the `-nostdlib` qemu link doesn't have. Now an exact bit-mask trunc.
3. **LLVM 19 aborts on packed compare results** (`Cannot select: v8i8 = bitcast <v8i1 HexagonISD::V2Q>`) when a few
   scalar float compares, ANDed with other bools, are stacked into a small char/int vector. On v65/v66 (MOCKDSP's
   target, no HVX float) each scalar float compare result now goes through an empty asm, so it stays a scalar
   predicate. Reproducer: `(u8x8){m[i] & (0.7f < a[i]*b[i]), ...}` with `-mhvx=v65`.
5. **Float reassociation** (`uop/symbolic.py`, all backends): "move add/mul consts to end" ((x + c) + y -> (x + y) + c)
   and two-stage constant folding ((x + c1) + c2 -> x + (c1 + c2)) are real-arithmetic rewrites that change a float
   expression's rounding. New `FLOAT_REASSOC` context var (default 1 = unchanged); `FLOAT_REASSOC=0` skips them for
   floats. The qemu families set it (conftest.py). MSDA's `r.x*W - 0.5 + off` was the case that found it.
6. **Right-hand operand parens** (`renderer/cstyle.py`, all C backends): the ALU renderer dropped the parens of any
   same-op operand, so a float `a + (b + c)` rendered as `a+b+c`, which C evaluates as `(a+b)+c`. Now only the left
   operand's parens are dropped for floats.
7. **`-x` rendered without parens** (`cstyle.py`): `a - -b` came out as `a--b`, a compile error. Now `(-x)`.
4. **`-Wunused-variable` under `-Werror`**: `_splat_of_loaded_lane` re-reads a splatted lane as a scalar, which can
   leave the vector load it replaced declared but unused. The DSP compile now passes `-Wno-unused-variable`.

### Provenance

Copied verbatim from onnx-simplifier `origin/master` at `6927c104`:

| here | onnx-simplifier |
|---|---|
| `rpn/pd_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/proposal_decode/pd_kernel.h` |
| `rpn/topk_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/topk/topk_kernel.h` |
| `rpn/nms_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/nms/nms_kernel.h` |
| `roialign/roialign_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/roialign_fast/roialign_kernel.h` |
| `roialign/roialign_u8_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/roialign_fast/roialign_u8_kernel.h` |
| `msda/msda_kernel.h`, `msda/msda_shape.h` | `scripts/android/msda_hvx/` |
| `layout/layout_kernels.h` | `scripts/android/tinygrad_hexagon_bridge/fpn_channels_last/layout_kernels.h` |

