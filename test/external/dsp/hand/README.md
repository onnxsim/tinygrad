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
| TopK n 163 200, k 1000 | exact | 1 710 527 | 3 874 019 724 | 2265x | 190 |
| TopK n 10 200, k 1000 | exact | 223 817 | 725 581 225 | 3242x | 120 |
| TopK n 1 465, k 1000 | exact | 112 446 | 521 571 | 4.6x | 78 |
| TopK n 663, k 663 | exact | 61 666 | 220 816 | 3.6x | 66 |
| NMS pairwise SuppressByIOU, 1000 boxes (iou 0.7) | exact | 26 005 765 | 84 413 024 | 3.2x | 1 |
| NMS pairwise SuppressByIOU, 300 boxes (iou 0.5) | exact | 2 448 821 | 6 759 624 | 2.8x | 1 |
| NMS greedy, 96 boxes, 96 unrolled sweeps | exact | 109 302 | 32 704 413 | 299x | 137 |
| NMS greedy at the real size (1000 boxes) | **xfail** | | | | |

- Proposal decode: exact through `exact.py`'s QuantizeLinear-by-thresholds (tinygrad's float division is `x * (1/y)`,
  and its symbolic rewrites treat float algebra as real algebra, so `round(x / scale)` can't be written directly) and
  the hand kernel's own Cephes `expf` polynomial written as Tensor ops. The gap is the gathers: each of the 4 deltas and
  the anchor base is a separate one-hot-folded gather kernel.
- TopK: one `uint64` sort key (`tk_key(value) << 32 | ~index`) through the bitonic network is exactly ORT's order,
  ties included, without `Tensor.sort`'s O(n^2) index recovery (26.6 G mask elements at n = 163 200). The hand kernel
  is select-then-sort (threshold, stream compaction of ~1.5 k survivors, counting sort): tinygrad has no stream
  compaction (data-dependent output size), so it sorts all n. The jump between n = 1 465 and n = 2 550 (0.5 M to 78 M
  instructions) is worth a look.
- NMS: the pairwise test is the same fp32 ops in the same order as ORT's `SuppressByIOU` (tinygrad renders
  `a * (1/b)` as a true division, `FDIV`, on the C backends). Greedy selection is the lexicographically-first
  maximal independent set -- inherently sequential. **Missing capability: a device-side data-dependent loop** (sweep
  the keep mask until it stops changing, or the hand kernel's `break` on the first suppressing box). Without it the
  exact form is rounds = n Jacobi sweeps, O(n^3), so the real-size case is `xfail(run=False)`.
- Not covered: `rpn_fused` itself (it composes these three kernels plus the level merge; checking it needs ORT
  captures of the real rest.onnx span, as onnx-simplifier's own CI notes).

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
4. **`-Wunused-variable` under `-Werror`**: `_splat_of_loaded_lane` re-reads a splatted lane as a scalar, which can
   leave the vector load it replaced declared but unused. The DSP compile now passes `-Wno-unused-variable`.

### Provenance

Copied verbatim from onnx-simplifier `origin/master` at `6927c104`:

| here | onnx-simplifier |
|---|---|
| `rpn/pd_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/proposal_decode/pd_kernel.h` |
| `rpn/topk_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/topk/topk_kernel.h` |
| `rpn/nms_kernel.h` | `scripts/android/tinygrad_hexagon_bridge/nms/nms_kernel.h` |

