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

A driver never needs onnx/onnxruntime: tests write their own case directories (for the QDQ convs, `hmx/qc_case.h`'s format:
`meta.txt` = `M K N zx zy relu sx sy [H W k stride]` with hex floats, `x.bin`, `w.bin` (ONNX layout), `bq.bin`, `sw.bin`,
`ref.bin`), with the reference computed in numpy.

Run: `HMX=1 DEV=DSP MOCKDSP=1 TC=1 HVX_ARCH=v69 CC=clang-19 HEXAGON_TOOLS=... python -m pytest -s test/external/dsp/hand`
(`-s` shows the cycle lines). onnxsim's Hexagon tinygrad CI job runs this directory against the pinned fork.

| family | hand kernel | tinygrad | hexagon-sim (hand / tinygrad pcycles) |
|---|---|---|---|
| `qconv` | 1x1 QDQ conv, `QC_EXACT` (`hmx_qconv.h`) | int8 `:cm` TensorCore + fused exact requant | 256x128x128: 24710 / 64842; 128x256x64 relu: 25358 / 21345 |
| `qconv` | 3x3 QDQ conv s1/s2, `QC_EXACT` (`hmx_qconv3.h`: shifted copies / phase split, `:single` windows, stitch) | grid-form conv (`TC_OPT=1`) + fused exact requant | 16x16x64->64 s1: 21088 / 87147; 16x16x128->128 s2 relu: 28388 / 114694 |
