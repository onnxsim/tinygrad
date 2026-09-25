"""MCC's decoder HVX steps (onnx-simplifier scripts/android/mcc_hmx/mb_hvx.h: LayerNorm, base-2 softmax + self term,
GELU on the HMX tile layout) -- a tracked target, not yet an oracle.

These are V69 qfloat kernels: every step computes in qf16 / qf32 (the hf x hf -> qf32 widening multiply, qf32 row sums
and reciprocals, a 1536.0 magic-number exp2), whose rounding is the phone's. Neither harness here can reproduce it:
qemu 8.2 can't decode qfloat at all, and hexagon-sim runs V69's IEEE-named (`.sf`) vector ops as IEEE fp32 while the
phone computes them in qf32 (the HMX exact-requant oracles hit exactly this: exact on the sim, 94% wrong on the phone).
Their scalar-C fallbacks (mcc_block.h) aren't a substitute oracle either: they need libm's erff and __fp16 conversion
helpers that the freestanding qemu link doesn't have, and they compute a different function (IEEE fp32, erf GELU).

What would make these real oracles: golden outputs captured on the phone -- run mb_hvx.h's steps on fixed tile-layout
inputs through a small FastRPC harness (mcc_hmx's client already packs the tiles) and commit the input/output tiles
here; the test then compares tinygrad's lowering of the same step (the `tg/` port in onnx-simplifier, onnxsim/tinygrad#7)
against those bytes under the family's contract (fp16-level: the decoder's occupancy logits within 0.045 of float64,
mcc_hmx/README.md). Not done here: this pass had no phone access.
"""
import pytest

STEPS = ["layernorm", "softmax_base2_self_term", "gelu"]

@pytest.mark.xfail(strict=True, run=False, reason=(
  "V69 qfloat kernel: needs phone-captured golden tiles (qemu can't decode qfloat; hexagon-sim runs V69's .sf vector ops "
  "as IEEE fp32, not the phone's qf32)"))
@pytest.mark.parametrize("step", STEPS)
def test_mcc_hvx_step(step):
  raise NotImplementedError(step)
