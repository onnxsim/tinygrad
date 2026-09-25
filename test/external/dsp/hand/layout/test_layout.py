"""FPN NCHW -> NHWC transpose (layout_kernels.h, the conversion roialign/ needs): hand HVX 32x32-block register transpose
(five vshuff rounds) vs tinygrad's `x.T.contiguous()`, on the real FPN output shapes (800x1088 input: P2..P6, C = 256).
A bit-exact copy by construction; the numbers say how close tinygrad's generic transpose gets to the register one."""
import pathlib
import numpy as np
import pytest
import harness
from harness import build_hand, run_hand, run_tinygrad, record, mismatch

HERE = pathlib.Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(harness.missing(intrinsics=True) is not None, reason=str(harness.missing(intrinsics=True)))
LEVELS = [("P2", 200 * 272), ("P3", 100 * 136), ("P4", 50 * 68), ("P5", 25 * 34), ("P6", 13 * 17)]
C = 256

@pytest.fixture(scope="module")
def exe(tmp_path_factory):
  return build_hand(HERE / "layout_oracle.c", tmp_path_factory.mktemp("layout") / "layout_oracle", cpu="v65", hvx=True)

@pytest.mark.parametrize("lvl,HW", LEVELS, ids=[l for l, _ in LEVELS])
def test_chw_to_hwc(exe, lvl, HW):
  x = np.random.default_rng(HW).standard_normal((C, HW)).astype(np.float32)
  hand = run_hand(exe, [np.concatenate([x.reshape(-1), np.zeros(32, np.float32)])], [(np.float32, (HW, C))], ints=(C, HW))
  assert mismatch(hand.outputs[0], np.ascontiguousarray(x.T)) == 0
  tg = run_tinygrad(lambda t: t.T.contiguous(), x)
  bad = mismatch(tg.outputs[0], hand.outputs[0])
  record("layout", f"chw_hwc_{lvl}_C{C}", mismatched=bad, hand_insns=hand.insns["hvx"], tg_insns=tg.insns, tg_kernels=tg.kernels)
  assert bad == 0
