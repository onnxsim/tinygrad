"""onnxsim hmx_gemm's hand-written fp16 GEMM (hmx_gemm.h) as the oracle for tinygrad's HMX fp16 TensorCore: both keep the
accumulator in HMX over all of K and round to fp16 once, so they must agree bit for bit. Cycles: the hand kernel's is the target
(it prepacks the weights on the host; tinygrad packs row-major B on every call)."""
import pathlib, tempfile, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from test.external.dsp.hand import hexsim

HMX = hexsim.HERE / "hmx"

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(), "needs the Hexagon toolchain (HEXAGON_TOOLS) + clang")
class TestHandHmxGemmF16(unittest.TestCase):
  def _run(self, M, K, N, seed=0):
    rng = np.random.default_rng(seed)
    A = ((rng.integers(-1000, 1001, (M, K))) / 1000).astype(np.float16)
    B = ((rng.integers(-1000, 1001, (K, N))) / 4000).astype(np.float16)
    ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float16)  # exact accumulation, one rounding
    with tempfile.TemporaryDirectory() as d:
      work = pathlib.Path(d)
      A.tofile(work / "a.bin"); B.tofile(work / "w.bin")
      out = hexsim.run_hand(HMX / "hand_gemm_f16.c", work, M, K, N, includes=[HMX])
      hand = np.fromfile(work / "c.bin", np.float16).reshape(M, N)
      hand_cyc = int(out.split("pcycles ")[1].split()[0])
      with hexsim.capture_dsp() as calls: Tensor(A).matmul(Tensor(B), dtype=dtypes.half).realize()
      self.assertEqual(len(calls), 1)
      tg, tg_cyc = hexsim.run_captured(*calls[0], work)
      tg = np.frombuffer(tg, np.float16)[:M * N].reshape(M, N)
    self.assertEqual(int((hand.view(np.uint16) != ref.view(np.uint16)).sum()), 0, "the hand oracle disagrees with exact accumulation")
    self.assertEqual(int((tg.view(np.uint16) != hand.view(np.uint16)).sum()), 0, "tinygrad disagrees with the hand kernel")
    print(f"\nfp16 GEMM {M}x{K}x{N}: hand {hand_cyc} pcycles, tinygrad {tg_cyc} ({tg_cyc / hand_cyc:.2f}x)")

  def test_gemm_f16(self): self._run(128, 576, 256)
  def test_gemm_f16_small(self): self._run(64, 64, 64, seed=1)

if __name__ == "__main__":
  unittest.main()
