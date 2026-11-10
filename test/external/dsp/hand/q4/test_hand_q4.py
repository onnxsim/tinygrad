"""The group-wise int4 HMX GEMM (hmx_q4.h, the PLaMo 3 layout) as a tinygrad DSP kernel (nn/onnx.py _matmul_q4) on hexagon-sim,
against a double reference on the dequantized weights (s * q rounded to fp16, the fp16 output rounded once)."""
import pathlib, tempfile, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from test.external.dsp.hand import hexsim
from tinygrad.nn.onnx import pack_int4_hmx, _matmul_q4

def quantize(W, G=128):
  """per (group of G rows, column) symmetric int4: s = fp16(max|W| / 7), q = clip(round(W / s), -8, 7). -> q (K, N), s (K/G, N)"""
  K, N = W.shape
  s = np.zeros((K // G, N), np.float16); q = np.zeros((K, N), np.int64)
  for g in range(K // G):
    blk = W[g * G:(g + 1) * G]
    mx = np.abs(blk).max(axis=0)
    s[g] = np.where(mx > 0, mx / 7, 1.0).astype(np.float16)
    q[g * G:(g + 1) * G] = np.clip(np.rint(blk / s[g].astype(np.float32)), -8, 7)
  return q, s

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(), "needs the Hexagon toolchain (HEXAGON_TOOLS) + clang")
class TestHandQ4Gemm(unittest.TestCase):
  def _run(self, M, K, N, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    W = rng.uniform(-0.05, 0.05, (K, N)).astype(np.float32)
    q, s = quantize(W)
    Wd = (q * np.repeat(s.astype(np.float32), 128, axis=0)).astype(np.float16)
    ref = A.astype(np.float64) @ Wd.astype(np.float64)
    codes, sc = pack_int4_hmx(q, s)
    with tempfile.TemporaryDirectory() as d:
      work = pathlib.Path(d)
      with hexsim.capture_dsp() as calls:
        _matmul_q4(Tensor(A), Tensor(codes), Tensor(sc)).realize()
      self.assertEqual(len(calls), 1)
      got, cyc = hexsim.run_captured(*calls[0], work)
      got = np.frombuffer(got, np.float16)[:M * N].reshape(M, N).astype(np.float64)
    err = np.abs(got - ref)
    self.assertTrue((err <= np.abs(ref) / 512 + 2e-3).all(), f"max abs err {err.max():.3g} (max |ref| {np.abs(ref).max():.3g})")
    print(f"\nhmx_q4 {M}x{K}x{N}: {cyc} pcycles, max abs err {err.max():.3g}")

  def test_one_group(self): self._run(32, 128, 32)
  def test_partial_rows(self): self._run(5, 512, 64, seed=1)
  def test_k_256(self): self._run(32, 256, 64, seed=2)
  def test_k_10240_down(self): self._run(32, 10240, 32, seed=3)  # PLaMo's down projection (one column block)

if __name__ == "__main__":
  unittest.main()
