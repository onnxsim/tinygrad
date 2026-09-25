"""The hand-written QDQ-exact 1x1 conv (onnxsim hmx_gemm/hmx_qconv.h, QC_EXACT) as the oracle for tinygrad's lowering of the
same layer -- the int8 :cm TensorCore with the fused ORT-exact requantization:
  y = clamp(rne(fp32(fp32(x @ w + b - zx sum w) * M)) + zy, lo, 255),  M = fp32(fp32(sx sw) / sy)
Both run on hexagon-sim on the same data; they must agree byte for byte (and with numpy's fp32 of ORT's formula). Cycles are
reported with the hand kernel's as the target."""
import os, pathlib, struct, tempfile, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from test.external.dsp.hand import hexsim

HMX = hexsim.HERE / "hmx"
f32 = np.float32

def case(M, K, N, zx, zy, relu, seed):
  rng = np.random.default_rng(seed)
  x = rng.integers(0, 256, (M, K), dtype=np.uint8); w = rng.integers(-127, 128, (N, K), dtype=np.int8)
  bq = rng.integers(-20000, 20000, N, dtype=np.int32)
  sx, sy = f32(0.0213), f32(0.0917); sw = rng.uniform(0.002, 0.01, N).astype(f32)
  m = (sx * sw / sy).astype(f32)
  bex = bq.astype(np.int64) - zx * w.astype(np.int64).sum(1)
  acc = x.astype(np.int64) @ w.T.astype(np.int64) + bex
  lo = zy if relu else 0
  ref = (np.clip(np.rint(acc.astype(f32) * m), lo - zy, 255 - zy) + zy).astype(np.uint8)
  return dict(x=x, w=w, bq=bq, sw=sw, sx=sx, sy=sy, zx=zx, zy=zy, lo=lo, m=m, bex=bex.astype(np.int32), ref=ref)

def tinygrad_layer(c):
  acc = Tensor(c["x"]).matmul(Tensor(np.ascontiguousarray(c["w"].T)), dtype=dtypes.int32) + Tensor(c["bex"])
  return ((acc.cast(dtypes.float32) * Tensor(c["m"])).round() + float(c["zy"])).clip(c["lo"], 255).cast(dtypes.uint8)

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(), "needs the Hexagon toolchain (HEXAGON_TOOLS) + clang")
class TestHandHmxQconv1x1(unittest.TestCase):
  def _run(self, M, K, N, zx, zy, relu, seed=0):
    c = case(M, K, N, zx, zy, relu, seed)
    with tempfile.TemporaryDirectory() as d:
      work = pathlib.Path(d)
      (work / "meta.txt").write_text(f"{M} {K} {N} {zx} {zy} {int(relu)} {float(c['sx']).hex()} {float(c['sy']).hex()}\n")
      for n in ("w", "bq", "sw"): c[n].tofile(work / f"{n}.bin")
      c["x"].tofile(work / "x.bin"); c["ref"].tofile(work / "ref.bin")
      out = hexsim.run_hand(HMX / "hand_qconv1x1.c", work, ".", includes=[HMX])
      hand = np.fromfile(work / "y.bin", np.uint8).reshape(M, N)
      hand_cyc = int(out.split("pcycles ")[1].split()[0])
      with hexsim.capture_dsp() as calls: tinygrad_layer(c).realize()
      self.assertEqual(len(calls), 1)
      tg, tg_cyc = hexsim.run_captured(*calls[0], work)
      tg = np.frombuffer(tg, np.uint8)[:M * N].reshape(M, N)
    self.assertEqual(int((hand != c["ref"]).sum()), 0, "the hand oracle disagrees with ORT's formula")
    self.assertEqual(int((tg != hand).sum()), 0, "tinygrad disagrees with the hand kernel")
    print(f"\n1x1 QDQ conv {M}x{K}x{N}: hand {hand_cyc} pcycles, tinygrad {tg_cyc} ({tg_cyc / hand_cyc:.2f}x)")
    return hand_cyc, tg_cyc

  def test_qconv1x1(self): self._run(256, 128, 128, 118, 131, False)
  def test_qconv1x1_relu(self): self._run(128, 256, 64, 0, 7, True, seed=1)

if __name__ == "__main__":
  unittest.main()
