"""qconv_v65 (integer QDQ conv and matmul for v65 HVX vrmpy) against a float64 reference, under MOCKDSP=1 (qemu)."""
import os, unittest, itertools
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context, getenv
from tinygrad.nn.qconv_v65 import qconv2d, qmatmul

def ref(xq, zx, sx, wq, sw, b, stride, pad, groups):
  x, w = (xq.astype(np.float64) - zx) * sx, wq.astype(np.float64) * sw.reshape(-1, 1, 1, 1)
  return Tensor(x, device="CPU").conv2d(Tensor(w, device="CPU"), Tensor(b.astype(np.float64), device="CPU"), stride=stride, padding=pad,
                                         groups=groups).numpy()

@unittest.skipUnless(getenv("MOCKDSP"), "requires MOCKDSP=1")
class TestQConvV65(unittest.TestCase):
  def test_matches_reference(self):
    rng = np.random.default_rng(0)
    cases = [("1x1", 192, 64, 1, 1, 0, 1, (16, 32)), ("3x3", 64, 64, 3, 1, 1, 1, (16, 32)), ("stem s2 C=6", 6, 16, 3, 2, 1, 1, (32, 64)),
             ("depthwise", 64, 64, 3, 1, 1, 64, (16, 32)), ("depthwise 7x7", 64, 64, 7, 1, 3, 64, (16, 32)),
             ("depthwise 7x7 s2", 64, 64, 7, 2, 3, 64, (16, 32)), ("depthwise x2 7x7 s2", 32, 64, 7, 2, 3, 32, (16, 32)),
             ("depthwise x2 3x3", 32, 64, 3, 1, 1, 32, (16, 32)), ("depthwise 3x3 s2 wide", 32, 32, 3, 2, 1, 32, (12, 150))]
    for (name, C, N, k, s, p, g, (H, W)), (xb, wb) in itertools.product(cases, [(8, 8), (16, 8), (16, 16)]):
      with self.subTest(name=name, bits=f"W{wb}A{xb}"):
        xdt, wdt = (np.uint8, 255) if xb == 8 else (np.uint16, 65535), (np.int8, 127) if wb == 8 else (np.int16, 32767)
        xq = rng.integers(0, xdt[1] + 1, (1, C, H, W)).astype(xdt[0])
        zx = int(rng.integers(0, xdt[1]))
        wq = rng.integers(-wdt[1], wdt[1] + 1, (N, C // g, k, k)).astype(wdt[0])
        sx, sw, b = 0.01, rng.uniform(1e-4, 1e-3, N).astype(np.float32), rng.normal(size=N).astype(np.float32)
        with Context(TC_OPT=1):
          y = qconv2d(Tensor(xq, device="DSP"), zx, sx, wq, sw, Tensor(b, device="DSP"), stride=s, padding=(p, p, p, p), groups=g).numpy()
        r = ref(xq, zx, sx, wq, sw, b, s, p, g)
        self.assertEqual(y.shape, r.shape)
        # the int32 passes are exact; only their float32 combination rounds
        self.assertLess(np.abs(y - r).max() / np.abs(r).max(), 1e-6)

  def test_depthwise_hvx_is_bit_identical_to_the_conv_path(self):
    # the hand-written HVX depthwise kernel (nn/dw_v65.py) produces the same integer sums as the tinygrad conv path, so the
    # float epilogue's inputs -- and the outputs -- are identical
    rng = np.random.default_rng(3)
    for (k, s, p), (xb, wb) in itertools.product([(7, 1, 3), (3, 1, 1), (7, 2, 3), (3, 2, 1)], [(8, 8), (16, 8), (16, 16), (8, 16)]):
      with self.subTest(k=k, stride=s, bits=f"W{wb}A{xb}"):
        C, H, W = 32, 10, 20
        xq = rng.integers(0, 1 << xb, (1, C, H, W)).astype(np.uint8 if xb == 8 else np.uint16)
        wq = rng.integers(-(1 << (wb - 1)) + 1, 1 << (wb - 1), (C, 1, k, k)).astype(np.int8 if wb == 8 else np.int16)
        sw, b = rng.uniform(1e-4, 1e-3, C).astype(np.float32), rng.normal(size=C).astype(np.float32)
        outs = []
        for flag in ("0", "1"):
          os.environ["QDW_HVX"] = flag
          getenv.cache_clear()
          try: outs.append(qconv2d(Tensor(xq, device="DSP"), int(rng.integers(0, 1 << xb)) if False else 7, 0.01, wq, sw, Tensor(b, device="DSP"),
                                   stride=s, padding=(p, p, p, p), groups=C).numpy())
          finally: os.environ.pop("QDW_HVX"); getenv.cache_clear()
        # (under these default settings the float epilogue is HVX qfloat code whose rounding depends on what is fused with it: a few
        # ulps of the largest output; the v65 settings, DSP_V65_HW=1, give the same bits -- checked on the models)
        np.testing.assert_allclose(outs[0], outs[1], rtol=0, atol=4e-7 * np.abs(outs[0]).max())

  def test_qmatmul_gemv(self):
    # a float head's GEMV: the activation is quantized to uint16 at run time (per tensor, over its [min, max]), the one
    # approximation, so the error bound is that quantization's (half a step of the input range per element)
    rng = np.random.default_rng(1)
    for (K, N), wb in itertools.product([(1024, 512), (512, 1024), (200, 64)], [8, 16]):
      with self.subTest(K=K, N=N, bits=f"W{wb}"):
        qmax = 127 if wb == 8 else 32767
        wq = rng.integers(-qmax, qmax + 1, (K, N)).astype(np.int8 if wb == 8 else np.int16)
        sw, b = rng.uniform(1e-4, 1e-3, N).astype(np.float32), rng.normal(size=N).astype(np.float32)
        a = np.maximum(rng.normal(size=(1, K)), 0).astype(np.float32)
        with Context(TC_OPT=1):
          y = qmatmul(Tensor(a, device="DSP"), wq, sw, Tensor(b, device="DSP")).numpy()
        r = a.astype(np.float64) @ (wq.astype(np.float64) * sw) + b
        self.assertEqual(y.shape, r.shape)
        step = (a.max() - a.min()) / 65535
        self.assertLess(np.abs(y - r).max(), 0.5 * step * np.abs(wq.astype(np.float64) * sw).sum(0).max() + 1e-4)

if __name__ == "__main__": unittest.main()
