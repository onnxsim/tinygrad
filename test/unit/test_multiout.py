import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import run_linear

def _kernels(*ts:Tensor) -> list:
  """schedule ts and run it -> the kernels"""
  linear, var_vals = ts[0].linear_with_vars(*ts[1:])
  run_linear(linear, var_vals)
  return [si for si in linear.src if si.op is Ops.CALL and si.src[0].op is Ops.SINK]

def _stores(k) -> int: return len([u for u in k.src[0].toposort() if u.op is Ops.STORE])

class TestMultiOutput(unittest.TestCase):
  """MULTIOUT (schedule/rangeify.py merge_outputs): an elementwise consumer that reads a buffer only where it stores becomes a
  second STORE of the buffer's producer"""
  def setUp(self):
    np.random.seed(0)
    self.xn = np.random.randn(8, 16).astype(np.float32)

  def _ln(self):
    x, g = Tensor(self.xn).realize(), Tensor(np.linspace(0.5, 2, 16, dtype=np.float32)).realize()
    return (x.layernorm() * g).contiguous()

  def _ref(self):
    xn = self.xn
    return (xn - xn.mean(1, keepdims=True)) / np.sqrt(xn.var(1, keepdims=True) + 1e-5) * np.linspace(0.5, 2, 16, dtype=np.float32)

  def test_cast_and_add_into_producer(self):
    # the fp16 copy of y + p and y * 2 become stores of y's normalize kernel; the row sum (a reduce) stays a kernel of its own
    for mo in (0, 1):
      with Context(MULTIOUT=mo, SCACHE=0):
        y = self._ln()
        p = Tensor(np.ones((8, 16), np.float32)).realize()
        a, b, c = (y + p).half().contiguous(), (y * 2).contiguous(), y.sum(1).contiguous()
        ks = _kernels(a, b, c)
        self.assertEqual(max(_stores(k) for k in ks), 1 if mo == 0 else 3)
        if mo: self.assertEqual(len(ks), 4)  # mean, var, normalize (+ 2 stores), the row sum
        np.testing.assert_allclose(a.numpy(), (self._ref() + 1).astype(np.float16), rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(b.numpy(), self._ref() * 2, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(c.numpy(), self._ref().sum(1), rtol=1e-4, atol=1e-4)

  def test_store_into_rows_of_padded_buffer(self):
    # the GEMM_FP16 A operand form (nn/onnx.py pad16): the cast stored into the first rows of a bigger buffer
    with Context(MULTIOUT=1, SCACHE=0):
      y = self._ln()
      b = (y * 2).contiguous()
      yp = Tensor.empty(10, 16, dtype=dtypes.half)
      yp[:8].assign(y.half())
      ks = _kernels(b, yp)
      self.assertEqual(max(_stores(k) for k in ks), 3)
      np.testing.assert_allclose(yp.numpy()[:8], self._ref().astype(np.float16), rtol=1e-2, atol=1e-2)
      np.testing.assert_allclose(b.numpy(), self._ref() * 2, rtol=1e-4, atol=1e-4)

  def test_dead_store(self):
    # y read only by the merged consumers: MULTIOUT=2 drops its store (a whole-program compile), MULTIOUT=1 keeps it (y is realized)
    for mo, n in ((1, 3), (2, 2)):
      with Context(MULTIOUT=mo, SCACHE=0):
        y = self._ln()
        a, b = y.half().contiguous(), (y * 2).contiguous()
        self.assertEqual(max(_stores(k) for k in _kernels(a, b)), n)
        np.testing.assert_allclose(a.numpy(), self._ref().astype(np.float16), rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(b.numpy(), self._ref() * 2, rtol=1e-4, atol=1e-4)

  def test_no_merge(self):
    # a transposed read, a masked (padded) producer, a reduce consumer, a transposing producer: one store per kernel
    with Context(MULTIOUT=1, SCACHE=0):
      x = Tensor(self.xn).realize()
      y = (x * 3).contiguous()
      t, s = (y.T * 2).contiguous(), y.sum(0).contiguous()
      z = (x.pad(((0, 0), (0, 4))) + 1).contiguous()
      w = (z * 2).contiguous()
      xt = (x.T * 2).contiguous()
      h = xt.half().contiguous()
      ks = _kernels(t, s, w, h)
      self.assertEqual(max(_stores(k) for k in ks), 1)
      np.testing.assert_allclose(t.numpy(), (self.xn * 3).T * 2, rtol=1e-5)
      np.testing.assert_allclose(w.numpy(), (np.pad(self.xn, ((0, 0), (0, 4))) + 1) * 2, rtol=1e-5)
      np.testing.assert_allclose(h.numpy(), (self.xn.T * 2).astype(np.float16), rtol=1e-3)

  def test_copy_not_a_producer(self):
    # a copy from another device (NPY -> CPU) is a store too: a consumer merged into it read across devices
    with Context(MULTIOUT=1, SCACHE=0):
      k = Tensor(self.xn[0]) + Tensor(self.xn[1])
      _kernels(k)
      np.testing.assert_allclose(k.numpy(), self.xn[0] + self.xn[1])

class TestMultiOutputDSPRender(unittest.TestCase):
  def test_cast_store_stays_vectorized(self):
    # a LayerNorm-style producer with its fp16 copy merged in (BEVFormer's GEMM_FP16 A operands): both stores whole HVX vectors
    from test.unit.test_dsp_render import MockDSPRenderer, Target, _NoCompile, to_program
    with Context(MULTIOUT=2, SCACHE=0):
      x, m, g = Tensor.empty(900, 256), Tensor.empty(900, 1), Tensor.empty(256)
      y = ((x - m) * g).contiguous()
      a, r = y.half().contiguous(), (y + x).contiguous()
      ks = [k for k in Tensor.schedule_linear(a, r).src if k.src[0].op is Ops.SINK]
      self.assertEqual(len(ks), 1)
      ren = MockDSPRenderer(Target(device="DSP"))
      ren.compiler = _NoCompile()
      src = to_program(ks[0].src[0], ren).src[2].arg
    self.assertIn("*((__fp16128*)", src)
    self.assertIn("*((float128*)", src)

if __name__ == "__main__":
  unittest.main()
