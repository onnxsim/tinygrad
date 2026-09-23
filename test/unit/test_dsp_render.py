import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.runtime import ops_dsp
from tinygrad.runtime.ops_dsp import MockDSPRenderer

class _NoCompile:
  def compile_cached(self, src:str) -> bytes: return b""
  def compile(self, src:str) -> bytes: return b""

def dsp_source(t:Tensor) -> str:
  ast = t.schedule_linear().src[-1].src[0]
  ren = MockDSPRenderer(Target(device="DSP"))
  ren.compiler = _NoCompile()  # render only: the point is the source text, and a bad source shouldn't cost a compile
  return to_program(ast, ren).src[2].arg

def max_chain(n:int, dtype) -> Tensor:
  x = Tensor.empty(4096, dtype=dtype)
  for _ in range(n): x = x.maximum(Tensor.empty(4096, dtype=dtype))
  return x

class TestDSPRender(unittest.TestCase):
  def test_float_max_chain_source_is_linear(self):
    # a float MAX rendered as a ternary names each operand twice, and single-use ALU results are inlined into their
    # consumer, so a chain of n maxes grew as 2**n in source text (>6 GB host memory for one test_ops case)
    for dt in (dtypes.float32, dtypes.int32):
      with self.subTest(dtype=dt):
        s8, s16 = len(dsp_source(max_chain(8, dt))), len(dsp_source(max_chain(16, dt)))
        self.assertLess(s16, 3 * s8, f"{dt}: source grew {s8} -> {s16} chars for 8 -> 16 chained maxes")

  def test_int_add_is_one_vector_op(self):
    src = dsp_source(Tensor.empty(4096, dtype=dtypes.int32) + Tensor.empty(4096, dtype=dtypes.int32))
    self.assertIn("(val0+val1)", src)
    self.assertNotIn("val0[1]", src)  # no per-lane constructor

class TestDSPQfloat(unittest.TestCase):
  # HVX_ARCH>=v68 turns on qfloat lowering; flip the module flag directly so this runs without a v68 toolchain
  def setUp(self): self.prev, ops_dsp.HVX_QFLOAT = ops_dsp.HVX_QFLOAT, True
  def tearDown(self): ops_dsp.HVX_QFLOAT = self.prev

  def test_product_of_computed_values_is_renormalized(self):
    # qf32 x qf32 without renormalizing is the badly inaccurate case (21% worst-case relative error on hardware):
    # both operands of such a multiply must go through the sf barrier helper
    a, b, c, d = (Tensor.empty(4096) for _ in range(4))
    src = dsp_source((a - b) * (c + d))
    self.assertIn("__hvx_mul_f", src)
    self.assertIn('__asm__("" : "+v"', src)

  def test_plain_arithmetic_is_left_to_llvm(self):
    # a multiply with a loaded operand, and adds of products, are accurate as LLVM lowers them: no helper
    a, b, w = Tensor.empty(4096), Tensor.empty(4096), Tensor.empty(4096)
    src = dsp_source(a * w + b * w)
    self.assertNotIn("__hvx_", src)

  def test_off_below_v68(self):
    ops_dsp.HVX_QFLOAT = False
    a, b, c, d = (Tensor.empty(2048) for _ in range(4))
    self.assertNotIn("__hvx_", dsp_source((a - b) * (c + d)))

if __name__ == '__main__':
  unittest.main()
