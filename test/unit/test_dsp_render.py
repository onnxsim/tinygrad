import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, Context
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

class TestDSPVrmpyGemv(unittest.TestCase):
  # a W8A8 decode GEMV: one uint8 activation row times int8 weights, int32 accumulation -- exactly vrmpybusv (u8 x s8 dot4)
  K, N = 576, 1536

  def test_rowmajor_gemv_uses_vrmpybusv(self):
    # M=1 (no range of its own on the activation side) and mixed u8 x s8 dtypes both used to keep the TC from matching
    src = dsp_source(Tensor.empty(1, self.K, dtype=dtypes.uint8).matmul(Tensor.empty(self.K, self.N, dtype=dtypes.int8), dtype=dtypes.int32))
    self.assertIn("__builtin_HEXAGON_V6_vrmpybusv_acc_128B", src)

  def test_packed_gemv_is_one_weight_vector_per_vrmpy(self):
    # over Wp[N/32][K/4][32][4] each vrmpy's weights are one contiguous 128-byte load, and the 32-lane int32 accumulator
    # stays one HVX register (an aligned register array accessed as a vector, not 32 scalar loads/stores per step)
    x = Tensor.empty(1, self.K, dtype=dtypes.uint8).reshape(1, self.K//4, 1, 4)
    wp = Tensor.empty(self.N//32, self.K//4, 32, 4, dtype=dtypes.int8)
    with Context(TC_OPT=1): src = dsp_source((x * wp).sum((1, 3), dtype=dtypes.int32).reshape(self.N))
    self.assertIn("__builtin_HEXAGON_V6_vrmpybusv_acc_128B", src)
    self.assertIn("*((signed_char128*)((data2_", src)
    self.assertNotIn("signed_char32 ", src)
    self.assertIn("__attribute__((aligned(128)))", src)
    self.assertIn("*((int32*)((buf0+0)))", src)
    self.assertNotIn("*(buf0+1)", src)
