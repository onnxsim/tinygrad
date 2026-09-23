import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
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

if __name__ == '__main__':
  unittest.main()
