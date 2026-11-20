"""Tensor.stage: a register table computed inside its consumer's kernel (two loop nests joined by an on-chip buffer). Numerics on the
CPU backend, structure in the DSP renderer's source."""
import unittest, re, os, contextlib
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, getenv
from tinygrad.codegen import to_program
from tinygrad.runtime.ops_dsp import MockDSPRenderer

@contextlib.contextmanager
def _env(**kw):
  old = {k:os.environ.get(k) for k in kw}
  os.environ.update(kw)
  getenv.cache_clear()
  try: yield
  finally:
    for k,v in old.items():
      if v is None: os.environ.pop(k)
      else: os.environ[k] = v
    getenv.cache_clear()

class _NoCompile:
  def compile_cached(self, src:str) -> bytes: return b""
  def compile(self, src:str) -> bytes: return b""

def dsp_sources(t:Tensor) -> list[str]:
  ren = MockDSPRenderer(Target(device="DSP"))
  ren.compiler = _NoCompile()
  return [to_program(c.src[0], ren).src[2].arg for c in t.schedule_linear().src if c.src[0].op.name == "SINK"]

def table_reduce(a, b, X, vis=None):
  # out[i, d] = sum_k tab[i, k] * X[i, k, d], the table a * b + 1 per row i
  tab = (a * b + 1.0).stage(1)
  t = tab.unsqueeze(-1) * X
  if vis is not None: t = (vis != 0).reshape(-1, 1, 1).where(t, 0.0)
  return t.sum(1)

class TestStage(unittest.TestCase):
  def setUp(self):
    rng = np.random.default_rng(0)
    self.a, self.b = rng.standard_normal((16, 32)).astype(np.float32), rng.standard_normal((16, 32)).astype(np.float32)
    self.X, self.vis = rng.standard_normal((16, 32, 32)).astype(np.float32), (rng.uniform(0, 1, 16) < 0.5).astype(np.uint8)

  def test_numerics(self):
    out = table_reduce(Tensor(self.a), Tensor(self.b), Tensor(self.X)).numpy()
    np.testing.assert_allclose(out, ((self.a * self.b + 1)[..., None] * self.X).sum(1), rtol=1e-5, atol=1e-4)

  def test_gated_numerics(self):
    out = table_reduce(Tensor(self.a), Tensor(self.b), Tensor(self.X), Tensor(self.vis)).numpy()
    ref = np.where(self.vis[:, None, None] != 0, (self.a * self.b + 1)[..., None] * self.X, 0).sum(1)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)

  def test_one_kernel_with_vector_table(self):
    # the table is a 32-lane register array, stored as one vector per row and read as scalars by the reduce's loop (unrolled by
    # STAGE_UNROLL=4 only, not into one vector load and lane extractions)
    srcs = dsp_sources(table_reduce(Tensor.empty(16, 32), Tensor.empty(16, 32), Tensor.empty(16, 32, 32)))
    self.assertEqual(len(srcs), 1)
    src = srcs[0]
    tab = re.search(r"\(\(float\*\)\((buf\d+)\+\(Ridx\d+<<2\)\)\)\[0\]", src)   # (STAGE_UNROLL=4: four entries a step)
    self.assertIsNotNone(tab)
    self.assertIn(f"float {tab.group(1)}[32] __attribute__((aligned(128)));", src)
    self.assertIn(f"*((float32*)(({tab.group(1)}+0))) =", src)

  def test_gate_around_table(self):
    # a gated reduce over the table: the gate's IF goes around the table walk and the table's computation (codegen hoist_ifs,
    # sink_into_ifs), not inside the walk
    src = dsp_sources(table_reduce(Tensor.empty(16, 32), Tensor.empty(16, 32), Tensor.empty(16, 32, 32), Tensor.empty(16, dtype=dtypes.uint8)))[0]
    body = src[src.index("if ("):]
    self.assertLess(body.index("*((float32*)((buf"), body.index("for (int Ridx"))
    self.assertEqual(src.count("if ("), 1)

  def test_qf32_accumulator(self):
    # with qfloat (v68+) the reduce over the table accumulates in qf32 (one vmpy + vadd a step), converted to sf once at the end
    from tinygrad.runtime import ops_dsp
    old, ops_dsp.HVX_QFLOAT = ops_dsp.HVX_QFLOAT, 1
    # (8 rows: a kernel no other test renders -- programs are cached by AST)
    try: src = dsp_sources(table_reduce(Tensor.empty(8, 32), Tensor.empty(8, 32), Tensor.empty(8, 32, 32)))[0]
    finally: ops_dsp.HVX_QFLOAT = old
    self.assertRegex(src, r"= \(float32\)__builtin_bit_cast\(__hvx_f, __QADD\(__QB\(.*\), __QADD\(__QADD\(__QADD\(__QMUL\(")
    self.assertIn("__QSF(", src)

  def test_int_lanes_through_bits(self):
    # (HVX_REVEC_BITCAST) a table built through a float's bits stays vectors: the bitcast re-vectorizes, forced to IEEE sf first
    from tinygrad.runtime import ops_dsp
    old, ops_dsp.HVX_QFLOAT = ops_dsp.HVX_QFLOAT, 1
    a, X = Tensor.empty(16, 32), Tensor.empty(16, 32, 32)
    tab = (((a + 12582912.0).bitcast(dtypes.int32) - 0x4B400000) & 15).stage(1)
    try:
      with _env(HVX_REVEC_BITCAST="1"):
        src = dsp_sources((Tensor(X.reshape(-1).uop.index((tab.unsqueeze(-1) * 32 + Tensor.arange(32, dtype=dtypes.int32)).reshape(-1).uop)))
                          .reshape(16, 32, 32).sum(1))[0]
    finally: ops_dsp.HVX_QFLOAT = old
    self.assertIn("__builtin_bit_cast(int32, (float32)__hvx_sfbar(", src)
    self.assertRegex(src, r"\*\(\(int32\*\)\(\(buf\d+\+0\)\)\) =")

  def test_table_loop_outside_lanes(self):
    # a table of more than STAGE_LANES (128) entries: its innermost axes are the vector lanes and a loop goes around them; what
    # doesn't depend on that loop's axis is computed once, outside it. Gated, the whole loop moves into the IF (sink_into_ifs)
    a, X, vis = Tensor.empty(8, 128), Tensor.empty(8, 4, 128, 32), Tensor.empty(8, dtype=dtypes.uint8)
    tab = ((a.exp2().reshape(8, 1, 128) + Tensor.arange(4).reshape(1, 4, 1)) * 0.5).stage(2)               # (8, 4, 128): 4 x 128
    src = dsp_sources((vis != 0).reshape(8, 1, 1, 1).where(tab.unsqueeze(-1) * X, 0.0).sum((1, 2)))[0]
    body = src[src.index("if ("):]
    loop = re.search(r"for \(int (Lidx\d+) = 0; \1 < 4; \1\+\+\) \{", body)
    self.assertIsNotNone(loop)
    self.assertNotIn("exp2", body[loop.end():body.index("for (int Ridx")])      # (v65 render: the exp2 is a polynomial, outside the loop)
    self.assertRegex(body[loop.end():], r"\*\(\(float128\*\)\(\(buf\d+\+\(Lidx\d+<<7\)\)\)\) =")
    self.assertEqual(src.count("if ("), 1)

  def test_norm_add_col(self):
    # (HVX_REVEC_BITCAST) lanes x + c, c + x and x (c = 0 folded away) line up as x + c
    from tinygrad.runtime import ops_dsp
    from tinygrad.uop.ops import UOp, Ops
    x = UOp.range(100, 0).cast(dtypes.int32)
    with _env(HVX_REVEC_BITCAST="1"): col = ops_dsp._norm_add_col((x, x + 3, UOp(Ops.ADD, x.dtype, (x.const_like(5), x))))
    self.assertEqual([c.op for c in col], [Ops.ADD]*3)
    self.assertEqual([c.src[1].arg for c in col], [0, 3, 5])
    self.assertTrue(all(c.src[0] is x for c in col))

  def test_msda_staged(self):
    # the generated MSDA through register tables (MSDA_STAGE=1) against numpy, with and without a visibility mask
    from tinygrad.nn.onnx import onnx_ops
    rng = np.random.default_rng(1)
    for NV, H, W, R, NO, P, masked in ((3, 5, 7, 2, 1, 4, True), (2, 6, 6, 1, 2, 2, False)):
      Q, M, C = 6, 2, 16
      value = rng.standard_normal((NV, H * W, C)).astype(np.float32)
      ref = rng.uniform(-0.1, 1.1, (NV, Q, R, 2)).astype(np.float32)
      off = (rng.standard_normal((Q, M, NO, P, 2)) * 2).astype(np.float32)
      attw = rng.uniform(0, 1, (Q, M, NO, P)).astype(np.float32)
      vis = (rng.uniform(0, 1, (NV, Q)) < 0.5).astype(np.uint8) if masked else None
      args = (Tensor(value), Tensor(ref), Tensor(off), Tensor(attw), None if vis is None else Tensor(vis))
      with _env(MSDA_GATED="0"): want = onnx_ops["MSDA"](*args, H=H, W=W).numpy()
      for stage in ("1", "2"):
        with _env(MSDA_STAGE=stage): out = onnx_ops["MSDA"](*args, H=H, W=W).numpy()
        np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)

if __name__ == "__main__": unittest.main()
