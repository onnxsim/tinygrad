"""codegen GATED_REDUCE: a sum of where(mask, term, 0) with the mask the same for every lane of a step accumulates under an IF,
and sink_into_ifs moves the term's computation into it. Numerics against numpy on the CPU backend."""
import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.codegen import sink_into_ifs
from tinygrad.uop.ops import UOp, Ops

class TestGatedReduce(unittest.TestCase):
  def test_masked_sum(self):
    rng = np.random.default_rng(0)
    a, m = rng.standard_normal((6, 50, 8, 32)).astype(np.float32), rng.uniform(0, 1, (6, 50)) < 0.2
    out = Tensor(m).reshape(6, 50, 1, 1).where(Tensor(a) * Tensor(a) + Tensor(a), 0.0).sum((0, 2)).numpy()
    ref = np.where(m[:, :, None, None], a * a + a, 0).sum((0, 2))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  def test_sink_into_ifs(self):
    # x = a + b used only inside the IF moves in; the gate and a value also used after the ENDIF stay
    a, b = UOp.variable("a", 0, 9), UOp.variable("b", 0, 9)
    g = a < 5
    x, y = a + b, a * 2
    mif = UOp(Ops.IF, src=(g, b))
    st, end = UOp(Ops.SINK, src=(x,)), UOp(Ops.ENDIF, src=(mif,))
    after = UOp(Ops.SINK, src=(y,), arg="after")
    lst = sink_into_ifs([a, b, g, x, y, mif, st, end, after])
    self.assertLess(lst.index(mif), lst.index(x))
    self.assertLess(lst.index(y), lst.index(mif))
    self.assertLess(lst.index(g), lst.index(mif))

  def test_merge_same_gate(self):
    # adjacent IF blocks under one gate become one block
    a = UOp.variable("a", 0, 9)
    g = a < 5
    i1, i2 = UOp(Ops.IF, src=(g, a)), UOp(Ops.IF, src=(g, a + 1))
    s1, s2 = UOp(Ops.SINK, src=(a,), arg=1), UOp(Ops.SINK, src=(a,), arg=2)
    lst = sink_into_ifs([a, g, i1, s1, UOp(Ops.ENDIF, src=(i1,)), i2, s2, UOp(Ops.ENDIF, src=(i2,))])
    self.assertEqual([u.op for u in lst].count(Ops.IF), 1)
    self.assertEqual([u.op for u in lst].count(Ops.ENDIF), 1)

if __name__ == "__main__": unittest.main()
