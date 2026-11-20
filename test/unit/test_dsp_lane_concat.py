"""A 128-lane int32 row is loaded as two 64-lane vectors; the STACK rebuilding it (an int32 -> uint16 narrowing) must render as a
shufflevector concatenation of those vectors, not 128 scalar lane reads (256 vinserts: 23x slower on the phone)."""
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, getenv

@unittest.skipUnless(getenv("MOCKDSP"), "requires MOCKDSP=1")
class TestLaneConcat(unittest.TestCase):
  def test_narrowing_cast_is_exact_and_concatenates(self):
    x = np.random.default_rng(0).integers(-70000, 140000, (128, 128)).astype(np.int32)
    with Context(DEBUG=4):
      import io, contextlib
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        t = Tensor(x, device="DSP").maximum(0)
        y = (65535 - (65535 - t).maximum(0)).cast(dtypes.uint16).contiguous().numpy()
    np.testing.assert_array_equal(y, np.clip(x, 0, 65535).astype(np.uint16))
    src = buf.getvalue()
    self.assertIn("void E_", src)
    self.assertIn("__builtin_shufflevector", src)  # the two 64-lane halves concatenated
    self.assertNotIn("[127])", src)  # no per-lane reads of the loaded row

if __name__ == "__main__": unittest.main()
