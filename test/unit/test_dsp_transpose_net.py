"""HVX_TRANSPOSE_NET: the butterfly network ops_dsp renders for the columns of an n x n block of vectors -- executed here in numpy
from the generated C (each stage's two shuffles per row pair), so the shuffle patterns themselves are checked."""
import re, unittest
import numpy as np
from tinygrad.runtime.ops_dsp import _transpose_helper

def run_helper(src:str, rows:list[np.ndarray]) -> list[np.ndarray]:
  r = {f"r{j}": v.copy() for j, v in enumerate(rows)}
  for a, c, lo, hi in re.findall(r"a = (r\d+), c = (r\d+); \w+ = __builtin_shufflevector\(a, c, ([\d,]+)\); \w+ = __builtin_shufflevector\(a, c, ([\d,]+)\)", src):
    cat = np.concatenate([r[a], r[c]])
    r[a], r[c] = cat[[int(x) for x in lo.split(",")]], cat[[int(x) for x in hi.split(",")]]
  return [r[f"r{j}"] for j in range(len(rows))]

class TestTransposeNet(unittest.TestCase):
  def test_transposes(self):
    for n in (8, 16, 32):
      m = np.arange(n * n).reshape(n, n)
      src = _transpose_helper("float32", n)
      self.assertEqual(src.count("__builtin_shufflevector"), 2 * (n // 2) * n.bit_length() - n)  # n/2 pairs x 2 x log2(n) stages
      np.testing.assert_array_equal(np.stack(run_helper(src, list(m))), m.T)

if __name__ == "__main__": unittest.main()
