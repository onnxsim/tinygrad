"""DSP_V65_VGATHER: the table-lookup kernels of dsp_graph_v65's emit become vgather kernels with a scalar fallback. qemu cannot run
vgather (the phone can: driving's 20 lookup calls 8.6 -> 1.25 ms, bit-exact), so this checks what qemu can: the emitted plumbing (a
resident-table slot, the graph.h defines, a kernel with both paths) and that the program, on its scalar path, still matches the JIT."""
import os, subprocess, sys, tempfile, textwrap, unittest, pathlib
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.runtime.support import dsp_graph_v65 as g

N = 98304  # 4 cores x 24576 elements: the shape of driving's E_192_4_128 lookups

# the v65 codegen (threads, 128-lane upcasts) is selected by DSP_V65_HW=1 in the environment at import, which the rest of the render
# tests must not have: the emit runs in a child process that does
CHILD = textwrap.dedent("""
  import os, sys, tempfile, pathlib
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.runtime.support import dsp_graph, dsp_graph_v65 as g
  N = %d
  rng = np.random.default_rng(0)
  idx = Tensor(rng.integers(0, 65536, N).astype(np.uint16), device="DSP").realize()
  tab = Tensor(rng.integers(0, 65536, 65536).astype(np.uint16), device="DSP").realize()
  ref = tab[idx.cast(dtypes.int32)].contiguous().numpy()
  res = []
  calls, bufs = dsp_graph.capture(lambda: res.append(tab[idx.cast(dtypes.int32)].contiguous().realize()))
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    os.environ["DSP_V65_VGATHER"] = "1"
    g.emit(d, calls, bufs, [idx.uop.buffer], res[0].uop.buffer)
    h, k = (d / "graph.h").read_text(), "".join(p.read_text() for p in d.glob("k*.c"))
    assert "#define G_NVTAB 1" in h and "G_VTAB(0)" in h, h
    assert "Q6_vgather_ARMWw" in k and "_scalar(" in k    # the HVX path and the scalar loop it falls back to
    y = g.run_qemu(d, [idx.numpy().tobytes()])             # qemu has no VTCM: the scalar path, matching the JIT
    np.testing.assert_array_equal(np.frombuffer(y, np.uint16)[:N], ref)
    os.environ["DSP_V65_VGATHER"] = "0"                    # off: nothing changes
    g.emit(d, calls, bufs, [idx.uop.buffer], res[0].uop.buffer)
    assert "G_NVTAB" not in (d / "graph.h").read_text()
    assert "Q6_vgather" not in "".join(p.read_text() for p in d.glob("k*.c"))
  print("ok")
""" % N)

@unittest.skipUnless(getenv("MOCKDSP") and os.environ.get("CC"), "requires MOCKDSP=1 and CC (a v65-capable clang)")
class TestDspVgather(unittest.TestCase):
  def test_emit_and_scalar_fallback(self):
    env = {**os.environ, "DEV": "DSP", "DSP_V65_HW": "1", "DSP_THREADS": "4", "NOLOCALS": "1", "PYTHONPATH": os.getcwd()}
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
    self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-3000:])
    self.assertTrue(r.stdout.strip().endswith("ok"))

  def test_recognizer_rejects_other_shapes(self):
    # a body that is not the canonical 128-lane lookup (no vgather rewrite: the scalar kernel is kept as it is)
    body = "__attribute__((noinline)) void kK(unsigned short* restrict a data0_128, unsigned short* restrict b data1_128, int x) {}"
    self.assertIsNone(g.vgather_kernel("kK", body, 4))

if __name__ == "__main__": unittest.main()
