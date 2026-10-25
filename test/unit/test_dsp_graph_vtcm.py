"""DSP_GRAPH_VTCM_KB: dsp_graph_v65.emit places a program's intermediates in VTCM past the HMX tile pool, by liveness, when every
kernel touches them as whole HVX vectors. Pure emitter checks (graph.h): no DSP, no compiler."""
import os, re, tempfile, unittest, pathlib
from tinygrad.runtime.support import dsp_graph_v65 as g

class FakeBuf:
  def __init__(self, nbytes:int, base=None, offset:int=0):
    self.nbytes, self.offset, self._base = nbytes, offset, base
  @property
  def base(self): return self._base or self
  def as_memoryview(self): return memoryview(bytes(self.nbytes))

def kern(name:str, loads:list[str], hmx:bool=False) -> str:
  # a kernel body in the renderer's shape: buffer j is data{j}_<n>; `loads` are the accesses, data0 is stored as a float128
  params = ", ".join(f"float* restrict __attribute__((align_value(128))) data{j}_4096" for j in range(len(loads) + 1))
  body = " ".join(l.replace("@", f"data{j + 1}_4096") for j, l in enumerate(loads))
  return (f"__attribute__((noinline)) void {name}({params}) {{ {'__hmx_call_start(); ' if hmx else ''}{body} "
          f"*((float128*)((data0_4096+0))) = x; }}\n/* DSP boilerplate */")

VEC, F4, SCALAR = "float128 a = (*((float128*)((@+0))));", "float4 a = (*((float4*)((@+4))));", "float a = (*(@+3));"

class TestVtcmAccess(unittest.TestCase):
  def test_access_widths(self):
    self.assertTrue(g._vtcm_access_ok(kern("k", [VEC]), 1))
    self.assertTrue(g._vtcm_access_ok(kern("k", [VEC]), 0))                    # the float128 store
    self.assertFalse(g._vtcm_access_ok(kern("k", [F4]), 1))                    # 16 bytes: the scalar core's load
    self.assertFalse(g._vtcm_access_ok(kern("k", [SCALAR]), 1))
    self.assertTrue(g._vtcm_access_ok(kern("k", ["__fp16128 a = (*((__fp16128*)((@+0))));"]), 1))
    self.assertTrue(g._vtcm_access_ok(kern("k", ["(__builtin_HEXAGON_Y2_dcfetch((char*)(@+0)+2048), 0);" + VEC]), 1))  # a prefetch
    # a pointer handed to helper code: HMX helpers pack / store with HVX; a hand kernel's code is unknown
    self.assertTrue(g._vtcm_access_ok(kern("k", ["__hmx_pack2(_a+0, (@+0), (@+32));"], hmx=True), 1))
    self.assertFalse(g._vtcm_access_ok(kern("k", ["__msda_call((@+0));"]), 1))

class TestVtcmPlacement(unittest.TestCase):
  def emit(self, calls, bufs, inp, out, kb:int) -> str:
    old = os.environ.get("DSP_GRAPH_VTCM_KB")
    os.environ["DSP_GRAPH_VTCM_KB"] = str(kb)
    try:
      with tempfile.TemporaryDirectory() as d:
        g.emit(d, calls, bufs, [inp], out)
        return (pathlib.Path(d) / "graph.h").read_text()
    finally:
      if old is None: os.environ.pop("DSP_GRAPH_VTCM_KB")
      else: os.environ["DSP_GRAPH_VTCM_KB"] = old

  def test_liveness_and_widths(self):
    x, w, a, b, c, y = (FakeBuf(16384) for _ in range(6))
    bufs = {id(t): [t, t.nbytes, None] for t in (x, w, a, b, c, y)}
    hmx = kern("h", [VEC, VEC], hmx=True)
    calls = [("h", hmx, [id(a), id(x), id(w)], {0}, 0),           # a = f(x, w)
             ("e", kern("e", [VEC]), [id(b), id(a)], {0}, 0),     # b = f(a): a dies here
             ("e", kern("e", [VEC]), [id(c), id(b)], {0}, 0),     # c = f(b): b dies, c may reuse a's VTCM
             ("f", kern("f", [F4]), [id(y), id(c)], {0}, 0)]      # c read as float4: c stays in DDR
    h = self.emit(calls, bufs, x, y, 64)
    regs = [int(v) for v in re.search(r"G_REG_VTCM\[G_NREG\] = \{([^}]*)\}", h).group(1).split(",")]
    order = list(dict.fromkeys(i for _, _, ids, *_ in calls for i in ids))  # the emitter's region order
    at = {t: regs[order.index(id(t))] for t in (x, w, a, b, c, y)}
    self.assertEqual((at[x], at[w], at[y], at[c]), (-1, -1, -1, -1))   # input, weight, output, narrow
    self.assertGreaterEqual(min(at[a], at[b]), 0)
    self.assertNotEqual(at[a], at[b])                                   # live together in call 1
    # off, or nothing fits: no VTCM defines at all
    self.assertNotIn("G_REG_VTCM", self.emit(calls, bufs, x, y, 0))
    self.assertNotIn("G_REG_VTCM", self.emit(calls, bufs, x, y, 8))

if __name__ == "__main__": unittest.main()
