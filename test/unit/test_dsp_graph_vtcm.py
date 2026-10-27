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

  def test_eviction_by_traffic(self):
    # a big buffer four kernels touch, live across short-lived ones placed first (more uses per byte-call): it takes the place
    # of the ones in its way when they carry less traffic in all (uses x bytes)
    x, w, y = FakeBuf(16384), FakeBuf(16384), FakeBuf(16384)
    s, t1, t2 = FakeBuf(32768), FakeBuf(24576), FakeBuf(24576)
    bufs = {id(b): [b, b.nbytes, None] for b in (x, w, y, s, t1, t2)}
    hmx, e = kern("h", [VEC, VEC], hmx=True), kern("e", [VEC])
    calls = [("h", hmx, [id(s), id(x), id(w)], {0}, 0),     # s = f(x, w)
             ("e", e, [id(t1), id(x)], {0}, 0),             # t1, t2: short-lived, between s's uses
             ("e", e, [id(t2), id(t1)], {0}, 0),
             ("e", e, [id(s), id(s)], {0}, 0),              # s in place
             ("e", e, [id(y), id(s)], {0}, 0),
             ("e", e, [id(x), id(t2)], {0}, 0)]
    h = self.emit(calls, bufs, w, y, 48)
    regs = [int(v) for v in re.search(r"G_REG_VTCM\[G_NREG\] = \{([^}]*)\}", h).group(1).split(",")]
    order = list(dict.fromkeys(i for _, _, ids, *_ in calls for i in ids))
    self.assertGreaterEqual(regs[order.index(id(s))], 0)

  def test_gathered_param_priority(self):
    # two regions live over the same calls, room for one: the one a kernel gathers from (/* vtcm-gathered */, from the renderer)
    # wins over a plain one read as often
    x, w, y = FakeBuf(16384), FakeBuf(16384), FakeBuf(16384)
    a, b = FakeBuf(32768), FakeBuf(32768)
    bufs = {id(t): [t, t.nbytes, None] for t in (x, w, y, a, b)}
    hmx, e = kern("h", [VEC, VEC], hmx=True), kern("e", [VEC])
    gath = kern("g", [VEC, VEC]).replace("/* DSP boilerplate", "/* vtcm-gathered: 1 */ /* DSP boilerplate")
    calls = [("h", hmx, [id(a), id(x), id(w)], {0}, 0), ("e", e, [id(b), id(x)], {0}, 0),
             ("g", gath, [id(y), id(a), id(b)], {0}, 0)]
    h = self.emit(calls, bufs, x, y, 40)
    regs = [int(v) for v in re.search(r"G_REG_VTCM\[G_NREG\] = \{([^}]*)\}", h).group(1).split(",")]
    order = list(dict.fromkeys(i for _, _, ids, *_ in calls for i in ids))
    self.assertGreaterEqual(regs[order.index(id(a))], 0)   # (unmarked, b takes the room)
    self.assertEqual(regs[order.index(id(b))], -1)

  def test_refill_after_eviction(self):
    # DSP_GRAPH_VTCM_REFILL: a region the first-fit pass crowded out and the eviction pass passed over gets the room an eviction
    # freed later (BEVFormer's decoder: each layer's value projection output)
    x, w, y = FakeBuf(4096), FakeBuf(4096), FakeBuf(4096)
    r0, r1, r2, r3 = FakeBuf(32768), FakeBuf(8192), FakeBuf(8192), FakeBuf(16384)
    bufs = {id(t): [t, t.nbytes, None] for t in (x, w, y, r0, r1, r2, r3)}
    e = kern("e", [VEC])
    calls = [("h", kern("h", [VEC, VEC], hmx=True), [id(r0), id(x), id(w)], {0}, 0), ("e", e, [id(r1), id(r0)], {0}, 0),
             ("e", e, [id(r2), id(r1)], {0}, 0), ("e", e, [id(r3), id(r2)], {0}, 0), ("e", e, [id(r1), id(r2)], {0}, 0),
             ("e", e, [id(r2), id(r1)], {0}, 0), ("e", e, [id(y), id(r3)], {0}, 0)]
    order = list(dict.fromkeys(i for _, _, ids, *_ in calls for i in ids))
    def at(h, t): return [int(v) for v in re.search(r"G_REG_VTCM\[G_NREG\] = \{([^}]*)\}", h).group(1).split(",")][order.index(id(t))]
    old = os.environ.get("DSP_GRAPH_VTCM_REFILL")
    try:
      os.environ["DSP_GRAPH_VTCM_REFILL"] = "0"
      self.assertEqual(at(self.emit(calls, bufs, x, y, 48), r1), -1)
      os.environ["DSP_GRAPH_VTCM_REFILL"] = "1"
      self.assertGreaterEqual(at(self.emit(calls, bufs, x, y, 48), r1), 0)
    finally:
      if old is None: os.environ.pop("DSP_GRAPH_VTCM_REFILL")
      else: os.environ["DSP_GRAPH_VTCM_REFILL"] = old

  def test_resident_request(self):
    # two regions live over the same calls, room for one: the plain one carries more traffic, but the model asked for the other
    # (/* vtcm-resident: j=p */, the renderer's listing of Tensor.empty(..., resident=p)); a refusal (p < 0) keeps a region out,
    # and a request exempts a region no single call writes whole (the model vouches for the writes covering what is read)
    x, w, y = FakeBuf(16384), FakeBuf(16384), FakeBuf(16384)
    a, b, c = FakeBuf(32768), FakeBuf(32768), FakeBuf(16384)
    cv = FakeBuf(8192, base=c)
    bufs = {id(t): [t, t.nbytes, None] for t in (x, w, y, a, b, c, cv)}
    hmx, e = kern("h", [VEC, VEC], hmx=True), kern("e", [VEC])
    def calls(ra:str, rc:str): return [
      ("h", hmx, [id(a), id(x), id(w)], {0}, 0), ("e", e, [id(b), id(x)], {0}, 0),
      ("e2", e, [id(cv), id(x)], {0}, 0),                                               # c: half of it written
      ("e3", kern("e3", [VEC, VEC, VEC]).replace("/* DSP boilerplate", ra + rc + "/* DSP boilerplate"),
       [id(y), id(a), id(b), id(c)], {0}, 0),
      ("e4", e, [id(w), id(b)], {0}, 0)]                                                # b used once more
    order = list(dict.fromkeys(id(bufs[i][0].base) for _, _, ids, *_ in calls("", "") for i in ids))
    def at(h, t): return [int(v) for v in re.search(r"G_REG_VTCM\[G_NREG\] = \{([^}]*)\}", h).group(1).split(",")][order.index(id(t))]
    h = self.emit(calls("", ""), bufs, x, y, 40)
    self.assertEqual((at(h, a) >= 0, at(h, b) >= 0, at(h, c)), (False, True, -1))
    h = self.emit(calls("/* vtcm-resident: 1=1 */ ", "/* vtcm-resident: 3=1 */ "), bufs, x, y, 56)
    self.assertEqual((at(h, a) >= 0, at(h, b) >= 0, at(h, c) >= 0), (True, False, True))
    h = self.emit(calls("/* vtcm-resident: 1=-1 */ ", ""), bufs, x, y, 64)
    self.assertEqual(at(h, a), -1)

  def test_raw_tile_alignment_and_prefetch(self):
    # a raw-tile buffer (listed by the renderer: /* hmx-tile: j */) goes to a 2 KB boundary, where the HMX reads and writes it in
    # place; the kernels' software prefetches of parameters placed in VTCM become no-ops
    x, w, y, a = FakeBuf(16384), FakeBuf(16384), FakeBuf(16384), FakeBuf(16512)
    s = FakeBuf(1024 * 8 * 2)
    sv = FakeBuf(1024 * 8 * 2, base=s)
    bufs = {id(b): [b, b.nbytes, None] for b in (x, w, y, a, s, sv)}
    pf = "(__builtin_HEXAGON_Y2_dcfetch((char*)(@+0)+2048), 0);" + VEC
    calls = [("e", kern("e", [VEC]), [id(a), id(x)], {0}, 0),
             ("h", "/* hmx-tile: 0 */ " + kern("h", [VEC, VEC], hmx=True), [id(sv), id(a), id(w)], {0}, 0),   # a stays live across s's write
             ("e2", "/* hmx-tile: 1 */ " + kern("e2", [pf, VEC]), [id(y), id(sv), id(a)], {0}, 0)]
    with tempfile.TemporaryDirectory() as d:
      old = os.environ.get("DSP_GRAPH_VTCM_KB")
      os.environ["DSP_GRAPH_VTCM_KB"] = "64"
      try: g.emit(d, calls, bufs, [x], y)
      finally:
        if old is None: os.environ.pop("DSP_GRAPH_VTCM_KB")
        else: os.environ["DSP_GRAPH_VTCM_KB"] = old
      h = (pathlib.Path(d) / "graph.h").read_text()
      regs = [int(v) for v in re.search(r"G_REG_VTCM\[G_NREG\] = \{([^}]*)\}", h).group(1).split(",")]
      order = list(dict.fromkeys(id(bufs[i][0].base) for _, _, ids, *_ in calls for i in ids))
      self.assertGreaterEqual(regs[order.index(id(s))], 0)
      self.assertEqual(regs[order.index(id(s))] % 2048, 0)
      k = re.search(r"case 2: (k\d+)\(", h).group(1)
      src = (pathlib.Path(d) / f"{k}.c").read_text()
      self.assertIn("__tg_nofetch((char*)(data1_", src)
      self.assertNotIn("__builtin_HEXAGON_Y2_dcfetch((char*)(data1_", src)

class TestHoist(unittest.TestCase):
  def test_hoist_out_of_hmx_chain(self):
    # calls between an HMX producer and the HMX consumer of its output that don't depend on the chain move before the producer
    sys_path = str(pathlib.Path(__file__).resolve().parents[2] / "examples")
    import sys
    if sys_path not in sys.path: sys.path.insert(0, sys_path)
    from dsp_graph_onnx import hoist_out_of_hmx_chains
    b = {n: FakeBuf(4096) for n in "iswvoz"}
    bufs = {id(t): [t] for t in b.values()}
    H, E = "__hmx_call_start();", "plain"
    calls = [("S", H, [id(b["s"]), id(b["i"])], {0}), ("max", E, [id(b["w"]), id(b["s"])], {0}),
             ("vproj", E, [id(b["v"]), id(b["i"])], {0}), ("vT", E, [id(b["z"]), id(b["v"])], {0}),
             ("PV", H, [id(b["o"]), id(b["s"]), id(b["w"]), id(b["z"])], {0})]
    names = [c[0] for c in hoist_out_of_hmx_chains(calls, bufs)]
    self.assertEqual(names, ["vproj", "vT", "S", "max", "PV"])

  def test_pull_raw_tile_readers(self):
    # a call reading a raw-tile buffer moves up to just after the last call it depends on (P @ V right after the exp writing P)
    sys_path = str(pathlib.Path(__file__).resolve().parents[2] / "examples")
    import sys
    if sys_path not in sys.path: sys.path.insert(0, sys_path)
    from dsp_graph_onnx import pull_raw_tile_readers
    b = {n: FakeBuf(4096) for n in "ixzo"}
    b["p"] = FakeBuf(1024 * 2)
    bufs = {id(t): [t] for t in b.values()}
    calls = [("exp", "plain", [id(b["p"]), id(b["i"])], {0}), ("other", "plain", [id(b["z"]), id(b["x"])], {0}),
             ("tail", "plain", [id(b["x"]), id(b["i"])], {0}), ("PV", "/* hmx-tile: 1 */ __hmx_", [id(b["o"]), id(b["p"])], {0})]
    self.assertEqual([c[0] for c in pull_raw_tile_readers(calls, bufs)], ["exp", "PV", "other", "tail"])

class TestCoreParam(unittest.TestCase):
  def test_core_position(self):
    # the threaded kernel's core id goes where its signature declares it (a buffer index skipped: declared before the last buffer)
    def sig(params): return f"__attribute__((noinline)) void E_1({', '.join(params)}) {{ }}\n/* DSP boilerplate */"
    self.assertEqual(g._core_param(sig(["float* data0_4", "float* data1_4", "const int data2_"]), 2), 2)
    self.assertEqual(g._core_param(sig(["float* data0_4", "const int data2_", "float* data2_4"]), 2), 1)
    self.assertEqual(g._core_param(sig(["float* data0_4", "float* data1_4"]), 2), 2)

if __name__ == "__main__": unittest.main()
