"""A captured tinygrad graph as one standalone Hexagon v65 program: the SDM845 cDSP's path (integer-only HVX, scalar float,
four hardware threads). dsp_graph.py is the v69 / HMX sibling; this one shares its capture() and differs in what it emits:

  calls, bufs = dsp_graph.capture(lambda: jit(**inputs))     every kernel recorded, not run (MOCKDSP=1). A TinyJit replay works:
                                                             its memory plan (views into one arena) is kept, not re-allocated
  info = emit(outdir, calls, bufs, [input Buffers], output)  k<n>.c per distinct kernel + graph.h + blob.bin (the weights)
  y = run_qemu(outdir, x_bytes)                              the whole program under qemu-hexagon, no phone needed
  build(outdir)                                              FastRPC skel (tg_graph.so) + Android client

Buffers are laid out as regions, one per base Buffer: a view becomes (region, byte offset), so kernels that alias one arena
through different views (a BITCAST of a float arena read as half, say) still see one memory. Regions no kernel writes are the
blob, uploaded once in chunks and read in place; the rest are allocated once and zeroed. A DSP_THREADS kernel (a trailing
core_id) runs its slices on a qurt thread pool on the phone, serially under qemu.
"""
from __future__ import annotations
import os, re, pathlib, subprocess, shutil, threading

FILES = pathlib.Path(__file__).parent / "dsp_graph_files" / "v65"
ALIGN = 128
def _rnd(n:int) -> int: return (n + ALIGN - 1) // ALIGN * ALIGN + ALIGN  # + ALIGN: a final HVX load may run past the end

VTAB_BYTES = 65536 * 2
_VG_HEADER = re.compile(r"void (\w+)\(unsigned short\* restrict [^,]*?data0_(\d+), unsigned short\* restrict [^,]*?data1_(\d+), "
                        r"unsigned short\* restrict [^,]*?data2_65536(, const int data3_)?\)")
_VG_LOOP = re.compile(r"for \(int Lidx1 = 0; Lidx1 < (\d+); Lidx1\+\+\) \{\n\s+int alu0 = "
                      r"(?:\(\(data3_(?:\*(\d+)|<<(\d+))\)\+\(Lidx1<<7\)\)|\(Lidx1<<7\));")

def vgather_kernel(kn:str, body:str, nthreads:int) -> str|None:
  """The DSP_V65_VGATHER rewrite of a table-lookup kernel: out[i] = tab[idx[i]] over u16 with a 65536-entry table (ONNX_QDQ_LUT's
  gathers), 128 lanes per step and the core_id slices contiguous. It becomes kn(out, idx, tab, [core,] vtab, vslot): with vtab (the
  table resident in VTCM, set by the skel when VTCM was granted) the HVX vgather path, else the original scalar loop (kept as
  kn_scalar, which is also all the qemu check ever runs). None when the kernel is not that shape. vgather takes 64 halfwords per
  instruction from word offsets (a 16-bit offset would reach only 32K entries), one VMEM store of its result, and only reads VTCM."""
  m, l = _VG_HEADER.search(body), _VG_LOOP.search(body)
  if not m or not l or bool(m.group(4)) != bool(nthreads) or m.group(2) != m.group(3): return None
  n, loops = int(m.group(2)), int(l.group(1))
  stride = int(l.group(2)) if l.group(2) else (1 << int(l.group(3))) if l.group(3) else n  # elements per core (all of them unthreaded)
  # one loop of 128-lane steps over a core's contiguous slice, the same offsets for the index load and the result store
  if n % max(nthreads, 1) or stride != n // max(nthreads, 1) or loops * 128 != stride or stride % 256: return None
  if body.count("(data2_65536+(((unsigned short*)(data1_") != 128 or "*((unsigned_short128*)((data0_" not in body: return None
  scalar = re.sub(rf"\bvoid {re.escape(kn)}\(", f"static void {kn}_scalar(", body, count=1)
  core_p, core_a, core_s = (", const int core", ", core", "core") if nthreads else ("", "", "0")
  return f"""#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
{scalar}
/* {stride} elements per core in {stride // 64} vectors of 64: four gathers in flight, each into its own 128-byte VTCM slot (the skel
 * keeps 512 bytes per core after the tables) */
void {kn}(unsigned char* out, unsigned char* idx, unsigned char* tab{core_p}, unsigned char* vtab, unsigned char* vslot) {{
  if (!vtab) {{ {kn}_scalar((unsigned short*)out, (unsigned short*)idx, (unsigned short*)tab{core_a}); return; }}
  const HVX_Vector* ip = (const HVX_Vector*)(idx + {core_s} * {stride * 2});
  HVX_Vector* op = (HVX_Vector*)(out + {core_s} * {stride * 2});
  HVX_Vector* t = (HVX_Vector*)(vslot + {core_s} * 512);
  unsigned rt = (unsigned)(unsigned long)vtab;
  for (int i = 0; i < {stride // 64}; i += 4) {{
    for (int j = 0; j < 4; j++) {{
      HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(ip[i + j]);
      HVX_Vector lo = Q6_V_lo_W(w), hi = Q6_V_hi_W(w);
      Q6_vgather_ARMWw(t + j, rt, {VTAB_BYTES - 1}, Q6_W_vcombine_VV(Q6_Vw_vadd_VwVw(hi, hi), Q6_Vw_vadd_VwVw(lo, lo)));
    }}
    for (int j = 0; j < 4; j++) op[i + j] = t[j];
  }}
}}
"""

_ELEM = {"float": 4, "int": 4, "unsigned_int": 4, "__fp16": 2, "half": 2, "short": 2, "unsigned_short": 2, "char": 1,
         "unsigned_char": 1, "signed_char": 1, "_Bool": 1}
_VEC_ACCESS = re.compile(r"\*\(\((" + "|".join(sorted(_ELEM, key=len, reverse=True)) + r")(\d+)(?:_u)?\*\)\(+$")

def _vtcm_access_ok(src:str, j:int) -> bool:
  """every access the kernel makes to its j-th buffer is a whole HVX vector (>= 128 bytes) or an HMX helper's (HVX pack / store).
  VTCM isn't cached in L1: a scalar or a 16-byte float4 load from it is slower than a cached DDR one (BEVFormer's encoder: a
  float4-reading reduce 0.66 -> 0.92 ms per call with its input in VTCM)"""
  body = src.split("/* DSP boilerplate */")[0]
  m = re.search(rf"\bdata{j}_\d+\b", body)
  if m is None: return False
  # a hand kernel can declare the parameters it reads and writes only as whole HVX vectors (/* vtcm-params: 1,2 */ in its call)
  if (d := re.search(r"/\* vtcm-params: ([\d,]+) \*/", body)) is not None and j in {int(x) for x in d.group(1).split(",")}: return True
  name, hmx, seen = m.group(0), "__hmx_" in body, False
  for o in re.finditer(rf"\b{name}\b", body):
    pre = body[max(0, o.start() - 48):o.start()]
    if pre.endswith("align_value(128))) ") or pre.endswith("(char*)("): continue  # the parameter, a prefetch
    seen = True
    if (v := _VEC_ACCESS.search(pre)) is not None:
      if _ELEM.get(v.group(1), 0) * int(v.group(2)) < 128: return False
    elif pre.endswith("*(") or not hmx: return False  # a scalar access; a pointer handed to a non-HMX (hand) kernel's code
  return seen

def _core_param(src:str, nbufs:int) -> int:
  """the position of a threaded kernel's core id (its `const int` parameter) among its parameters (default: after the buffers)"""
  body = src.split("/* DSP boilerplate */")[0]
  m = re.search(r"noinline\)\) void\s+\w+\(([^{]*)\)\s*\{", body)
  if m is None: return nbufs
  params = [p.strip() for p in m.group(1).split(",")]
  return next((i for i, p in enumerate(params) if p.startswith("const int")), nbufs)

def emit(outdir, calls, bufs, inputs:list, output, state:list|None=None) -> dict:
  """calls/bufs from dsp_graph.capture; inputs: the graph's input Buffers in order, output: its output Buffer"""
  o = pathlib.Path(outdir)
  o.mkdir(parents=True, exist_ok=True)
  for p in o.glob("k*.c"): p.unlink()
  views = {i: v[0] for i, v in bufs.items()}
  base = {i: b.base for i, b in views.items()}
  regions: list = list(dict.fromkeys(base[i] for _, _, ids, *_ in calls for i in ids))
  rid = {id(r): n for n, r in enumerate(regions)}
  written = {id(base[b]) for _, _, ids, outs, _ in calls for j, b in enumerate(ids) if j in outs}
  in_ids, out_id = [id(x.base) for x in inputs], id(output.base)
  for x in inputs:
    if x.offset != 0 or x.nbytes != x.base.nbytes: raise ValueError("a graph input must be a whole buffer")
  missing = [n for n in [*in_ids, out_id] if n not in rid]
  if missing: raise ValueError(f"{len(missing)} graph input/output buffers are not used by any captured kernel")
  r_id_to_reg = {n: id(r) for n, r in enumerate(regions)}
  blob, blob_off = bytearray(), {}
  for r in regions:
    if id(r) in written or id(r) in in_ids: continue
    blob_off[id(r)] = len(blob)
    blob += bytes(r.as_memoryview()) + bytes(-r.nbytes % ALIGN)

  knames: dict[str, str] = {}
  thunks, cases = [], []
  vg: dict[str, str|None] = {}  # src -> the vgather rewrite of its kernel, when every call of it looks its table up in a constant
  vtabs: dict[tuple[int, int], int] = {}  # (region, byte offset) of a resident table -> its VTCM slot
  if os.environ.get("DSP_V65_VGATHER", "0") != "0":
    for name, src, ids, _, nthreads in calls:
      if src in vg: continue
      body = src.split("/* DSP boilerplate */")[0]
      m = re.search(r"noinline\)\) void\s+(\w+)\(", body)
      alt = vgather_kernel("kK", re.sub(rf"\bvoid\s+{re.escape(m.group(1))}\(", "void kK(", body, count=1), nthreads) if m else None
      same = [c for c in calls if c[1] == src]
      const_table = all(len(c[2]) == 3 and id(base[c[2][2]]) in blob_off and c[4] == nthreads for c in same)
      vg[src] = alt if alt and const_table else None
  # HMX kernels (a v69 capture with HMX=1): the skel takes the HMX lock on the graph thread only, so they can't be sliced over the pool
  hmx = any("__hmx_" in src for _, src, *_ in calls)
  for name, src, _, _, nthreads in calls:
    if nthreads and "__hmx_" in src: raise ValueError(f"{name}: an HMX kernel can't run on the DSP_THREADS pool")
  for n, (name, src, ids, _, nthreads) in enumerate(calls):
    if src not in knames:
      kn = knames[src] = f"k{len(knames)}"
      body = src.split("/* DSP boilerplate */")[0]
      # a tiny kernel may come without the noinline attribute (a one-element int64 fill: `void E(long* restrict data0_1)`)
      m = re.search(r"noinline\)\) void\s+(\w+)\(", body) or re.search(r"^void\s+(\w+)\(", body, re.M)
      if m is None: raise ValueError(f"no kernel function in {name}")
      text = re.sub(rf"\bvoid\s+{re.escape(m.group(1))}\(", f"void {kn}(", body, count=1)
      if vg.get(src): text = vg[src].replace("kK", kn)
      (o / f"{kn}.c").write_text(text)
    args = ", ".join(f"R[{rid[id(base[b])]}]+{views[b].offset}" for b in ids)
    if vg.get(src):
      tb = ids[2]
      slot = vtabs.setdefault((rid[id(base[tb])], views[tb].offset), len(vtabs))
      if nthreads:
        thunks.append(f"static void g_t{n}(unsigned char** R, int core) {{ {knames[src]}({args}, core, G_VTAB({slot}), G_VSLOT); }}")
        cases.append(f"  case {n}: G_PARALLEL(g_t{n}, R, {nthreads}); break;  /* {name} (vgather) */")
      else: cases.append(f"  case {n}: {knames[src]}({args}, G_VTAB({slot}), G_VSLOT); break;  /* {name} (vgather) */")
    elif nthreads:
      # the core id goes where the kernel declares it: usually last, but a kernel whose buffer indices skip one (eight per-head
      # inputs numbered 0-7 and 9) declares it before the last buffer
      a = [f"R[{rid[id(base[b])]}]+{views[b].offset}" for b in ids]
      a.insert(_core_param(src, len(ids)), "core")
      thunks.append(f"static void g_t{n}(unsigned char** R, int core) {{ {knames[src]}({', '.join(a)}); }}")
      cases.append(f"  case {n}: G_PARALLEL(g_t{n}, R, {nthreads}); break;  /* {name} */")
    elif "__hmx_" in src: cases.append(f"  case {n}: g_hmx_lock(); {knames[src]}({args}); g_hmx_unlock(); break;  /* {name} */")
    else: cases.append(f"  case {n}: {knames[src]}({args}); break;  /* {name} */")
  out_view = output
  # DSP_GRAPH_REUSE: the regions that aren't weights, inputs or the output share one arena, placed by liveness (first fit over
  # the call range each is used in), so a region freed by its last use is reused. One region per buffer was 85 MB of scratch
  # for BEVFormer's encoder layer (180 MB for its decoder), and a kernel writing fresh memory paid for it in TLB / cache misses
  reg_off: dict[int, int] = {}
  arena = 0
  if os.environ.get("DSP_GRAPH_REUSE", "0") != "0":
    live: dict[int, list[int]] = {}
    for n, (_, _, ids, _, _) in enumerate(calls):
      for b in ids: live.setdefault(rid[id(base[b])], [n, n])[1] = n
    pinned = {rid[i] for i in in_ids} | {rid[out_id]}
    placed: list[tuple[int, int, int, int]] = []  # (offset, size, first call, last call)
    for r in sorted((r for r in live if id(regions[r]) not in blob_off and r not in pinned), key=lambda r: live[r][0]):
      size, (f, l) = _rnd(regions[r].nbytes), live[r]
      busy = sorted((o, o + sz) for o, sz, f2, l2 in placed if not (l2 < f or l < f2))
      off = 0
      for lo, hi in busy:
        if off + size <= lo: break
        off = max(off, hi)
      placed.append((off, size, f, l)); reg_off[r] = off; arena = max(arena, off + size)
  # DSP_GRAPH_VTCM_KB (HMX programs): intermediates in VTCM. The skel acquires that much VTCM past the HMX tile pool (HMX_VTCM_KB),
  # which no kernel touches, so a region there survives every call between its first and last use -- the schedule-level fact a
  # kernel can't see on its own. Regions are placed by liveness (first fit), most calls per byte-call held first; only regions some one
  # call writes whole (VTCM isn't zeroed, and a partly written region may rely on the zeros of a fresh allocation) and every
  # kernel reads and writes as whole HVX vectors (_vtcm_access_ok), never weights, inputs or the output
  vtcm_off: dict[int, int] = {}
  vtcm_bytes = 0
  if hmx and (vkb := int(os.environ.get("DSP_GRAPH_VTCM_KB", "0"))) > 0:
    uses: dict[int, list[int]] = {}
    whole: set[int] = set()
    narrow: set[int] = set()
    weight: dict[int, int] = {}
    for n, (_, src, ids, outs, _) in enumerate(calls):
      declared = re.search(r"/\* vtcm-params: ([\d,]+) \*/", src)
      for j, b in enumerate(ids):
        r = rid[id(base[b])]
        uses.setdefault(r, []).append(n)
        # a parameter a hand kernel declares (MSDA's value maps) is gathered from many times per call: worth 16 plain uses
        weight[r] = weight.get(r, 0) + (16 if declared and j in {int(x) for x in declared.group(1).split(",")} else 1)
        if j in outs and views[b].offset == 0 and views[b].nbytes == base[b].nbytes: whole.add(r)
        if not _vtcm_access_ok(src, j): narrow.add(r)
    pinned = {rid[i] for i in in_ids} | {rid[out_id]}
    cand = [r for r in uses if r in whole and r not in narrow and r not in pinned and id(regions[r]) not in blob_off and
            _rnd(regions[r].nbytes) <= vkb * 1024]
    vplaced: list[tuple[int, int, int, int]] = []
    # most uses per byte-call of VTCM held first: a small buffer read twice 200 calls apart holds its bytes the whole time
    # declared hand-kernel parameters first (their gathers gain the most), then the rest by uses per byte-call held
    for r in sorted(cand, key=lambda r: (weight[r] < 16, -weight[r] / (regions[r].nbytes * (uses[r][-1] - uses[r][0] + 1)), uses[r][0])):
      size, f, l = _rnd(regions[r].nbytes), uses[r][0], uses[r][-1]
      busy = sorted((o, o + sz) for o, sz, f2, l2 in vplaced if not (l2 < f or l < f2))
      off = 0
      for lo, hi in busy:
        if off + size <= lo: break
        off = max(off, hi)
      if off + size > vkb * 1024: continue
      vplaced.append((off, size, f, l))
      vtcm_off[r], vtcm_bytes = off, max(vtcm_bytes, off + size)
    for r in vtcm_off: reg_off.pop(r, None)
  h = [f"/* generated by tinygrad/runtime/support/dsp_graph_v65.py: {len(calls)} calls, {len(knames)} kernels, {len(regions)} regions */",
       f"#define G_NCALLS {len(calls)}", f"#define G_NREG {len(regions)}", f"#define G_BLOB_BYTES {len(blob)}",
       f"#define G_NIN {len(inputs)}", f"#define G_OUT_REG {rid[out_id]}", f"#define G_OUT_OFF {out_view.offset}",
       f"#define G_OUT_BYTES {out_view.nbytes}",
       *([f"#define G_HMX 1", f"#define G_VTCM_KB {int(os.environ.get('HMX_VTCM_KB', 256))}", "void g_hmx_lock(void);", "void g_hmx_unlock(void);"]
         if hmx else []),
       "static const unsigned G_REG_BYTES[G_NREG] = {" + ", ".join(str(_rnd(r.nbytes)) for r in regions) + "};",
       "static const int G_REG_BLOB[G_NREG] = {" + ", ".join(str(blob_off.get(id(r), -1)) for r in regions) + "};",
       *([f"#define G_ARENA_BYTES {arena}", "static const int G_REG_OFF[G_NREG] = {" + ", ".join(str(reg_off.get(i, -1)) for i in range(len(regions))) + "};"]
         if reg_off else []),
       *([f"#define G_GVTCM_BYTES {vtcm_bytes}",
          "static const int G_REG_VTCM[G_NREG] = {" + ", ".join(str(vtcm_off.get(i, -1)) for i in range(len(regions))) + "};"]
         if vtcm_off else []),
       "static const int G_IN_REG[G_NIN] = {" + ", ".join(str(rid[i]) for i in in_ids) + "};",
       "static const unsigned G_IN_BYTES[G_NIN] = {" + ", ".join(str(x.nbytes) for x in inputs) + "};",
       *([f"#define G_NVTAB {len(vtabs)}", f"#define G_VTCM_BYTES ({len(vtabs)} * {VTAB_BYTES} + 4096)",
          "static const int G_VTAB_BLOB[G_NVTAB] = {" + ", ".join(str(blob_off[r_id_to_reg[k[0]]] + k[1]) for k in vtabs) + "};",
          "extern unsigned char* g_vtcm;", f"#define G_VTAB_BYTES {VTAB_BYTES}",
          f"#define G_VTAB(s) (g_vtcm ? g_vtcm + (s) * {VTAB_BYTES} : 0)", f"#define G_VSLOT (g_vtcm ? g_vtcm + G_NVTAB * {VTAB_BYTES} : 0)"]
         if vtabs else []),
       # recurrent state kept on the DSP: (input index, offset of the output slice that feeds it, the slice's padded length), ascending
       # by offset; tg_graph_run loops each slice back into its input region after every run (see tg_graph.idl `flags`)
       *([f"#define G_NSTATE {len(state)}",
          "static const int G_ST_IN[G_NSTATE] = {" + ", ".join(str(x[0]) for x in state) + "};",
          "static const int G_ST_OFF[G_NSTATE] = {" + ", ".join(str(x[1]) for x in state) + "};",
          "static const int G_ST_LEN[G_NSTATE] = {" + ", ".join(str(x[2]) for x in state) + "};"] if state else []),
       *[f"void {k}();" for k in knames.values()],
       *([f"#define G_PERF_VOTE {int(os.environ['DSP_V65_PERF_VOTE'])}"] if int(os.environ.get("DSP_V65_PERF_VOTE", "0")) else []),
       "#ifndef G_PARALLEL\n#define G_PARALLEL(f, R, n) for (int _c = 0; _c < (n); _c++) f(R, _c)\n#endif",
       *thunks,
       "static void g_call(int i, unsigned char** R) {", "  switch (i) {", *cases, "  }", "}"]
  (o / "graph.h").write_text("\n".join(h) + "\n")
  (o / "blob.bin").write_bytes(bytes(blob))
  return {"calls": len(calls), "kernels": len(knames), "regions": len(regions), "blob": len(blob), "hmx": hmx,
          "threaded_calls": sum(1 for c in calls if c[4]), "input_bytes": [x.nbytes for x in inputs], "output_bytes": out_view.nbytes,
          "scratch": sum(_rnd(r.nbytes) for n, r in enumerate(regions) if id(r) not in blob_off and n not in reg_off and n not in vtcm_off) + arena,
          "vtcm_regions": len(vtcm_off), "vtcm_bytes": vtcm_bytes}

def _input_blob(xs:list[bytes]) -> bytes:
  return b"".join(x + bytes(-len(x) % ALIGN) for x in xs)  # inputs packed back to back, each 128-byte aligned

def _cc() -> str: return os.environ.get("CC", "clang")
def _hvx_args(arch:str="v65") -> list[str]: return ["--target=hexagon", f"-mcpu=hexagon{arch}", f"-mhvx={arch}", "-mhvx-length=128b"]

QEMU_MAIN = r"""
#include "graph.h"
static unsigned char* R[G_NREG];
static void rd(void* d, unsigned n) { for (unsigned r = 0; r < n; ) { int k = read(0, (unsigned char*)d + r, n - r); if (k <= 0) exit(2); r += k; } }
void _start(void) {
  unsigned char* blob = mmap2(0, G_BLOB_BYTES + 4096, 3, 0x21, -1, 0);
  rd(blob, G_BLOB_BYTES);
#ifdef G_ARENA_BYTES
  unsigned char* arena = mmap2(0, G_ARENA_BYTES + 4096, 3, 0x21, -1, 0);
  for (int i = 0; i < G_NREG; i++) R[i] = G_REG_BLOB[i] >= 0 ? blob + G_REG_BLOB[i] : G_REG_OFF[i] >= 0 ? arena + G_REG_OFF[i] : mmap2(0, G_REG_BYTES[i] + 4096, 3, 0x21, -1, 0);
#else
  for (int i = 0; i < G_NREG; i++) R[i] = G_REG_BLOB[i] >= 0 ? blob + G_REG_BLOB[i] : mmap2(0, G_REG_BYTES[i] + 4096, 3, 0x21, -1, 0);
#endif
  for (int i = 0; i < G_NIN; i++) {
    rd(R[G_IN_REG[i]], G_IN_BYTES[i]);
    if (G_IN_BYTES[i] % 128) { unsigned char pad[128]; rd(pad, 128 - G_IN_BYTES[i] % 128); }
  }
  for (int i = 0; i < G_NCALLS; i++) g_call(i, R);
  write(1, R[G_OUT_REG] + G_OUT_OFF, G_OUT_BYTES);
  exit(0);
}
"""

def _kernel_flags(arch:str="v65") -> list[str]:
  return [*_hvx_args(arch), "-O2", "-fPIC", "-ffreestanding", "-nostdlib", "-fno-stack-protector", "-Wno-deprecated-non-prototype"]

def compile_kernels(outdir, arch:str="v65", cc:str|None=None, extra:tuple[str, ...]=()) -> list[str]:
  """k<n>.c -> k<n>.o, once for both the qemu program and the skel (same flags). Objects are cached across compiles by source,
  flags and compiler (DSP_GRAPH_OBJ_CACHE, default ~/.cache/tinygrad/dsp_graph_v65): a recompile after a small codegen change
  rebuilds only the kernels whose source changed. The largest sources start first, since they set the wall time"""
  import hashlib, shutil
  from concurrent.futures import ThreadPoolExecutor
  o = pathlib.Path(outdir)
  cache = pathlib.Path(os.environ.get("DSP_GRAPH_OBJ_CACHE", pathlib.Path.home() / ".cache/tinygrad/dsp_graph_v65"))
  cache.mkdir(parents=True, exist_ok=True)
  flags, cc = [*_kernel_flags(arch), *extra], cc or _cc()
  ver = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
  ks = sorted((p for p in o.glob("k*.c")), key=lambda p: -p.stat().st_size)
  def one(k:pathlib.Path):
    obj = k.with_suffix(".o")
    key = hashlib.sha256("\0".join([ver, *flags, k.read_text()]).encode()).hexdigest()
    cached = cache / f"{key}.o"
    if not cached.exists():
      tmp = cache / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
      subprocess.run([cc, "-c", *flags, "-o", str(tmp), str(k)], check=True)
      os.replace(tmp, cached)
    shutil.copyfile(cached, obj)
  with ThreadPoolExecutor(os.cpu_count()) as ex: list(ex.map(one, ks))
  return sorted(k.with_suffix(".o").name for k in ks)

def run_qemu(outdir, xs:list[bytes], timeout:int=3600) -> bytes:
  """the emitted program under qemu-hexagon-static (DSP_THREADS slices run in turn) -> the output bytes"""
  from tinygrad.runtime import ops_dsp
  o = pathlib.Path(outdir)
  boiler = ops_dsp.mockdsp_boilerplate.replace("{{", "{").replace("}}", "}")
  (o / "qemu_main.c").write_text(boiler + "\nunsigned char* g_vtcm = 0;  /* the vgather kernels take their scalar path here */\n" + QEMU_MAIN)
  objs = compile_kernels(o)
  libgcc = ops_dsp._find_libgcc()
  subprocess.run([_cc(), "-static", "-fuse-ld=lld", *_kernel_flags(), "-o", "graph_qemu.elf", "qemu_main.c", *objs,
                  *([libgcc] if libgcc else [])], cwd=o, check=True)
  stdin = (o / "blob.bin").read_bytes() + _input_blob(xs)
  return subprocess.run(["qemu-hexagon-static", str(o / "graph_qemu.elf")], input=stdin, stdout=subprocess.PIPE, check=True,
                        timeout=timeout).stdout

def build(outdir, skel_arch:str="v68", kernel_arch:str="v65") -> tuple[pathlib.Path, pathlib.Path]:
  # kernel_arch v68+ (qfloat HVX, HMX): the kernels are built with the SDK's hexagon-clang (-mhmx when graph.h has G_HMX) and the skel
  # for the same arch; v65 kernels keep $CC (the SDK 6.x compiler starts at v68)
  """the FastRPC skel (tg_graph.so) + Android client. The kernels are built for kernel_arch with $CC (a v65-capable clang; the
  Hexagon SDK 6.x compiler starts at v68); the skel for skel_arch with the SDK's. Needs HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN and an
  NDK clang (NDK_CLANG). On the phone: client 'file:///tg_graph.so?tg_graph_skel_handle_invoke&_modver=1.0&_dom=cdsp' <dir with
  blob.bin input.bin ref.bin> [iters] [threads] [batch] [prof], ADSP_LIBRARY_PATH at the skel"""
  o, sdk, tc = pathlib.Path(outdir), pathlib.Path(os.environ["HEXAGON_SDK_ROOT"]), pathlib.Path(os.environ["HEXAGON_TOOLCHAIN"])
  ndk = os.environ.get("NDK_CLANG", "/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang")
  for f in ("skel.c", "client.c", "tg_graph.idl"): shutil.copy(FILES / f, o / f)
  new_arch = int(kernel_arch.lstrip("v")) >= 68
  hmx = new_arch and "#define G_HMX 1" in (o / "graph.h").read_text()
  if new_arch: skel_arch = kernel_arch
  if hmx: shutil.copy(FILES.parent / "hmx_runtime.h", o / "hmx_runtime.h")
  inc = ["-I", ".", "-I", str(sdk / "incs"), "-I", str(sdk / "incs/stddef")]
  qurt = ["-I", str(sdk / f"rtos/qurt/compute{skel_arch}/include/qurt"), "-I", str(sdk / f"rtos/qurt/compute{skel_arch}/include/posix")]
  def run(cmd): subprocess.run(cmd, cwd=o, check=True)
  run([str(sdk / "ipc/fastrpc/qaic/Ubuntu/qaic"), "-I", str(sdk / "incs"), "-I", str(sdk / "incs/stddef"), "tg_graph.idl"])
  hc = [str(tc / "bin/hexagon-clang"), "-c", "-O2", "-fPIC", f"-mcpu=hexagon{skel_arch}"]
  run([*hc, *inc, "-o", "skel_rpc.o", "tg_graph_skel.c"])
  run([*hc, f"-mhvx={skel_arch}", "-mhvx-length=128b", *(["-mhmx"] if hmx else []), "-Wno-deprecated-non-prototype", *inc, *qurt,
       "-o", "impl.o", "skel.c"])
  ks = [k[:-2] + ".c" for k in (compile_kernels(o, kernel_arch, cc=str(tc / "bin/hexagon-clang"), extra=("-mhmx",) if hmx else ())
                                if new_arch else compile_kernels(o, kernel_arch))]
  lib = tc / f"target/hexagon/lib/{skel_arch}/G0"
  run([str(tc / "bin/hexagon-link"), "-Bdynamic", "-shared", "-export-dynamic", "-o", "tg_graph.so", "skel_rpc.o", "impl.o",
       *[k[:-2] + ".o" for k in ks], str(lib / "pic/libgcc.a"), str(lib / "pic/libgcc.so")])
  run([ndk, "-O2", *inc, "-o", "client", "client.c", "tg_graph_stub.c", "-L", str(sdk / "ipc/fastrpc/remote/ship/android_aarch64"),
       "-lcdsprpc"])
  return o / "tg_graph.so", o / "client"

def write_case(outdir, xs:list[bytes], ref:bytes) -> None:
  """input.bin (the inputs as the client sends them) and ref.bin (the expected output) next to blob.bin"""
  o = pathlib.Path(outdir)
  (o / "input.bin").write_bytes(_input_blob(xs))
  (o / "ref.bin").write_bytes(ref)

ARTIFACT_MAGIC = b"TGHXV65\0"
def pack(outdir, artifact_path, program_lines:list[str], arch:str="v65") -> int:
  """the runner artifact for a compiler service (tools/onnx-remote's onnx-remote-compiler in onnxsim): tg_graph.so, blob.bin and
  program.txt (program_lines: the I/O contract, one record per line) in one little-endian container -- magic TGHXV65\\0, u32 file
  count, then per file u32 name length, name, u64 size, bytes -- so a runner unpacks it with no archive or JSON library"""
  import struct
  o = pathlib.Path(outdir)
  files = [("tg_graph.so", (o / "tg_graph.so").read_bytes()), ("blob.bin", (o / "blob.bin").read_bytes()),
           ("program.txt", ("\n".join(program_lines) + "\n").encode())]
  with open(artifact_path, "wb") as f:
    # the same container and tg_graph ABI for every arch; the magic names the DSP arch the program needs (TGHXV65\0, TGHXV69\0)
    f.write((ARTIFACT_MAGIC if arch == "v65" else f"TGHX{arch.upper()}".encode() + b"\0") + struct.pack("<I", len(files)))
    for name, data in files: f.write(struct.pack("<I", len(name)) + name.encode() + struct.pack("<Q", len(data)) + data)
  return pathlib.Path(artifact_path).stat().st_size
