"""Shared harness for the hand-written DSP kernels kept as test oracles (test/external/dsp/hand/): build a hand C driver or a
captured tinygrad kernel for hexagon-sim (-mv69, HMX on), run it, return outputs and cycles. Tests skip when the Hexagon
toolchain (HEXAGON_TOOLS, default ~/.cache/hexagon-oa-19/Tools) or a Hexagon-capable clang for MOCKDSP isn't there.

  tools()                       the toolchain dir, or None
  capture_dsp()                 context manager: MOCKDSP kernel calls whose source matches are recorded (source, argument
                                bytes) instead of run -- qemu can't run HMX; the rest runs normally
  run_captured(src, bufs, work) hexagon-sim run of one captured kernel -> (bytes of buffer 0, pcycles of one call)
  run_hand(c_file, work, *args, includes=(), defines=()) build + run a hand driver -> stdout (drivers print their own results)
"""
from __future__ import annotations
import os, re, shutil, subprocess, pathlib, contextlib

HERE = pathlib.Path(__file__).resolve().parent

def tools() -> pathlib.Path|None:
  t = pathlib.Path(os.environ.get("HEXAGON_TOOLS", os.path.expanduser("~/.cache/hexagon-oa-19/Tools")))
  return t if (t / "bin" / "hexagon-sim").exists() and (t / "bin" / "hexagon-clang").exists() else None

def mockdsp_ok() -> bool:
  # CC may carry flags: conftest.py appends -ffp-contract=off to it. shutil.which() on the whole string
  # ("/usr/bin/clang-19 -ffp-contract=off") finds nothing, so the HMX tests would skip themselves
  # everywhere - including CI, where the toolchain was installed and the skip looked inexplicable.
  # Probe the compiler itself, not the whole command line.
  cc = (os.environ.get("CC") or "clang").split()[0]
  return shutil.which(cc) is not None

def _sim(work:pathlib.Path, elf:str, *args) -> str:
  from tinygrad.runtime import ops_dsp
  t = tools()
  r = subprocess.run([str(t / "bin/hexagon-sim"), "-mv69", "--mhmx", "1", "--timing", elf, "--", *map(str, args)], cwd=work,
                     env=ops_dsp._hexsim_env(t, work), capture_output=True, text=True, check=True)
  return r.stdout

def _cc(work:pathlib.Path, srcs, out:str, extra=()):
  subprocess.run([str(tools() / "bin/hexagon-clang"), "-mv69", "-mhmx", "-mhvx", "-mhvx-length=128B", "-O2", "-Wno-deprecated-non-prototype",
                  *extra, *map(str, srcs), "-o", out, "-lhexagon", "-lm"], cwd=work, check=True)

@contextlib.contextmanager
def capture_dsp(match=("__hmx_", "WMMA")):
  """record MOCKDSP kernel calls whose source contains one of `match` as (source, [argument bytes]) instead of running them.
  Sources are keyed by the compiled binary (compile_cached sees every source, cached or not; tinygrad compiles a whole schedule
  before it creates any runtime, so "the last rendered source" would be wrong)"""
  from tinygrad.helpers import to_mv
  from tinygrad.runtime import ops_dsp
  calls: list[tuple[str, list[bytes]]] = []
  src_of: dict[bytes, str] = {}
  o_cc, o_init, o_call = ops_dsp.DSPCompiler.compile_cached, ops_dsp.MockDSPProgram.__init__, ops_dsp.MockDSPProgram.__call__
  def compile_cached(self, src:str) -> bytes:
    lib = o_cc(self, src); src_of[bytes(lib)] = src; return lib
  def init(self, dev, obj): o_init(self, dev, obj); self._src = src_of.get(bytes(obj.lib), "")
  def call(self, *bufs, vals=(), **kw):
    if any(m in getattr(self, "_src", "") for m in match):
      calls.append((self._src, [bytes(to_mv(b.va_addr, b.size)) for b in bufs])); return 0.0
    return o_call(self, *bufs, vals=vals, **kw)
  ops_dsp.DSPCompiler.compile_cached, ops_dsp.MockDSPProgram.__init__, ops_dsp.MockDSPProgram.__call__ = compile_cached, init, call
  from tinygrad.helpers import Context
  try:
    with Context(PARALLEL=0): yield calls  # compile in this process, where the hook is
  finally: ops_dsp.DSPCompiler.compile_cached, ops_dsp.MockDSPProgram.__init__, ops_dsp.MockDSPProgram.__call__ = o_cc, o_init, o_call

MAIN = r"""#include <stdio.h>
unsigned char* __hmx_vtcm;
unsigned int __hmx_gen = 1;
extern unsigned long long hexagon_sim_read_pcycles(void);
@BODY@
@DECL@
int main(void) {
  unsigned base; __asm__ volatile("%0 = cfgbase" : "=r"(base));
  __hmx_vtcm = (unsigned char*)(*(volatile unsigned*)((base << 16) + 0x38) << 16);  /* VTCM base from the config table */
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" :: "r"(r));
@LOAD@
  @NAME@(@ARGS@);  /* warm-up; the second call is timed */
  unsigned long long t0 = hexagon_sim_read_pcycles();
  @NAME@(@ARGS@);
  printf("pcycles %llu\n", hexagon_sim_read_pcycles() - t0);
@LOAD@
  @NAME@(@ARGS@);  /* the checked output: from the original inputs */
  { FILE* f = fopen("out.bin", "wb"); fwrite(b0, 1, @N0@, f); fclose(f); }
  return 0;
}
"""

def run_captured(src:str, bufs:list[bytes], work:pathlib.Path) -> tuple[bytes, int]:
  """a captured tinygrad kernel on hexagon-sim with its captured arguments -> (buffer 0 after one call, pcycles per call)"""
  body = src.split("/* DSP boilerplate */")[0]
  name = re.search(r"noinline\)\) void\s+(\w+)\(", body) or re.search(r"\bvoid\s+(\w+)\(unsigned char\*", body)
  for i, b in enumerate(bufs): (work / f"buf{i}.bin").write_bytes(b)
  decl = "\n".join(f"static unsigned char b{i}[{len(b) + 128}] __attribute__((aligned(128)));" for i, b in enumerate(bufs))
  load = "\n".join(f'  {{ FILE* f = fopen("buf{i}.bin", "rb"); fread(b{i}, 1, {len(b)}, f); fclose(f); }}' for i, b in enumerate(bufs))
  c = (MAIN.replace("@BODY@", body).replace("@DECL@", decl).replace("@LOAD@", load).replace("@NAME@", name.group(1))
       .replace("@ARGS@", ", ".join(f"(void*)b{i}" for i in range(len(bufs)))).replace("@N0@", str(len(bufs[0]))))
  (work / "k.c").write_text(c)
  _cc(work, ["k.c"], "k.elf")
  out = _sim(work, "k.elf")
  return (work / "out.bin").read_bytes(), int(re.search(r"pcycles (\d+)", out).group(1))

def run_hand(c_file:pathlib.Path, work:pathlib.Path, *args, includes=(), defines=()) -> str:
  """build a hand driver (its headers from `includes`) for hexagon-sim and run it in `work` -> its stdout"""
  _cc(work, [c_file], "hand.elf", [*(f"-I{i}" for i in includes), *(f"-D{d}" for d in defines)])
  return _sim(work, "hand.elf", *args)
