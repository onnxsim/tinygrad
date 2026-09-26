"""Hand-written Hexagon kernels as oracles for tinygrad's DSP backend.

Each family directory holds verbatim copies of hand-written HVX/scalar kernels from onnx-simplifier
(scripts/android/...), a small freestanding qemu driver per kernel (<name>_oracle.c, built on qemu_rt.h), and a
test that runs the hand kernel and a tinygrad implementation of the same op on the same inputs and asserts the
family's exactness contract. See README.md for the layout, the metric and the status table.

Everything runs under qemu-hexagon (linux-user): the hand kernel as a freestanding ELF, tinygrad through its own
MOCKDSP=1 path (DEV=DSP, set by conftest.py before tinygrad is imported). The performance number for both sides is
QEMU's executed-instruction count on the same real-shaped data (hand: around the kernel call inside the driver;
tinygrad: the sum over its kernels, GlobalCounters.time_sum_s * 1e9, since MockDSPProgram returns inscount/1e9).
It is an instruction count, not a cycle count: hexagon-sim's --timing mode would give cycles, but tinygrad's
HEXSIM=1 path runs kernels on zero-filled buffers, which is meaningless for the data-dependent gathers most of
these kernels are about. Treat the hand count as the target and the ratio as the signal.
"""
from __future__ import annotations
import functools, json, os, pathlib, re, shutil, subprocess, tempfile
from dataclasses import dataclass, field
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(os.environ.get("HAND_ORACLE_RESULTS", HERE / ".results.jsonl"))

def _run(cmd, timeout=1800, **kw) -> subprocess.CompletedProcess:
  return subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
                        check=False, **kw)

@functools.cache
def qemu() -> str|None: return shutil.which("qemu-hexagon-static") or shutil.which("qemu-hexagon")

@functools.cache
def clang() -> str|None:
  """An LLVM clang that builds freestanding hexagonv65..v73 ELFs (the Qualcomm toolchain's hexagon-clang 19 dropped
  v65, which tinygrad's MOCKDSP targets). HEXAGON_CLANG or CC first, then clang-19/20/18/clang."""
  if not shutil.which("ld.lld"): return None
  for c in filter(None, [os.environ.get("HEXAGON_CLANG"), os.environ.get("CC"), "clang-19", "clang-20", "clang-18", "clang"]):
    if (path := shutil.which(c)) is None: continue
    probe = [path, "--target=hexagon", "-mcpu=hexagonv65", "-mhvx=v65", "-x", "c", "-c", "-o", os.devnull, "-"]
    if _run(probe, timeout=60).returncode == 0: return path
  return None

@functools.cache
def hexagon_include() -> pathlib.Path|None:
  """The Hexagon toolchain's target headers (hexagon_types.h, hvx_hexagon_protos.h), for kernels written with the
  Q6_* intrinsics. HEXAGON_TOOLS (or HEXAGON_TOOLCHAIN) = the toolchain's Tools/ dir."""
  root = os.environ.get("HEXAGON_TOOLS") or os.environ.get("HEXAGON_TOOLCHAIN")
  if root and (p := pathlib.Path(root) / "target" / "hexagon" / "include").is_dir(): return p
  return None

@functools.cache
def libgcc() -> pathlib.Path|None:
  """The toolchain's soft-float archive, for a -nostdlib -ffreestanding build. A Hexagon float division
  (`float / float`, and transitively anything sigmoid/softmax-shaped) is not always a native instruction
  sequence: clang >= 19 inlines a Newton-Raphson sequence, clang 15/17/18 emit a call to
  `__hexagon_divsf3` instead, which then fails to link. The toolchain ships the routine in libgcc.a (a plain
  static archive, so only referenced symbols are pulled in and this is free for every kernel that does not
  need it). Same list, and the same caveat, as tinygrad's runtime/ops_dsp.py _find_libgcc: some SDK
  snapshots ship no v65 archive, and the Hexagon scalar ISA these target has been stable v65-v81, so the
  lowest available version is used."""
  root = os.environ.get("HEXAGON_TOOLS") or os.environ.get("HEXAGON_TOOLCHAIN")
  if not root: return None
  libdir = pathlib.Path(root) / "target" / "hexagon" / "lib"
  for arch in ["v65", "v66", "v67", "v68", "v69", "v71", "v73", "v75", "v77", "v79", "v81"]:
    if (p := libdir / arch / "libgcc.a").exists(): return p
  return None

def missing(intrinsics:bool=False) -> str|None:
  """Why this machine can't run the oracles (a skip reason), or None."""
  if int(os.environ.get("HVX_ARCH", "v65").lstrip("v")) >= 68:
    return "the qemu oracles need HVX_ARCH < v68 (qemu 8.2 can't decode the qfloat ops MOCKDSP emits for v68+)"
  if qemu() is None: return "needs qemu-hexagon(-static)"
  if clang() is None: return "needs an LLVM clang with the Hexagon target and ld.lld (set HEXAGON_CLANG)"
  if intrinsics and hexagon_include() is None: return "needs the Hexagon toolchain's target headers (set HEXAGON_TOOLS)"
  return None

def build_hand(driver:pathlib.Path, out:pathlib.Path, cpu:str="v73", hvx:bool=True, flags=(), includes=()) -> pathlib.Path:
  """Freestanding qemu build of an oracle driver. -ffp-contract=off everywhere: the hand kernels' exactness contracts
  (ORT bit-exact) assume no FMA contraction, and so does their DSP build."""
  cmd = [clang(), "--target=hexagon", f"-mcpu=hexagon{cpu}", *([f"-mhvx={cpu}", "-mhvx-length=128b"] if hvx else []),
         "-O2", "-ffp-contract=off", "-static", "-nostdlib", "-ffreestanding", "-fuse-ld=lld", "-Wno-unused-function",
         f"-I{HERE}", f"-I{driver.parent}", *[f"-I{i}" for i in includes], *flags]
  if hexagon_include() is not None: cmd += ["-isystem", hexagon_include()]
  res = _run([*cmd, *(["-x", "none", libgcc()] if libgcc() is not None else []), "-o", out, driver])
  assert res.returncode == 0, f"hand build of {driver.name} failed:\n{res.stdout}{res.stderr}"
  return out

@dataclass
class HandRun:
  outputs: list[np.ndarray]
  insns: dict[str, int] = field(default_factory=dict)
  stdout: str = ""

def run_hand(exe:pathlib.Path, inputs:list[np.ndarray], outputs:list[tuple[np.dtype, tuple]], ints=(), expect_exit:int=0) -> HandRun:
  """argv: one file per input, one file per output, then the integers. Outputs are read back with the given dtype
  and shape; "insns <label> <n>" stdout lines become HandRun.insns."""
  with tempfile.TemporaryDirectory() as d:
    tmp = pathlib.Path(d)
    ins = []
    for i, a in enumerate(inputs):
      np.ascontiguousarray(a).tofile(p := tmp / f"in{i}.bin")
      ins.append(p)
    outs = [tmp / f"out{i}.bin" for i in range(len(outputs))]
    res = _run([qemu(), exe, *ins, *outs, *[str(int(v)) for v in ints]])
    assert res.returncode == expect_exit, f"{exe.name} exited {res.returncode}:\n{res.stdout[-4000:]}{res.stderr[-4000:]}"
    got = [np.fromfile(p, dtype=dt).reshape(shape) for p, (dt, shape) in zip(outs, outputs)]
  insns = {m.group(1): int(m.group(2)) for m in re.finditer(r"^insns (\S+) (\d+)$", res.stdout, re.M)}
  return HandRun(got, insns, res.stdout)

@dataclass
class TinyRun:
  outputs: list[np.ndarray]
  insns: int
  kernels: int

def run_tinygrad(fn, *args, **kwargs) -> TinyRun:
  """fn builds tinygrad Tensors (inputs realized on DSP first, so input copies aren't counted) and returns a Tensor or
  a tuple of Tensors. Only the kernels realizing the outputs are counted."""
  from tinygrad import Tensor, GlobalCounters
  from tinygrad.helpers import Context
  targs = [Tensor(a).realize() if isinstance(a, np.ndarray) else a for a in args]
  GlobalCounters.reset()
  with Context(DEBUG=0):
    outs = fn(*targs, **kwargs)
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    Tensor.realize(*outs)
    insns, kernels = int(round(GlobalCounters.time_sum_s * 1e9)), GlobalCounters.kernel_count
  return TinyRun([o.numpy() for o in outs], insns, kernels)

def record(family:str, case:str, **kv):
  """One line per case in .results.jsonl (README table source), and on stdout for pytest -s."""
  row = {"family": family, "case": case, **kv}
  print(json.dumps(row))
  with RESULTS.open("a") as f: f.write(json.dumps(row) + "\n")

def mismatch(a:np.ndarray, b:np.ndarray) -> int:
  """Elements that differ bit for bit (floats compared as their bit patterns, so -0.0 != 0.0 and NaN == same NaN)."""
  a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
  assert a.shape == b.shape and a.dtype == b.dtype, f"{a.shape} {a.dtype} vs {b.shape} {b.dtype}"
  if a.dtype.kind == "f": a, b = a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}")
  return int((a != b).sum())
