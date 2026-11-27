"""AxelerScript text for a tiled int8 GEMM, emitted from a tinygrad tile plan (no device needed).

tinygrad's own scheduler fuses a tiled matmul graph into one compute kernel with no tile boundaries,
slots, or waits. This module keeps tinygrad's graph and scheduler for the arithmetic and produces the
DMA/slot/wait structure from an explicit per-tile plan, emitting the AxelerScript the Metis pipelined
GEMM uses (overlapped waits after both launches, two L1 slots, one DDR output buffer cast into row slices).

Verified against the reference generator `scripts/axelera/pipe_unroll.py` (OVERLAP=1, OUTSLICE=1, NSLOTS=2).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

SCALE = 1.0 / 4096.0

@dataclass(frozen=True)
class TilePlan:
  index: int
  slot: int
  row0: int
  row1: int

def tile_plan(M:int, NT:int, nslots:int=2) -> list[TilePlan]:
  TILE = M // NT
  return [TilePlan(i, i % nslots, i * TILE, (i + 1) * TILE) for i in range(NT)]

def schedule_tiled_gemm(M:int, K:int, N:int, NT:int) -> tuple[int, int]:
  """tinygrad schedule of the tiled GEMM graph: (compute kernels, copies). No device is touched."""
  TILE = M // NT
  rng = np.random.default_rng(0)
  x = Tensor(rng.integers(-128, 128, (M, K), dtype=np.int8), device="CPU")
  w = Tensor(rng.integers(-128, 128, (K, N), dtype=np.int8), device="CPU")
  tiles = []
  for i in range(NT):
    acc = x[i*TILE:(i+1)*TILE].cast(dtypes.int32).matmul(w.cast(dtypes.int32))
    tiles.append(((acc * SCALE) + 0.5).floor().clip(-128, 127).cast(dtypes.int8))
  calls = [u for u in Tensor.cat(*tiles, dim=0).schedule_linear().src if u.op is Ops.CALL]
  return (sum(1 for c in calls if c.src and c.src[0].op is Ops.SINK),
          sum(1 for c in calls if c.src and c.src[0].op is Ops.COPY))

def render_gemm(M:int, K:int, N:int, NT:int, nslots:int=2) -> str:
  assert M % NT == 0 and K % 64 == 0 and N % 64 == 0 and nslots >= 2
  TILE, KB, NB = M // NT, K // 64, N // 64
  tp = tile_plan(M, NT, nslots)
  params = ", ".join(f'x{i}: A.Buffer(({TILE}, {KB}), "int8x64", scope=A.DDR)' for i in range(NT))
  body = [f'        w_l1 = A.alloc_buffer(({K}, {NB}), "int8x64", scope=A.L1)',
          "        axol.atomic.copy(w, output=w_l1)",
          "        w_imc = axol.atomic.load_imc(w_l1)"]
  body += [f'        s{k} = A.alloc_buffer(({TILE}, {KB}), "int8x64", scope=A.L1)' for k in range(nslots)]
  body.append(f'        o_full = A.alloc_buffer(({NT * TILE}, {NB}), "int8x64", scope=A.DDR)')
  copy_line = lambda j: f"        axol.atomic.copy(x{j}, output=s{tp[j].slot}, id='c{j}')"
  body += [copy_line(0), "        A.waits('c0')"]
  for t in tp:
    i = t.index
    if i + 1 < NT: body.append(copy_line(i + 1))
    body.append(f"        acc{i} = axol.atomic.matmul(matrix_ifd=s{t.slot}, matrix_imc=w_imc, id='mm{i}')")
    waits = [f"'mm{i}'"] + ([f"'c{i + 1}'"] if i + 1 < NT else [])
    body.append("        A.waits(" + ", ".join(waits) + ")")
    body.append(f"        sc{i} = axol.atomic.mul(acc{i}, {SCALE!r})")
    body.append(f"        axol.atomic.dpu_cast(sc{i}, o_full[{t.row0} : {t.row1}, 0 : {NB}])")
  body.append("        A.output(o_full)")
  return ("@A.ir_module\nclass PipeUnroll:\n    @A.prim_func\n"
          f"    def main({params}, w: A.Buffer(({K}, {NB}), \"int8x64\", scope=A.DDR)):\n" + "\n".join(body) + "\n")
