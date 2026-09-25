"""tinygrad implementation of multi-scale deformable attention (msda/msda_kernel.h's contract), for the oracle test.

The hand kernel's scalar body, op for op: per (query, head), acc starts at 0 and adds value * w for every visible map
v, level l, point p and bilinear tap t in that order (skipped points/taps add nothing). The sum is written as an explicit
chain of adds in that order -- a Tensor reduction may split or reorder a float sum -- and the per-tap gathers are
Tensor indexing by computed row indices (one head's D channels of one pixel), folded by tinygrad into direct loads.
"""
from dataclasses import dataclass
import numpy as np
from tinygrad import Tensor, dtypes

MSDA_LOC, MSDA_REF_PIX, MSDA_REF_BOX = 0, 1, 2

@dataclass
class MsdaShape:
  NV:int
  L:int
  H:list
  W:list
  start:list
  S:int
  M:int
  D:int
  P:int
  Q:int
  NO:int
  mode:int
  NVR:int=1
  RL:int=1
  R:int=1
  RD:int=2
  vdtype:int=0
  has_vis:bool=False

  def packed(self) -> np.ndarray:
    hdr = [self.NV, self.L, self.S, self.M, self.D, self.P, self.Q, self.NO, self.mode, self.NVR, self.RL, self.R, self.RD,
           int(self.has_vis), self.vdtype]
    return np.array(hdr + [x for l in range(self.L) for x in (self.H[l], self.W[l], self.start[l])], np.int32)

def msda(s:MsdaShape, value:Tensor, vscale:Tensor|None, vzp:Tensor|None, loc:Tensor, ref:Tensor|None, attw:Tensor,
         vis:Tensor|None) -> Tensor:
  """value: (NV, S, M*D) float32, or uint8 with per-map vscale/vzp. loc (Q, M, NO, L, P, 2), ref (NVR, Q, RL, R, RD),
  attw (Q, M, NO, L, P), vis (NV, Q) uint8. Returns (Q, M*D) float32."""
  Q, M, D = s.Q, s.M, s.D
  rows = value.reshape(s.NV * s.S * M, D)                        # row (v, pixel, head) -> the head's D channels
  if vis is not None:
    visb = vis != 0                                              # (NV, Q)
    n = visb.cast(dtypes.int32).sum(0)
  else:
    visb, n = None, Tensor.full((Q,), s.NV, dtype=dtypes.int32)
  # inv = 1.0f / (float)max(n, 1), then attw * inv: materialized, because tinygrad rewrites a * (1/b) into a / b (codegen/
  # decomp/op.py, backends with FDIV) -- a different rounding from the hand kernel's multiply by the rounded reciprocal
  inv = (1.0 / (n > 1).where(n, 1).cast(dtypes.float32)).contiguous()
  head = Tensor.arange(M, dtype=dtypes.int32).reshape(1, M)
  acc = None
  for v in range(s.NV):
    o = v if s.NO > 1 else 0
    for l in range(s.L):
      Wl, Hl = float(s.W[l]), float(s.H[l])
      for p in range(s.P):
        lx, ly = loc[:, :, o, l, p, 0], loc[:, :, o, l, p, 1]     # (Q, M)
        if s.mode == MSDA_LOC:
          x, y = -0.5 + lx * Wl, -0.5 + ly * Hl
        else:
          r = ref[v if s.NVR > 1 else 0, :, l if s.RL > 1 else 0, p % s.R, :].reshape(Q, 1, s.RD)
          ax, ay = r[..., 0] * Wl - 0.5, r[..., 1] * Hl - 0.5
          if s.mode == MSDA_REF_PIX: x, y = ax + lx * 1.0, ay + ly * 1.0
          else:
            sx, sy = ((r[..., 2] * 0.5) / float(s.P)) * Wl, ((r[..., 3] * 0.5) / float(s.P)) * Hl
            x, y = ax + lx * sx, ay + ly * sy
        ok = (x > -1.0) & (x < Wl) & (y > -1.0) & (y < Hl)
        if visb is not None: ok = ok & visb[v].reshape(Q, 1)
        x0 = (x + 1.0).cast(dtypes.int32) - 1                    # floor for x > -1
        y0 = (y + 1.0).cast(dtypes.int32) - 1
        fx, fy = x - x0.cast(dtypes.float32), y - y0.cast(dtypes.float32)
        a = attw[:, :, o, l, p] * inv.reshape(Q, 1)
        ws = [((1.0 - fy) * (1.0 - fx)) * a, ((1.0 - fy) * fx) * a, (fy * (1.0 - fx)) * a, (fy * fx) * a]
        for t, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
          xt, yt = x0 + dx, y0 + dy
          tap_ok = ok & (xt >= 0) & (xt < s.W[l]) & (yt >= 0) & (yt < s.H[l])
          pix = (v * s.S + s.start[l]) + tap_ok.where(yt * s.W[l] + xt, 0)
          val = rows[(pix * M + head)]                            # (Q, M, D)
          if s.vdtype == 1: val = (val.cast(dtypes.float32) - vzp[v].cast(dtypes.float32)) * vscale[v]
          term = tap_ok.unsqueeze(-1).where(val * ws[t].unsqueeze(-1), 0.0)
          # the hand kernel's acc starts at +0.0: 0 + (-0.0) is +0.0, and tinygrad would fold a literal `0.0 + term` away
          acc = (term == 0.0).where(0.0, term) if acc is None else acc + term
    # one kernel per value map: with the one-hot gathers folded into direct loads, the whole NV x L x P x 4-tap chain fuses
    # into one kernel, and at BEVFormer SCA's 6 maps x 8 points (u8) its 700 KB of C kept clang busy for minutes
    if s.NV > 1: acc = acc.contiguous()
  return acc.reshape(Q, M * D)
