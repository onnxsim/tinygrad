"""Depthwise convolution for the v65 DSP as a hand-written HVX kernel (runtime/ops_dsp.py _DW_HVX_HELPERS): the integer
convolution of unsigned byte planes with int8 / int16 weights, exactly (int32), one channel per loop iteration, channels split
over the DSP threads."""
from __future__ import annotations
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

def pack_dw_weights(wq:np.ndarray) -> np.ndarray:
  """(C, 1, kh, kw) int8 / int16 -> int32 words [C][Q][kh][G]: 4 consecutive kx taps per word (zero padded), Q = 1 (signed bytes)
  or 2 (low bytes unsigned, high bytes signed) for int16"""
  C, _, kh, kw = wq.shape
  G = (kw + 3) // 4
  w = np.zeros((C, kh, G * 4), dtype=np.int32)
  w[:, :, :kw] = wq[:, 0]
  planes = [w] if wq.dtype == np.int8 else [w & 255, w >> 8]
  out = np.zeros((C, len(planes), kh, G), dtype=np.uint32)
  for q, pl in enumerate(planes):
    b = (pl.astype(np.int64) & 255).astype(np.uint32).reshape(C, kh, G, 4)
    out[:, q] = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
  return out.view(np.int32)

def dw_flat_len(H:int, Wp:int, kh:int, kw:int) -> tuple[int, int]:
  """(number of 128-output blocks, padded input length) for output rows H (stride 1) of a Wp-wide image"""
  G = (kw + 3) // 4
  nb = (H * Wp + 127) // 128
  return nb, nb * 128 + (kh - 1) * Wp + 4 * G + 8 + 128

def dw_hvx(planes:list[Tensor], wpacked:Tensor, C:int, nb:int, Wp:int, kh:int, kw:int, Q:int, nthreads:int, mult:int=1) -> Tensor:
  """planes: 1 or 2 (low, high) u8 tensors (C // mult, Lin) of flat padded images; wpacked: int32 (C, Q, kh, G), C output channels, output
  channel c reading the image of input channel c // mult (a depthwise conv with a channel multiplier). Returns int32 (C, nb*128):
  sum over activation planes p and weight planes q of shift(8p + 8q) * (plane_p (*) w_q), flat output j = y*Wp + x"""
  G, P = (kw + 3) // 4, len(planes)
  Lin, Lout = planes[0].shape[1], nb * 128
  assert C % nthreads == 0 and all(p.shape == (C // mult, Lin) and p.dtype == dtypes.uint8 for p in planes)
  def kern(Y, X0, X1, W):
    t = UOp.range(nthreads, 0, AxisType.THREAD)
    ci = UOp.range(C // nthreads, 1)
    c = t * (C // nthreads) + ci
    Y, X0, X1, W = Y.flatten(), X0.flatten(), X1.flatten(), W.flatten()
    cu = UOp(Ops.CUSTOM, dtypes.void, (Y.index(c * Lout), X0.index((c // mult) * Lin), X1.index((c // mult) * Lin), W.index(c * (Q * kh * G)), ci, t),
             arg=f"__dw_hvx({{0}}, {{1}}, {{2}}, {{3}}, {nb}, {Wp}, {kh}, {G}, {P}, {Q});")
    return cu.end(ci, t).sink(arg=KernelInfo(name=f"dw_{kh}x{kw}_{P}{Q}" + (f"_m{mult}" if mult != 1 else ""), opts_to_apply=()))
  y = Tensor.empty(C, Lout, dtype=dtypes.int32, device=planes[0].device)
  return Tensor.custom_kernel(y, planes[0], planes[-1], wpacked, fxn=kern)[0]

def dw_prep(x:Tensor, Hp:int, Wp:int, pt:int, pl:int, Lin:int, pv:int, nthreads:int) -> list[Tensor]:
  """x: (1, C, H, W) uint8 / uint16 values < 256 / < 65536 -> flat padded byte planes (C, Lin) [low] or [low, high]: the image placed at
  (pt, pl) of a Hp x Wp image filled with pv, then Lin - Hp * Wp more pv bytes (the kernel's read-ahead)"""
  _, C, H, W = x.shape
  two = x.dtype == dtypes.uint16
  assert x.dtype in (dtypes.uint8, dtypes.uint16) and C % nthreads == 0 and Lin % 128 == 0
  def kern(*bufs):
    outs, S = bufs[:-1], bufs[-1].flatten()
    t = UOp.range(nthreads, 0, AxisType.THREAD)
    ci = UOp.range(C // nthreads, 1)
    c = t * (C // nthreads) + ci
    dst = [o.flatten().index(c * Lin) for o in outs]
    src = S.index(c * (H * W))
    tail = f"{H}, {W}, {Lin}, {Wp}, {pt}, {pl}, {pv});"
    arg = f"__dw_prep16({{0}}, {{1}}, {{2}}, {tail}" if two else f"__dw_prep8({{0}}, {{1}}, {tail}"
    cu = UOp(Ops.CUSTOM, dtypes.void, (*dst, src, ci, t), arg=arg)
    return cu.end(ci, t).sink(arg=KernelInfo(name=f"dwprep_{H}x{W}_{2 if two else 1}", opts_to_apply=()))
  outs = [Tensor.empty(C, Lin, dtype=dtypes.uint8, device=x.device) for _ in range(2 if two else 1)]
  return Tensor.custom_kernel(*outs, x.contiguous(), fxn=kern)[:len(outs)]
