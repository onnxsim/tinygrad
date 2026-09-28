"""Integer QDQ convolution for Hexagon v65 HVX (vrmpy): Conv(DequantizeLinear(xq), DequantizeLinear(wq)) with u8/u16 activations
and s8/s16 per-channel weights, accumulated exactly in int32 on the DSP's integer vector unit instead of in scalar float.

  y[n] = sx * sw[n] * (sum_{c,dy,dx} xq[c, y*s+dy, x*s+dx] * wq[n,c,dy,dx]  -  zx * sum wq[n]) + b[n]

- The input is padded with its own zero point, so a padded tap contributes (zx - zx) * w = 0 after the zx * sum(w) correction.
- A 16-bit operand is two byte planes (x = hi*256 + lo, lo unsigned): each (activation plane, weight plane) pair is one u8 x s8 /
  u8 x u8 vrmpy pass (tinygrad's hexagon_v65 TensorCore), realized on its own so the TensorCore matches. W8A8 is one pass,
  W8A16 two, W16A16 four. The passes' int32 sums are exact; they are combined, shifted, in float32 with the correction.
- vrmpy multiplies 4 consecutive reduce elements (input channels) of one pixel by the same 4 of 32 output channels, so the
  weights are prepacked [C/4][kh][kw][N][4] (one 128-byte operand per instruction) and each activation plane is stored
  channel-4-interleaved [C/4][H][W][4] (one 4-byte operand). Both are views back to the logical NCHW / NCHW-weight shapes.
- Depthwise (groups == C) has no channel reduction for vrmpy: it is plain int32 multiply-adds, which HVX vectorizes.
"""
from __future__ import annotations
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

def _planes(q:np.ndarray|Tensor, bits:int) -> list[tuple[int, object]]:
  """(shift, plane) byte planes of an integer array/tensor, low plane unsigned, top plane keeps the sign"""
  if bits == 8: return [(0, q)]
  lo, hi = (q & 255), (q >> 8)  # arithmetic shift: hi carries the sign of a signed 16-bit value
  return [(0, lo), (8, hi)]

def _np_plane_dtype(p:np.ndarray, signed:bool): return p.astype(np.int8 if signed else np.uint8)

def pack_weight_planes(wq:np.ndarray) -> list[tuple[int, np.ndarray, bool]]:
  """(N, C, kh, kw) int8/int16 -> [(shift, packed (C4, kh, kw, N, 4) plane, signed)], C padded to a multiple of 4 with zeros"""
  N, C, kh, kw = wq.shape
  C4 = (C + 3) // 4
  w = np.zeros((N, C4 * 4, kh, kw), dtype=np.int32)
  w[:, :C] = wq
  bits = 8 if wq.dtype == np.int8 else 16
  out = []
  for shift, plane in _planes(w, bits):
    signed = shift == 8 or bits == 8
    packed = np.ascontiguousarray(plane.reshape(N, C4, 4, kh, kw).transpose(1, 3, 4, 0, 2))
    out.append((shift, _np_plane_dtype(packed, signed), signed))
  return out

def qconv2d(xq:Tensor, zx:int, sx:float, wq:np.ndarray, sw:np.ndarray, bias:Tensor|None=None, stride=1, dilation=1,
            padding:tuple[int, int, int, int]=(0, 0, 0, 0), groups:int=1) -> Tensor:
  """xq: (1, C, H, W) uint8/uint16, wq: (N, C/groups, kh, kw) int8/int16 (zero point 0), sw: (N,) float, bias: float (N,) or None.
  padding is tinygrad's (left, right, top, bottom). Returns float32 (1, N, OH, OW)"""
  N, Cg, kh, kw = wq.shape
  C = xq.shape[1]
  xbits = 8 if xq.dtype == dtypes.uint8 else 16
  if xq.dtype not in (dtypes.uint8, dtypes.uint16): raise NotImplementedError(f"qconv2d activations must be u8/u16, not {xq.dtype}")
  # the zero-point correction per output channel and the per-channel scale, in float64 then float32
  scale = Tensor((np.float64(sx) * sw.astype(np.float64)).astype(np.float32), device=xq.device).reshape(1, N, 1, 1)
  corr = Tensor((-float(zx) * wq.reshape(N, -1).astype(np.float64).sum(1)).astype(np.float32), device=xq.device).reshape(1, N, 1, 1)
  xp = xq.cast(dtypes.int32).pad(((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])), value=zx)
  acc: Tensor|None = None
  if groups == 1:
    C4 = (C + 3) // 4
    if C4 * 4 != C: xp = xp.pad(((0, 0), (0, C4 * 4 - C), (0, 0), (0, 0)), value=zx)  # extra channels meet zero weights
    Hp, Wp = xp.shape[2], xp.shape[3]
    wplanes = pack_weight_planes(wq)
    for ashift, aplane in _planes(xp, xbits):
      # channel-4-interleaved activation plane, viewed back as (1, C4*4, Hp, Wp)
      a = aplane.cast(dtypes.uint8).reshape(C4, 4, Hp, Wp).permute(0, 2, 3, 1).contiguous()
      a = a.permute(0, 3, 1, 2).reshape(1, C4 * 4, Hp, Wp)
      for wshift, wpacked, signed in wplanes:
        wt = Tensor(wpacked, device=xq.device).permute(3, 0, 4, 1, 2).reshape(N, C4 * 4, kh, kw)
        part = a.cast(dtypes.int32).conv2d(wt.cast(dtypes.int32), stride=stride, dilation=dilation).realize()
        term = part.cast(dtypes.float32) * float(1 << (ashift + wshift))
        acc = term if acc is None else acc + term
  elif groups == C and Cg == 1:
    # 32 channels per vector: the padded input (kept at its own width) is stored [C/32][H][W][32] and the weights [kh][kw][C], so
    # one tap of 32 channels is one contiguous load of each. Both are views back to the logical NCHW / (C, 1, kh, kw) shapes.
    # The copy also materializes the padding (padding masks on every tap cost more than the multiply-adds)
    xpu = xq.pad(((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])), value=zx)
    Hp, Wp = xpu.shape[2], xpu.shape[3]
    if C % 32 == 0:
      xpu = xpu.reshape(C // 32, 32, Hp, Wp).permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2).reshape(1, C, Hp, Wp)
    elif any(padding): xpu = xpu.contiguous()
    xp = xpu.cast(dtypes.int32)
    wt = Tensor(np.ascontiguousarray(wq[:, 0].astype(np.int32).transpose(1, 2, 0)), device=xq.device).permute(2, 0, 1).reshape(N, 1, kh, kw)
    # u16 x s16 x taps can pass 2^31: split the activation into byte planes then (each pass stays under 2^27 for 3x3)
    for ashift, aplane in (_planes(xp, xbits) if xbits == 16 and wq.dtype == np.int16 else [(0, xp)]):
      term = aplane.conv2d(wt, stride=stride, dilation=dilation, groups=groups).cast(dtypes.float32) * float(1 << ashift)
      acc = term if acc is None else acc + term
  else: raise NotImplementedError(f"qconv2d: groups={groups} with {Cg} channels per group")
  assert acc is not None
  y = (acc + corr) * scale
  return y if bias is None else y + bias.reshape(1, N, 1, 1)

def qmatmul(a:Tensor, wq:np.ndarray, sw:np.ndarray, bias:Tensor|None=None) -> Tensor:
  """a (..., K) float @ dequantized wq (K, N) int8/int16 (per output channel scale sw (N,), zero point 0) on the vrmpy passes.
  The activation has no QDQ of its own (a float head), so it is quantized here at run time: per tensor, asymmetric, uint16
  over its [min, max]. That quantization is the one approximation; the integer passes are exact. Returns float32 (..., N)"""
  K, N = wq.shape
  lead = a.shape[:-1]
  M = int(np.prod(lead)) if lead else 1
  K4 = (K + 3) // 4 * 4
  a2 = a.reshape(M, K).float()
  lo, hi = a2.min(), a2.max()
  s = ((hi - lo) / 65535.0).maximum(1e-30)
  z = (-lo / s).round().clip(0, 65535)
  aq = ((a2 / s).round() + z).clip(0, 65535).cast(dtypes.int32)
  if K4 != K: aq = aq.pad(((0, 0), (0, K4 - K)))
  w = np.zeros((K4, N), dtype=np.int32)
  w[:K] = wq
  colsum = Tensor(w.sum(0).astype(np.float32), device=a.device)
  acc: Tensor|None = None
  for ashift, aplane in _planes(aq, 16):
    ap = aplane.cast(dtypes.uint8).contiguous()
    for wshift, plane in _planes(w, 8 if wq.dtype == np.int8 else 16):
      signed = wshift == 8 or wq.dtype == np.int8
      # [K/4][N][4]: one vrmpy operand (32 outputs x 4 k) is one 128-byte load
      packed = np.ascontiguousarray(plane.reshape(K4 // 4, 4, N).transpose(0, 2, 1)).astype(np.int8 if signed else np.uint8)
      wv = Tensor(packed, device=a.device).permute(1, 0, 2).reshape(1, N, K4)
      part = (ap.reshape(M, 1, K4).cast(dtypes.int32) * wv.cast(dtypes.int32)).sum(-1, dtype=dtypes.int32).realize()
      term = part.cast(dtypes.float32) * float(1 << (ashift + wshift))
      acc = term if acc is None else acc + term
  assert acc is not None
  y = (acc - z * colsum.reshape(1, N)) * (s * Tensor(sw.astype(np.float32), device=a.device).reshape(1, N))
  if bias is not None: y = y + bias.reshape(1, N)
  return y.reshape(*lead, N) if lead else y.reshape(N)
