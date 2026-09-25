"""tinygrad implementations of the RPN post-processing kernels (rpn/*_kernel.h), for the oracle tests."""
import numpy as np
from tinygrad import Tensor, dtypes
import exact

# ------------------------------------------------------------------------------------------ proposal decode

def pd_constants(P:dict, base:np.ndarray, stride:float) -> dict:
  """Everything pd_prepare/pd_prepare_anchors derive from the level's constants, computed on the host with the
  same float32 arithmetic: the 256-entry delta LUT (backbone uint8 -> twice re-quantized fp32 delta) and the
  box-grid QuantizeLinear thresholds (see exact.py for why a threshold count and not x / scale)."""
  q = np.arange(256, dtype=np.float32)
  d = (q - np.float32(P["bb_z"])) * np.float32(P["bb_s"])
  lut = exact.qdq(exact.qdq(d, P["s1"], P["z1"]), P["s2"], P["z2"])
  return {"lut": lut, "box_t": exact.quantize_thresholds(P["box_s"], P["box_z"]), "base": base.astype(np.float32),
          "stride": np.float32(stride)}

def _rint(x:Tensor) -> Tensor: return x.round()  # half to even, like pd_rint

def pd_expf(x:Tensor) -> Tensor:
  """pd_expf_nb, op for op (Cephes polynomial, 2^n by building the exponent bits)."""
  x = (x > 88.0).where(88.0, x)
  xl = (x < -87.0).where(-87.0, x)
  n = _rint(xl * 1.44269504088896341)
  r = xl - n * 0.693359375
  r = r - n * -2.12194440e-4
  p = Tensor.full_like(r, 1.9875691500e-4)
  for c in (1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1): p = p * r + c
  p = p * r * r + r + 1.0
  s = ((n.cast(dtypes.int32) + 127) << 23).bitcast(dtypes.float32)
  return (x < -87.0).where(0.0, p * s)

def _count_ge(v:Tensor, t:Tensor) -> Tensor:
  """count(v >= t) for the 255 ascending QuantizeLinear thresholds, as a branch-free binary search: 8 table loads and
  compares per value instead of 255 (the thresholds are sorted, so the count is the insertion position)."""
  pos = v.zeros_like(dtype=dtypes.int32)
  for step in (128, 64, 32, 16, 8, 4, 2, 1):
    cand = pos + step
    take = (cand <= 255) & (v >= t[(cand - 1).clip(0, 254)])
    pos = take.where(cand, pos)
  return pos

def proposal_decode(nchw:Tensor, idx:Tensor, H:int, W:int, P:dict, C:dict) -> Tensor:
  """pd_decode's shipped path: anchors computed from the level grid, deltas gathered straight from the backbone's
  uint8 [1, 3*4, H, W] map through the LUT, decode, clip, box-grid QDQ. Returns [k, 4] float32."""
  HW = H * W
  a, hw = idx % 3, idx // 3
  h = hw // W
  fx, fy = (hw - h * W).cast(dtypes.float32) * float(C["stride"]), h.cast(dtypes.float32) * float(C["stride"])
  base = Tensor(C["base"], device=nchw.device)                      # [3, 4]
  bsel = [base[:, c][a] for c in range(4)]
  a0, a1, a2, a3 = bsel[0] + fx, bsel[1] + fy, bsel[2] + fx, bsel[3] + fy
  flat, lut = nchw.flatten(), Tensor(C["lut"], device=nchw.device)
  d = [lut[flat[(a * 4 + c) * HW + hw].cast(dtypes.int32)] for c in range(4)]
  w, h_ = (a2 - a0) + 1.0, (a3 - a1) + 1.0
  cx, cy = a0 + 0.5 * w, a1 + 0.5 * h_
  pcx, pcy = d[0] * w + cx, d[1] * h_ + cy
  ec = float(np.float32(P["exp_clip"]))
  ew, eh = pd_expf((d[2] < ec).where(d[2], ec)), pd_expf((d[3] < ec).where(d[3], ec))
  hw2, hh2 = 0.5 * (ew * w), 0.5 * (eh * h_)
  box = [pcx - hw2, pcy - hh2, (pcx + hw2) - 1.0, (pcy + hh2) - 1.0]
  cxl, cyl = float(np.float32(P["clip_x"])), float(np.float32(P["clip_y"]))
  box = [(v < 0.0).where(0.0, (v > lim).where(lim, v)) for v, lim in zip(box, (cxl, cyl, cxl, cyl))]
  t = Tensor(C["box_t"], device=nchw.device)
  bz, bs = float(P["box_z"]), float(np.float32(P["box_s"]))
  q = [_count_ge(v, t).cast(dtypes.float32) for v in box]
  return Tensor.stack(*[(qc - bz) * bs for qc in q], dim=-1)

# ------------------------------------------------------------------------------------------------- TopK

def _bitonic_desc(x:Tensor) -> Tensor:
  """Tensor.sort's bitonic network (descending, along the last axis, which must be a power of two) without its index
  recovery: Tensor.sort finds each sorted value's index with an n x n equality mask, which is O(n^2) memory (26.6 G
  elements at the real n = 163200). Here the index rides inside the key instead, so the network alone is enough."""
  *batch, n = x.shape
  n_stages = n.bit_length() - 1
  assert 1 << n_stages == n
  b = len(batch)
  x = x.reshape(*batch, *((2,) * n_stages))
  for stage in range(1, n_stages + 1):
    if stage != n_stages:
      cdim = b + n_stages - stage - 1
      blue, green = x.split(1, cdim)
      flip_dims = tuple(-i for i in range(1, stage + 2))
      x = blue.cat(green.flip(flip_dims), dim=cdim).contiguous()
    for substage in range(stage - 1, -1, -1):
      pdim = b + n_stages - substage - 1
      top, bottom = x.split(1, pdim)
      x = top.maximum(bottom).cat(top.minimum(bottom), dim=pdim).contiguous()
    if stage != n_stages:
      blue, fgreen = x.split(1, cdim)
      x = blue.cat(fgreen.flip(flip_dims), dim=cdim)
  return x.reshape(*batch, n)

def _merge_desc(x:Tensor) -> Tensor:
  """x (..., 2m): two descending runs of m. Reversing the second makes the whole row bitonic; log2(2m) half-cleaner
  stages then sort it descending."""
  *batch, n = x.shape
  m = n // 2
  x = x[..., :m].cat(x[..., m:].flip(-1), dim=-1)
  s = n.bit_length() - 1
  b = len(batch)
  x = x.reshape(*batch, *((2,) * s))
  for sub in range(s - 1, -1, -1):
    pdim = b + s - sub - 1
    top, bottom = x.split(1, pdim)
    x = top.maximum(bottom).cat(top.minimum(bottom), dim=pdim).contiguous()
  return x.reshape(*batch, n)

def _topk_keys(key:Tensor, k:int) -> Tensor:
  """The k largest uint64 keys of a 1-D tensor, descending. A tournament instead of one sort of all n: sort chunks of
  B = next_pow2(k) keys, then repeatedly merge pairs of sorted chunks and keep each merge's top B -- ~n log2(B)^2 / 2
  compare-exchanges plus log2(n/B) merge rounds, instead of n log2(n)^2 / 2 for the full network."""
  n = key.shape[0]
  B = 1 << max(0, (k - 1).bit_length())
  nc = 1 << max(0, (-(-n // B) - 1).bit_length())                 # chunk count, a power of two
  x = key.pad(((0, nc * B - n),), value=0).reshape(nc, B)          # key 0 sorts last (see topk_desc)
  x = _bitonic_desc(x)
  while x.shape[0] > 1:
    x = _merge_desc(x.reshape(x.shape[0] // 2, 2 * B))[:, :B]
  return x.reshape(B)[:k]

def topk_desc(x:Tensor, k:int) -> tuple[Tensor, Tensor]:
  """ORT TopK (largest, sorted) with its tie order -- value descending, then index ascending -- as one uint64 sort key
  (tk_key(value) << 32 | ~index): a descending sort of the keys is exactly that order, no stability needed. Every real
  key is > 0 (tk_key sets the top bit of non-negative floats and ~index > 0 below 2^32 - 1), so padding with 0 is safe.
  The index comes from an arange over x's own length: an arange that is later padded isn't folded by tinygrad's
  symbolic and stays an O(n^2) reduce (the n = 2550 case spent 99% of its instructions there)."""
  n = x.shape[0]
  u = x.bitcast(dtypes.uint32)
  u = (u == 0x80000000).where(0, u)                                           # -0 ties with +0, like tk_key
  key = (u >= 0x80000000).where(u ^ 0xFFFFFFFF, u | 0x80000000)
  i = Tensor.arange(n, dtype=dtypes.uint32)
  s = _topk_keys(((key.cast(dtypes.uint64) << 32) | (i ^ 0xFFFFFFFF).cast(dtypes.uint64)).contiguous(), k)
  hi, lo = (s >> 32).cast(dtypes.uint32), (s & 0xFFFFFFFF).cast(dtypes.uint32)
  vals = (hi >= 0x80000000).where(hi ^ 0x80000000, hi ^ 0xFFFFFFFF).bitcast(dtypes.float32)
  return vals, (lo ^ 0xFFFFFFFF).cast(dtypes.int64)

# -------------------------------------------------------------------------------------------------- NMS

def _maxmin(lhs:Tensor, rhs:Tensor) -> tuple[Tensor, Tensor]:  # nms_maxmin: (min, max), lhs >= rhs picks rhs as min
  ge = lhs >= rhs
  return ge.where(rhs, lhs), ge.where(lhs, rhs)

def _fmax(a:Tensor, b:Tensor) -> Tensor: return (a < b).where(b, a)  # std::max
def _fmin(a:Tensor, b:Tensor) -> Tensor: return (b < a).where(b, a)  # std::min

def suppress_matrix(boxes:Tensor, thr:float) -> Tensor:
  """[i, j] = ORT's SuppressByIOU(box i, box j), the same fp32 ops in the same order (nms_suppress_exact), for all
  pairs at once. Boxes are [n, 4] as [y1, x1, y2, x2]."""
  b1, b2 = boxes.unsqueeze(1), boxes.unsqueeze(0)            # [n, 1, 4], [1, n, 4]
  x1_min, x1_max = _maxmin(b1[..., 1], b1[..., 3])
  x2_min, x2_max = _maxmin(b2[..., 1], b2[..., 3])
  ix_min, ix_max = _fmax(x1_min, x2_min), _fmin(x1_max, x2_max)
  y1_min, y1_max = _maxmin(b1[..., 0], b1[..., 2])
  y2_min, y2_max = _maxmin(b2[..., 0], b2[..., 2])
  iy_min, iy_max = _fmax(y1_min, y2_min), _fmin(y1_max, y2_max)
  inter = (ix_max - ix_min) * (iy_max - iy_min)
  area1, area2 = (x1_max - x1_min) * (y1_max - y1_min), (x2_max - x2_min) * (y2_max - y2_min)
  union = area1 + area2 - inter
  ok = (ix_max > ix_min) & (iy_max > iy_min) & (inter > 0.0) & (area1 > 0.0) & (area2 > 0.0) & (union > 0.0)
  iou = inter / ok.where(union, 1.0)                          # IEEE division: tinygrad lowers a*(1/b) to a/b (FDIV)
  return ok & (iou > float(np.float32(thr)))

def nms_greedy_blocked(boxes:Tensor, scores:Tensor, thr:float, max_out:int, B:int=32) -> tuple[Tensor, Tensor]:
  """Greedy NMS, exact for any input, without a device-side data-dependent loop: the visit order is cut into blocks of B.
  A block's boxes are first checked against every kept box of the earlier (already final) blocks -- one masked row
  reduction -- and then the recurrence inside the block needs at most B Jacobi sweeps over a B x B matrix. That is
  O(n^2 + n B^2) work and n/B * (B + 1) kernels, instead of O(n^3) for plain Jacobi sweeps over the whole n x n
  matrix (n sweeps are needed for exactness)."""
  n = boxes.shape[0]
  _, order = topk_desc(scores, n)
  order = order.cast(dtypes.int32)
  Sp = suppress_matrix(boxes[order], thr).cast(dtypes.int32).contiguous()   # Sp[a, b]: visit a suppressed by visit b
  keep_blocks: list[Tensor] = []
  for b0 in range(0, n, B):
    b1 = min(n, b0 + B)
    if b0:
      kept = Tensor.cat(*keep_blocks) if len(keep_blocks) > 1 else keep_blocks[0]
      pre = ((Sp[b0:b1, :b0] * kept.unsqueeze(0)).sum(1) == 0).cast(dtypes.int32)
    else:
      pre = Tensor.ones(b1 - b0, dtype=dtypes.int32)
    inner = (Sp[b0:b1, b0:b1] * Tensor.ones(b1 - b0, b1 - b0, dtype=dtypes.int32).tril(-1)).contiguous()
    pre = pre.contiguous()
    k = pre
    for _ in range(b1 - b0): k = (pre * ((inner * k.unsqueeze(0)).sum(1) == 0).cast(dtypes.int32)).contiguous()
    keep_blocks.append(k)
  keep = Tensor.cat(*keep_blocks) if len(keep_blocks) > 1 else keep_blocks[0]
  rank = keep.cumsum() - 1
  keep = keep * (rank < max_out).cast(dtypes.int32)
  slot = Tensor.arange(n, dtype=dtypes.int32).unsqueeze(1)
  hit = (rank.unsqueeze(0) == slot) & (keep.unsqueeze(0) == 1)
  sel = (hit.cast(dtypes.int32) * (order + 1).unsqueeze(0)).sum(1) - 1
  return sel, keep.sum().reshape(1)
