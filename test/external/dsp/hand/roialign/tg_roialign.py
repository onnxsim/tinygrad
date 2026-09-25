"""tinygrad implementations of the RoiAlign kernels (roialign/*_kernel.h), for the oracle tests.

Both follow ORT's RoiAlign (opset < 16, mode avg, output_half_pixel) expression for expression, like the hand
kernels: sample position y = y1 + ph*bin_h + (iy + .5)*bin_h/sr, clamp, y_low = (int)y, bilinear weights hy*hx etc.
Two things tinygrad needs help with to stay bit-exact:
- the divisor of bin_h = roi_h / OH is a constant, and tinygrad folds `x / c` into `x * (1/c)` (not IEEE division
  unless c is a power of two). The divisors come in as a tiny device buffer instead, so `x / d` stays a division
  (FDIV, rendered as C `/`);
- the per-sample gathers are data-dependent: each tap reads the C channels of one (y, x) pixel of the map. That is
  Tensor indexing by a computed int tensor -- tinygrad folds the one-hot form into a direct load.
"""
from tinygrad import Tensor, dtypes

def _maxf(a:Tensor, b) -> Tensor: return (a > b).where(a, b)  # roi_maxf / ru8_maxf: a > b ? a : b

def _axis(v:Tensor, n:int):
  """ru8_axis / the hand kernels' per-sample clamp: (valid, lo, hi, l, h) for sample coordinates v."""
  valid = ((v < -1.0) | (v > float(n))).logical_not()
  v = (v <= 0.0).where(0.0, v)
  lo = v.cast(dtypes.int32)                                     # (int)v, v >= 0 here
  top = lo >= n - 1
  lo = top.where(n - 1, lo)
  hi = top.where(n - 1, lo + 1)
  v = top.where(lo.cast(dtypes.float32), v)
  l = v - lo.cast(dtypes.float32)
  return valid, lo, hi, l, 1.0 - l

def _samples(rois:Tensor, scale:float, n_out:int, sr:int, divs:Tensor, axis:int):
  """Sample coordinates along one axis, [R, n_out*sr] in (p, i) order: y1 + p*bin + ((i + .5)*bin)/sr."""
  c1 = rois[:, axis] * scale
  c2 = rois[:, axis + 2] * scale
  bin_ = _maxf(c2 - c1, 1.0) / divs[0]                          # / (float)OW  (true division, see module docstring)
  p = Tensor.arange(n_out, dtype=dtypes.float32).reshape(1, n_out, 1)
  i = (Tensor.arange(sr, dtype=dtypes.float32) + 0.5).reshape(1, 1, sr)
  b = bin_.reshape(-1, 1, 1)
  v = (c1.reshape(-1, 1, 1) + p * b) + (i * b) / divs[1]        # / (float)sr
  return v.reshape(rois.shape[0], n_out * sr)

def _grid(rois, scale, H, W, OH, OW, sr, divs_h, divs_w):
  """Per-(roi, ph, iy, pw, ix) axis data, broadcast to [R, OH, sr, OW, sr]."""
  R = rois.shape[0]
  ya = _axis(_samples(rois, scale, OH, sr, divs_h, 1), H)
  xa = _axis(_samples(rois, scale, OW, sr, divs_w, 0), W)
  ya = [t.reshape(R, OH, sr, 1, 1) for t in ya]
  xa = [t.reshape(R, 1, 1, OW, sr) for t in xa]
  return ya, xa

def _taps(feat_rows:Tensor, W:int, ya, xa):
  """The 4 bilinear taps' channel rows, each [R, OH, sr, OW, sr, C]."""
  (_, ylo, yhi, _, _), (_, xlo, xhi, _, _) = ya, xa
  return [feat_rows[(y * W + x)] for y, x in ((ylo, xlo), (ylo, xhi), (yhi, xlo), (yhi, xhi))]

def roialign_hwc(feat:Tensor, rois:Tensor, scale:float, OH:int, OW:int, sr:int, divs_h:Tensor, divs_w:Tensor) -> Tensor:
  """roialign_hwc: fp32 (H, W, C) map -> (R, OH, OW, C). Per sample (((p1*w1 + p2*w2) + p3*w3) + p4*w4), accumulated
  over the sr*sr samples in (iy, ix) order starting from 0, skipped samples adding nothing -- the hand kernel's order."""
  H, W, C = feat.shape
  ya, xa = _grid(rois, scale, H, W, OH, OW, sr, divs_h, divs_w)
  (yv, _, _, ly, hy), (xv, _, _, lx, hx) = ya, xa
  inv_count = 1.0 / float(sr * sr)
  ws = [(hy * hx) * inv_count, (hy * lx) * inv_count, (ly * hx) * inv_count, (ly * lx) * inv_count]
  p = _taps(feat.reshape(H * W, C), W, ya, xa)
  s = None
  for t in range(4):
    term = p[t] * ws[t].unsqueeze(-1)
    s = term if s is None else s + term
  s = (yv & xv).unsqueeze(-1).where(s, 0.0)                     # [R, OH, sr, OW, sr, C]
  acc = None
  for iy in range(sr):
    for ix in range(sr):
      term = s[:, :, iy, :, ix, :]
      acc = (0.0 + term) if acc is None else acc + term
  return acc                                                    # [R, OH, OW, C]

RU8_WBITS, RU8_PRESHIFT = 14, 7
RU8_ONE = 1 << RU8_WBITS

def ru8_requant_params(s_in:float, s_out:float, count:int) -> tuple[int, int]:
  """ru8_requant_params (host side, double precision like the C)."""
  M = float(s_in) / (float(count) * float(RU8_ONE) * float(s_out)) * float(1 << RU8_PRESHIFT)
  sh, lim = 0, 2147483647.0 / float((4 * 255 * RU8_ONE >> RU8_PRESHIFT) + 1)
  while M * 2.0 < lim and sh < 60: M, sh = M * 2.0, sh + 1
  return int(M + 0.5), sh

def roialign_u8(fmap:Tensor, rois:Tensor, scale:float, OH:int, OW:int, sr:int, z_in:int, z_out:int, mult:int, shift:int,
                divs_h:Tensor, divs_w:Tensor) -> Tensor:
  """ru8_roi for a batch of RoIs of one level: uint8 (H, W, C) map -> uint8 (R, OH, OW, C). Q14 bilinear weights
  (the 4th absorbs the rounding), exact int32 accumulation, zero point once per bin, one fixed-point requant."""
  H, W, C = fmap.shape
  ya, xa = _grid(rois, scale, H, W, OH, OW, sr, divs_h, divs_w)
  (yv, _, _, ly, hy), (xv, _, _, lx, hx) = ya, xa
  q = [((f * float(RU8_ONE)) + 0.5).cast(dtypes.int32) for f in (hy * hx, hy * lx, ly * hx)]
  w3 = RU8_ONE - q[0] - q[1] - q[2]
  w = [(w3 < 0).where(q[0] + w3, q[0]), q[1], q[2], (w3 < 0).where(0, w3)]
  p = _taps(fmap.reshape(H * W, C), W, ya, xa)
  valid = yv & xv                                                # [R, OH, sr, OW, sr]
  s = sum(p[t].cast(dtypes.int32) * w[t].unsqueeze(-1) for t in range(4))
  acc = valid.unsqueeze(-1).where(s, 0).sum((2, 4), dtype=dtypes.int32)       # [R, OH, OW, C]
  nvalid = valid.cast(dtypes.int32).sum((2, 4)).unsqueeze(-1)
  a = (acc - z_in * nvalid * RU8_ONE + (1 << (RU8_PRESHIFT - 1))) >> RU8_PRESHIFT
  a = ((a * mult + (1 << (shift - 1))) >> shift) + z_out
  return a.clip(0, 255).cast(dtypes.uint8)
