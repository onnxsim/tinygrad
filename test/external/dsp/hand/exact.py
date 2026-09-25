"""Host-side helpers that turn ORT's exactness contracts into forms tinygrad can compute exactly.

tinygrad's float division is x * (1/y) (mixin/elementwise.py `div`), and its symbolic rewrites treat float algebra
as real algebra, so an op that ORT defines through an IEEE division -- QuantizeLinear's round(x / scale) -- can't be
written as `x / scale` and stay bit-exact. For a *constant* scale it doesn't need to be: QuantizeLinear onto uint8 is
a monotone step function of x with 256 levels, so it equals the number of per-level thresholds x reaches. The
thresholds are found here, once per (scale, zero point), by bisection over float32 bit patterns using NumPy's IEEE
float32 division -- the same arithmetic ORT (and the hand kernels, built -ffp-contract=off) do per element.
"""
import numpy as np

def _key(x:np.ndarray) -> np.ndarray:
  """float32 -> int64, monotone over the finite floats (-0.0 and +0.0 map to the same key)."""
  u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.int64)
  return np.where(u >= 0x80000000, 0x80000000 - u, u)

def _unkey(k:np.ndarray) -> np.ndarray:
  k = np.asarray(k, np.int64)
  return np.where(k < 0, 0x80000000 - k, k).astype(np.uint32).view(np.float32)

def rint(x:np.ndarray) -> np.ndarray: return np.rint(x)  # NumPy's rint is round-half-to-even, like the kernels' pd_rint

def quantize_u8(x, scale:float, zero:int) -> np.ndarray:
  """ORT QuantizeLinear to uint8 in float32: clamp(rint(x / scale) + zero, 0, 255), IEEE division."""
  with np.errstate(over="ignore"):
    q = rint(np.float32(x) / np.float32(scale)) + np.float32(zero)
  return np.clip(q, 0, 255).astype(np.float32)

def qdq(x, scale:float, zero:int) -> np.ndarray:
  return ((quantize_u8(x, scale, zero) - np.float32(zero)) * np.float32(scale)).astype(np.float32)

def quantize_thresholds(scale:float, zero:int) -> np.ndarray:
  """T[0..254] (float32) with quantize_u8(x) == count(x >= T) for every finite float32 x. T[j] is the smallest
  float reaching level j+1; levels no finite float reaches get +inf (and level 0 always holds)."""
  lo = np.full(255, _key(np.float32(-np.finfo(np.float32).max)), np.int64)
  hi = np.full(255, _key(np.float32(np.finfo(np.float32).max)), np.int64)
  level = np.arange(1, 256, dtype=np.float32)
  reach_hi = quantize_u8(_unkey(hi), scale, zero) >= level
  # invariant: f(lo) < level (or lo is the minimum), f(hi) >= level
  for _ in range(34):
    mid = (lo + hi) // 2
    ok = quantize_u8(_unkey(mid), scale, zero) >= level
    hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
  t = _unkey(hi)
  # the minimum float itself may already reach the level
  t = np.where(quantize_u8(_unkey(lo), scale, zero) >= level, _unkey(lo), t)
  return np.where(reach_hi, t, np.float32(np.inf)).astype(np.float32)
