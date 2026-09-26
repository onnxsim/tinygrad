"""tinygrad lowerings of mb_hvx.h's three HVX steps, on the HMX tile layout, for the mcc oracle.

  layernorm(x, gamma, beta)        -> (rows, 512) half, the kernel's per-row mean / rstd in float
  softmax_base2_self_term(s, q, k)  -> (rows, 224) half P, (rows,) half p_self
  gelu(x)                           -> the tanh form the kernel computes, in half

These are the *tinygrad side* of test_mcc.py: the same op the hand kernel computes, written as plain
Tensor code, with the casts in the places the kernel does them (LayerNorm's statistics and its apply in
float, the softmax's exponentials in half on the -13 clamp, GELU's intermediates in half). What is not
replicated is the tile layout itself -- the hand kernel's argument to each step is an HMX tile buffer,
where rows 2p and 2p+1 are interleaved in one vector, and the widening hf x hf -> qf32 multiply splits
exactly that way. tinygrad has no such layout, so the test gives each side the same (rows, cols) matrix in
plain row-major order and the comparison is of the *values*: tinygrad's result is packed back into tiles
(untiles) and compared against the phone's golden tile bytes. That is the claim the oracle makes -- the
lowering computes the same function, to fp16-level accuracy, on the data the kernel saw -- and it is why
the comparison is a tolerance and not bit equality (see test_mcc.py for the contract and the measured
numbers).
"""

import numpy as np
from tinygrad import Tensor, dtypes

D, KT, ST, HD, SEEN, SEEN_PAD, TILE, TH = 512, 16, 7, 32, 197, 224, 2048, 1024
SCALE = 0.17677669529663687
LOG2E = 1.4426950408889634
SCALE2 = SCALE * LOG2E
EPS = 1e-6
PAD = -65504.0  # the padded score columns, as mcc_block.h's ts6 biases them

_I, _J = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
IDX = 64 * (_I // 2) + 2 * _J + _I % 2


def T(x, dtype=None):
    """numpy array / numpy scalar / Tensor -> Tensor. The three lowerings take their arguments this way
    so a caller can hand them either the case's raw numpy tiles or another Tensor. Two things to know:
    tinygrad's `Tensor(other_tensor)` is a *cast*, not an adopt (it silently reinterprets the other's
    buffer as this dtypes' -- which is how the first version of these functions lost the dtypes of the
    qf16 chain), and a numpy scalar is not a data source at all (a 0-d array is)."""
    if isinstance(x, Tensor): return x
    if dtype is not None: return Tensor(np.array(x, dtype=np.float16 if dtype == dtypes.half else np.float32), dtype=dtype)
    return Tensor(x)


def tiles(a, stride=None):
    """(R, C) float -> fp16 tiles [R/32][C/32 or stride][1024] (mcc_case.py's, re-derived here so this
    module stands alone: the test imports it for the goldens' packing and the kernel copies for the
    packing the golden side uses)."""
    a = np.asarray(a)
    r, c = a.shape
    n = c // 32 if stride is None else stride
    t = a.astype(np.float16).reshape(r // 32, 32, c // 32, 32).transpose(0, 2, 1, 3)
    out = np.zeros((t.shape[0], n, TH), np.float16)
    out[:, : t.shape[1], IDX] = t
    return out


def untiles(t, r, c, stride=None):
    """[R/32][C/32 or stride][1024] fp16 tiles -> (R, C) float32, dropping the padding tiles of a wider
    stride (the block's S has 8 tiles per row block, 7 of them columns)."""
    t = np.asarray(t, np.float16)
    n = c // 32 if stride is None else stride
    return t.reshape(r // 32, n, TH)[:, : c // 32][:, :, IDX].transpose(0, 2, 1, 3).reshape(r, c).astype(np.float32)


# the float64 references (mcc_case.py computes and commits these; re-derived here so test_mcc.py can check
# the committed ones against the math and then quote them, without depending on the case generator)
def gelu_tanh(x):
    """the tanh form mbv_tile_gelu computes: x * sigmoid(1.5957691 (x + 0.044715 x^3)), 0 for x < -4.
    1.5957691 * log2(e) = 2.3022082, the constant the kernel splats, so the inner argument is written in
    log2 units here and exponentiated with exp2 -- the same function, the same constant (mb_hvx.h spells
    it -2.3022082f in the kernel and mcc_case.py's reference spells it 1.5957691 * log2(e))."""
    k = 2.3022082
    return np.where(x < -4.0, 0.0, x * 1.0 / (1.0 + np.exp2(-k * (x + 0.044715 * x**3))))


def layernorm_ref(x, gamma, beta, eps=EPS):
    """float64, the kernel's one-pass order: sums of x and x^2, then var = E[x^2] - mean^2"""
    mean = x.sum(-1, keepdims=True) / D
    var = (x * x).sum(-1, keepdims=True) / D - mean * mean
    return (x - mean) / np.sqrt(np.maximum(var, 0.0) + eps) * gamma + beta


def softmax_ref(s, s_self, seen=SEEN):
    """float64 reference of mbv_softmax with its clamps: m7 = max(row max, s_self) - 7 (so the largest e
    is 2^7 = 128) and t = max(s - m7, -13), over the *seen* scores only (mcc_case.py spells out why the
    27 padded columns are out of the max and the sum). Returns (P over the 224 columns, p_self)."""
    live = np.arange(s.shape[-1]) < seen
    seen_max = np.where(live, s, -np.inf).max(-1, keepdims=True)
    m7 = np.maximum(seen_max, s_self) - 7.0
    e = np.exp2(np.maximum(s - m7, -13.0))
    e_self = np.exp2(np.maximum(s_self - m7, -13.0))
    tot = np.where(live, e, 0.0).sum(-1, keepdims=True) + e_self
    return e / tot, (e_self / tot)[:, 0]


def layernorm(x, gamma, beta, eps=EPS):
    """(rows, 512) half x -> (rows, 512) half. The statistics in float (as mbv_layernorm's qf32 sums),
    the apply in float too, and the result rounded back to half once (the kernel's qf32 -> hf narrow)."""
    f = T(x, dtypes.half).float()
    mean = f.mean(-1, keepdim=True)
    var = (f * f).mean(-1, keepdim=True) - mean * mean  # one pass, as the kernel's sums do
    rstd = (var.maximum(0.0) + eps).rsqrt()
    return ((f - mean) * rstd * T(gamma, dtypes.half).float() + T(beta, dtypes.half).float()).cast(dtypes.half)


def softmax_base2_self_term(s, q, k, seen=SEEN, pad=PAD):
    """the kernel's base-2 softmax + self term, in the units it uses: s in log2 units (its K is
    pre-scaled by scale * log2(e)), the self score recomputed as q . k * scale * log2(e), the padded
    columns at -65504, the max taken over the whole row and the self score, e = 2^[s - (max - 7)] with
    the -13 floor, and P = e / (sum over the 224 columns + the self term).

    The padded columns (>= seen) are hidden exactly as mbv_softmax hides them: the kernel's row max and
    its qf32 sum only ever run over the first MB_ST * 32 == 224 lanes of a vector, which are the 197 seen
    scores plus the self term. A lowering that took the max (or the sum) over all 224 columns would see
    -65504 and get both wrong: 27 padded lanes pull the max down by 27 * 2^-13 == 0.0033, which is
    0.23% off every P in the row, and they add 27 * 2^-13 == 3.3e-3 to the sum. The clamp is applied
    over all 224 columns, because the kernel does apply it there: the padded ones land on the -13 floor
    and then contribute exp2(-13) == 1.2e-4 of the row's total, which is the kernel's own stand-in for
    0 on them, and the golden shows the same residue in its own P (test_mcc.py measures it).

    Returns (P (rows, 224) half, p_self (rows,) half). The exponentials run in half (the kernel's exp2 is
    an hf helper) and the sums / reciprocal in float (its qf32 sums and srecip)."""
    q, k = T(q, dtypes.half).float(), T(k, dtypes.half).float()
    s_self = (q * k).sum(-1, keepdim=True) * SCALE2
    s = T(s, dtypes.half).float()
    live = Tensor(np.arange(s.shape[-1]) < seen).reshape(1, -1)  # bool, the 197 seen columns
    # the row max is over the seen scores only, then over the self score
    m = Tensor.where(live, s, Tensor(-np.inf, dtype=s.dtype)).max(-1, keepdim=True).maximum(s_self) - 7.0
    # the exponentials in half, as the kernel's hf exp2 is, with the kernel's -13 floor
    e = (s - m).cast(dtypes.half).maximum(-13.0).cast(dtypes.half).exp2().contiguous()
    e_self = (s_self - m).cast(dtypes.half).maximum(-13.0).cast(dtypes.half).exp2()
    tot = Tensor.where(live, e.float(), Tensor(0.0, dtype=s.dtype)).sum(-1, keepdim=True) + e_self.float()
    p = (e.float() / tot).cast(dtypes.half)
    return p, (e_self / tot).cast(dtypes.half)


def gelu(x, half_intermediates=True):
    """the tanh form mbv_tile_gelu computes: x * 1 / (1 + 2^t) with t = -2.3022082 x (1 + 0.044715 x^2)
    clamped to [-12, 12] (2.3022082 = 1.5957691 * log2(e)), the -12 floor clamped back up to -12, and
    x < -4 -> 0. In half when half_intermediates, as the kernel's qf16 chain is; the result is half."""
    half = dtypes.half
    fl = half if half_intermediates else dtypes.float
    c = T(np.float16(0.044715) if half_intermediates else np.float32(0.044715), fl)
    k = T(np.float16(-2.3022082) if half_intermediates else np.float32(-2.3022082), fl)
    lo, hi = T(np.float16(-12.0) if half_intermediates else np.float32(-12.0), fl), \
        T(np.float16(12.0) if half_intermediates else np.float32(12.0), fl)
    cut = T(np.float16(-4.0) if half_intermediates else np.float32(-4.0), fl)
    xh = T(x, half).cast(half)
    # the clamps are on t, before its exp2 (a float maximum of a half tensor would widen it)
    t = (((xh * xh * c + 1.0) * (xh * k)).cast(half).maximum(lo).minimum(hi))
    g = xh * (t.exp2() + 1.0).reciprocal()
    return Tensor.where(xh > cut, g, Tensor.zeros_like(g)).cast(half)
