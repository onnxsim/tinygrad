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

D, KT, ST, HD, SEEN, SEEN_PAD, TILE, TH = 512, 16, 7, 32, 197, 224, 2048, 1024
SCALE = 0.17677669529663687
LOG2E = 1.4426950408889634
SCALE2 = SCALE * LOG2E
EPS = 1e-6
PAD = -65504.0  # the padded score columns, as mcc_block.h's ts6 biases them

_I, _J = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
IDX = 64 * (_I // 2) + 2 * _J + _I % 2


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
    is 2^7 = 128) and t = max(s - m7, -13). Returns (P over the 224 columns, p_self)."""
    m7 = np.maximum(s.max(-1, keepdims=True), s_self) - 7.0
    e = np.exp2(np.maximum(s - m7, -13.0))
    e_self = np.exp2(np.maximum(s_self - m7, -13.0))
    tot = e.sum(-1, keepdims=True) + e_self
    return e / tot, (e_self / tot)[:, 0]


def layernorm(x, gamma, beta, eps=EPS):
    """(rows, 512) half x -> (rows, 512) half. The statistics in float (as mbv_layernorm's qf32 sums),
    the apply in float too, and the result rounded back to half once (the kernel's qf32 -> hf narrow)."""
    from tinygrad import Tensor, dtypes

    f = x.float()
    mean = f.mean(-1, keepdim=True)
    var = (f * f).mean(-1, keepdim=True) - mean * mean  # one pass, as the kernel's sums do
    rstd = (var.maximum(0.0) + eps).rsqrt()
    return ((f - mean) * rstd * gamma.float() + beta.float()).cast(dtypes.half)


def softmax_base2_self_term(s, q, k, seen=SEEN, pad=PAD):
    """the kernel's base-2 softmax + self term, in the units it uses: s in log2 units (its K is
    pre-scaled by scale * log2(e)), the self score recomputed as q . k * scale * log2(e), the padded
    columns at -65504, the max taken over the whole row and the self score, e = 2^[s - (max - 7)] with
    the -13 floor, and P = e / (sum over the 224 columns + the self term).

    Returns (P (rows, 224) half, p_self (rows,) half). The exponentials run in half (the kernel's exp2 is
    an hf helper) and the sums / reciprocal in float (its qf32 sums and srecip). `pad` is the tile filler of
    the score columns past the first: the committed S tiles hold -65504 there (mcc_case.py), and that
    subtract in half overflows to -inf, which is the same 0 * 1/N the kernel's own -13 floor produces there.
    """
    from tinygrad import Tensor, dtypes

    q, k = q.float(), k.float()
    s_self = (q * k).sum(-1, keepdim=True) * SCALE2
    s = s.float()
    s = Tensor.where(Tensor.arange(s.shape[-1]).reshape(1, -1) < seen, s, Tensor(pad).float())
    # the row max is over the 224 columns of S (the padded ones hold -65504) and the self score
    m = s.max(-1, keepdim=True).maximum(s_self)
    # the exponentials in half, as the kernel's hf exp2 is, with the kernel's -13 floor
    e = (s - m).cast(dtypes.half).maximum(-13.0).cast(dtypes.half).exp2().contiguous()
    e_self = (s_self - m).cast(dtypes.half).maximum(-13.0).cast(dtypes.half).exp2()
    tot = e.float().sum(-1, keepdim=True) + e_self.float()
    p = (e.float() / tot).cast(dtypes.half)
    return p, (e_self / tot).cast(dtypes.half)


def gelu(x, half_intermediates=True):
    """the tanh form mbv_tile_gelu computes: x * 1 / (1 + 2^t) with t = -2.3022082 x (1 + 0.044715 x^2)
    clamped to [-12, 12] (2.3022082 = 1.5957691 * log2(e)), the -12 floor clamped back up to -12, and
    x < -4 -> 0. In half when half_intermediates, as the kernel's qf16 chain is; the result is half."""
    from tinygrad import Tensor, dtypes

    half = dtypes.half
    c = Tensor(np.float16(0.044715) if half_intermediates else np.float32(0.044715))
    k = Tensor(np.float16(-2.3022082) if half_intermediates else np.float32(-2.3022082))
    lo, hi = Tensor(np.float16(-12.0) if half_intermediates else np.float32(-12.0)), \
        Tensor(np.float16(12.0) if half_intermediates else np.float32(12.0))
    cut = Tensor(np.float16(-4.0) if half_intermediates else np.float32(-4.0))
    xh = x.cast(half)
    # the clamps are on t, before its exp2 (a float maximum of a half tensor would widen it)
    t = (((xh * xh * c + 1.0) * (xh * k)).cast(half).maximum(lo).minimum(hi))
    g = xh * (t.exp2() + 1.0).reciprocal()
    return Tensor.where(xh > cut, g, Tensor.zeros_like(g)).cast(half)
