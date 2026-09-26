"""Fixed tile-layout inputs + float64 references for mb_hvx.h's three HVX steps, and the phone-captured
goldens test_mcc.py compares against.

  python mcc_case.py export --out <dir> [--rt 4] [--gelu-tiles 4] [--x0 <ref.py export dir>/x0.bin]

Host-side and offline. `--out` gets

| file | what |
|---|---|
| `case.json` | the case: rt (row blocks), nt (GELU tiles), the seed, the source commit, how the goldens were captured |
| `x.bin` | LayerNorm's input, rt x 16 HMX tiles of fp16 (element (r, c) at halfword IDX(r%32, c%32)) |
| `ln.bin` | gamma then beta, MB_D fp16 each (mcc_block.h's ln1 / ln2 layout) |
| `s.bin` | the softmax's scores, rt x 8 tiles (stride 8, 7 used): 224 columns of log2-unit scores with the 197..223 padding at -65504, exactly what the block's S tile 6 holds after the ts6 column table |
| `q.bin`, `k.bin` | the head's q and k tiles, rt tiles each (in the block: the qkv GEMM's column blocks h and 16+h) |
| `g.bin` | GELU's input, nt fp16 tiles (in the block: the fc1 output) |
| `ref_h.bin` | LayerNorm's float64 reference, fp32, rt*32 x 512 |
| `ref_s.bin` | the softmax's float64 reference P, fp32, rt*32 x 224 |
| `ref_pv.bin` | the float64 reference p_self, fp32, rt*32 |
| `ref_gelu.bin` | GELU's float64 reference, fp32, nt*32 x 32 |
| `gold_h.bin` | the phone's LayerNorm output tiles (captured by mcc_golden_client) |
| `gold_spv.bin` | the phone's softmax output P, rt x 8 tiles |
| `gold_pv.bin` | the phone's p_self, rt tiles (the interleaved pair the softmax writes) |
| `gold_gelu.bin` | the phone's GELU output tiles (GELU is in place, so the skel copies the result out) |
| `gold_times.txt` | the skel's timings for that capture |

The inputs are real data where there is real data: x is the first `rt * 32` rows of a ref.py export's x0
(the fp16 query chunk after the positional embedding) and LayerNorm's gamma / beta and the softmax's q /
k / S are derived from those same rows (a LayerNorm output is the block's input to Wqkv, so the head's
q, k and the scores are in their real ranges). GELU's input is those rows through a fixed affine, with
row 0 of every tile an explicit sweep of the kernel's edges.

A golden is a *step* output, not a graph: mb_hvx.h's steps are what they are, and the oracle's claim is
that tinygrad's lowering of the same step, on the same bytes, lands inside the family's fp16-level
contract. test_mcc.py reads this dir and says which comparisons that buys.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# the kernel's own constants, kept in step with mcc_block.h (there is no other way to import them)
D, KT, ST, HD, SEEN, SEEN_PAD, TILE, TH = 512, 16, 7, 32, 197, 224, 2048, 1024
SCALE = 0.17677669529663687
LOG2E = 1.4426950408889634
SCALE2 = SCALE * LOG2E  # MB_SCALE2: the block's K is pre-scaled by this, so its scores are log2 units
EPS = 1e-6
PAD16 = np.float16(-65504.0)  # what mcc_block.h's ts6 column table biases the padded score columns to

# IDX(i, j) = 64*(i/2) + 2*j + i%2: halfword of element (i, j) in a 32x32 HMX tile
_I, _J = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
IDX = 64 * (_I // 2) + 2 * _J + _I % 2


def mb_idx(i, j):
    """mcc_block.h's scalar element index in a tile, vectorized (i, j may be arrays)"""
    return 64 * (np.asarray(i) // 2) + 2 * np.asarray(j) + np.asarray(i) % 2


def tiles(a, stride=None):
    """(R, C) -> fp16 tiles, element (i, j) of tile at halfword IDX(i, j): [R/32][C/32 or stride][1024].
    `stride` pads the tile count per row block (the block's S has 8 tiles per row block, 7 used)."""
    r, c = a.shape
    n = c // 32 if stride is None else stride
    t = a.astype(np.float16).reshape(r // 32, 32, c // 32, 32).transpose(0, 2, 1, 3)
    out = np.zeros((t.shape[0], n, TH), np.float16)
    out[:, : t.shape[1], IDX] = t
    return out


def untiles(t, r, c):
    """[R/32][C/32][1024] fp16 tiles -> (R, c)"""
    return t.reshape(r // 32, c // 32, TH)[:, :, IDX].transpose(0, 2, 1, 3).reshape(r, c)


def f16(a):
    return np.ascontiguousarray(a.astype(np.float16))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gelu_tanh(x):
    """the tanh form mbv_tile_gelu computes, x * sigmoid(1.5957691 (x + 0.044715 x^3)), and 0 for x < -4.
    1.5957691 = 2 * sqrt(2/pi), so its inner argument is x + 0.044715 x^3 scaled like the hand kernel's; the
    kernel splats 1.5957691 * log2(e) = 2.3022082 and exponentiates in base 2 (mbv_hexp2), which is the
    same function -- the constant is the one both sides spell, to 8 digits."""
    return np.where(x < -4.0, 0.0, x * sigmoid(1.5957691 * (x + 0.044715 * x**3)))


def layernorm_ref(x, gamma, beta, eps=EPS):
    """float64, the kernel's one-pass order: sums of x and x^2, then var = E[x^2] - mean^2"""
    mean = x.sum(-1, keepdims=True) / D
    var = (x * x).sum(-1, keepdims=True) / D - mean * mean
    return (x - mean) / np.sqrt(np.maximum(var, 0.0) + eps) * gamma + beta


def softmax_ref(s, s_self, seen=SEEN):
    """float64 reference of mbv_softmax, including its two clamps: m7 = max(row max, s_self) - 7 (so the
    largest e is 2^7 = 128) and t = max(s - m7, -13) (the floor stands in for 0: the padded columns and
    anything that small). Returns (P over the 224 columns, p_self)."""
    m7 = np.maximum(s.max(-1, keepdims=True), s_self) - 7.0
    e = np.exp2(np.maximum(s - m7, -13.0))
    e_self = np.exp2(np.maximum(s_self - m7, -13.0))
    tot = e.sum(-1, keepdims=True) + e_self
    return e / tot, (e_self / tot)[:, 0]


def git_rev(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          check=False).stdout.strip() or None


def cmd_export(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rt, nt = a.rt, a.gelu_tiles
    rows = rt * 32
    if a.x0 and Path(a.x0).exists():
        x0 = np.fromfile(a.x0, np.float16).reshape(-1, D)[:rows]
        source = str(a.x0)
    else:
        rng = np.random.default_rng(a.seed)
        x0 = (rng.standard_normal((rows, D)) * 0.5).astype(np.float16)
        source = f"synthetic, seed {a.seed}"
    # the kernel sees fp16: every reference below is computed from the fp16 values, not the pre-rounding ones
    x = x0.astype(np.float64)

    # LayerNorm: gamma / beta as a trained LayerNorm has them (near 1 / near 0)
    rng = np.random.default_rng(a.seed + 1)
    gamma, beta = 1.0 + 0.1 * rng.standard_normal(D), 0.05 * rng.standard_normal(D)
    ln16 = f16(np.stack([gamma, beta]))
    h_ref = layernorm_ref(x0.astype(np.float64), ln16[0].astype(np.float64), ln16[1].astype(np.float64))

    # softmax, head 0: q is head 0's qkv column block, k the seen tokens' K (pre-scaled by scale*log2e in
    # the block, so the scores come out of HMX in log2 units). K's 197 tokens are this chunk's rows tiled,
    # which keeps them in the real range without needing the encoder cache.
    q16, k16 = f16(h_ref[:, :HD]), np.zeros((SEEN, HD), np.float16)
    k16[:] = f16(np.resize(h_ref[:, HD : 2 * HD], (SEEN, HD)))
    q, k = q16.astype(np.float64), k16.astype(np.float64)
    s_self = (q * k[:rows]).sum(-1, keepdims=True) * SCALE2
    # The scores the softmax sees are HMX's fp16 *output*: the GEMM accumulates 32 products in its fp16
    # accumulator, so every column is already a rounded intermediate. The float64 reference has to be built
    # on those bytes (the kernel only ever sees them). The first version of this case built it on the exact
    # fp32 product rounded once at the end, which differs from the tiles it then shipped by ~0.5 ulp, and
    # from the phone by more; test_mcc.py's own check of ref_s against the tiles is what caught it. */
    s = np.full((rows, SEEN_PAD), float(PAD16), np.float64)
    s[:, :SEEN] = (q @ k.T) * SCALE2
    s16 = f16(s)
    s_ref, pself_ref = softmax_ref(s16.astype(np.float64), (q * k[:rows]).sum(-1, keepdims=True) * SCALE2)

    # GELU: the fc1 output's range, with two row pairs of every tile an explicit sweep of the kernel's
    # edges (the x < -4 cutoff, the exp2 clamp at +-12 and its saturation, the fp16 top). mbv_tile_gelu
    # takes a tile pointer and sweeps its 16 vectors, so a tile is the unit: nt tiles of 32x32.
    g = f16(np.resize(x0.astype(np.float64) * 1.5 + 0.3, (nt, TH)))
    # 32 values, laid down in the tile's (row pair, column) halfword order, over row pair 0 of each tile
    sweep = f16(np.array([-4.5, -4.0, -3.999, -3.0, -2.0, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 12.0, 12.5, 20.0, 64.0, 1000.0, 1e4, 65504.0]))
    for i in range(nt):
        g[i, mb_idx(0, np.arange(32))] = sweep
    g_ref = gelu_tanh(g.astype(np.float64))

    (out / "x.bin").write_bytes(f16(tiles(x0)).tobytes())
    (out / "ln.bin").write_bytes(ln16.tobytes())
    (out / "s.bin").write_bytes(f16(tiles(s16, stride=8)).tobytes())
    (out / "q.bin").write_bytes(f16(tiles(q16)).tobytes())
    (out / "k.bin").write_bytes(f16(tiles(k16[:rows])).tobytes())
    (out / "g.bin").write_bytes(g.tobytes())  # already tiles: flat, nt x 1024 halfwords
    (out / "ref_h.bin").write_bytes(h_ref.astype(np.float32).tobytes())
    (out / "ref_s.bin").write_bytes(s_ref.astype(np.float32).tobytes())
    (out / "ref_pv.bin").write_bytes(pself_ref.astype(np.float32).tobytes())
    (out / "ref_gelu.bin").write_bytes(untiles(g, nt * 32, 32).astype(np.float32).ravel().tobytes())
    (out / "case.json").write_text(json.dumps({
        "rt": rt, "nt": nt, "rows": rows, "seen": SEEN, "seed": a.seed, "x0": source,
        "source_commit": git_rev(HERE.parents[4]), "tile_bytes": TILE, "scale2": SCALE2, "eps": EPS,
        "goldens": None,  # filled in by test_mcc.py's README / the capture commit message
    }, indent=2) + "\n")
    sizes = {p.name: p.stat().st_size for p in sorted(out.glob("*.bin"))}
    print(f"-> {out}: rt {rt} (Q {rows}), nt {nt}, seen {SEEN}, {source}")
    for n, s_ in sizes.items():
        print(f"   {n:16s} {s_:8d} B")
    print(f"   total {sum(sizes.values())} B")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--rt", type=int, default=4)
    e.add_argument("--gelu-tiles", type=int, default=4)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--x0", help="a ref.py export dir's x0.bin: real query rows instead of synthetic")
    a = ap.parse_args()
    {"export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
