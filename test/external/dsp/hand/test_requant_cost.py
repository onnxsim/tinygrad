"""What the requantization of ONE conv family costs, with the same instrument on both arms.

The whole-graph A/B in test/external/dsp/hand/REQUANT_RECONCILE.md (HMX_RQ_STUB on the phone) prices the requant
arithmetic of every HMX kernel in the ResNet-18 QDQ graph at once.  That is the wrong unit for a per-family question:
one 3x3 family can be 0.1% of the graph, and a term that small is inside the run-to-run noise.

This test prices it per kernel instead.  It renders ONE conv (the 3x3 128->128 s1 family the HMX README bisects -
onnxsim's build.sh --conv 32 64 128 128 1), captures the generated kernel with its real arguments, and runs it on
hexagon-sim twice: once as rendered, once with HMX_RQ_STUB=1, which replaces only the ORT-exact arithmetic of
__hmx_rqwf with a plain cast and leaves the bias add, the flags, the packs, the stores and the exact redo alone.

**Each arm is a separate process, and it has to be.**  `HMX_RQ_STUB` is substituted into the C helper block when
`tinygrad.runtime.ops_dsp` is *imported* (`_HMX_ACC_HELPERS`), so flipping os.environ inside one process changes
nothing - both arms render byte-identical C and the "stub" measures nothing at all, which is exactly the failure mode
that made the first version of this file report 100948 pcycles for both arms.  helpers.getenv is also
functools.cache'd, a second trap on the same road.  This is the same reason the phone harness (/tmp/bisect.sh) builds
twice, and it is worth writing down: any HMX build flag that reaches the C through a module-level template is
import-time, not call-time.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC=clang-19 HEXAGON_TOOLS=<Tools> HMX_VTCM_KB=4096 \\
    python -m pytest -s -v test/external/dsp/hand/test_requant_cost.py

It prints the split and asserts that the stub arm came out faster, so a rewrite regression that made the arithmetic
free (or a flag that stopped reaching the kernel) would show up rather than silently report 0%.
"""
import json, os, subprocess, sys, unittest
import numpy as np
# the arm subprocesses run this file directly, so the repo root has to be importable (pytest's rootdir handling puts
# test/external/dsp on sys.path instead, and "test.external.dsp.hand" is not a package from there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from test.external.dsp.hand import hexsim  # noqa: E402

# 3x3 128->128 stride 1 on a 32x64 image: the layer the HMX README's 682k pcycle bisection and its 654 us phone number
# describe as "3x3 128->128, s1"
H, W, C, N, ST = 32, 64, 128, 128, 1
ZY, LO = 131, 0

def grid(H:int, W:int, st:int) -> tuple[int, int, int]:
  """(Wp, P64, L): row stride of the padded image, output grid pixels padded to 64, input pixels needed"""
  Wp = W + 2
  P64 = ((H // st) * Wp + 63) // 64 * 64
  return Wp, P64, st * (P64 - 1) + 2 * Wp + 3

def case(H:int, W:int, C:int, N:int, st:int, seed:int = 2) -> dict:
  """one QDQ conv: random uint8 activations, int8 weights, an int32 bias and a per-column scale. The scale spread is
  qdq_layer.py's: m puts most outputs inside [0, 255] with some saturating, and every 8th column on a power of two,
  so the .5-tie paths are exercised too (a real per-tensor QDQ model produces ordinary scales and never ties)."""
  rng = np.random.default_rng(seed)
  Wp, P64, L = grid(H, W, st)
  X = rng.integers(0, 256, (L, C), dtype=np.uint8)
  Wt = rng.integers(-128, 128, (3, 3, C, N), dtype=np.int8)
  b = rng.integers(-20000, 20000, N, dtype=np.int32)
  A = np.stack([X[dy * Wp + dx: dy * Wp + dx + st * (P64 - 1) + 1: st] for dy in range(3) for dx in range(3)], 1).reshape(P64, 9 * C)
  acc0 = A.astype(np.int64) @ Wt.reshape(9 * C, N).astype(np.int64) + b
  m = (rng.uniform(0.5, 2.0, N) * 50 / (np.abs(acc0).max(axis=0) + 1)).astype(np.float32)
  m[::8] = np.float32(2.0 ** -8)
  v = (acc0.astype(np.float32) * m).astype(np.float32)
  ref = (np.clip(np.rint(v), np.float32(LO - ZY), np.float32(255 - ZY)) + ZY).astype(np.uint8)
  return dict(X=X, Wt=Wt, b=b, m=m, ref=ref, acc=acc0, Wp=Wp, P64=P64)

def _arm(stub:int, int32:bool = False) -> dict:
  """render + run one arm in THIS process (the caller runs one arm per process); -> pcycles, off-ORT count"""
  import tempfile, pathlib
  from tinygrad import Tensor, dtypes
  from tinygrad.helpers import getenv
  from tinygrad.runtime import ops_dsp
  getenv.cache_clear()
  c = case(H, W, C, N, ST)
  def conv3x3_grid(x:Tensor, w:Tensor) -> Tensor:
    # tinygrad's lowering of a QDQ conv (tinygrad/nn/onnx_qdq.py) in grid form: the padded NHWC image flattened at
    # row stride Wp, the output on the same grid, dx and c merged into one reduce axis (hence TC_OPT=1)
    v = x.permute(1, 0)._pool((3,), 1, 1).permute(0, 2, 1)._pool((3,), ST, c["Wp"])
    v = v.shrink(((0, C), (0, 3), (0, c["P64"]), (0, 3))).permute(2, 3, 1, 0)
    return (v.reshape(c["P64"], 1, 3, 3, C).cast(dtypes.int32) * w.permute(3, 0, 1, 2).reshape(1, N, 3, 3, C).cast(dtypes.int32)).sum((2, 3, 4))
  acc = conv3x3_grid(Tensor(c["X"]), Tensor(c["Wt"])) + Tensor(c["b"])
  y = acc if int32 else ((acc.cast(dtypes.float32) * Tensor(c["m"])).round() + float(ZY)).clip(LO, 255).cast(dtypes.uint8)
  with hexsim.capture_dsp() as calls:
    y.realize()
    assert len(calls) == 1, f"expected one HMX kernel, got {len(calls)}"
    src, bufs = calls[0]
  assert "__hmx_i8_mac" in src, "not lowered to the int8 TensorCore: the HMX acc rewrite bailed"
  if not int32:
    want = f"#define HMX_RQ_STUB {stub}"
    assert want in src, f"HMX_RQ_STUB did not reach the kernel: {want!r} not in the rendered C (is it set in os.environ?)"
    assert "__hmx_rq4f(" in src, "the requantization was not fused into the kernel"
  with tempfile.TemporaryDirectory() as d: out, cyc = hexsim.run_captured(src, bufs, pathlib.Path(d))
  bad = 0
  if int32: bad = int((np.frombuffer(out, np.int32)[:c["P64"] * N].reshape(c["P64"], N) != c["acc"]).sum())
  else: bad = int((np.frombuffer(out, np.uint8)[:c["P64"] * N].reshape(c["P64"], N) != c["ref"]).sum())
  return dict(pcycles=cyc, bad=bad, stub=stub, int32=int32, hmx_acc=bool(ops_dsp.DSPRenderer.hmx_acc))

def _run_arm(stub:int, int32:bool = False) -> dict:
  # the arm subprocesses need the same environment the conftest builds for the parent. conftest.py sets
  # DEV=DSP / MOCKDSP / HVX_ARCH in its own process only, so passing os.environ through is not enough - the
  # explicit set is what makes the arm render an HMX kernel. Without DEV=DSP the arm captures zero HMX
  # calls and fails with "expected one HMX kernel, got 0", which is what CI reported.
  env = {k: v for k, v in os.environ.items() if k != "HMX_RQ_STUB"}
  env.update({"DEV": "DSP", "MOCKDSP": "1", "TC": "1", "TC_OPT": "1", "HVX_ARCH": "v69"})
  if not env.get("HEXAGON_TOOLS"): env["HEXAGON_TOOLS"] = str(hexsim.tools() or "")
  if stub: env["HMX_RQ_STUB"] = "1"
  cmd = [sys.executable, __file__, "--arm", str(stub)] + (["--int32"] if int32 else [])
  r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=os.getcwd())
  if r.returncode: raise RuntimeError(f"arm {stub} failed:\n{r.stdout}\n{r.stderr}")
  return json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(),
                     "needs the Hexagon toolchain (HEXAGON_TOOLS) + a Hexagon-capable clang for MOCKDSP")
@unittest.skip("a cost measurement, not a correctness gate: one arm is a 254M-instruction 3x3 conv on "
               "hexagon-sim and takes >20 min, and the suite runs four arms. Run it by hand - the module "
               "docstring has the command. CI's job here is correctness, and REQUANT_RECONCILE.md carries "
               "the measured numbers.")
class TestRequantCost(unittest.TestCase):
  def test_requant_cost_3x3_128_128_s1(self):
    base = _run_arm(0)
    stub = _run_arm(1)
    self.assertEqual(base["bad"], 0, "the rendered kernel disagrees with ORT's formula")
    d = base["pcycles"] - stub["pcycles"]
    print(f"\n3x3 {H}x{W}x{C}->{N} s{ST} (grid {grid(H, W, ST)[1]} px, K = 9 x {C}): {base['pcycles']} pcycles with the "
          f"requantization, {stub['pcycles']} with its arithmetic stubbed -> the requantization arithmetic is "
          f"{d} pcycles, {100 * d / base['pcycles']:.1f}% of the family, "
          f"{d / (grid(H, W, ST)[1] * N):.2f} cycles/output (stub arm: {stub['bad']} off ORT, as expected)")
    self.assertGreater(d, 0, "stubbing the requantization arithmetic did not make the kernel faster")

  def test_family_cost_without_the_whole_epilogue(self):
    """HMX_RQ_STUB is only an honest instrument when the fast path actually runs. With the qdq_layer.py-style scales
    (every 8th column a power of two, zy 131) every group sets the tie flag and the whole family is redone by the
    out-of-line exact path __hmx_rq4x, which the stub does not touch - so the stubbed arm then measures the exact
    path twice over and the "arithmetic" delta is exactly 0. Read that 0 as "the stub never ran", not as "the
    arithmetic is free".

    This is the HMX README's 682k family, and at 682k for 2112 x 128 = 270 336 outputs the README's 1.2 cycles/output
    for the requantization is arithmetically impossible (it would be 120M pcycles), so the 682k is not this layer with
    this data either - see REQUANT_RECONCILE.md. Emitting the accumulator instead of the requantized bytes (the
    --int32 form, which drops the whole epilogue: the packs and the stores as well) is the reading of "3x3 128->128 s1
    minus its requantization" that does not depend on which of the two requantization paths the data happens to take."""
    i32 = _run_arm(0, int32=True)
    full = _run_arm(0)
    d = full["pcycles"] - i32["pcycles"]
    print(f"\n3x3 {H}x{W}x{C}->{N} s{ST}: {full['pcycles']} pcycles with the whole epilogue (requantization + packs + "
          f"stores), {i32['pcycles']} emitting the int32 accumulator instead -> the epilogue is {d} pcycles, "
          f"{100 * d / full['pcycles']:.1f}% of the family, {d / (grid(H, W, ST)[1] * N):.2f} cycles/output")
    self.assertGreater(d, 0, "the epilogue did not cost anything: the conv was not lowered to the HMX path")

if __name__ == "__main__":
  if "--arm" in sys.argv:
    print(json.dumps(_arm(int(sys.argv[sys.argv.index("--arm") + 1]), "--int32" in sys.argv)))
  else: unittest.main()
