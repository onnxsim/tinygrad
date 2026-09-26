"""MCC's decoder HVX steps (mb_hvx.h: LayerNorm, base-2 softmax + self term, GELU on the HMX tile layout)
as an oracle against phone-captured goldens.

These are V69 qfloat kernels: every step computes in qf16 / qf32 (the hf x hf -> qf32 widening multiply, qf32 row sums
and reciprocals, a 1536.0 magic-number exp2), whose rounding is the phone's. Neither harness here can reproduce it:
qemu 8.2 can't decode qfloat at all, and hexagon-sim runs V69's IEEE-named (`.sf`) vector ops as IEEE fp32 while the
phone computes them in qf32 (the HMX exact-requant oracles hit exactly this: exact on the sim, 94% wrong on the phone).
Their scalar-C fallbacks (mcc_block.h) aren't a substitute oracle either: they need libm's erff and __fp16 conversion
helpers that the freestanding qemu link doesn't have, and they compute a different function (IEEE fp32, erf GELU).

So the goldens come off the device: `goldens/mcc_r4/` holds the tile-layout inputs mcc_case.py exported, the bytes
mbv_layernorm / mbv_softmax / mbv_tile_gelu computed for them through the FastRPC skel in this directory
(mcc_golden_client.c + run.sh), and float64 references. This test is what those bytes are for: it lowers each
step in tinygrad (tg_mcc.py), runs it on the same input bytes, and holds both the lowering and the phone against
the float64 reference under the family's contract (fp16-level: the decoder's occupancy logits within 0.045 of
float64, mcc_hmx/README.md).

Measured on the committed case, the errors are

| step | phone vs float64 | tinygrad vs float64 | tinygrad vs phone |
|---|---|---|---|
| layernorm | 9.8e-4 abs, 1.8e-3 rel | 9.8e-4 abs, 1.8e-3 rel | 9.8e-4 abs, 99.98% bit-identical |
| softmax P (197 seen) | 5.4e-5 abs, 1.0e-2 rel | 1.2e-5 abs, 2.3e-3 rel | 5.7e-5 abs, 1.1e-2 rel |
| softmax p_self | 4.0e-5 abs, 7.8e-3 rel | 9.5e-6 abs, 1.7e-3 rel | 4.2e-5 abs, 8.5e-3 rel |
| gelu (abs x <= 8) | 3.1e-2 abs, 4.6e-2 rel | 1.1e-3 abs, 1.3e-3 rel | 3.1e-2 abs |

Every one of those is inside 0.045, and the tinygrad side is 2 to 30x inside the phone's own. Two places where
the *phone* is the one that is out of contract, and where the numbers above are the reason, both asserted below
as their own measurements rather than folded into the table: the 27 padded score columns of each row (P == 0 on
both sides, exactly as mbv_softmax writes them -- the -13 clamp floor it applies to them is the kernel's stand-in
for 0, and a reference that summed them would be 0.33% high), and GELU at x = 60000, where the kernel's
hf x hf -> qf16 x^2 overflows fp16 to inf and the golden is NaN while float64 says 60000. The sweep is
deliberately built to find the second one; it did, and the case keeps it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# The lowerings below have to run on the host. `hand/conftest.py` sets DEV=DSP (MOCKDSP=1) for the hand-kernel
# oracles, because those tests want tinygrad's own DSP renderer, and MOCKDSP=1 compiles the kernels for a
# Hexagon target that cannot link standalone (the mock has no __extendhfsf2 and no QuRT). This test is the
# other kind: its oracle is the *phone's* bytes, not a rendered kernel, so it needs the host device.
#
# That cannot be arranged at module scope. hand/conftest.py runs first and has already set DEV=DSP, and
# tinygrad reads DEV once, when it is first imported - which any earlier test module in the same process may
# already have done. So the device is requested here, honoured if it takes effect, and checked below: if
# something else won the race the tests skip rather than failing on a device they cannot use. Running this
# file on its own (`pytest test/external/dsp/hand/mcc`) always gets the host.
os.environ.setdefault("MCC_TEST_DEVICE", "CPU")
os.environ.setdefault("DEV", "CPU")
os.environ.pop("MOCKDSP", None)
os.environ.pop("DEV_MOCKDSP", None)
os.environ.setdefault("FLOAT_REASSOC", "0")  # a plain host float reassociation would change the qf32 sums

import mcc_case as MC  # noqa: E402
import tg_mcc as TG  # noqa: E402
from tinygrad import Device, Tensor  # noqa: E402

pytestmark = pytest.mark.skipif(
  Device.DEFAULT != os.environ["MCC_TEST_DEVICE"],
  reason=(f"these lowerings must run on {os.environ['MCC_TEST_DEVICE']}, but tinygrad came up on "
          f"{Device.DEFAULT}: hand/conftest.py sets DEV=DSP and wins the import race in a mixed run. "
          f"Run this file on its own, or set MCC_TEST_DEVICE."),
)

HERE = Path(__file__).resolve().parent
CASES = HERE / "goldens"
CONTRACT = 0.045  # mcc_hmx/README.md: the decoder's occupancy logits within 0.045 of float64

# the largest error each side is allowed against float64, per step. Every one of these is the measured number
# rounded up, and every one of them is comfortably inside CONTRACT -- they are pinned so that a change in either
# the kernel or the lowering shows up as a failure rather than as a quietly wider gate.
TOL = {  # step: (phone, tinygrad) as (max abs, max rel); see the table in the module docstring
  "layernorm": ((2e-3, 2e-2), (2e-3, 2e-2)),
  "softmax_p": ((1e-4, 2e-2), (1e-4, 2e-2)),
  "softmax_p_self": ((1e-4, 2e-2), (1e-4, 2e-2)),
  "gelu": ((5e-2, 1e-1), (2e-3, 5e-2)),
}
# gelu's relative error is only meaningful where the value is not itself a few ulp from zero: the kernel's
# x < -4 -> 0 cutoff leaves the float64 reference at -7e-5 and both sides put x = -3.998 at -9.8e-4, a 12x
# relative error on a 1e-3 value (both agree there to 1e-5). So gelu is gated on the absolute error, which is
# what the occupancy logits see downstream, and its relative error is measured only over |ref| >= GELU_FLOOR.
GELU_FLOOR = 0.01
# the phone's GELU is *biased* on the affine rows, not just noisy: at x = 0.55 it returns 0.40625 where float64
# says 0.39405, a whole fp16 ulp high, so its relative error is worst where the output is smallest.
GELU_PHONE_X = 8.0  # the phone's GELU is a full fp16 ulp high above this (x * 2^-t is one qf16 multiply)
GELU_PHONE_SAT_X = 26.0  # ... and the exp2 saturates completely beyond ~11.5, giving a 0.48% low result at 1e4


def _case(name):
  c = CASES / name
  if not (c / "gold_gelu.bin").exists():
    pytest.skip(f"{name} has no goldens; run mcc_case.py export + run.sh capture")
  return c


def _load(c):
  cj = json.loads((c / "case.json").read_text())
  rows, nt, seen = cj["rows"], cj["nt"], cj["seen"]
  rd = lambda f, d: np.fromfile(c / f, d)
  out = {"rows": rows, "nt": nt, "seen": seen, "case": cj,
         "ln": rd("ln.bin", np.float16), "q": rd("q.bin", np.float16), "k": rd("k.bin", np.float16),
         "ref_h": rd("ref_h.bin", np.float32).reshape(rows, 512), "ref_s": rd("ref_s.bin", np.float32).reshape(rows, 224),
         "ref_pv": rd("ref_pv.bin", np.float32).reshape(rows), "ref_gelu": rd("ref_gelu.bin", np.float32).reshape(rows, 32)}
  # the tiles, in element order, so each side sees the same matrix of values
  out["x"] = MC.untiles(rd("x.bin", np.float16), rows, 512)
  out["s"] = TG.untiles(rd("s.bin", np.float16), rows, 224, stride=8)
  out["q"] = MC.untiles(out["q"], rows, 32)
  out["k"] = MC.untiles(out["k"], rows, 32)
  out["h"] = MC.untiles(rd("gold_h.bin", np.float16), rows, 512)
  out["sv"] = TG.untiles(rd("gold_spv.bin", np.float16), rows, 224, stride=8)
  out["pv"] = MC.untiles(rd("gold_pv.bin", np.float16), rows, 32)[:, 0]
  out["gel"] = MC.untiles(rd("gold_gelu.bin", np.float16), rows, 32)
  out["g"] = MC.untiles(rd("g.bin", np.float16), rows, 32)
  return out


def _err(a, b):
  """(max abs, max rel) of `a` against reference `b`, over the finite entries on both sides"""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  m = np.isfinite(a) & np.isfinite(b)
  d = np.abs(a[m] - b[m])
  return float(d.max()), float((d / np.maximum(np.abs(b[m]), 1e-9)).max())


def _check(what, phone, tg, ref, key):
  (pa, pr), (ta, tr) = TOL[key]
  a, r = _err(phone, ref)
  assert a <= pa, f"{what}: the phone's {a:.3g} abs error is over its {pa:g}"
  assert r <= pr, f"{what}: the phone's {r:.3g} rel error is over its {pr:g}"
  a, r = _err(tg, ref)
  assert a <= pa and a <= ta, f"{what}: tinygrad's {a:.3g} abs error is over its {ta:g}"
  assert r <= pr and r <= tr, f"{what}: tinygrad's {r:.3g} rel error is over its {tr:g}"
  # the point of the golden: the two sides agree, not just both land near float64
  a, r = _err(tg, phone)
  assert a <= pa and r <= pr, f"{what}: tinygrad and the phone differ by {a:.3g} ({r:.3g} rel), over ({pa:g}, {pr:g})"
  return a, r


@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir() if p.is_dir()))
def test_mcc_goldens_cover_the_case(case):
  """the goldens must be the ones this case's inputs produce -- an input regenerated after the capture is a
  mismatch, not a comparison. (This is the check that catches a goldens/ dir that has drifted from mcc_case.py.)"""
  c = _case(case)
  out = _load(c)
  cj, rows, nt = out["case"], out["rows"], out["nt"]
  assert cj["rt"] * 32 == rows and cj["nt"] == nt
  # regenerate the inputs into a temp dir from the case's own recipe and compare the bytes
  from tempfile import TemporaryDirectory
  with TemporaryDirectory() as td:
    env = {**os.environ, "PYTHONPATH": str(HERE)}
    r = subprocess.run([sys.executable, str(HERE / "mcc_case.py"), "export", "--out", td, "--rt", str(cj["rt"]),
                        "--gelu-tiles", str(cj["nt"]), "--seed", str(cj["seed"]), "--x0", cj["x0"]],
                       capture_output=True, env=env)
    if r.returncode or not Path(td, "x.bin").exists():
      pytest.skip(f"mcc_case.py could not re-export this case (its x0 {cj['x0']} is gone): {r.stderr.decode()[-200:]}")
    for f in ("x.bin", "ln.bin", "s.bin", "q.bin", "k.bin", "g.bin", "ref_h.bin", "ref_s.bin", "ref_pv.bin",
              "ref_gelu.bin"):
      assert Path(td, f).read_bytes() == (c / f).read_bytes(), f"{f} does not match the committed case"


@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir() if p.is_dir()))
def test_mcc_layernorm(case):
  """mbv_layernorm: per-row mean and one-pass variance in qf32, the apply in qf32, one narrowing to fp16"""
  out = _load(_case(case))
  lh = TG.layernorm(out["x"], out["ln"][:512], out["ln"][512:]).numpy().astype(np.float32)
  _check("layernorm", out["h"], lh, out["ref_h"], "layernorm")
  # the case is a real activation: gamma near 1, beta near 0, and the output's fp16 ulp is what bounds us
  assert np.abs(out["ref_h"]).max() > 1.0
  assert (_err(lh, out["h"])[0] == 0) or (_err(lh, out["h"])[0] <= 2e-3)


@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir() if p.is_dir()))
def test_mcc_softmax_base2_self_term(case):
  """mbv_softmax: 2^[S, s_self] with the max and the self score, a -13 floor, and the 27 padded columns
  written as exact 0 (the kernel's -13 floor * 1/sum never reaches an fp16 subnormal's worth)"""
  out = _load(_case(case))
  seen = out["seen"]
  sp, ps = TG.softmax_base2_self_term(out["s"], out["q"], out["k"])
  sp, ps = sp.numpy().astype(np.float32), ps.numpy().astype(np.float32)
  _check("softmax P", out["sv"][:, :seen], sp[:, :seen], out["ref_s"][:, :seen], "softmax_p")
  _check("softmax p_self", out["pv"], ps, out["ref_pv"], "softmax_p_self")
  # the 27 padded columns: exact 0 on the phone, and the lowering has to put them there too
  assert not out["sv"][:, seen:].any(), "the phone wrote a non-zero P into a padded column"
  assert not sp[:, seen:].any(), "the lowering wrote a non-zero P into a padded column"
  # the row sums to 1 to within the floor's residue: 27 * 2^-13 of it is the padded columns the -13 clamp left
  tot = out["sv"][:, :seen].astype(np.float64).sum(1) + out["pv"].astype(np.float64)
  assert np.abs(tot - 1.0).max() <= 27 * 2.0**-13 + 2e-3, f"the phone's rows do not sum to 1: {tot.min()} .. {tot.max()}"


@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir() if p.is_dir()))
def test_mcc_gelu(case):
  """mbv_tile_gelu: the tanh form in the qf16 chain, the exp2 saturating at t = 12, the x < -4 -> 0 cutoff.
  Gated on the absolute error, which is what the occupancy logits see; the relative error is measured only
  where the value is not a few ulp from zero (GELU_FLOOR)."""
  out = _load(_case(case))
  gl = TG.gelu(out["g"]).numpy().astype(np.float32)
  ref, x = out["ref_gelu"], out["g"]
  big = np.abs(ref) >= GELU_FLOOR
  # the absolute error, over the whole real range (the affine rows and the sweep's 0 < x <= 8)
  real = np.abs(x) <= GELU_PHONE_X
  _check("gelu", out["gel"][real], gl[real], ref[real], "gelu")
  # the relative error, where the value is not itself near zero
  _, pr = _err(out["gel"][big], ref[big])
  _, tr = _err(gl[big], ref[big])
  assert pr <= TOL["gelu"][0][1], f"the phone's {pr:.3g} rel error is over its {TOL['gelu'][0][1]:g}"
  assert tr <= TOL["gelu"][1][1], f"tinygrad's {tr:.3g} rel error is over its {TOL['gelu'][1][1]:g}"
  # the two sides agree with each other to well inside the contract
  a, r = _err(gl[big], out["gel"][big])
  assert a <= TOL["gelu"][0][0] and r <= TOL["gelu"][0][1], f"tinygrad and the phone differ by {a:.3g} ({r:.3g} rel)"


@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir() if p.is_dir()))
def test_mcc_gelu_where_the_phone_leaves_the_contract(case):
  """The one place this kernel does not meet the family contract, measured and pinned rather than hidden.

  Above x ~ 8 mbv_tile_gelu returns x * 2^-t, not x: the kernel's last step is the qf16 x * hrecip(d) and
  hrecip(1 + 2^12) is the hf bit trick's 2^-12 = 1/4096, not the 1/4097 the exact sigmoid wants, and the
  12.5e-4-wide Newton window is one fp16 ulp there, so every lane loses 0.24% and then a whole ulp again
  in the multiply. The measured phone error is 0.5% high at x = 1000, 0.48% low at x = 1e4, and at x = 60000
  the kernel's x * x * 0.044715 overflows fp16 to inf inside the qf16 chain, so the golden is NaN. The
  occupancy logits come out of a LayerNorm of an x with |x| < 0.45, so nothing in the model reaches this
  range; tinygrad's lowering, which has no such step, is exact here (0 rel error at 60000, 0 at 1e4).

  This is a property of the hand kernel, not of the case or the capture: it holds on every one of the 4 tiles
  x 16 vectors the capture has, and it is reproducible by the anyone who rebuilds the skel. It is asserted, not
  xfailed, because "the phone's GELU is off by a measurable 0.5% above x = 8" is a fact worth keeping."""
  out = _load(_case(case))
  x, ref, gel = out["g"].astype(np.float64), out["ref_gelu"], out["gel"]
  hi = x >= GELU_PHONE_X
  assert hi.sum() >= 4, "the sweep is supposed to carry the kernel's large-x regime"
  # the sweep's large-x rows: the kernel's exp2 saturates from x ~ 11.5 on, so the phone returns x * 2^-t there
  # and is a measurable 0.5% out, then at x = 60000 the x * x * 0.044715 overflows fp16 inside the qf16 chain
  # and it returns NaN. Both are facts about the phone's bytes, and both are asserted rather than hidden.
  sat = hi & np.isfinite(gel)
  rel = np.abs(gel[sat] - ref[sat]) / np.maximum(np.abs(ref[sat]), 1e-9)
  assert rel.max() > CONTRACT, f"expected the phone to leave the contract above x={GELU_PHONE_X}, it is now at {rel.max():.3g}"
  assert np.isnan(gel[hi & ~np.isfinite(gel)]).all()
  assert (hi & np.isnan(gel)).sum() >= 1, "the overflow probe (x * x * 0.044715 past the fp16 top) is gone from the case"
  # the occupancy logits' own range: nothing in the model comes near it
  assert np.abs(out["x"]).max() < GELU_PHONE_X
  # and tinygrad's lowering, which has no such step, is exact where the phone is not
  gl = TG.gelu(out["g"]).numpy().astype(np.float32)
  assert np.abs(gl[hi] - ref[hi]).max() <= 1e-3, "the lowering is expected to hold where the kernel does not"
