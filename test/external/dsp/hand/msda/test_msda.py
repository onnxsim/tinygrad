"""Multi-scale deformable attention: msda/msda_kernel.h (scalar body) vs tinygrad, on onnx-simplifier's model-shaped
synthetic calls (msda_ref.py `synthetic`): RT-DETR-r18's decoder cross-attention (300 box queries, 3 levels), BEVFormer's
TSA (2-frame queue, per-frame offsets) and SCA (6 cameras, visibility mask, a query no camera sees), each with float32 and
uint8 (per-map scale / zero point) value maps. Edge cases are mixed in like msda_ref.py does: points off the map, on
pixel centers and borders. Contract: tinygrad equals the hand scalar body bit for bit.

The hand kernel's HVX body (V68+ qf32) is not covered: qemu 8.2 can't decode qfloat, and hexagon-sim runs qf32 as IEEE
fp32, so neither gives the phone's rounding. onnx-simplifier checks it on the phone (msda_hvx/README.md)."""
import pathlib
import numpy as np
import pytest
from tinygrad import Tensor
import harness
from harness import build_hand, run_hand, run_tinygrad, record, mismatch

HERE = pathlib.Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(harness.missing() is not None, reason=str(harness.missing()))

def synthetic(kind:str, seed:int=0):
  from tg_msda import MsdaShape, MSDA_REF_PIX, MSDA_REF_BOX
  rng = np.random.default_rng(seed)
  if kind == "rtdetr_decoder":
    q, levels, m, d, nl, p, nv, no, mode, r = 300, [(80, 80), (40, 40), (20, 20)], 8, 32, 3, 4, 1, 1, MSDA_REF_BOX, 1
    ref = np.concatenate([rng.random((1, q, 1, 1, 2)) * 0.9 + 0.05, rng.random((1, q, 1, 1, 2)) * 0.5 + 0.02], -1)
    loc = rng.standard_normal((q, m, no, nl, p, 2)) * 2.0
    ref[0, 0, 0, 0] = [0.0, 1.0, 0.1, 0.1]                               # corners: zero-padding taps
  else:
    tsa = kind == "bevformer_tsa"
    q, levels, m, d, nl = 96, ([(50, 50)] if tsa else [(15, 25)]), 8, 32, 1
    nv, no, p, r, mode = (2, 2, 4, 1, MSDA_REF_PIX) if tsa else (6, 1, 8, 4, MSDA_REF_PIX)
    h, w = levels[0]
    ref = rng.random((nv, q, 1, r, 2)) * 1.4 - 0.2
    loc = rng.standard_normal((q, m, no, nl, p, 2)) * 2.0
    ref[:, :4] = 0.0                                                       # exact pixel centers / borders / half pixels
    loc[:4] = np.array([-0.5, 0.0, 0.5, float(w) - 0.5]).reshape(4, 1, 1, 1, 1, 1)
    loc[4, :, :, :, :, 0] = float(w) + 0.5 - ref[0, 4, 0, 0, 0] * w      # x = w: just past the right edge
  S = sum(h * w for h, w in levels)
  starts = list(np.cumsum([0] + [h * w for h, w in levels])[:-1])
  value = rng.standard_normal((nv, S, m * d)).astype(np.float32)
  att = rng.standard_normal((q, m, no, nl * p))
  attw = (np.exp(att) / np.exp(att).sum(-1, keepdims=True)).reshape(q, m, no, nl, p).astype(np.float32)
  vis = None
  if kind == "bevformer_sca":
    vis = (rng.random((nv, q)) < 0.3).astype(np.uint8)
    vis[:, 5], vis[:, 6] = 0, 1                                            # no camera sees query 5, every camera sees 6
  s = MsdaShape(NV=nv, L=nl, H=[h for h, _ in levels], W=[w for _, w in levels], start=[int(x) for x in starts], S=S, M=m, D=d,
                P=p, Q=q, NO=no, mode=mode, NVR=ref.shape[0], RL=1, R=r, RD=ref.shape[-1], has_vis=vis is not None)
  return s, value, loc.astype(np.float32), ref.astype(np.float32), attw, vis

def quantize(value):
  """per-map uint8 (msda_ref.py `quantize(per_map=True)`-like): real = (u8 - zp) * scale"""
  lo, hi = value.min((1, 2)), value.max((1, 2))
  scale = ((hi - lo) / 255.0).astype(np.float32)
  zp = np.clip(np.round(-lo / scale), 0, 255).astype(np.int32)
  u8 = np.clip(np.round(value / scale[:, None, None]) + zp[:, None, None], 0, 255).astype(np.uint8)
  return u8, scale, zp

@pytest.fixture(scope="module")
def msda_exe(tmp_path_factory):
  return build_hand(HERE / "msda_oracle.c", tmp_path_factory.mktemp("msda") / "msda_oracle", cpu="v65", hvx=True)

CASES = [(k, u8) for k in ("rtdetr_decoder", "bevformer_tsa", "bevformer_sca") for u8 in (False, True)]

@pytest.mark.parametrize("kind,u8", CASES, ids=[f"{k}_{'u8' if u else 'f32'}" for k, u in CASES])
def test_msda(msda_exe, kind, u8):
  from tg_msda import msda
  s, value, loc, ref, attw, vis = synthetic(kind, seed=len(kind) + u8)
  dummy = np.zeros(1, np.float32)
  vu8, vscale, vzp = quantize(value) if u8 else (np.zeros(4, np.uint8), np.ones(s.NV, np.float32), np.zeros(s.NV, np.int32))
  s.vdtype = int(u8)
  hand = run_hand(msda_exe, [s.packed(), dummy if u8 else value, vu8, vscale, vzp, loc, ref,
                             attw, vis if vis is not None else np.zeros(4, np.uint8)],
                  [(np.float32, (s.Q, s.M * s.D))])
  args = [vu8 if u8 else value, vscale if u8 else None, vzp if u8 else None, loc, ref, attw, vis]
  tg = run_tinygrad(lambda *a: msda(s, *a), *[Tensor(a).realize() if a is not None else None for a in args])
  bad = mismatch(tg.outputs[0], hand.outputs[0])
  record("msda", f"{kind}_{'u8' if u8 else 'f32'}_Q{s.Q}", mismatched=bad, hand_insns=hand.insns["msda_scalar"],
         tg_insns=tg.insns, tg_kernels=tg.kernels)
  assert bad == 0, f"{bad} of {s.Q*s.M*s.D} outputs differ"
