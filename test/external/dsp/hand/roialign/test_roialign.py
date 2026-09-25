"""RoiAlign (Mask R-CNN box/mask heads): hand kernels roialign_kernel.h (fp32) / roialign_u8_kernel.h (uint8, the one
the e2e pipeline ships) vs tinygrad.

Feature maps have the real FPN level shapes (800x1088 input: P2..P5, C = 256), box head 7x7 and mask head 14x14,
sampling_ratio 2. RoIs are synthetic (some partly outside the map), 16 per call -- the real count is data-dependent
(up to 1000 per image). Contracts: the hand kernels agree with ORT (fp32 to 1e-4, uint8 within 1 LSB of
QuantizeLinear(ORT fp32)); tinygrad must equal the hand kernel bit for bit.
"""
import pathlib
import numpy as np
import pytest
from tinygrad import Tensor
import harness
from harness import build_hand, run_hand, run_tinygrad, record, mismatch

HERE = pathlib.Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(harness.missing(intrinsics=True) is not None, reason=str(harness.missing(intrinsics=True)))

# (level, H, W, 1/spatial_scale) for 800x1088
LEVELS = [("P2", 200, 272, 4), ("P3", 100, 136, 8), ("P4", 50, 68, 16), ("P5", 25, 34, 32)]
CASES = [(lvl, H, W, inv, oh) for lvl, H, W, inv in LEVELS for oh in (7, 14)]
C, R, SR = 256, 16, 2

def _rois(rng, H, W, inv, r):
  img_h, img_w = H * inv, W * inv
  x1, y1 = rng.uniform(-8, img_w, r), rng.uniform(-8, img_h, r)
  return np.stack([x1, y1, x1 + rng.uniform(0, img_w / 2, r), y1 + rng.uniform(0, img_h / 2, r)], 1).astype(np.float32)

def ort_roialign(feat_hwc, rois, oh, sr, scale):
  import onnxruntime as ort
  from onnx import parser
  h, w, c = feat_hwc.shape
  m = parser.parse_model(f"""<ir_version: 7, opset_import: ["" : 12]>
    roialign (float[1,{c},{h},{w}] X, float[{len(rois)},4] rois, int64[{len(rois)}] bidx) => (float[{len(rois)},{c},{oh},{oh}] Y) {{
      Y = RoiAlign <mode = "avg", output_height = {oh}, output_width = {oh}, sampling_ratio = {sr},
                    spatial_scale = {float(scale)!r}> (X, rois, bidx)
    }}""")
  s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
  y = s.run(None, {"X": np.ascontiguousarray(feat_hwc.transpose(2, 0, 1))[None], "rois": rois,
                   "bidx": np.zeros(len(rois), np.int64)})[0]
  return np.ascontiguousarray(y.transpose(0, 2, 3, 1))           # (R, OH, OW, C)

def _divs(oh:int) -> Tensor: return Tensor(np.array([oh, SR], np.float32)).realize()

@pytest.fixture(scope="module")
def exes(tmp_path_factory):
  d = tmp_path_factory.mktemp("roialign")
  # fp32: -mhvx=v65 (no HVX float: the vector-extension FMA is scalarized, and qemu 8.2 can't decode v68+ qfloat)
  return {"f32": build_hand(HERE / "roialign_oracle.c", d / "roi_f32", cpu="v65", hvx=True),
          "u8": build_hand(HERE / "roialign_u8_oracle.c", d / "roi_u8", cpu="v65", hvx=True)}

@pytest.mark.parametrize("lvl,H,W,inv,oh", CASES, ids=[f"{l}_{oh}x{oh}" for l, _, _, _, oh in CASES])
def test_roialign_f32(exes, lvl, H, W, inv, oh):
  from tg_roialign import roialign_hwc
  rng = np.random.default_rng(H * oh)
  feat = rng.standard_normal((H, W, C)).astype(np.float32)
  rois = _rois(rng, H, W, inv, R)
  scale = float(np.float32(1.0) / np.float32(inv))
  hand = run_hand(exes["f32"], [feat, rois], [(np.float32, (R, oh, oh, C))], ints=(H, W, C, R, oh, oh, SR, inv))
  np.testing.assert_allclose(hand.outputs[0], ort_roialign(feat, rois, oh, SR, scale), atol=1e-4, err_msg="hand vs ORT")
  tg = run_tinygrad(lambda f, r, d: roialign_hwc(f, r, scale, oh, oh, SR, d, d), feat, rois, _divs(oh))
  bad = mismatch(tg.outputs[0], hand.outputs[0])
  record("roialign", f"f32_{lvl}_{oh}x{oh}_R{R}", mismatched=bad, hand_insns=hand.insns["roialign_hwc"], tg_insns=tg.insns,
         tg_kernels=tg.kernels)
  assert bad == 0, f"{bad} of {R*oh*oh*C} values differ"

@pytest.mark.parametrize("lvl,H,W,inv,oh", CASES, ids=[f"{l}_{oh}x{oh}" for l, _, _, _, oh in CASES])
def test_roialign_u8(exes, lvl, H, W, inv, oh):
  from tg_roialign import roialign_u8, ru8_requant_params
  rng = np.random.default_rng(1000 + H * oh)
  fmap = rng.integers(0, 256, (H, W, C), dtype=np.uint8)
  rois = _rois(rng, H, W, inv, R)
  scale = float(np.float32(1.0) / np.float32(inv))
  z_in, s_in, z_out, s_out = int(rng.integers(90, 160)), 0.0417 * (1 + H % 3), int(rng.integers(60, 140)), 0.0213
  mult, shift = ru8_requant_params(s_in, s_out, SR * SR)
  sbits = int(np.array([scale], np.float32).view(np.int32)[0])
  hand = run_hand(exes["u8"], [fmap, rois], [(np.uint8, (R, oh, oh, C))],
                  ints=(H, W, C, R, oh, oh, SR, z_in, z_out, mult, shift, sbits))
  ref = ort_roialign(((fmap.astype(np.float32) - z_in) * np.float32(s_in)).astype(np.float32), rois, oh, SR, scale)
  qref = np.clip(np.rint(ref / np.float32(s_out)) + z_out, 0, 255)
  assert np.abs(hand.outputs[0].astype(np.int32) - qref.astype(np.int32)).max() <= 1, "hand vs QuantizeLinear(ORT fp32)"
  tg = run_tinygrad(lambda f, r, d: roialign_u8(f, r, scale, oh, oh, SR, z_in, z_out, mult, shift, d, d), fmap, rois, _divs(oh))
  bad = mismatch(tg.outputs[0], hand.outputs[0])
  record("roialign", f"u8_{lvl}_{oh}x{oh}_R{R}", mismatched=bad, hand_insns=hand.insns["roialign_u8"], tg_insns=tg.insns,
         tg_kernels=tg.kernels)
  assert bad == 0, f"{bad} of {R*oh*oh*C} bytes differ"
