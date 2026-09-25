"""RPN post-processing (Mask R-CNN rest.onnx): hand kernels pd_kernel.h / topk_kernel.h / nms_kernel.h vs tinygrad.

Shapes are the real graph's (800x1088 input, 5 FPN levels, k=1000 per level; see onnx-simplifier's
scripts/android/tinygrad_hexagon_bridge/dynamic_ops_survey.md); values are synthetic in the real ranges, no model
download. Contract for every kernel here: bit-exact with ONNX Runtime's CPU output, so hand == tinygrad bit for bit.
"""
import pathlib
import numpy as np
import pytest
from tinygrad import dtypes
import harness
from harness import build_hand, run_hand, run_tinygrad, record, mismatch

HERE = pathlib.Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(harness.missing() is not None, reason=str(harness.missing()))

# (H, W, k) per FPN level P2..P6 and the anchor stride
LEVELS = [(200, 272, 1000, 4), (100, 136, 1000, 8), (50, 68, 1000, 16), (25, 34, 1000, 32), (13, 17, 663, 64)]

@pytest.fixture(scope="module")
def pd_exe(tmp_path_factory):
  return build_hand(HERE / "pd_oracle.c", tmp_path_factory.mktemp("pd") / "pd_oracle", cpu="v73", hvx=False)

PD_FIELDS = [("s1", np.float32), ("s2", np.float32), ("z1", np.int32), ("z2", np.int32), ("exp_clip", np.float32),
             ("clip_x", np.float32), ("clip_y", np.float32), ("box_s", np.float32), ("box_z", np.int32),
             ("bb_s", np.float32), ("bb_z", np.int32)]

def pd_case(lvl:int, seed:int=0):
  H, W, k, stride = LEVELS[lvl]
  rng = np.random.default_rng(seed + lvl)
  f = np.float32
  # rest.onnx-like constants (the same ranges as onnx-simplifier's ci/pd_selfcheck.c)
  P = {"s1": f(0.0078125) * f(1 + lvl), "z1": 120 + lvl, "s2": f(0.00390625) * f(2 + lvl), "z2": 128,
       "exp_clip": f(4.135166556742356), "clip_x": f(1087.0), "clip_y": f(799.0), "box_s": f(5.0) + f(lvl),
       "box_z": 0, "bb_s": f(0.0137) * f(1 + lvl), "bb_z": 131 - 3 * lvl}
  params = b"".join(np.array([P[n]], dt).tobytes() for n, dt in PD_FIELDS)
  base = np.array([[-s * (a + 1), -s * (3 - a), s * (a + 1), s * (3 - a)] for a in range(3) for s in [np.float32(stride)]],
                  np.float32)
  i = np.arange(3 * H * W)
  hw = i // 3
  anchors = (base[i % 3] + np.stack([hw % W, hw // W, hw % W, hw // W], 1).astype(np.float32) * np.float32(stride)).astype(np.float32)
  nchw = rng.integers(0, 256, (12, H * W), dtype=np.uint8)
  deltas = ((nchw.reshape(3, 4, H * W).transpose(2, 0, 1).reshape(-1, 4).astype(np.float32) - np.float32(P["bb_z"]))
            * np.float32(P["bb_s"])).astype(np.float32)
  idx = rng.permutation(3 * H * W)[:k].astype(np.int32)
  return P, params, base, anchors, nchw, deltas, idx

@pytest.mark.parametrize("lvl", range(len(LEVELS)), ids=[f"P{i+2}" for i in range(len(LEVELS))])
def test_proposal_decode(pd_exe, lvl):
  from tg_rpn import pd_constants, proposal_decode
  H, W, k, stride = LEVELS[lvl]
  P, params, base, anchors, nchw, deltas, idx = pd_case(lvl)
  hand = run_hand(pd_exe, [np.frombuffer(params, np.uint8), anchors, idx, deltas, nchw],
                  [(np.float32, (k, 4)), (np.float32, (k, 4))], ints=(3 * H * W, k, H, W))
  ref, fast = hand.outputs
  assert mismatch(ref, fast) == 0, "hand kernel: reference and fast paths disagree"
  C = pd_constants(P, base, stride)
  tg = run_tinygrad(lambda n, i: proposal_decode(n, i, H, W, P, C), nchw, idx)
  bad = mismatch(tg.outputs[0], fast)
  record("rpn", f"proposal_decode_P{lvl+2}", k=k, mismatched=bad, hand_insns=hand.insns["fast"], tg_insns=tg.insns,
         tg_kernels=tg.kernels)
  assert bad == 0, f"tinygrad differs from the hand kernel in {bad}/{4*k} values"

# ------------------------------------------------------------------------------------------------- TopK
# (n, k, distinct values): the real calls -- per-level pre-NMS selects (P2..P6) and the post-NMS/final ones -- with
# dequantized-int8-like scores (few distinct values: ties at the k-th value are the normal case, not an edge case)
TOPK_CASES = [(163200, 1000, 213), (40800, 1000, 180), (10200, 1000, 213), (2550, 1000, 51), (663, 663, 90),
              (1465, 1000, 120), (106, 100, 60)]

@pytest.fixture(scope="module")
def topk_exe(tmp_path_factory):
  return build_hand(HERE / "topk_oracle.c", tmp_path_factory.mktemp("topk") / "topk_oracle", cpu="v73", hvx=True)

@pytest.mark.parametrize("n,k,distinct", TOPK_CASES, ids=[f"n{n}_k{k}" for n, k, _ in TOPK_CASES])
def test_topk(topk_exe, n, k, distinct):
  from tg_rpn import topk_desc
  rng = np.random.default_rng(n)
  x = (rng.integers(-distinct // 2, distinct // 2 + 1, n) * 0.0625).astype(np.float32)
  x[x == 0] = 0.0  # no -0.0: the kernel canonicalizes it to +0.0 (tk_key), ORT would return -0.0
  order = np.lexsort((np.arange(n), -x))[:k]                                  # ORT: value desc, then index asc
  hand = run_hand(topk_exe, [x], [(np.float32, (k,)), (np.int64, (k,))], ints=(n, k))
  assert mismatch(hand.outputs[0], x[order]) == 0 and mismatch(hand.outputs[1], order.astype(np.int64)) == 0, \
    "hand kernel differs from the ORT-order reference"
  tg = run_tinygrad(lambda t: topk_desc(t, k), x)
  bad = mismatch(tg.outputs[0], hand.outputs[0]) + mismatch(tg.outputs[1], hand.outputs[1])
  record("rpn", f"topk_n{n}_k{k}", mismatched=bad, hand_insns=hand.insns["vec-rot"], tg_insns=tg.insns, tg_kernels=tg.kernels)
  assert bad == 0, f"tinygrad differs from the hand kernel in {bad} of {2*k} values/indices"

# -------------------------------------------------------------------------------------------------- NMS
# The real graph's two NMS groups: 5 per-level RPN calls (iou 0.7, up to 1000 boxes each) and 80 per-class box-head
# calls (iou 0.5, a few to a few hundred boxes). Boxes here are synthetic clusters on the uint8 box grid (the RPN
# boxes are box-grid QDQ'd, so exact ties in coordinates and in IoU are common); scores dequantized-int8-like.
NMS_CASES = [("level", 1000, 0.7, 2000), ("level", 663, 0.7, 2000), ("class", 300, 0.5, 100), ("class", 40, 0.5, 100)]

def nms_case(n:int, seed:int):
  rng = np.random.default_rng(seed)
  centers = rng.uniform(0, 1000, (max(1, n // 12), 2))
  c = centers[rng.integers(0, len(centers), n)] + rng.normal(0, 12, (n, 2))
  wh = rng.uniform(8, 160, (n, 2))
  y1x1, y2x2 = c - wh / 2, c + wh / 2
  boxes = np.round(np.concatenate([y1x1[:, :1], y1x1[:, 1:], y2x2[:, :1], y2x2[:, 1:]], 1) / 5) * 5  # box grid, scale 5
  swap = rng.random(n) < 0.1                                                  # some boxes given as [y2, x2, y1, x1]
  boxes[swap] = boxes[swap][:, [2, 3, 0, 1]]
  scores = rng.integers(0, 120, n).astype(np.float32) * np.float32(0.0078125)
  return boxes.astype(np.float32), scores

def ort_nms(boxes, scores, thr, max_out):
  import onnxruntime as ort
  from onnx import parser
  m = parser.parse_model(f"""<ir_version: 8, opset_import: ["" : 13]>
    nms (float[1, N, 4] b, float[1, 1, N] s) => (int64[K, 3] o) {{
      m = Constant <value = int64[1] {{{max_out}}}> ()
      t = Constant <value = float[1] {{{float(np.float32(thr))!r}}}> ()
      o = NonMaxSuppression(b, s, m, t)
    }}""")
  sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
  return sess.run(None, {"b": boxes[None], "s": scores[None, None]})[0][:, 2].astype(np.int32)

@pytest.fixture(scope="module")
def nms_exe(tmp_path_factory):
  return build_hand(HERE / "nms_oracle.c", tmp_path_factory.mktemp("nms") / "nms_oracle", cpu="v73", hvx=False)

def _nms_hand(nms_exe, boxes, scores, thr, max_out):
  n = len(scores)
  thr_bits = int(np.array([thr], np.float32).view(np.int32)[0])
  return run_hand(nms_exe, [boxes, scores], [(np.int32, (n,)), (np.int32, (1,)), (np.uint8, (n, n))], ints=(n, max_out, thr_bits))

@pytest.mark.parametrize("group,n,thr,max_out", NMS_CASES, ids=[f"{g}_n{n}" for g, n, _, _ in NMS_CASES])
def test_nms_suppress_matrix(nms_exe, group, n, thr, max_out):
  """The parallel half of NMS: every pairwise SuppressByIOU decision, bit for bit (ties at the threshold included)."""
  from tg_rpn import suppress_matrix
  boxes, scores = nms_case(n, n)
  hand = _nms_hand(nms_exe, boxes, scores, thr, max_out)
  sel, cnt, S = hand.outputs
  assert mismatch(sel[:cnt[0]], ort_nms(boxes, scores, thr, max_out)) == 0, "hand kernel differs from ORT"
  # int32, not uint8: a bool vector stored as 4/8 x uchar makes LLVM 19's Hexagon backend abort with HVX on
  # ("Cannot select: v8i8 = bitcast <v8i1 HexagonISD::V2Q>"; README.md, known gaps)
  tg = run_tinygrad(lambda b: suppress_matrix(b, thr).cast(dtypes.int32), boxes)
  bad = mismatch(tg.outputs[0].astype(np.uint8), S)
  record("rpn", f"nms_suppress_matrix_{group}_n{n}", mismatched=bad, hand_insns=hand.insns["suppress_matrix"],
         tg_insns=tg.insns, tg_kernels=tg.kernels)
  assert bad == 0, f"{bad} of {n*n} pairwise decisions differ"

@pytest.mark.parametrize("group,n,thr,max_out", NMS_CASES, ids=[f"{g}_n{n}" for g, n, _, _ in NMS_CASES])
def test_nms_greedy_blocked(nms_exe, group, n, thr, max_out):
  """Greedy NMS at the real sizes, exact for any input: blocks of 32 in visit order, each checked against the earlier
  blocks' kept boxes in one reduction, then at most 32 sweeps inside the block (tg_rpn.nms_greedy_blocked). This is what
  replaced the old xfail (a device-side data-dependent loop is still missing, but the blocked form doesn't need one)."""
  from tg_rpn import nms_greedy_blocked
  boxes, scores = nms_case(n, n)
  hand = _nms_hand(nms_exe, boxes, scores, thr, max_out)
  tg = run_tinygrad(lambda b, s: nms_greedy_blocked(b, s, thr, max_out), boxes, scores)
  bad = mismatch(tg.outputs[0], hand.outputs[0]) + mismatch(tg.outputs[1], hand.outputs[1])
  record("rpn", f"nms_greedy_blocked_{group}_n{n}", mismatched=bad, hand_insns=hand.insns["hvx_portable"], tg_insns=tg.insns,
         tg_kernels=tg.kernels)
  assert bad == 0
