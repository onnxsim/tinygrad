"""The grouped int4 MatMul of PLaMo 3's ONNX export (plamo3_onnx._qconst4_grouped: packed nibbles, Mod / Div, Concat, Reshape,
Sub 8, Mul by the group scales) run through tinygrad's ONNX importer on the DSP (ONNX_Q4_HMX=1 -> _matmul_q4) on hexagon-sim,
against numpy on the dequantized weights."""
import os, pathlib, tempfile, unittest
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from tinygrad import Tensor, dtypes
from test.external.dsp.hand import hexsim
from test.external.dsp.hand.q4.test_hand_q4 import quantize
from tinygrad.nn.onnx import OnnxRunner, pack_int4_hmx

def grouped_int4_matmul_model(q, s, M, K, N):
  """the same node chain as plamo3_onnx._qconst4_grouped, then MatMul(x, W); float32 x (M, K) -> y (M, N)"""
  G = K // s.shape[0]
  inits, nodes = [], []
  def const(name, arr): inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name
  def node(op, ins, out, **kw): nodes.append(helper.make_node(op, ins, [out], **kw)); return out
  packed = (((q + 8)[0::2]) + 16 * ((q + 8)[1::2])).astype(np.uint8)
  x = node("Cast", [const("w_p", packed)], "w_x", to=TensorProto.INT32)
  lo = node("Mod", [x, const("w_c16", np.array(16, np.int32))], "w_lo")
  hi = node("Div", [x, const("w_c16b", np.array(16, np.int32))], "w_hi")
  ax = const("w_ax1", np.array([1], np.int64))
  stacked = node("Concat", [node("Unsqueeze", [lo, ax], "w_ul"), node("Unsqueeze", [hi, ax], "w_uh")], "w_st", axis=1)
  w = node("Reshape", [stacked, const("w_shape", np.array([K, N], np.int64))], "w_r")
  f = node("Sub", [node("Cast", [w], "w_f", to=TensorProto.FLOAT), const("w_c8", np.array(8.0, np.float32))], "w_sub")
  f = node("Reshape", [f, const("w_gshape", np.array([K // G, G, N], np.int64))], "w_g")
  f = node("Mul", [f, const("w_s", s.reshape(K // G, 1, N).astype(np.float32))], "w_m")
  W = node("Reshape", [f, const("w_kshape", np.array([K, N], np.int64))], "w_full")
  nodes.append(helper.make_node("MatMul", ["x", W], ["y"]))
  g = helper.make_graph(nodes, "q4", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
                        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])], initializer=inits)
  return helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(), "needs the Hexagon toolchain (HEXAGON_TOOLS) + clang")
class TestHandQ4Onnx(unittest.TestCase):
  def _run(self, M, K, N, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    W = rng.uniform(-0.05, 0.05, (K, N)).astype(np.float32)
    q, s = quantize(W)
    Wd = (q * np.repeat(s.astype(np.float32), 128, axis=0)).astype(np.float16)
    ref = A.astype(np.float16).astype(np.float64) @ Wd.astype(np.float64)
    model = grouped_int4_matmul_model(q, s, M, K, N)
    with tempfile.TemporaryDirectory() as d:
      work = pathlib.Path(d)
      path = work / "m.onnx"; path.write_bytes(model.SerializeToString())
      old = os.environ.get("ONNX_Q4_HMX"); os.environ["ONNX_Q4_HMX"] = "1"
      try:
        runner = OnnxRunner(path)
        with hexsim.capture_dsp() as calls:
          # fp16 input: the captured run does not execute the fp32 -> fp16 cast in front of the kernel (see the note below)
          runner({"x": Tensor(A.astype(np.float16), device="DSP")})["y"].realize()
      finally:
        if old is None: del os.environ["ONNX_Q4_HMX"]
        else: os.environ["ONNX_Q4_HMX"] = old
      q4 = [c for c in calls if "__hmx_q4_gemm(" in c[0]]
      self.assertEqual(len(q4), 1, f"the MatMul did not take the HMX int4 path ({len(calls)} kernels captured)")
      got, cyc = hexsim.run_captured(*q4[0], work)
      got = np.frombuffer(got, np.float16)[:M * N].reshape(M, N).astype(np.float64)
    err = np.abs(got - ref)
    self.assertTrue((err <= np.abs(ref) / 512 + 2e-3).all(), f"max abs err {err.max():.3g} (max |ref| {np.abs(ref).max():.3g})")
    print(f"\nhmx_q4 onnx {M}x{K}x{N}: {cyc} pcycles, max abs err {err.max():.3g}")

  def test_plamo_style(self): self._run(32, 256, 64)
  # NOTE: the activation is fp16 here. With fp32 activations the matcher inserts an fp32 -> fp16 cast kernel in front of the
  # HMX kernel; under the qemu mock that cast did not run before the capture (zeros), so this path is checked on the phone.

if __name__ == "__main__":
  unittest.main()
