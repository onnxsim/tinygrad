"""ONNX_QDQ_LUT's table rule for Q(f(DQ(x))) on a u16 camera-style input: a Sub of a per-channel constant whose channels all
hold one value (openpilot's vision mean) is a table like a scalar Sub; a varying constant keeps the float path."""
import os, tempfile, unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from tinygrad import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn.onnx import OnnxRunner

S = np.float32(1 / 257)

def mean_sub_model(mean_q:np.ndarray):
  inits = [numpy_helper.from_array(S, "s"), numpy_helper.from_array(np.uint16(0), "z0"), numpy_helper.from_array(np.float32(S / 2), "sm"),
           numpy_helper.from_array(mean_q, "mq"), numpy_helper.from_array(np.uint16(32768), "zy")]
  nodes = [helper.make_node("DequantizeLinear", ["x", "s", "z0"], ["xd"]), helper.make_node("DequantizeLinear", ["mq", "sm", "z0"], ["md"]),
           helper.make_node("Sub", ["xd", "md"], ["d"]), helper.make_node("QuantizeLinear", ["d", "s", "zy"], ["y"])]
  g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.UINT16, [1, 4, 8, 8])],
                        [helper.make_tensor_value_info("y", TensorProto.UINT16, [1, 4, 8, 8])], inits)
  return helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)])

def run(model, x, lut:bool):
  os.environ["ONNX_QDQ_LUT"] = "1" if lut else "0"
  getenv.cache_clear()
  try:
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
      onnx.save(model, f.name)
      runner = OnnxRunner(f.name)
      y = runner({"x": Tensor(x)})["y"].numpy()
      plan = getattr(runner, "_qconv_w", {}).get(("q", "y"))
      return y, plan[0] if plan else None
  finally: os.environ.pop("ONNX_QDQ_LUT"); getenv.cache_clear()

def ref(x, mean_q):
  # the ONNX ops in float32, each rounding to float32, then round half to even and saturate
  d = (x.astype(np.float32) * S) - (mean_q.astype(np.float32) * np.float32(S / 2))
  return np.clip(np.rint(d / S) + 32768, 0, 65535).astype(np.uint16)

class TestOnnxQdqLut(unittest.TestCase):
  def test_uniform_channel_constant(self):
    x = np.random.default_rng(0).integers(0, 65536, (1, 4, 8, 8)).astype(np.uint16)
    x.reshape(-1)[:4] = [0, 1, 65534, 65535]
    mq = np.full((1, 4, 1, 1), 65535, np.uint16)
    y, plan = run(mean_sub_model(mq), x, lut=True)
    self.assertEqual(plan, "lut")
    np.testing.assert_array_equal(y, ref(x, mq))

  def test_varying_channel_constant_keeps_float_path(self):
    x = np.random.default_rng(1).integers(0, 65536, (1, 4, 8, 8)).astype(np.uint16)
    mq = np.array([100, 200, 300, 400], np.uint16).reshape(1, 4, 1, 1)
    (y, plan), (y_float, _) = run(mean_sub_model(mq), x, lut=True), run(mean_sub_model(mq), x, lut=False)
    self.assertIsNone(plan)
    np.testing.assert_array_equal(y, y_float)

if __name__ == "__main__": unittest.main()
