"""A static QDQ ONNX through tinygrad's OnnxRunner onto the DSP's HMX as one program (nn/onnx_qdq.py + runtime/support/dsp_graph.py),
on hexagon-sim, bit-exact against the ORT-CPU semantics (nn/onnx_qdq.qdq_emulate; and against onnxruntime when installed).
The model has every op the lowering covers: a 7x7 s2 stem on 3 channels, MaxPool, 3x3 s1 / s2 convs, a 1x1 s2 downsample,
residual Adds, Relus folded into zero-point-0 outputs (onnxsim full_qdq + quantized_io form, built directly here)."""
import os, pathlib, tempfile, unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from test.external.dsp.hand import hexsim
os.environ.setdefault("QDQ_HMX", "1")

def qdq_model(seed=0, H=32, W=32, fold_relu=False):
  """fold_relu: no Relu nodes (a zero-point-0 Q already clamps at 0; onnxsim's full_qdq form, which the hand runner reads)"""
  rng = np.random.default_rng(seed)
  inits, nodes = [], []
  def init(name, a): inits.append(numpy_helper.from_array(a, name)); return name
  def dq(x, s, z, name): nodes.append(helper.make_node("DequantizeLinear", [x, s, z], [name])); return name
  def q(x, s, z, name): nodes.append(helper.make_node("QuantizeLinear", [x, s, z], [name])); return name
  def act(name, scale, zp): return (init(name + "_s", np.float32(scale)), init(name + "_z", np.uint8(zp)))
  def conv(x, xs, xz, cin, cout, k, s, relu, name):
    w = rng.integers(-127, 128, (cout, cin, k, k), dtype=np.int8)
    sw = rng.uniform(0.002, 0.02, cout).astype(np.float32)
    ws, wz = init(name + "_ws", sw), init(name + "_wz", np.zeros(cout, np.int8))
    sx = np.float32(numpy_helper.to_array(next(i for i in inits if i.name == xs)))
    b = rng.integers(-3000, 3000, cout, dtype=np.int32)
    bs, bz = init(name + "_bs", (sx * sw).astype(np.float32)), init(name + "_bz", np.zeros(cout, np.int32))
    wd = dq(init(name + "_w", w), ws, wz, name + "_wd")
    nodes[-1].attribute.append(helper.make_attribute("axis", 0))
    bd = dq(init(name + "_b", b), bs, bz, name + "_bd")
    nodes[-1].attribute.append(helper.make_attribute("axis", 0))
    xd = dq(x, xs, xz, name + "_xd")
    nodes.append(helper.make_node("Conv", [xd, wd, bd], [name + "_c"], kernel_shape=[k, k], strides=[s, s], pads=[k // 2] * 4))
    ys, yz = act(name + "_y", float(rng.uniform(0.05, 0.2)), 0 if relu else int(rng.integers(90, 170)))
    y = name + "_c"
    if relu and not fold_relu: nodes.append(helper.make_node("Relu", [y], [name + "_r"])); y = name + "_r"
    return q(y, ys, yz, name + "_q"), ys, yz
  def add(a, as_, az, b, bs, bz, name):
    ad, bd = dq(a, as_, az, name + "_ad"), dq(b, bs, bz, name + "_bd")
    nodes.append(helper.make_node("Add", [ad, bd], [name + "_a"]))
    y = name + "_a"
    if not fold_relu: nodes.append(helper.make_node("Relu", [y], [name + "_r"])); y = name + "_r"
    ys, yz = act(name + "_y", float(rng.uniform(0.1, 0.3)), 0)
    return q(y, ys, yz, name + "_q"), ys, yz
  xs, xz = act("x", 0.0187, 114)
  nodes.append(helper.make_node("Transpose", ["x"], ["x_t"], perm=[0, 3, 1, 2]))
  c1, s1, z1 = conv("x_t", xs, xz, 3, 64, 7, 2, True, "c1")
  pd = dq(c1, s1, z1, "p_d")
  nodes.append(helper.make_node("MaxPool", [pd], ["p_m"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
  p = q("p_m", s1, z1, "p_q")
  c2, s2, z2 = conv(p, s1, z1, 64, 64, 3, 1, True, "c2")
  c3, s3, z3 = conv(c2, s2, z2, 64, 64, 3, 1, False, "c3")
  a3, sa3, za3 = add(c3, s3, z3, p, s1, z1, "a3")
  c4, s4, z4 = conv(a3, sa3, za3, 64, 128, 3, 2, True, "c4")
  c5, s5, z5 = conv(c4, s4, z4, 128, 128, 3, 1, False, "c5")
  d5, sd5, zd5 = conv(a3, sa3, za3, 64, 128, 1, 2, False, "d5")
  y, ys, yz = add(c5, s5, z5, d5, sd5, zd5, "a5")
  nodes.append(helper.make_node("Identity", [y], ["y"]))
  for i, n in enumerate(nodes): n.name = f"n{i}_{n.op_type}"  # named, as real exports are (qdq_graph.py keys nodes by name)
  g = helper.make_graph(nodes, "qdq_tiny", [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, H, W, 3])],
                        [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 128, H // 8, W // 8])], inits)
  m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
  m.ir_version = 8
  return m

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(), "needs the Hexagon toolchain (HEXAGON_TOOLS) + clang")
class TestQDQOnnxDSP(unittest.TestCase):
  def test_tiny_resnet_on_hexsim(self):
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.nn.onnx_qdq import qdq_emulate
    from tinygrad.runtime.support import dsp_graph
    m = qdq_model()
    m.graph.node.pop()  # the Identity: the output is the last Add's Q
    m.graph.output[0].name = m.graph.node[-1].output[0]
    with tempfile.TemporaryDirectory() as d:
      work = pathlib.Path(d)
      onnx.save(m, work / "m.onnx")
      runner = OnnxRunner(work / "m.onnx")
      net = runner._qdq_grid()  # on (HMX=1 on the DSP; QDQ_HMX=1 anywhere)
      self.assertIsNotNone(net, "the QDQ grid lowering didn't take the model")
      x = np.random.default_rng(1).integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)
      ref = qdq_emulate(net, x)
      try:
        import onnxruntime as ort
        o = ort.InferenceSession(str(work / "m.onnx"), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        self.assertEqual(int((o != ref).sum()), 0, "the emulator disagrees with onnxruntime")
      except ImportError: pass
      xt = Tensor(x).realize()
      for c in net.consts: c.realize()
      out: list = []
      calls, bufs = dsp_graph.capture(lambda: out.append(runner({net.in_name: xt})[runner.graph_outputs[0]]))
      info = dsp_graph.emit(work / "g", calls, bufs, xt.uop.buffer, out[0].uop.buffer)
      got, cyc = dsp_graph.run_sim(work / "g", x.tobytes(), ref.nbytes)
      got = np.frombuffer(got, np.uint8).reshape(ref.shape)
    self.assertEqual(int((got != ref).sum()), 0, "tinygrad's program disagrees with ORT's semantics")
    print(f"\nQDQ tiny ResNet ({len(net.ops)} ops, {info['calls']} kernels): hexagon-sim {cyc} pcycles")

if __name__ == "__main__":
  unittest.main()
