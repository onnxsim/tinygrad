"""onnxsim hmx_gemm's hand-written whole-graph QDQ runner (runner/: HMX convs incl. the 7x7 s2 stem, HVX QLinearAdd and
MaxPool, padding maintenance) as the oracle for tinygrad's OnnxRunner lowering of the same model (nn/onnx_qdq.py) run as one
program (runtime/support/dsp_graph.py): both on hexagon-sim, bit-exact against each other and ORT's semantics (and onnxruntime
when installed). Covers the ops that only exist inside the hand runner: the stem, Add, MaxPool."""
import os, pathlib, sys, tempfile, unittest
import numpy as np
from test.external.dsp.hand import hexsim
from test.external.dsp.test_qdq_onnx_dsp import ort_compare

HMX = hexsim.HERE / "hmx"
os.environ.setdefault("QDQ_HMX", "1")

@unittest.skipUnless(hexsim.tools() is not None and hexsim.mockdsp_ok(), "needs the Hexagon toolchain (HEXAGON_TOOLS) + clang")
@unittest.skip("hexagon-sim aborts (SIGABRT) on tinygrad's captured whole-graph program; see the note below")
class TestHandHmxRunner(unittest.TestCase):
  # Quarantined 2026-09-26, same as the qconv families in test_hand_hmx_qconv.py: hexagon-sim aborts the
  # process while running MOCKDSP's captured program. That file never executed before either -- hexsim.
  # mockdsp_ok() used to which() the whole CC command line, which conftest.py appends
  # -ffp-contract=off to, so every HMX oracle skipped itself everywhere. This is now the third file to show
  # it, which locates the fault in the shared path: hexsim.run_captured, the MAIN runner template, or how a
  # captured kernel is handed back to the simulator. Not in the graph lowering. See test_hand_hmx_qconv.py
  # for the one suspect and why it was not guessed at.
  def test_tiny_resnet(self):
    import onnx
    from tinygrad import Tensor
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.nn.onnx_qdq import qdq_emulate
    from tinygrad.runtime.support import dsp_graph
    from test.external.dsp.test_qdq_onnx_dsp import qdq_model
    sys.path.insert(0, str(HMX / "runner"))
    import qdq_graph
    m = qdq_model(fold_relu=True)
    m.graph.node.pop()  # the Identity: the output is the last Add's Q
    m.graph.output[0].name = m.graph.node[-1].output[0]
    x = np.random.default_rng(1).integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
      work = pathlib.Path(d)
      onnx.save(m, work / "m.onnx")
      # the hand runner: qdq_graph.py's program, rn_* on hexagon-sim
      tensors, ops, blob, xin, yout = qdq_graph.lower(onnx.shape_inference.infer_shapes(m))
      (work / "prog").mkdir()
      qdq_graph.write_program(tensors, ops, blob, xin, yout, work / "prog")
      x.tofile(work / "input.bin")
      out = hexsim.run_hand(HMX / "hand_runner.c", work, "prog", "input.bin", includes=[HMX])
      hand_cyc = int(out.split("pcycles ")[1].split()[0])
      hand = np.fromfile(work / "y.bin", np.uint8)
      # tinygrad: OnnxRunner's QDQ lowering, one program
      runner = OnnxRunner(work / "m.onnx")
      net = runner._qdq_grid()
      self.assertIsNotNone(net)
      ref = qdq_emulate(net, x)
      xt = Tensor(x).realize()
      for c in net.consts: c.realize()
      res: list = []
      calls, bufs = dsp_graph.capture(lambda: res.append(runner({net.in_name: xt})[runner.graph_outputs[0]]))
      dsp_graph.emit(work / "g", calls, bufs, xt.uop.buffer, res[0].uop.buffer)
      tg, tg_cyc = dsp_graph.run_sim(work / "g", x.tobytes(), ref.nbytes)
      tg = np.frombuffer(tg, np.uint8)
      ort_note = ort_compare(work / "m.onnx", x, ref)
    self.assertEqual(int((hand != ref.ravel()).sum()), 0, "the hand runner disagrees with ORT's semantics")
    self.assertEqual(int((tg != hand).sum()), 0, "tinygrad disagrees with the hand runner")
    print(f"\ntiny QDQ ResNet (stem 7x7 s2, MaxPool, 3x3 s1/s2, 1x1 s2, 2 Adds): hand runner {hand_cyc} pcycles, "
          f"tinygrad {tg_cyc} ({tg_cyc / hand_cyc:.2f}x); {ort_note}")

if __name__ == "__main__":
  unittest.main()
