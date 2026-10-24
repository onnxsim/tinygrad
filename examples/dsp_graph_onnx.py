"""Any ONNX model -> one standalone Hexagon program (tinygrad/runtime/support/dsp_graph_v65.py's emitter and skel) packed as a
compiler-service artifact for onnxsim's onnx-remote-hexagon-worker, for cDSPs from v68 on (qfloat HVX, HMX with HMX=1).

  python3 examples/dsp_graph_onnx.py model.onnx outdir --arch v69 --artifact model.tghx [--manifest manifest.json]

Unlike examples/openpilot/dsp_graph_v65.py this needs no qemu run: v68+ code can't run under qemu (qfloat, HMX), so the kernels
are captured without executing anything (dsp_graph.capture) from tinygrad's ONNX runner on placeholder inputs, and the runner's
outputs on the phone are the correctness check (compare them with onnxruntime's). The output is the ONNX outputs back to back:
a single output keeps its dtype (FLOAT or UINT8), several are each cast to FLOAT. Inputs must have static shapes.
Env: DSP_THREADS (the pool size baked into the program, default 4), HMX / QDQ_HMX / HMX_VTCM_KB as for the HMX lowering, and
HEXAGON_SDK_ROOT / HEXAGON_TOOLCHAIN / NDK_CLANG for the build.
"""
import argparse, json, os, pathlib, sys

ONNX_DTYPES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 9: "bool", 10: "float16"}

def pretranspose_gemm_weights(model) -> int:
  """Gemm(A, B, transB=1) with B a constant -> B stored transposed and transB=0, when every use of B is such a Gemm's B (a weight
  shared by two Gemms, like BEVFormer's two-frame value_proj, is transposed once). Same result; the generated kernel's output
  lanes then read B contiguously. Kept (N, K), each reduction step loaded its 128 lanes one scalar at a time from 128 rows
  (BEVFormer-tiny's encoder FFN: 2.2 s per 2500x256x512 Gemm on the V69). -> the number of Gemms rewritten"""
  import onnx
  from onnx import numpy_helper
  init = {t.name: t for t in model.graph.initializer}
  def trans_b(n): return next((a for a in n.attribute if a.name == "transB"), None)
  uses: dict[str, list] = {}
  for n in model.graph.node:
    for k, i in enumerate(n.input): uses.setdefault(i, []).append((n, k))
  for o in model.graph.output: uses.setdefault(o.name, []).append((None, -1))
  done = 0
  for name, us in uses.items():
    if name not in init or not all(n is not None and n.op_type == "Gemm" and k == 1 and (tb := trans_b(n)) is not None and tb.i == 1
                                   for n, k in us): continue
    t = init[name]
    t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).T.copy(), t.name))
    for n, _ in us: trans_b(n).i = 0
    done += len(us)
  onnx.checker.check_model(model)
  return done

def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("onnx"); p.add_argument("outdir")
  p.add_argument("--arch", default="v69", help="the cDSP's HVX arch (v68 or later)")
  p.add_argument("--artifact", required=True); p.add_argument("--manifest")
  args = p.parse_args()
  if int(args.arch.lstrip("v")) < 68: sys.exit("--arch must be v68 or later (v65 programs: examples/openpilot/dsp_graph_v65.py)")
  # before tinygrad is imported: these are read at import time
  os.environ.update(DEV="DSP", MOCKDSP="1", HVX_ARCH=args.arch)
  os.environ.setdefault("DSP_THREADS", "4")
  import numpy as np, onnx
  from tinygrad import Tensor, dtypes
  from tinygrad.nn.onnx import OnnxRunner
  from tinygrad.runtime.support import dsp_graph, dsp_graph_v65

  model = onnx.load(args.onnx)
  out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
  if os.environ.get("GEMM_PRETRANSPOSE", "1") != "0" and (n := pretranspose_gemm_weights(model)):
    onnx.save(model, out / "model.onnx")
    print(f"pre-transposed the constant B of {n} transB=1 Gemms", file=sys.stderr)
    args.onnx = str(out / "model.onnx")
  init = {i.name for i in model.graph.initializer}
  graph_inputs = [vi for vi in model.graph.input if vi.name not in init]
  run = OnnxRunner(args.onnx)
  # constants first: lowerings read them with .numpy(), and inside the capture no kernel runs
  for v in run.graph_values.values():
    if isinstance(v, Tensor): v.realize()
  if hasattr(run, "_qdq_grid"): run._qdq_grid()  # the QDQ lowering (when the graph has one) is built on the first call: build it here
  feeds = {}
  for vi in graph_inputs:
    tt = vi.type.tensor_type
    shape = [d.dim_value for d in tt.shape.dim]
    if any(d <= 0 for d in shape): sys.exit(f"input {vi.name} has a dynamic shape {shape}; export with static shapes")
    feeds[vi.name] = Tensor(np.zeros(shape, ONNX_DTYPES[tt.elem_type])).realize()

  holder: dict = {}
  def fn():
    r = run(feeds)
    outs = [r[vo.name] for vo in model.graph.output] if isinstance(r, dict) else [r]
    holder["y"] = (outs[0] if len(outs) == 1 else Tensor.cat(*[o.cast(dtypes.float32).flatten() for o in outs])).contiguous().realize()
  calls, bufs = dsp_graph.capture(fn)
  y = holder["y"]
  used = {id(b[0].base) for b in bufs.values()}
  ins = [(name, t) for name, t in feeds.items() if id(t.uop.buffer.base) in used]
  info = dsp_graph_v65.emit(out, calls, bufs, [t.uop.buffer for _, t in ins], y.uop.buffer)
  print(f"emitted {info['calls']} calls ({info['threaded_calls']} threaded), {info['kernels']} kernels, hmx={info['hmx']}, "
        f"{info['blob']/1e6:.1f} MB weights, {info['scratch']/1e6:.2f} MB scratch -> {out}", file=sys.stderr)
  dsp_graph_v65.build(out, kernel_arch=args.arch)

  # program.txt: the runner's I/O contract (see examples/openpilot/dsp_graph_v65.py and onnxsim's remote_hexagon_worker.cpp)
  slot = {name: i for i, (name, _) in enumerate(ins)}
  threads = max(1, int(os.environ["DSP_THREADS"]))
  lines = [f"ncalls {info['calls']}", f"threads {threads}", f"output_bytes {info['output_bytes']}"]
  for i, vi in enumerate(graph_inputs):
    nbytes = feeds[vi.name].nbytes() if vi.name in slot else 0
    lines.append(f"input {i} {vi.type.tensor_type.elem_type} {nbytes} {slot.get(vi.name, -1)}")
  lines += [f"name {i} {c[0]}" for i, c in enumerate(calls)]
  single = len(model.graph.output) == 1
  total = 0
  for vo in model.graph.output:
    tt = vo.type.tensor_type
    dims = [d.dim_value or 1 for d in tt.shape.dim]
    dt = tt.elem_type if single and tt.elem_type in (1, 2) else 1
    n = int(np.prod(dims))
    total += n * (1 if dt == 2 else 4)
    lines.append(f"output {dt} {n} " + " ".join(map(str, dims)))
  if total != info["output_bytes"]: sys.exit(f"ONNX outputs ({total} bytes) don't match the program output ({info['output_bytes']} bytes)")
  size = dsp_graph_v65.pack(out, args.artifact, lines, arch=args.arch)
  manifest = {"schema_version": 1,
              "compiler": {"name": "tinygrad-dsp_graph_onnx", "version": "1", "id": f"tinygrad-hexagon-{args.arch}"},
              "target": {"backend": "hexagon-fastrpc", "device": "cdsp", "chip": args.arch},
              "artifact": {"format": f"tghx-{args.arch}", "abi": "tg_graph-idl-1"},
              "io": {"dtype": "mixed", "layout": "onnx", "dynamic_shapes": False},
              "capabilities": {"ops": [], "dtypes": ["float32", "uint8"]},
              "legalization": {"profile": "none", "version": 1},
              "program": {"calls": info["calls"], "kernels": info["kernels"], "threads": threads, "hmx": info["hmx"],
                          "weights_bytes": info["blob"], "scratch_bytes": info["scratch"]}}
  if args.manifest: pathlib.Path(args.manifest).write_text(json.dumps(manifest))
  print(f"artifact {args.artifact}: {size/1e6:.1f} MB", file=sys.stderr)

if __name__ == "__main__":
  main()
