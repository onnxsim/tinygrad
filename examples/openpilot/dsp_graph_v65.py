"""An openpilot model captured by compile3.py (DEV=DSP MOCKDSP=1) as one standalone Hexagon v65 program
(tinygrad/runtime/support/dsp_graph_v65.py), plus the reference to check it against.

  MOCKDSP=1 DEV=DSP DSP_V65_HW=1 [DSP_THREADS=4] python3 examples/openpilot/dsp_graph_v65.py model.pkl outdir [--qemu] [--build]

The capture's own memory plan and weights are used as they are, so any compile3 capture works (driving, DM, ...). The reference
is the JIT itself replayed under qemu on seeded inputs; --qemu also runs the emitted program under qemu and compares, --build
makes the FastRPC skel + client (see dsp_graph_v65.build) for the phone.
"""
import argparse, json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compile3
from tinygrad import Tensor
from tinygrad.runtime.support import dsp_graph, dsp_graph_v65

def seeded_inputs(jit, seed:int) -> dict[str, Tensor]:
  rng, ins = np.random.default_rng(seed), {}
  for name, (view, _, dt, dev) in zip(jit.captured.expected_names, jit.captured.expected_input_info):
    # a 1-D input is recorded as the bare buffer (a NOOP view): its size isn't in the capture, so take the inputs from the npz
    # compile3.py writes next to the pickle (--inputs)
    if view._shape is None: raise ValueError(f"input {name} has no recorded shape; pass --inputs (compile3.py writes <pkl>_inputs.npz)")
    shape, np_dt = tuple(view.shape), np.dtype(dt.fmt)
    a = rng.integers(0, 256, shape) if np.issubdtype(np_dt, np.integer) else rng.standard_normal(shape) * 8
    ins[name] = Tensor(a.astype(np_dt), device=dev).realize()
  return ins

if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("pickle"); p.add_argument("outdir")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--inputs", help="an .npz of the inputs by name (default: seeded random ones)")
  p.add_argument("--qemu", action="store_true", help="run the emitted program under qemu and compare with the reference")
  p.add_argument("--build", action="store_true", help="build the FastRPC skel and Android client")
  p.add_argument("--onnx", help="the ONNX model the pickle came from: its input order and output shapes define the artifact's I/O")
  p.add_argument("--artifact", help="with --onnx and --build: pack a compiler-service artifact (dsp_graph_v65.pack) here")
  p.add_argument("--manifest", help="with --artifact: write the compiler manifest (docs/remote-manifest.md in onnxsim) here")
  args = p.parse_args()
  out = pathlib.Path(args.outdir)
  with open(args.pickle, "rb") as f: jit = compile3.load_pickle(f)
  if args.inputs is None and pathlib.Path(args.pickle.rsplit(".", 1)[0] + "_inputs.npz").exists():
    args.inputs = args.pickle.rsplit(".", 1)[0] + "_inputs.npz"
  if args.inputs:
    npz = np.load(args.inputs)
    inputs = {name: Tensor(npz[name], device=dev).realize()
              for name, (_, _, _, dev) in zip(jit.captured.expected_names, jit.captured.expected_input_info)}
  else: inputs = seeded_inputs(jit, args.seed)
  st = time.perf_counter()
  ref = jit(**inputs).numpy().copy()
  print(f"reference: the JIT under qemu, output {ref.shape} {ref.dtype}, {time.perf_counter()-st:.1f} s")
  res: list = []
  st = time.perf_counter()
  calls, bufs = dsp_graph.capture(lambda: res.append(jit(**inputs)))
  print(f"timing: capture {time.perf_counter()-st:.1f} s")
  used = {id(b[0].base) for b in bufs.values()}
  unused = [k for k, t in inputs.items() if id(t.uop.buffer.base) not in used]
  if unused: print(f"inputs no kernel reads (left out of the program): {unused}")
  inputs = {k: t for k, t in inputs.items() if k not in unused}
  info = dsp_graph_v65.emit(out, calls, bufs, [t.uop.buffer for t in inputs.values()], res[0].uop.buffer)
  xs = [t.numpy().tobytes() for t in inputs.values()]
  dsp_graph_v65.write_case(out, xs, ref.tobytes())
  info |= {"inputs": list(inputs), "output_shape": list(ref.shape), "seed": args.seed, "pickle": str(pathlib.Path(args.pickle).resolve())}
  (out / "graph.json").write_text(json.dumps(info, indent=2) + "\n")
  print(f"emitted {info['calls']} calls ({info['threaded_calls']} threaded), {info['kernels']} kernels, {info['regions']} regions, "
        f"{info['blob']/1e6:.1f} MB weights, {info['scratch']/1e6:.2f} MB scratch -> {out}")
  # the kernels are compiled once (dsp_graph_v65.compile_kernels, object-cached); then the qemu check and the skel build, which
  # only link them, run side by side
  from concurrent.futures import ThreadPoolExecutor
  st = time.perf_counter()
  if args.qemu or args.build: dsp_graph_v65.compile_kernels(out)
  print(f"timing: kernels compiled {time.perf_counter()-st:.1f} s")
  with ThreadPoolExecutor(2) as ex:
    def check():
      st = time.perf_counter()
      return dsp_graph_v65.run_qemu(out, xs), time.perf_counter() - st
    def build():
      st = time.perf_counter()
      return dsp_graph_v65.build(out), time.perf_counter() - st
    qemu_f = ex.submit(check) if args.qemu else None
    build_f = ex.submit(build) if args.build else None
    if qemu_f is not None:
      y, dt = qemu_f.result()
      same = y == ref.tobytes()
      print(f"emitted program under qemu ({dt:.1f} s): {'bit-exact' if same else 'MISMATCH'} vs the JIT")
      if not same:
        yy = np.frombuffer(y, dtype=ref.dtype)[:ref.size]
        print(f"  {np.sum(yy != ref.ravel())} of {ref.size} differ, max |diff| {np.nanmax(np.abs(yy - ref.ravel()))}")
        sys.exit(1)
    if build_f is not None:
      paths, dt = build_f.result()
      print("built", *paths, f"({dt:.1f} s)")
  if args.artifact:
    import onnx
    from tinygrad.helpers import getenv
    model = onnx.load(args.onnx, load_external_data=False)
    slot = {name: i for i, name in enumerate(inputs)}  # the program's input order (compile3 sorts by name)
    # program.txt, one record per line. The runner receives ONNX inputs in graph order (the transport's tensors are positional):
    #   input <onnx index> <onnx dtype> <bytes> <program slot, or -1 when no kernel reads it>
    #   output <onnx dtype of the returned tensor> <elements> <dims...>   the program's output is the ONNX outputs back to back:
    #     compile3 ALL_OUTPUTS=1 casts every one to FLOAT; ALL_OUTPUTS=2 (a uint8 program output) keeps UINT8/INT8 ones as they are
    #   name <call index> <kernel name>
    lines = [f"ncalls {info['calls']}", f"threads {max(1, getenv('DSP_THREADS', 1))}", f"output_bytes {info['output_bytes']}"]
    for i, vi in enumerate(model.graph.input):
      nbytes = inputs[vi.name].nbytes() if vi.name in inputs else 0
      lines.append(f"input {i} {vi.type.tensor_type.elem_type} {nbytes} {slot.get(vi.name, -1)}")
    # one line per call: its kernel name, for the runner's per-call profile events
    lines += [f"name {i} {c[0]}" for i, c in enumerate(calls)]
    total, packed = 0, ref.dtype == np.uint8
    for vo in model.graph.output:
      dims = [d.dim_value or 1 for d in vo.type.tensor_type.shape.dim]
      dt = vo.type.tensor_type.elem_type if packed and vo.type.tensor_type.elem_type in (2, 3) else 1  # UINT8 / INT8, else FLOAT
      total += int(np.prod(dims)) * (1 if dt != 1 else 4); lines.append(f"output {dt} {int(np.prod(dims))} " + " ".join(map(str, dims)))
    if total != info["output_bytes"]: raise ValueError(f"ONNX outputs ({total} bytes) don't match the program output ({info['output_bytes']} bytes)")
    size = dsp_graph_v65.pack(out, args.artifact, lines)
    manifest = {"schema_version": 1,
                "compiler": {"name": "tinygrad-dsp_graph_v65", "version": "1", "id": "tinygrad-hexagon-v65"},
                "target": {"backend": "hexagon-fastrpc", "device": "cdsp", "chip": "v65"},
                "artifact": {"format": "tghx-v65", "abi": "tg_graph-idl-1"},
                "io": {"dtype": "mixed", "layout": "onnx", "dynamic_shapes": False},
                "capabilities": {"ops": [], "dtypes": ["float32", "uint8"]},
                "legalization": {"profile": "none", "version": 1},
                "program": {"calls": info["calls"], "kernels": info["kernels"], "threads": max(1, getenv("DSP_THREADS", 1)),
                            "weights_bytes": info["blob"], "scratch_bytes": info["scratch"]}}
    if args.manifest: pathlib.Path(args.manifest).write_text(json.dumps(manifest))
    print(f"artifact {args.artifact}: {size/1e6:.1f} MB")
