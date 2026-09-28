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
  args = p.parse_args()
  out = pathlib.Path(args.outdir)
  with open(args.pickle, "rb") as f: jit = compile3.load_pickle(f)
  inputs = seeded_inputs(jit, args.seed)
  if args.inputs:
    npz = np.load(args.inputs)
    inputs = {k: Tensor(npz[k].astype(t.numpy().dtype), device=t.device).realize() for k, t in inputs.items()}
  st = time.perf_counter()
  ref = jit(**inputs).numpy().copy()
  print(f"reference: the JIT under qemu, output {ref.shape} {ref.dtype}, {time.perf_counter()-st:.1f} s")
  res: list = []
  calls, bufs = dsp_graph.capture(lambda: res.append(jit(**inputs)))
  info = dsp_graph_v65.emit(out, calls, bufs, [t.uop.buffer for t in inputs.values()], res[0].uop.buffer)
  xs = [t.numpy().tobytes() for t in inputs.values()]
  dsp_graph_v65.write_case(out, xs, ref.tobytes())
  info |= {"inputs": list(inputs), "output_shape": list(ref.shape), "seed": args.seed, "pickle": str(pathlib.Path(args.pickle).resolve())}
  (out / "graph.json").write_text(json.dumps(info, indent=2) + "\n")
  print(f"emitted {info['calls']} calls ({info['threaded_calls']} threaded), {info['kernels']} kernels, {info['regions']} regions, "
        f"{info['blob']/1e6:.1f} MB weights, {info['scratch']/1e6:.2f} MB scratch -> {out}")
  if args.qemu:
    st = time.perf_counter()
    y = dsp_graph_v65.run_qemu(out, xs)
    same = y == ref.tobytes()
    print(f"emitted program under qemu ({time.perf_counter()-st:.1f} s): {'bit-exact' if same else 'MISMATCH'} vs the JIT")
    if not same:
      yy = np.frombuffer(y, dtype=ref.dtype)[:ref.size]
      print(f"  {np.sum(yy != ref.ravel())} of {ref.size} differ, max |diff| {np.nanmax(np.abs(yy - ref.ravel()))}")
      sys.exit(1)
  if args.build: print("built", *dsp_graph_v65.build(out))
