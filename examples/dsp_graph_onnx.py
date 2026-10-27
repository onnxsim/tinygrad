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

def constants_to_initializers(model) -> int:
  """Constant nodes -> initializers, so they are realized (and their python values read, see main) before the capture: a
  Constant node's output is made inside it, where a value the runner reads on the host comes back as zeros. -> the number moved"""
  from onnx import helper, numpy_helper
  keep, moved = [], 0
  for n in model.graph.node:
    attr = {a.name: a for a in n.attribute}
    if n.op_type == "Constant" and n.domain in ("", "ai.onnx") and "value" in attr:
      t = helper.get_attribute_value(attr["value"])
      model.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(t), n.output[0]))
      moved += 1
    else: keep.append(n)
  del model.graph.node[:]
  model.graph.node.extend(keep)
  return moved

def sink_calls(calls:list, bufs:dict) -> list:
  """Each call moved to just before the first later call that depends on it (reads what it writes, or writes what it reads or
  writes), last calls first. The capture runs every kernel as early as possible, so a buffer lived from its producer near the
  start to a consumer much later (BEVFormer's camera value maps: calls 3 to 76): later producers shorten live ranges, and the
  graph emitter's VTCM arena (DSP_GRAPH_VTCM_KB) holds more of what is live at once. -> the reordered calls"""
  # byte ranges within base buffers: slices of one buffer (BEVFormer's merged value projection, read per layer, written per layer)
  # don't order each other
  def span(b): v = bufs[b][0]; return (id(v.base), v.offset, v.offset + v.nbytes)
  def writes(c): return [span(c[2][o]) for o in c[3]]
  def touches(c): return [span(b) for b in c[2]]
  def hit(xs, ys): return any(a == c and lo < hi2 and lo2 < hi for a, lo, hi in xs for c, lo2, hi2 in ys)
  out = list(calls)
  for i in range(len(out) - 2, -1, -1):
    c = out[i]
    w, t = writes(c), touches(c)
    j = next((j for j in range(i + 1, len(out)) if hit(touches(out[j]), w) or hit(writes(out[j]), t)), len(out))
    if j > i + 1: out.insert(j - 1, out.pop(i))
  return out

def hoist_out_of_hmx_chains(calls:list, bufs:dict) -> list:
  """A buffer one HMX call writes and a later HMX call reads (the generated attention's per-head S^T: the S^T product, its max and
  exp, P @ V) is what the graph emitter's VTCM arena gains most from; every call in between that doesn't depend on what was
  written since the producer moves before it (sinking put BEVFormer's whole V projection chain between a layer's first S^T and
  its P @ V, and S^T no longer fit next to its buffers). -> the reordered calls"""
  def span(b): v = bufs[b][0]; return (id(v.base), v.offset, v.offset + v.nbytes)
  def hit(xs, ys): return any(a == c and lo < hi2 and lo2 < hi for a, lo, hi in xs for c, lo2, hi2 in ys)
  def writes(c): return [span(c[2][o]) for o in c[3]]
  def touches(c): return [span(b) for b in c[2]]
  from tinygrad.runtime.support.dsp_graph_v65 import _listed
  out, i = list(calls), 0
  while i < len(out):
    c = out[i]
    if "__hmx_" in c[1]:
      # the chain: c's output's last toucher within 32 calls; when c writes a raw-tile buffer (the generated attention's S^T), every
      # call that depends on it directly or through another such call (S^T -> max -> exp into P -> P @ V), up to the last HMX one.
      # (followed for every HMX call, the longer chains cost BEVFormer's encoder layer 1.5 ms: other matmuls' operands out of VTCM)
      chain, j = writes(c), None
      general = bool(_listed(c[1], "hmx-tile") & set(c[3]))
      for k in range(i + 1, min(i + 32, len(out))):
        if hit(touches(out[k]), chain):
          if not general:
            j = k if "__hmx_" in out[k][1] else None
            continue
          chain += writes(out[k])
          if "__hmx_" in out[k][1]: j = k
      if j is not None:
        wrote, keep, move = writes(c), [], []
        for d in out[i + 1:j]:
          # depends on the chain: reads or overwrites what it wrote; or is overwritten... by it (moved first, it would read late)
          if hit(touches(d), wrote) or hit(writes(d), [sp for k in keep for sp in touches(k)] + touches(c)): keep.append(d); wrote += writes(d)
          else: move.append(d)
        if move:
          out[i:j] = move + [c] + keep
          i += len(move)
    i += 1
  return out

def pull_raw_tile_readers(calls:list, bufs:dict) -> list:
  """Each call that reads a raw-tile buffer (ops_dsp HMX_TILE, listed as /* hmx-tile: j */ in its source: the generated attention's S^T and
  P) moved up to just after the last call it depends on. Sinking put a layer's last P @ V after the other heads' tails, P then
  lived across them, didn't fit the VTCM arena, and P @ V copied its tiles in from DDR (0.3 ms instead of 0.05). -> the reordered
  calls"""
  def span(b): v = bufs[b][0]; return (id(v.base), v.offset, v.offset + v.nbytes)
  def writes(c): return [span(c[2][o]) for o in c[3]]
  def touches(c): return [span(b) for b in c[2]]
  def hit(xs, ys): return any(a == c and lo < hi2 and lo2 < hi for a, lo, hi in xs for c, lo2, hi2 in ys)
  from tinygrad.runtime.support.dsp_graph_v65 import _listed
  out = list(calls)
  for j in range(len(out)):
    c = out[j]
    if not (_listed(c[1], "hmx-tile") - set(c[3])): continue
    # the last call c depends on: c reads what it writes, or c writes what it touches
    i = max((i for i in range(j) if hit(touches(c), writes(out[i])) or hit(writes(c), touches(out[i]))), default=-1)
    if i + 1 < j: out[i + 1:j + 1] = [c] + out[i + 1:j]
  return out

def keep_attention_vtcm(calls:list, bufs:dict) -> list:
  """The split hand attention (nn/onnx.py _hmx_attn_dsp) keeps a head's S tiles in VTCM from its S call (hmx_attn_s_*) through
  the pool's softmax (attn_sm_*) to its P @ V call (hmx_attn_pv_*); data flows between them only through that VTCM, which the
  schedule doesn't see, so it may put another HMX kernel in between, whose tile caches overwrite S (BEVFormer's decoder: a Gemm
  between head 2's S and P @ V, cls_scores 13% off). Every call in between that reads nothing the group wrote moves before the
  S call (the group writes only its tokens, so that's all but the softmax); one that writes what the group reads can't, and
  is an error. -> the reordered calls"""
  def base(b): return id(bufs[b][0].base)
  def writes(c): return {base(c[2][o]) for o in c[3]}
  def reads(c): return {base(b) for i, b in enumerate(c[2]) if i not in c[3]}
  out, i = list(calls), 0
  while i < len(out):
    if out[i][0].startswith("hmx_attn_s_"):
      j = next(j for j in range(i + 1, len(out)) if out[j][0].startswith("hmx_attn_pv_"))
      wrote, used, keep, move = writes(out[i]), reads(out[i]), [], []
      for c in out[i + 1:j]:
        if c[0].startswith("attn_sm_") or reads(c) & wrote:
          keep.append(c); wrote |= writes(c); used |= reads(c)
        else: move.append(c)
      used |= reads(out[j])
      if any(writes(c) & used for c in move): raise RuntimeError(f"call {i}: a call between an attention S and P @ V writes their input")
      out[i:j] = move + [out[i]] + keep
      i += len(move)
    i += 1
  return out

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
  os.environ.setdefault("PARALLEL", "0")  # compile in this process: the capture's stand-in compiler (below) doesn't reach a pool
  # layout choices that keep the generated kernels' vector lanes on unit-stride axes (each is 0 to turn off). BEVFormer-tiny on
  # the V69, all four against none: decoder 3.39 -> 1.19 s, encoder layer 0.91 -> 0.39 s, 3-layer encoder 3.55 -> 1.23 s
  # (REDUCE_TWO_STAGE on top: 1.12 / 0.32 / 0.97 s). HVX_UPCAST_ELEMWISE: an elementwise kernel whose small masked axis (a concat's)
  # was upcast also gets its unit-stride axis as vector lanes -- BEVFormer's temporal-attention input, concat(prev_bev, query) cast
  # to fp16 for GEMM_FP16, ran fully scalar at 17 ms per layer (3-layer encoder with onnxsim::MSDA + GEMM_FP16: 235 -> 162 ms)
  for flag in ("SOFTMAX_CONTIGUOUS", "MATMUL_CONTIGUOUS", "TRANSPOSE_CONTIGUOUS", "HVX_UPCAST_ODD", "REDUCE_TWO_STAGE", "HVX_UPCAST_ELEMWISE"):
    os.environ.setdefault(flag, "1")
  # intermediates in the VTCM the HMX tile pool leaves of DSP_VTCM_TOTAL_KB (the V69's 8 MB; dsp_graph_v65.emit picks them by
  # liveness and access width; a cDSP that grants less runs them from DDR). BEVFormer-tiny with a 3 MB pool: decoder 84 -> 64 ms,
  # 1 / 3 encoder layers 42 -> 36 / 111 -> 105 ms
  # buffer reuse planned by the graph emitter (DSP_GRAPH_REUSE: DDR by liveness, after the call sinking and the VTCM arena), not
  # by tinygrad's memory planner: its aliasing of unrelated tensors made calls depend on each other (a copy of BEVFormer's
  # camera value maps couldn't move past a reuse of its source region) and kept the maps out of VTCM. Scratch: 108 -> 37 MB
  os.environ.setdefault("NO_MEMORY_PLANNER", "1")
  # multi-output kernels (schedule/rangeify.py merge_outputs): a producer also stores the fp16, row-padded copy its GEMM_FP16 consumers
  # take (nn/onnx.py pad16), instead of a cast and a pad kernel re-reading its output; 2: a store no kernel reads is dropped (only
  # the final output is read back). BEVFormer-tiny on the V69: decoder 27.6 -> 26.5, enc1 13.7 -> 12.0, enc3 40.3 -> 36.9 ms
  os.environ.setdefault("MULTIOUT", "2")
  os.environ.setdefault("DSP_GRAPH_REUSE", "1")
  if os.environ.get("HMX"):
    os.environ.setdefault("DSP_GRAPH_VTCM_KB", str(max(0, int(os.environ.get("DSP_VTCM_TOTAL_KB", "8192")) - int(os.environ.get("HMX_VTCM_KB", "256")) - 72)))
  import numpy as np, onnx
  from tinygrad import Tensor, dtypes
  from tinygrad.nn.onnx import OnnxRunner
  from tinygrad.runtime.support import dsp_graph, dsp_graph_v65

  model = onnx.load(args.onnx)
  out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
  changed = False
  if os.environ.get("GEMM_PRETRANSPOSE", "1") != "0" and (n := pretranspose_gemm_weights(model)):
    print(f"pre-transposed the constant B of {n} transB=1 Gemms", file=sys.stderr)
    changed = True
  if n := constants_to_initializers(model):
    print(f"{n} Constant nodes made initializers", file=sys.stderr)
    changed = True
  if changed:
    onnx.save(model, out / "model.onnx")
    args.onnx = str(out / "model.onnx")
  init = {i.name for i in model.graph.initializer}
  graph_inputs = [vi for vi in model.graph.input if vi.name not in init]
  run = OnnxRunner(args.onnx)
  # constants first: lowerings read them with .numpy(), and inside the capture no kernel runs
  for v in run.graph_values.values():
    if isinstance(v, Tensor): v.realize()
  # and the python values the runner reads from small constants (Gather indices, Split sizes, Reshape shapes): read inside the
  # capture, .tolist() runs a copy kernel that doesn't run there and reads zeros -- BEVFormer's 3-layer encoder took every
  # layer's camera value maps from one merged value_proj output with scalar Gather indices: all three layers read slice 0
  from tinygrad.nn.onnx import _to_python_const
  for name, v in run.graph_values.items():
    if name in run.const_names and isinstance(v, Tensor) and v.numel() <= 4096: run._python_const_cache[name] = _to_python_const(v)
  if hasattr(run, "_qdq_grid"): run._qdq_grid()  # the QDQ lowering (when the graph has one) is built on the first call: build it here
  feeds = {}
  for vi in graph_inputs:
    tt = vi.type.tensor_type
    shape = [d.dim_value for d in tt.shape.dim]
    if any(d <= 0 for d in shape): sys.exit(f"input {vi.name} has a dynamic shape {shape}; export with static shapes")
    feeds[vi.name] = Tensor(np.zeros(shape, ONNX_DTYPES[tt.elem_type])).realize()

  # inside the capture nothing runs, so its kernels needn't compile for the mock (clang-19's v68+ HVX path, the qemu link): a
  # stand-in binary keyed by the source, as bevformer_tiny/tinygrad_port/capture_msda.py's NO_COMPILE in onnxsim. The device
  # build compiles every kernel with the Hexagon SDK's compiler. (clang-19 crashed on the hand MSDA kernel's HVX body, and the
  # mock link has no memcpy.)
  import hashlib
  from tinygrad.runtime import ops_dsp
  real_compile = ops_dsp.DSPCompiler.compile
  ops_dsp.DSPCompiler.compile = lambda self, src: b"NOCOMPILE" + hashlib.sha256(src.encode()).digest()
  holder: dict = {}
  def fn():
    r = run(feeds)
    outs = [r[vo.name] for vo in model.graph.output] if isinstance(r, dict) else [r]
    holder["y"] = (outs[0] if len(outs) == 1 else Tensor.cat(*[o.cast(dtypes.float32).flatten() for o in outs])).contiguous().realize()
  calls, bufs = dsp_graph.capture(fn)
  if os.environ.get("DSP_GRAPH_SINK", "1") != "0": calls = sink_calls(calls, bufs)
  if os.environ.get("DSP_GRAPH_HMX_HOIST", "1") != "0": calls = hoist_out_of_hmx_chains(calls, bufs)
  if os.environ.get("DSP_GRAPH_RAW_PULL", "1") != "0": calls = pull_raw_tile_readers(calls, bufs)
  calls = keep_attention_vtcm(calls, bufs)
  ops_dsp.DSPCompiler.compile = real_compile
  y = holder["y"]
  used = {id(b[0].base) for b in bufs.values()}
  ins = [(name, t) for name, t in feeds.items() if id(t.uop.buffer.base) in used]
  info = dsp_graph_v65.emit(out, calls, bufs, [t.uop.buffer for _, t in ins], y.uop.buffer)
  print(f"emitted {info['calls']} calls ({info['threaded_calls']} threaded), {info['kernels']} kernels, hmx={info['hmx']}, "
        f"{info['vtcm_regions']} regions in {info['vtcm_bytes']/1e6:.2f} MB VTCM, "
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
