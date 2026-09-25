"""Static QDQ ONNX graphs (onnxsim full_qdq + quantized_io NHWC input: uint8 activations, per-channel int8 weights) lowered
for the DSP's HMX int8 TensorCore, bit-exact against ONNX Runtime CPU. OnnxRunner uses this when every node is covered.

What ORT computes after its QDQ fusion, and what this reproduces exactly:
  Conv (k x k, stride 1/2, pad k//2, per-channel int8, int32 bias, a Relu folded into its output Q):
    y = clamp(rne(fp32(fp32(acc) * M)) + zy, lo, 255),  M = fp32(fp32(sx sw) / sy),  acc = sum (x - zx) w + b
  Add (QLinearAdd): y = clamp(rne(rb*b + (ra*a + fixed))) in separate fp32 ops (ops_dsp.hmx_qlinear_add)
  MaxPool 3x3 s2 p1 (same scale in and out)

Layout: every activation is a padded flat grid -- NHWC, a ring of pixels holding its zero point, flattened to (rows, C) at a
row stride Wp (W + 2 by default), plus a tail its consumers' windows may overrun into. A k x k / stride s conv is the ordinary
(A * W).sum() over the windowed view A(p, dy, dx, c) = x[base + s*p + dy*Wp + dx, c] (movement ops only), on an output grid
at the input's row stride, so the pixel axis is one axis for the TensorCore (TC_OPT=1: two reduce axes). A stride-1 conv's
grid is written straight into the next padded grid (its garbage columns land on the pad columns, then the ring is rewritten);
other convs crop through a copy. The stride-2 stem runs as a stride-1 conv on the input's 2x2 phase split (4 phases x 8 padded
channels = one 32-byte K block per tap).
"""
from __future__ import annotations
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv

f32 = np.float32
r64 = lambda n: (n + 63) // 64 * 64

class QT:
  """a quantized activation: channels, height, width, scale, zero point"""
  def __init__(self, name, c, h, w, scale, zp): self.name, self.c, self.h, self.w, self.scale, self.zp = name, c, h, w, f32(scale), int(zp)

def _np(t) -> np.ndarray: return t.numpy() if isinstance(t, Tensor) else np.asarray(t)

def qdq_program(nodes, values:dict, inputs:dict):
  """OnnxRunner's parsed graph -> (ops, xin, yout, in_name, out_name), or raises NotImplementedError naming the first
  unsupported node. ops: dicts {op: conv|add|maxpool, ...} in graph order; activations are QT"""
  prod = {o: n for n in nodes for o in n.outputs}
  cons: dict[str, list] = {}
  for n in nodes:
    for x in n.inputs: cons.setdefault(x, []).append(n)
  def dq(name):
    n = prod.get(name)
    if n is None or n.op != "DequantizeLinear": raise NotImplementedError(f"{name}: expected a DequantizeLinear")
    return n.inputs[0], _np(values[n.inputs[1]]), (_np(values[n.inputs[2]]) if len(n.inputs) > 2 and n.inputs[2] else np.zeros((), np.uint8))
  def q_after(name):
    qn = cons.get(name, [])
    # a Relu before a zero-point-0 QuantizeLinear is what the uint8 saturation already does (onnxsim's full_qdq folds it away)
    if len(qn) == 1 and qn[0].op == "Relu":
      r, qn = qn[0], cons.get(qn[0].outputs[0], [])
      if len(qn) == 1 and qn[0].op == "QuantizeLinear" and int(_np(values[qn[0].inputs[2]])) != 0:
        raise NotImplementedError(f"{name}: Relu before a QuantizeLinear with a nonzero zero point")
    if len(qn) != 1 or qn[0].op != "QuantizeLinear": raise NotImplementedError(f"{name}: output must go straight into one QuantizeLinear")
    return qn[0]
  if len(inputs) != 1: raise NotImplementedError("one graph input")
  gin, spec = next(iter(inputs.items()))
  tr = cons.get(gin, [])
  if len(tr) != 1 or tr[0].op != "Transpose" or list(tr[0].opts.get("perm", [])) != [0, 3, 1, 2]:
    raise NotImplementedError("expected a uint8 NHWC graph input followed by Transpose(0,3,1,2) (quantized_io nhwc)")
  _, h, w, c = spec.shape
  nchw = tr[0].outputs[0]
  dqin = cons[nchw][0]
  acts = {nchw: QT(nchw, c, h, w, _np(values[dqin.inputs[1]]), _np(values[dqin.inputs[2]]))}
  ops: list[dict] = []
  def qt(qn, c, h, w):
    t = QT(qn.outputs[0], c, h, w, _np(values[qn.inputs[1]]), _np(values[qn.inputs[2]]))
    acts[t.name] = t
    return t
  for n in nodes:
    if n.op in ("QuantizeLinear", "DequantizeLinear", "Transpose", "Relu") and (n.op != "Transpose" or n is tr[0]):
      if n.op == "Relu" and not (n.inputs[0] in prod and prod[n.inputs[0]].op in ("Conv", "Add")):
        raise NotImplementedError(f"Relu {n.outputs[0]} not after a Conv / Add")
      continue
    if n.op == "Conv":
      k, st = list(n.opts.get("kernel_shape", [])), list(n.opts.get("strides", [1, 1]))
      pads, grp = list(n.opts.get("pads", [0, 0, 0, 0])), n.opts.get("group", 1)
      if not k or k[0] != k[1] or st[0] != st[1] or st[0] not in (1, 2) or len(set(pads)) != 1 or pads[0] != k[0] // 2 or grp != 1:
        raise NotImplementedError(f"Conv {n.outputs[0]}: kernel {k} strides {st} pads {pads} group {grp}")
      xs, _, _ = dq(n.inputs[0])
      wq_name, sw, _ = dq(n.inputs[1])
      bq = _np(values[dq(n.inputs[2])[0]]) if len(n.inputs) > 2 and n.inputs[2] else None
      x, wq = acts[xs], _np(values[wq_name]).astype(np.int8)
      ho, wo = (x.h - 1) // st[0] + 1, (x.w - 1) // st[0] + 1
      y = qt(q_after(n.outputs[0]), wq.shape[0], ho, wo)
      ops.append(dict(op="conv", x=x, y=y, k=k[0], s=st[0], wq=wq, bq=(bq if bq is not None else np.zeros(wq.shape[0])).astype(np.int32),
                      swa=np.broadcast_to(np.asarray(sw, f32), (wq.shape[0],)).copy()))
    elif n.op == "Add":
      a, b = acts[dq(n.inputs[0])[0]], acts[dq(n.inputs[1])[0]]
      ops.append(dict(op="add", a=a, b=b, y=qt(q_after(n.outputs[0]), a.c, a.h, a.w)))
    elif n.op == "MaxPool":
      k, st, pads = list(n.opts.get("kernel_shape", [])), list(n.opts.get("strides", [1, 1])), list(n.opts.get("pads", [0, 0, 0, 0]))
      x = acts[dq(n.inputs[0])[0]]
      if k != [3, 3] or st != [2, 2] or pads[0] != 1: raise NotImplementedError(f"MaxPool {n.outputs[0]}: {k} {st} {pads}")
      y = qt(q_after(n.outputs[0]), x.c, (x.h - 1) // 2 + 1, (x.w - 1) // 2 + 1)
      if y.scale != x.scale or y.zp != x.zp: raise NotImplementedError(f"MaxPool {n.outputs[0]}: requantizing MaxPool")
      ops.append(dict(op="maxpool", x=x, y=y))
    else: raise NotImplementedError(f"{n.op} ({n.outputs[0]})")
  return ops, acts[nchw], ops[-1]["y"], gin

def add_consts(ta:QT, tb:QT, ty:QT):
  """ORT's QLinearAdd (MLAS): ra = sa/sy, rb = sb/sy, fixed = zy - (ra*za + rb*zb), all fp32"""
  ra, rb = f32(ta.scale / ty.scale), f32(tb.scale / ty.scale)
  return ra, rb, f32(f32(ty.zp) - f32(f32(ra * f32(ta.zp)) + f32(rb * f32(tb.zp))))

def add_ref(a, ta, b, tb, ty):
  ra, rb, fixed = add_consts(ta, tb, ty)
  v = f32(rb * b.astype(f32)) + f32(f32(ra * a.astype(f32)) + fixed)
  return np.clip(np.rint(v).astype(np.int64), 0, 255).astype(np.uint8)

class Act:
  """a padded flat grid: t (L, C) uint8, H x W pixels inside a ring of `pad` (wider on the right when Wp > W + 2 pad)"""
  def __init__(self, t, H, W, C, pad, zp, Wp=None):
    self.t, self.H, self.W, self.C, self.pad, self.zp = t, H, W, C, pad, zp
    self.Wp = Wp or W + 2 * pad

def need_rows(H, W, pad, k, s, Wp=None):
  """rows a k x k / s window over a padded grid reads (output grid Ho x Wp padded to 64, window centered in the ring)"""
  Wp, Ho = Wp or W + 2 * pad, (H - 1) // s + 1
  return (pad - k // 2) * (Wp + 1) + s * (r64(Ho * Wp) - 1) + (k - 1) * Wp + k

def window(x:Tensor, Wp, k, s, P64, base) -> Tensor:
  """(L, C) -> (P64, dy, dx, C): x[base + s*p + dy*Wp + dx]"""
  C = x.shape[1]
  v = x[base:].permute(1, 0)._pool((k,), 1, 1)                     # (C, L', k): dx
  v = v.permute(0, 2, 1)._pool((k,), s, Wp)                        # (C, k, P', k): dy, stride s over the grid
  return v.shrink(((0, C), (0, k), (0, P64), (0, k))).permute(2, 3, 1, 0)

def canon(g:Tensor, Ho, Wo, Wg, C, zp, L) -> Tensor:
  """an output grid (row stride Wg, Ho x Wo valid) -> a padded flat grid of L rows (ring 1 and tail = zp), one copy"""
  x = g[: Ho * Wg].reshape(Ho, Wg, C)[:, :Wo].pad(((1, 1), (1, 1), (0, 0)), value=zp).reshape(-1, C)
  return x.pad(((0, L - x.shape[0]), (0, 0)), value=zp).contiguous()

def into_grid(y:Tensor, H, W, C, zp, L, Wp=None) -> Tensor:
  """a stride-1 output grid (row stride Wp) straight into the next padded grid at offset Wp + 1: pixel (i, j) belongs at
  (i+1)*Wp + (j+1) = p + Wp + 1, so the grid's garbage columns land on the pad columns; then the ring is rewritten with zp"""
  Wp, P64 = Wp or W + 2, y.shape[0]
  assert L >= Wp + 1 + P64
  g = Tensor.empty(L, C, dtype=dtypes.uint8, device=y.device)
  g[Wp + 1: Wp + 1 + P64].assign(y)
  v = g[: (H + 2) * Wp].reshape(H + 2, Wp, C)
  for sl in ((slice(0, 1), slice(0, Wp)), (slice(H + 1, H + 2), slice(0, Wp)), (slice(1, H + 1), slice(0, 1)), (slice(1, H + 1), slice(W + 1, Wp))):
    v[sl].assign(Tensor.full(v[sl].shape, zp, dtype=dtypes.uint8, device=y.device))
  return g

def _requant(acc:Tensor, m:Tensor, zy:int) -> Tensor:
  # ORT's output, which the DSP renderer lowers per 32-lane row to its exact integer requantization (lo = zy under a Relu
  # folded into Q is the same clip, since zy = 0 there)
  return ((acc.cast(dtypes.float32) * m).round() + float(zy)).clip(0, 255).cast(dtypes.uint8)

class QDQGridNet:
  """the lowered graph: __call__(x (1, H, W, C) uint8 NHWC) -> (1, C, H, W) uint8, as ORT's output"""
  def __init__(self, nodes, values, inputs, device:str|None=None):
    self.ops, self.xin, self.yout, self.in_name = qdq_program(nodes, values, inputs)
    self.device = device
    st = self.ops[0]
    if not (st["op"] == "conv" and st["x"] is self.xin and st["s"] == 2 and self.xin.c <= 8):
      raise NotImplementedError("expected a stride-2 stem conv on the (<= 8 channel) input first")
    # rows each activation holds: the max over its consumers' windows, its own ring, and grids written into it
    L: dict[str, int] = {}
    def need(name, n): L[name] = max(L.get(name, 0), n)
    self.sk = (st["k"] + 1) // 2
    self.Hs, self.Ws = (self.xin.h + 2 * (st["k"] // 2) + 1) // 2, (self.xin.w + 2 * (st["k"] // 2) + 1) // 2
    need("S", max(self.Hs * self.Ws, r64(st["y"].h * self.Ws) - 1 + (self.sk - 1) * self.Ws + self.sk))
    Wg = {st["y"].name: self.Ws}  # grids whose row stride isn't W + 2: the stem's output
    for o in self.ops:
      if o["op"] == "conv" and o["x"] is not self.xin: need(o["x"].name, need_rows(o["x"].h, o["x"].w, 1, o["k"], o["s"]))
      elif o["op"] == "maxpool": need(o["x"].name, need_rows(o["x"].h, o["x"].w, 1, 3, 2, Wg.get(o["x"].name)))
      y = o["y"]
      Wp = Wg.get(y.name, y.w + 2)
      need(y.name, (y.h + 2) * Wp)
      if o["op"] == "conv" and (o["s"] == 1 or o is st): need(y.name, Wp + 1 + r64(y.h * Wp))
    for _ in range(2):  # an Add runs over whole buffers: its inputs and output the same length, a multiple of 128 bytes
      for o in self.ops:
        if o["op"] == "add":
          n = max(L[o[k].name] for k in ("a", "b", "y"))
          while (n * o["y"].c) % 128: n += 1
          for k in ("a", "b", "y"): L[o[k].name] = n
    self.L, self.Wg = L, Wg
    self._build_consts()

  def _t(self, a:np.ndarray) -> Tensor: return Tensor(a, device=self.device)

  def _build_consts(self):
    # weights in (dy, dx, C, N) with the activation zero point folded into the bias, M in ORT's fp32
    xin, st, sk = self.xin, self.ops[0], self.sk
    self.consts: list[Tensor] = []
    self.w: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
    for i, o in enumerate(self.ops):
      if o["op"] != "conv": continue
      wq, bq, N, k = o["wq"], o["bq"].astype(np.int64), o["wq"].shape[0], o["k"]
      if o is st:  # (ay, ax, by, bx, c, n): tap (2 ay + by, 2 ax + bx) of the phase split
        wk = np.zeros((sk, sk, 2, 2, 8, N), np.int8)
        for dy in range(k):
          for dx in range(k): wk[dy // 2, dx // 2, dy % 2, dx % 2, :xin.c] = wq[:, :, dy, dx].T
        wk = wk.reshape(sk, sk, 32, N)
      else: wk = np.ascontiguousarray(wq.transpose(2, 3, 1, 0))
      bias = (bq - int(o["x"].zp) * wq.reshape(N, -1).astype(np.int64).sum(1)).astype(np.int32)
      m = (f32(o["x"].scale) * o["swa"].astype(f32) / f32(o["y"].scale)).astype(f32)
      self.w[i] = (self._t(wk), self._t(bias), self._t(m))
      self.consts += list(self.w[i])

  def _conv(self, a:Act, i:int, k:int, s:int, base:int, Wp:int, C:int, P64:int) -> Tensor:
    W_, B_, M_ = self.w[i]
    N = W_.shape[3]
    v = window(a.t, Wp, k, s, P64, base)
    acc = (v.reshape(P64, 1, k, k, C).cast(dtypes.int32) * W_.permute(3, 0, 1, 2).reshape(1, N, k, k, C).cast(dtypes.int32)).sum((2, 3, 4))
    return _requant(acc + B_, M_, self.ops[i]["y"].zp)

  def __call__(self, x:Tensor) -> Tensor:
    from tinygrad.runtime.ops_dsp import hmx_qlinear_add
    xin, st, L = self.xin, self.ops[0], self.L
    # stem: the 2x2 phase split of the padded input (one copy), a stride-1 sk x sk conv on it, straight into its consumer's grid
    k0, p0, Hs, Ws, sk = st["k"], st["k"] // 2, self.Hs, self.Ws, self.sk
    xp = x.reshape(xin.h, xin.w, xin.c).pad(((p0, 2 * Hs - xin.h - p0), (p0, 2 * Ws - xin.w - p0), (0, 8 - xin.c)), value=xin.zp)
    S = xp.reshape(Hs, 2, Ws, 2, 8).permute(0, 2, 1, 3, 4).reshape(Hs * Ws, 32)
    S = Act(S.pad(((0, L["S"] - Hs * Ws), (0, 0)), value=xin.zp).contiguous(), Hs, Ws, 32, 0, xin.zp, Ws)
    yt = st["y"]
    y = self._conv(S, 0, sk, 1, 0, Ws, 32, r64(yt.h * Ws))
    vals = {yt.name: Act(into_grid(y, yt.h, yt.w, yt.c, yt.zp, L[yt.name], Ws), yt.h, yt.w, yt.c, 1, yt.zp, Ws)}
    for i, o in enumerate(self.ops[1:], start=1):
      yt = o["y"]
      if o["op"] == "conv":
        a, k, s = vals[o["x"].name], o["k"], o["s"]
        Ho, Wo = yt.h, yt.w
        y = self._conv(a, i, k, s, (a.pad - k // 2) * (a.Wp + 1), a.Wp, a.C, r64(Ho * a.Wp))
        if s == 1 and a.Wp == a.W + 2 and getenv("QDQ_INTO_GRID", 1): t = into_grid(y, Ho, Wo, yt.c, yt.zp, L[yt.name])
        # a copy crops the grid: materialized first (fused with the crop, tinygrad splits the pixel axis again)
        else: t = canon(y.contiguous(), Ho, Wo, a.Wp, yt.c, yt.zp, L[yt.name])
        vals[yt.name] = Act(t, Ho, Wo, yt.c, 1, yt.zp)
      elif o["op"] == "add":
        a, b = vals[o["a"].name], vals[o["b"].name]
        assert (a.H, a.W, a.C, a.Wp) == (b.H, b.W, b.C, b.Wp) and a.t.shape == b.t.shape
        # the pads hold za, zb: they must come out as zy (they are the next conv's padding)
        pz = add_ref(np.array([o["a"].zp], np.uint8), o["a"], np.array([o["b"].zp], np.uint8), o["b"], yt)[0]
        if pz != yt.zp: raise NotImplementedError(f"Add pads: {o['a'].zp} + {o['b'].zp} -> {pz}, not zy {yt.zp}")
        ra, rb, fixed = add_consts(o["a"], o["b"], yt)
        vals[yt.name] = Act(hmx_qlinear_add(a.t, b.t, float(ra), float(rb), float(fixed)), a.H, a.W, a.C, 1, yt.zp, a.Wp)
      else:  # maxpool 3x3 s2 p1: the ring holds the input's zero point, 0 (post-Relu): the pooling minimum
        a = vals[o["x"].name]
        if a.zp != 0: raise NotImplementedError("MaxPool needs a zero-point-0 input (its ring is the pooling minimum)")
        y = window(a.t, a.Wp, 3, 2, r64(yt.h * a.Wp), 0).max(axis=(1, 2)).contiguous()
        vals[yt.name] = Act(canon(y, yt.h, yt.w, a.Wp, yt.c, yt.zp, L[yt.name]), yt.h, yt.w, yt.c, 1, yt.zp)
    out = vals[self.yout.name]
    y = out.t[: (out.H + 2) * out.Wp].reshape(out.H + 2, out.Wp, out.C)[1:out.H + 1, 1:out.W + 1]
    return y.permute(2, 0, 1).reshape(1, out.C, out.H, out.W).contiguous()

def qdq_emulate(net:QDQGridNet, x_nhwc:np.ndarray) -> np.ndarray:
  """ORT CPU's semantics for the lowered program, in numpy (the tests' reference): x (1, H, W, C) -> (1, C, H, W)"""
  vals = {net.xin.name: x_nhwc[0].transpose(2, 0, 1)}
  for o in net.ops:
    if o["op"] == "conv":
      xq, k, s = vals[o["x"].name], o["k"], o["s"]
      c, h, w = xq.shape
      p = k // 2
      xp = np.pad(xq.astype(np.int64) - o["x"].zp, ((0, 0), (p, p), (p, p)))
      ho, wo = (h - 1) // s + 1, (w - 1) // s + 1
      acc = np.zeros((o["wq"].shape[0], ho, wo), np.int64)
      for dy in range(k):
        for dx in range(k):
          acc += (o["wq"][:, :, dy, dx].astype(np.int64) @ xp[:, dy:dy + s * ho:s, dx:dx + s * wo:s].reshape(c, -1)).reshape(-1, ho, wo)
      acc += o["bq"].astype(np.int64)[:, None, None]
      m = (f32(o["x"].scale) * o["swa"].astype(f32) / f32(o["y"].scale)).astype(f32)
      vals[o["y"].name] = np.clip(np.rint(acc.astype(f32) * m[:, None, None]).astype(np.int64) + o["y"].zp, 0, 255).astype(np.uint8)
    elif o["op"] == "add": vals[o["y"].name] = add_ref(vals[o["a"].name], o["a"], vals[o["b"].name], o["b"], o["y"])
    else:
      xq = vals[o["x"].name]
      c, h, w = xq.shape
      ho, wo = (h - 1) // 2 + 1, (w - 1) // 2 + 1
      xp = np.pad(xq, ((0, 0), (1, 1), (1, 1)), constant_values=0)
      y = np.zeros((c, ho, wo), np.uint8)
      for dy in range(3):
        for dx in range(3): y = np.maximum(y, xp[:, dy:dy + 2 * ho:2, dx:dx + 2 * wo:2])
      vals[o["y"].name] = y
  return vals[net.yout.name][None]
