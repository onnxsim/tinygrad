from dataclasses import replace, dataclass
import itertools, functools
from typing import Iterator
from tinygrad.helpers import getenv, DISABLE_FAST_IDIV, TRANSCENDENTAL, SPEC, DEBUG, VIZ, IMAGE, NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC
from tinygrad.helpers import ALLOW_TF32, DEFAULT_FLOAT, DEFAULT_INT, NUM_CPU_THREADS, TC_SELECT, TC_OPT, TracingKey, Context, panic, prod
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, Ops, UPat, rewrite_group, KernelInfo, ProgramInfo, GroupOp, AxisType
from tinygrad.uop.weak import pm_lower_index_dtype, pm_commit_weak, pm_cast_weak
from tinygrad.uop.render import pyrender
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program
from tinygrad.renderer import Renderer, Estimates
from tinygrad.renderer.isa import ISARenderer, IselContext, PreRegAllocContext
from tinygrad.dtype import dtypes, AddrSpace

# import all pattern matchers here
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, symbolic, pm_move_where_on_load, pm_clean_up_group_sink, pm_remove_invalid
from tinygrad.uop.movement import mop_cleanup
from tinygrad.codegen.decomp.dtype import pm_dtype_decomps
from tinygrad.codegen.decomp.op import get_late_rewrite_patterns, get_simplifying_rewrite_patterns
from tinygrad.codegen.decomp.transcendental import get_transcendental_patterns
from tinygrad.codegen.late.coalesce import indexing_simplify
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen.late.gater import pm_move_gates_from_index
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse, pm_reduce_unparented
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.rangeify import pm_mops
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
from tinygrad.codegen.late.coalesce import memory_coalescing, pm_simplify_add_image
from tinygrad.helpers import all_same, flatten, argsort, partition
from tinygrad.uop.ops import _broadcast_shape, identity_element
from tinygrad.schedule.rangeify import BufferizeOpts

def do_number_param(ctx:list[int], x:UOp):
  if x.arg.slot != -1: return None
  ctx[0] += 1
  return x.replace(arg=replace(x.arg, slot=ctx[0]-1))

pm_number_params = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), do_number_param),
])

def build_range_map(sink:UOp, store_order:bool=False) -> dict[int, int]:
  rngs = [x for x in sink.toposort() if x.op is Ops.RANGE and x.arg[1] in {AxisType.UNROLL, AxisType.UPCAST}]
  if store_order and (ordered:=_store_stride_order(sink, rngs)) is not None: rngs = ordered
  return {x.arg[0]:i for i,x in enumerate(rngs)}

def _store_stride_order(sink:UOp, rngs:list[UOp]) -> list[UOp]|None:
  # the expanded axes ordered by their stride in the (single) global store, largest first, so vector lanes -- and a tensor
  # core's register accumulator -- follow the output's memory order. Toposort order depends on how the index expression
  # happens to be built: for an HMX tile with a single M tile, or a permuted output, it put the N axes before M and the
  # accumulator came out column-major (every epilogue element a scalar lane gather). None (keep toposort) unless every
  # axis has a constant stride
  stores = [x for x in sink.toposort() if x.op is Ops.STORE and x.src[0].op is Ops.INDEX and len(x.src[0].src) >= 2]
  if len(stores) != 1 or not any(x.op is Ops.WMMA for x in sink.toposort()): return None
  idx = stores[0].src[0].src[1]
  zero = {r:r.const_like(0) for r in idx.toposort() if r.op is Ops.RANGE}
  base = idx.substitute(zero).simplify()
  strides = {}
  for r in rngs:
    if r not in zero:
      strides[r] = 0
      continue
    d = (idx.substitute({**zero, r:r.const_like(1)}).simplify() - base).simplify()
    if d.op is not Ops.CONST: return None
    strides[r] = d.arg
  pos = {r:i for i,r in enumerate(rngs)}
  return sorted(rngs, key=lambda r: (-strides[r], pos[r]))

def expand_reduce(r:UOp):
  range_srcs = []
  new_axes = []
  for u in r.src[1:]:
    if u.op == Ops.RANGE:
      range_srcs.append(u)
    else:
      for i,s in enumerate(u.shape):
        if s > 1: new_axes.append(i)
  if len(new_axes) == 0: return None
  assert r.arg[1] == 0
  # permute so new_axes come to front, then reduce
  perm = tuple(new_axes) + tuple(i for i in range(len(r.src[0].shape)) if i not in new_axes)
  out_shape = tuple([1 if i in new_axes else s for i,s in enumerate(r.src[0].shape)])
  return r.src[0].permute(perm).reduce(*range_srcs, arg=(r.arg[0], len(new_axes))).reshape(out_shape)

def contract_axis(ctx:dict[int, int], u:UOp, arg):
  permute_tail = [ctx[rn] for rn,_ in arg]
  permute_head = [i for i in range(len(u.shape)) if i not in permute_tail]
  out = u.permute(permute_head+permute_tail)
  return out.reshape(*out.shape[:len(permute_head)], -1)

def unroll_axis(ctx:dict[int, int], u:UOp, arg):
  permute_tail = [ctx[rn] for rn,_ in arg]
  out = u.reshape(*u.shape[:-1], *[nm for _,nm in arg])
  permute_head = [i for i in range(len(out.shape)) if i not in permute_tail]
  return out.permute(argsort(permute_head+permute_tail))

def expand_wmma(ctx:dict[int, int], u:UOp):
  if u.arg[4] is None: return None
  in0, in1, out0 = u.arg[4]
  wmma = u.replace(src=(contract_axis(ctx, u.src[0], in0), contract_axis(ctx, u.src[1], in1), u.src[2]),
                   arg=(*u.arg[:4], None))
  return unroll_axis(ctx, wmma, out0)

expander2 = PatternMatcher([
  (UPat(Ops.REDUCE, name="r"), expand_reduce),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp.const(tuple(range(r.vmax+1)), r.dtype) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg[0]] else 1 for i in range(len(ctx))])) if r.arg[0] in ctx else None),
  (UPat(Ops.WMMA, name="u"), expand_wmma),
])+pm_flatten_range+mop_cleanup

def expand_broadcast(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  return x.replace(src=tuple([u.expand(shape) for u in x.src]))

def broadcast_and_devec_wmma(b:UOp):
  shapes = [u.shape[:-1] for u in b.src]
  if all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  src_expanded = tuple([u.expand(shape+(u.shape[-1],)) for u in b.src])
  src = []
  for idx in itertools.product(*[range(i) for i in b.shape[:-1]]):
    src.append(b.replace(src=tuple([x.index(*idx) for x in src_expanded])))
  return UOp.stack(*src).reshape(b.shape)

pm_wmma_add = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
   lambda add, wmma: UOp(wmma.op, src=(wmma.src[0], wmma.src[1], wmma.src[2]+add), arg=wmma.arg)),
  # push permute/reshape to the other side of the add
  # (a bare reshape too: a v65 vrmpy tensor core's 32 lanes reshaped to the output axes were added to the loop accumulator outside
  # the WMMA, so every K step zeroed a register, ran the vrmpy from zero and added it back, instead of one accumulating vrmpy)
  (UPat(Ops.RESHAPE, src=(UPat(Ops.WMMA, name="wmma"), UPat()), name="reshape") + UPat.var("add"),
    lambda wmma,reshape,add: (wmma + add.reshape(wmma.shape)).reshape(reshape.shape)
      if wmma.arg[0] == (32, 1, 4) and add.shape == reshape.shape else None),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.WMMA, name="wmma"),), name="permute") + UPat.var("add"),
    lambda wmma,permute,add: (wmma + add.permute(argsort(permute.arg))).permute(permute.arg)),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.RESHAPE, src=(UPat(Ops.WMMA, name="wmma"), UPat()), name="reshape"),), name="permute") + UPat.var("add"),
    lambda wmma,reshape,permute,add: (wmma + add.permute(argsort(permute.arg)).reshape(wmma.shape)).reshape(reshape.shape).permute(permute.arg)),
])

pm_expand_broadcast = pm_wmma_add+PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), expand_broadcast),
  (UPat(Ops.WMMA, name="b"), broadcast_and_devec_wmma),
])

def do_devectorize(b:UOp):
  if b.shape == (): return None
  # broadcasting needs to be already unpacked, Invalid matches any dtype and shape
  if not all(x.shape == b.shape or x.base.is_invalid for x in b.src): return None
  src = []
  for idx_c in itertools.product(*[[UOp.const(i) for i in range(x)] for x in b.shape]):
    src.append(b.replace(dtype=None, src=tuple(x.base if x.base.is_invalid else x.index(*idx_c) for x in b.src)))
  return UOp.stack(*src).reshape(b.shape) if b.op is not Ops.STORE else UOp.group(*src)

def do_stack_wmma(u:UOp):
  if all(x.op in (Ops.STACK, Ops.WMMA) for x in u.src): return None
  assert len(u.shape) == 1
  src = []
  for b in u.src:
    if b.op != Ops.STACK:
      src.append(UOp.stack(*[b.index(i) for i in range(b.max_numel())]))
    else:
      src.append(b)
  return u.replace(src=tuple(src))

ew_devectorizer = PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise, name="b"), do_devectorize),
])

devectorizer2 = mop_cleanup+pm_mops+PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise|{Ops.LOAD,Ops.STORE}, name="b"), do_devectorize),
  # INDEX without src is nothing (TODO: this should be in mop_cleanup)
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
  # unpack WMMA
  (UPat(Ops.WMMA, name="u"), do_stack_wmma),
  # stacked INDEX is many INDEX
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.STACK, name="s")), name="x"),
   lambda b,s,x: UOp.stack(*[x.replace(src=(b,u)) for u in s.src])),
  # INDEX into RESHAPE moves the RESHAPE
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.RESHAPE, name="s"))),
   lambda b,s: b.index(s.src[0]).reshape(s.shape)),
  # RESHAPE a void is removed (hack for AFTER)
  (UPat(Ops.RESHAPE, dtype=dtypes.void, name="x"), lambda x: x.src[0]),
  # reshape of a single element shaped value to scalar is an index
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(0) if x.marg == () and x.src[0].shape == (1,) else None),
  # EXPAND on scalar -> STACK
  (UPat(Ops.EXPAND, src=(UPat.var("x"), UPat()), name="out"),
   lambda x,out: UOp.stack(*([x]*out.max_numel())) if x.shape == () and out.shape == (out.max_numel(),) else None),
])

def fix_group_for_reduce(x:UOp):
  reduce_gfr, reduce_r = partition(x.src[1:], lambda u: u.op is Ops.RANGE and u.arg[1] == AxisType.GROUP_REDUCE)
  if len(reduce_gfr) == 0: return None

  # NOTE: if there's other locals here, we need them in the buffer too
  upstream_locals = [u for u in x.toposort() if u.op is Ops.RANGE and u.arg[1] == AxisType.LOCAL]

  # do only the non grouped reduces early
  ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
  reduce_loop = [x.replace(arg=(x.arg[0]+100, AxisType.REDUCE)) for x in reduce_gfr]
  buf = ret.bufferize(*upstream_locals, *reduce_gfr, arg=BufferizeOpts(reduce_gfr[0].arg[0], AddrSpace.LOCAL)).index(*upstream_locals, *reduce_loop)

  # do the final reduce (if/barrier are added in gpudims step)
  # NOTE: we remove all horizontal reduces here, they remain in the first reduce
  return buf.reduce(*reduce_loop, arg=(x.arg[0], 0))

@dataclass
class ReduceContext:
  acc_num: int = 0

def merge_reduce_ends(sink:UOp):
  # merge ENDs that share the same range and nesting context (only those created by reduce_to_acc)
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END and u.tag == "mergeable": range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.arg[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1: continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(next_axis + j, *rr.arg[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

def _uniform_gate(inp:UOp) -> tuple[UOp, UOp]|None:
  # where(g, t, 0) with g the same for every lane (a scalar, or a vector of one scalar repeated), possibly under a horizontal sum
  # of unrolled terms (REDUCE / PERMUTE / RESHAPE: the gate then holds for all of them) -> (g, the same without the where)
  w = inp
  while w.op in (Ops.REDUCE, Ops.PERMUTE, Ops.RESHAPE) and (w.op is not Ops.REDUCE or w.arg[0] is Ops.ADD): w = w.src[0]
  if w.op is not Ops.WHERE or (z:=w.src[2]).op not in (Ops.CONST, Ops.STACK) or \
     any(c.op is not Ops.CONST or c.arg != 0 for c in ([z] if z.op is Ops.CONST else z.src)): return None
  g = w.src[0]
  if g.op is Ops.STACK:
    if not g.src or any(x is not g.src[0] for x in g.src): return None
    g = g.src[0]
  if g.dtype != dtypes.bool or (g._shape is not None and any(d != 1 for d in g.shape)): return None
  return g, inp.substitute({w: w.src[1]})

def reduce_ranges_to_acc(ctx:ReduceContext, r:UOp):
  acc = UOp.placeholder_like(r, ctx.acc_num, AddrSpace.REG)
  ctx.acc_num += 1
  topo = r.src[0].toposort()
  ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
  input_ranges = tuple(x for x in topo if x.op is Ops.RANGE and x not in r.src[1:] and x not in ended_ranges)
  acc_init = acc.after(*input_ranges).store(UOp.const(identity_element(r.arg[0], r.dtype)))
  acc_initted = acc.after(acc_init, *r.src[1:])
  inp = r.src[0].reduce(arg=r.arg) if r.arg[1] else r.src[0]
  # GATED_REDUCE: a sum of where(g, t, 0) with g the same for every lane (a data-dependent mask: BEVFormer's per-(camera, query)
  # visibility) accumulates t under g -- a gated store, an IF after linearizing, and sink_into_ifs moves t's computation into it
  if r.arg[0] is Ops.ADD and getenv("GATED_REDUCE", 1) and (gt:=_uniform_gate(inp)) is not None:
    acc_out = acc_initted.store(acc_initted.alu(r.arg[0], gt[1]), gt[0]).end(*r.src[1:]).rtag("mergeable")
  else: acc_out = acc_initted.store(acc_initted.alu(r.arg[0], inp)).end(*r.src[1:]).rtag("mergeable")
  return acc.after(acc_out)

def expand_horizontal_reduce(r:UOp):
  inp = r.src[0]
  vals = [inp.index(*idx) for idx in itertools.product(*[range(inp.max_shape[a]) for a in range(r.arg[1])])]
  return functools.reduce(lambda x,y: x.alu(r.arg[0], y), vals)

pm_reduce_local = pm_wmma_add+PatternMatcher([
  # fix group for reduce
  (UPat(Ops.REDUCE, name="x"), fix_group_for_reduce),
  # remove reduces
  (UPat(Ops.REDUCE, src=(UPat(), UPat()), allow_any_len=True, name="r"), reduce_ranges_to_acc),
  (UPat(Ops.REDUCE, src=(UPat(),), name="r"), expand_horizontal_reduce),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
])+pm_clean_up_group_sink

def is_shape_changing_bitcast(u:UOp): return u.op is Ops.BITCAST and u.shape != u.src[0].shape
def maybe_load(u:UOp): return u.load() if u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG) else u
pm_add_loads = PatternMatcher([
  (UPat(GroupOp.Elementwise|{Ops.REDUCE,Ops.WMMA,Ops.STACK}, name="x"),
   lambda x: None if is_shape_changing_bitcast(x) else x.replace(src=tuple(map(maybe_load, x.src)))),
  # (a gated store's gate too: GATED_REDUCE's gate can be a bool buffer read directly)
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(src=(x.src[0], maybe_load(x.src[1]))+tuple(map(maybe_load, x.src[2:])))),
])

def add_local_buffer(ctx, x:UOp):
  buf = UOp.placeholder(x.max_shape, x.dtype, slot=next(ctx), addrspace=x.arg.addrspace)
  return buf.after(buf.index(*x.src[1:]).store(x.src[0]).end(*x.src[1:]))

pm_add_local_buffers = PatternMatcher([
  (UPat(Ops.STAGE, name="x"), add_local_buffer),
])+pm_mops

STAGE_SLOT = 1 << 16  # register tables' (Tensor.stage) buffer slots start here
def _staged_buf(u:UOp) -> bool:
  while u.op in (Ops.SHRINK, Ops.INDEX, Ops.CAST, Ops.AFTER): u = u.src[0]
  return u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG and u.arg.slot >= STAGE_SLOT

def add_reg_stage(ctx:tuple[Iterator[int], dict], idx:UOp, x:UOp):
  # a register table (Tensor.stage) before the expander: its loops, fully upcast, become vector stores, and the reduces see a STORE /
  # END (their accumulators open outside it), not a STAGE. Slots apart from the accumulators' (ReduceContext.acc_num). The consumer's
  # upcast / unrolled ranges the table depends on (a vectorized output axis) are lanes, not loops: one table per lane of them
  if x.arg.addrspace is not AddrSpace.REG: return None
  if x not in ctx[1]:
    own = sorted(UOp.sink(*x.src[1:]).ranges, key=lambda r: r.arg)
    ext = sorted([r for r in x.src[0].ranges if r not in own and r.arg[-1] in (AxisType.UPCAST, AxisType.UNROLL)], key=lambda r: r.arg)
    n, lane = prod(x.max_shape), UOp.const(0, dtypes.weakint)
    for r in ext: lane = lane * r.src[0] + r
    buf = UOp.placeholder((n * prod([r.src[0].arg for r in ext]),), x.dtype, slot=next(ctx[0]), addrspace=AddrSpace.REG)
    # its innermost axes upcast up to STAGE_LANES (128) lanes, the rest loops around them: what doesn't depend on those loops is
    # computed once, outside (the generated MSDA's floors per point, not per tap: 256 lanes in one go spilled HVX registers; the
    # decoder's 4 x 32 in one go: 0.58 -> 0.50 ms a call vs as a loop around 32 lanes)
    zero = {r:r.const_like(0) for r in own}
    def stride(r:UOp) -> int: return x.src[1].substitute({**zero, r:r.const_like(1)}).simplify().arg
    up, lanes = {}, 1
    for r in sorted(own, key=stride):
      if lanes * r.src[0].arg > getenv("STAGE_LANES", 128) and lanes > 1: break
      up[r], lanes = r.replace(arg=r.arg[:-1]+(AxisType.UPCAST,)), lanes * r.src[0].arg
    ctx[1][x] = (buf.after(buf.index(lane * n + x.src[1]).store(x.src[0]).end(*own).substitute(up)), lane * n)
  st, base = ctx[1][x]
  return idx.replace(src=(st, base + idx.src[1])+idx.src[2:])
pm_add_reg_stages = PatternMatcher([(UPat(Ops.INDEX, src=(UPat(Ops.STAGE, name="x"),), allow_any_len=True, name="idx"), add_reg_stage)])

# float ALUs need a float operand
# make that cast explicit before the decomps, which expand SIN/LOG2/EXP2 into float polynomials and assert a float operand
pm_cast_float_alu = PatternMatcher([
  (UPat((Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.RECIPROCAL), src=(UPat(name="x"),), name="u"),
   lambda u,x: u.replace(src=(x.cast(u.dtype),)) if x.dtype != u.dtype else None),
])

def _is_local_store(x:UOp): return x.op is Ops.STORE and x.addrspace is AddrSpace.LOCAL

def add_raw_barrier(after:UOp):
  # loads from a LOCAL buffer that depend (via AFTER) on stores to LOCAL memory need a workgroup barrier
  if after.addrspace is not AddrSpace.LOCAL: return None
  # one toposort over all the deps
  deps = UOp.sink(*after.src[1:]).toposort(gate=lambda x: x.op is not Ops.BARRIER)
  if not any(_is_local_store(x) for x in deps): return None
  return after.src[0].after(UOp(Ops.BARRIER, src=after.src[1:]))

def add_war_barrier(end:UOp):
  # a LOCAL buffer stored and loaded in the same loop needs a barrier at the end of the loop body
  rngs = [r for r in end.src[1:] if r.op is Ops.RANGE and r.arg[1] in (AxisType.REDUCE, AxisType.WEAK, AxisType.LOOP) and r.vmax > 0]
  if not rngs or end.src[0].op is Ops.BARRIER: return None
  sl = end.src[0].backward_slice_with_self
  # only stores that are inside this loop body (not in the backward slice through AFTER chains from other loops)
  store_bufs = {x.buf_uop for x in sl if _is_local_store(x) and any(r in x.ranges for r in rngs)}
  # a load whose buffer matches a local store's buffer is necessarily a local load
  if not (loads:=[x for x in sl if x.op is Ops.LOAD and x.src[0].buf_uop in store_bufs]): return None
  return end.replace(src=(UOp(Ops.BARRIER, src=(end.src[0], *loads)),)+end.src[1:])

pm_implicit_barriers = PatternMatcher([
  (UPat(Ops.AFTER, name="after"), add_raw_barrier),
  (UPat(Ops.END, name="end"), add_war_barrier),
])

pm_casted_consts = PatternMatcher([
  (UPat(Ops.CONST, dtypes.all, name="c"), lambda c: UOp.cconst(c.val, c.dtype)),
])

def full_rewrite_to_sink(ast:UOp, ren:Renderer, optimize:bool=True) -> UOp:
  if VIZ: graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(ast))
  if SPEC: type_verify(ast, spec_tensor)

  # resolve UNSHARDs (multi-device UNSHARDs are already resolved by the scheduler; this handles in-kernel shards, e.g. fragments)
  sink = graph_rewrite(ast, multi_pm, name="multi_pm")

  # preprocess
  sink = graph_rewrite(sink, pm_mops, name="early movement ops", bottom_up=True)

  # first we optimize
  if optimize:
    # collapse loads reduce (indexing by a tensor)
    sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")

    # split ranges
    sink = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")

    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, sym+pm_flatten_range, name="initial symbolic")

    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges")

    # do postrange optimization, BEAM or hand_coded_optimizations
    sink = apply_opts(sink, ren, beam=ast.arg.beam)

  # ** expander (expand_rewrite) **
  # reduce_unparented: a REDUCE whose src folded to a CONST (e.g. x*0) has no parented ranges, collapse it before the expander
  sink = graph_rewrite(sink, sym+pm_move_where_on_load+pm_flatten_range+pm_reduce_unparented, name="postopt symbolic")

  # register tables (Tensor.stage) to stores before the expander
  if any(u.op is Ops.STAGE and u.arg.addrspace is AddrSpace.REG for u in sink.backward_slice):
    sink = graph_rewrite(sink, pm_add_reg_stages, ctx=(itertools.count(STAGE_SLOT), {}), name="add register stages")

  # expand
  sink = graph_rewrite(sink, expander2, ctx=build_range_map(sink, ren.target.device == "DSP"), name="expander")

  # remove reduce
  sink = graph_rewrite(sink, mop_cleanup+pm_reduce_local, ctx=ReduceContext(), name="remove reduces")

  # add locals
  sink = graph_rewrite(sink, pm_add_local_buffers, ctx=itertools.count(0), name="add local buffers")

  # add gpu dims (late). this works after devectorize, but it's faster here
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")

  # **** optimizations are done, now we lower to actual code ****

  sink = graph_rewrite(sink, symbolic_simple+pm_expand_broadcast+pm_add_loads, name="*** expand broadcast / add loads")

  # devectorize
  sink = graph_rewrite(sink, symbolic_simple+devectorizer2+indexing_simplify, ctx=ren, name="devectorize2")

  # some coalescing misses without this
  sink = graph_rewrite(sink, sym, name="early symbolic")

  # do memory coalescing (late)
  sink = memory_coalescing(sink, ren)
  sink = graph_rewrite(sink, symbolic_simple+ew_devectorizer+pm_simplify_add_image,
                       name="add images", ctx=({}, ren), bottom_up=True)

  # extra symbolic before decomp. crashes without this?
  # NOTE: also run indexing_simplify here, while the index is still weakint and (x+y)*c -> x*c+y*c applies
  sink = graph_rewrite(sink, sym+indexing_simplify, name="extra symbolic")

  # lower index dtype
  # NOTE: we need indexing_simplify to remove the cast to long using the Invalid
  sink = graph_rewrite(sink, symbolic_simple+pm_lower_index_dtype+indexing_simplify, ctx={}, name="lower all index dtypes")

  # final symbolic before decomp
  sink = graph_rewrite(sink, symbolic, name="final symbolic")

  sink = graph_rewrite(sink, pm_cast_float_alu, name="cast float alu operands")

  # **** decomps ****

  # floordiv+mod / dtype decomp (early)
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_simplifying_rewrite_patterns(supported_ops)
  sink = graph_rewrite(sink, pm_decomp, name="early decompositions")

  # late decomps + move gates from unrenderable INVALID where
  sink = graph_rewrite(sink, pm_dtype_decomps+pm_commit_weak, ctx=(set(), ren), name="decomp dtypes")
  pm_decomp = pm_decomp+\
    get_late_rewrite_patterns(supported_ops, bool(DISABLE_FAST_IDIV))+\
    get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="late decompositions")
  sink = graph_rewrite(sink, pm_move_gates_from_index, name="move gates from index")

  # final rules for the renderer (without sym)
  extra_matcher = ren.extra_matcher if ren.extra_matcher is not None else PatternMatcher([])
  pm_final_rewrite = pm_commit_weak+pm_cast_weak+pm_decomp+extra_matcher+pm_split_ends
  sink = graph_rewrite(sink, pm_final_rewrite+pm_remove_invalid, ctx=ren, name="final rewrite")

  # spell every literal as a casted const CAST(dt, CONST(value))
  # TODO: remove once consts are always weak
  sink = graph_rewrite(sink, pm_casted_consts, name="casted consts", walk=True)

  # add implicit barriers (stores/loads through LOCAL memory ordered by AFTER or across loop iterations need workgroup barriers)
  sink = graph_rewrite(sink, pm_implicit_barriers, name="add implicit barriers")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

  # put unnumbered variable PARAMs in slots
  num_params = len([x for x in sink.toposort() if x.op is Ops.PARAM and x.arg.slot != -1])
  sink = graph_rewrite(sink, pm_number_params, ctx=[num_params], name="number params with -1", walk=True)

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Output AST")
  if SPEC: type_verify(sink, spec_program)

  # return the rewritten sink
  return sink

# inject IF/ENDIF. only needed if device doesn't support gated stores
pm_linearize_cleanups = PatternMatcher([
  # if statements are not allowed in the graph
  (UPat((Ops.IF, Ops.ENDIF)), lambda: panic(RuntimeError, "if not allowed in graph")),
  # gated STORE becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
  (UPat(Ops.STORE, name="u", src=(UPat((Ops.INDEX, Ops.SHRINK)).or_casted(), UPat(), UPat(name="gate", dtype=dtypes.bool))),
   lambda u, gate: ((st:=u.replace(src=u.src[0:2])), [mif:=UOp(Ops.IF, src=(gate, u.src[0])), st, UOp(Ops.ENDIF, src=(mif,))]))
])

# requires lst be toposorted. like graph rewrite, but for lines
def line_rewrite(lst:list[UOp], pm:PatternMatcher, ctx=None) -> list[UOp]:
  newlst = []
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    nu = u.replace(src=tuple([replaced.get(x, x) for x in u.src]))
    ret: tuple[UOp, list[UOp]] = pm.rewrite(nu, ctx) or (nu, [nu])
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

def merge_ifs(lst:list[UOp]) -> list[UOp]:
  # adjacent blocks under the same gate become one (a gated vector store split into per-lane stores: one IF each)
  merged, alias = [], {}
  for u in lst:
    if u.op is Ops.IF and merged and merged[-1].op is Ops.ENDIF and merged[-1].src[0].src[0] is u.src[0]:
      alias[u] = merged.pop().src[0]
      continue
    merged.append(u)
  if not alias: return lst
  def opener(i:UOp) -> UOp:
    while i in alias: i = alias[i]
    return i
  return [UOp(Ops.ENDIF, src=(opener(u.src[0]),)) if u.op is Ops.ENDIF else u for u in merged]

_SINKABLE = GroupOp.ALU | {Ops.LOAD, Ops.CAST, Ops.BITCAST, Ops.INDEX, Ops.STACK, Ops.SHRINK, Ops.CUSTOMI}
def sink_into_ifs(lst:list[UOp]) -> list[UOp]:
  """move the values only an IF block uses into it: from the IF's own scope (after the innermost RANGE / IF still open at it), pure
  ops and loads -- a load only when no store lies between it and the IF. The IF of a gated store (GATED_REDUCE) then skips its
  value's whole computation, not just the store"""
  lst = merge_ifs(lst)
  users: dict[UOp, list[UOp]] = {}
  for u in lst:
    for x in u.src: users.setdefault(x, []).append(u)
  for _ in range(sum(u.op is Ops.IF for u in lst)):
    out, moved_any = list(lst), False
    for i in range(len(out) - 1, -1, -1):
      if out[i].op is not Ops.IF: continue
      j, depth = i + 1, 0
      while j < len(out) and not (out[j].op is Ops.ENDIF and depth == 0):
        depth += out[j].op in (Ops.IF, Ops.RANGE)
        depth -= out[j].op in (Ops.ENDIF, Ops.END)
        j += 1
      # the IF's scope begins after the innermost block still open at i
      s, depth = i - 1, 0
      while s >= 0:
        if out[s].op in (Ops.END, Ops.ENDIF): depth += 1
        elif out[s].op in (Ops.RANGE, Ops.IF):
          if depth == 0: break
          depth -= 1
        s -= 1
      inside = set(out[i + 1:j])
      move: list[int] = []
      store_after = False
      k = i
      while (k:=k - 1) > s:
        u = out[k]
        # a register table (Tensor.stage) only read inside: its stores move in too (and are no barrier to moving loads), and so does
        # the loop that builds it (a table's loop axes: STAGE_LANES) when it does nothing else
        if u.op in (Ops.STORE, Ops.AFTER) and _staged_buf(u.src[0]) and users.get(u) and all(x in inside for x in users[u]):
          move.append(k)
          inside.add(u)
          continue
        if u.op is Ops.END:
          r0, depth = k - 1, 0
          while not (out[r0].op is Ops.RANGE and depth == 0):
            depth += out[r0].op is Ops.END
            depth -= out[r0].op is Ops.RANGE
            r0 -= 1
          blk = out[r0:k + 1]
          if r0 > s and any(x.op is Ops.STORE for x in blk) and \
             all(x.op is not Ops.IF and (x.op is not Ops.STORE or _staged_buf(x.src[0])) for x in blk) and \
             all(y in inside or y in blk for x in blk for y in users.get(x, [])):
            move += range(r0, k + 1)
            inside.update(blk)
            k = r0
            continue
        if u.op is Ops.STORE: store_after = True
        if u.op not in _SINKABLE or (u.op is Ops.LOAD and store_after) or not users.get(u): continue
        if all(x in inside for x in users[u]):
          move.append(k)
          inside.add(u)
      if move:
        mv = [out[k] for k in sorted(move)]
        rest = [u for k, u in enumerate(out[:i + 1]) if k not in set(move)]
        out = rest + mv + out[i + 1:]
        moved_any = True
        break
    lst = out
    if not moved_any: break
  return lst

def _block_end(lst:list[UOp], i:int) -> int:
  # the index of the END / ENDIF closing the RANGE / IF at i
  j, depth = i + 1, 0
  while not (lst[j].op in (Ops.END, Ops.ENDIF) and depth == 0):
    depth += lst[j].op in (Ops.IF, Ops.RANGE)
    depth -= lst[j].op in (Ops.ENDIF, Ops.END)
    j += 1
  return j

def hoist_ifs(lst:list[UOp]) -> list[UOp]:
  """an IF whose gate is known before the loop around it, in a loop that does nothing outside the IF (no store, no inner block),
  goes around the loop: a gated reduce over a register table (Tensor.stage) then skips the whole table walk, and sink_into_ifs can
  move the table's computation in too"""
  lst = merge_ifs(lst)
  for _ in range(len(lst)):
    pos = {u:k for k,u in enumerate(lst)}
    for i,u in enumerate(lst):
      if u.op is not Ops.IF: continue
      j, a, depth = _block_end(lst, i), i - 1, 0
      while a >= 0 and not (lst[a].op in (Ops.RANGE, Ops.IF) and depth == 0):
        depth += lst[a].op in (Ops.END, Ops.ENDIF)
        depth -= lst[a].op in (Ops.RANGE, Ops.IF)
        a -= 1
      if a < 0 or lst[a].op is not Ops.RANGE or pos.get(u.src[0], len(lst)) >= a: continue
      e = _block_end(lst, a)
      if any(x.op in (Ops.STORE, Ops.RANGE, Ops.IF, Ops.END, Ops.ENDIF) for x in lst[a+1:i] + lst[j+1:e]): continue
      lst = lst[:a] + [u] + lst[a:i] + lst[i+1:j] + lst[j+1:e+1] + [lst[j]] + lst[e+1:]
      break
    else: break
  return lst

def do_linearize(ctx:Renderer, prg:UOp, sink:UOp) -> UOp:
  if DEBUG >= 3 and sink.arg.applied_opts: print(f"{sink.arg.function_name:<25} opts: {sink.arg.applied_opts}")
  lst = line_rewrite(linearize(sink), pm_linearize_cleanups)
  if getenv("GATED_REDUCE", 1) and any(u.op is Ops.IF for u in lst):
    # (IF_HOIST=1 for any kernel; on by itself in one with register tables)
    if getenv("IF_HOIST", 0) or any(u.op is Ops.BUFFER and _staged_buf(u) for u in lst): lst = hoist_ifs(lst)
    lst = sink_into_ifs(lst)
  # isa renderers need to allocate registers
  if isinstance(ctx, ISARenderer):
    if ctx.pre_regalloc_matcher is not None: lst = line_rewrite(lst, ctx.pre_regalloc_matcher, PreRegAllocContext())
    # register definitions (INS without srcs) move to the top so regalloc sees their live ranges span the whole program (callee saved regs)
    lst = sorted(lst, key=lambda u: u.op is not Ops.INS or bool(u.src))
    regalloc_ctx = LinearScanRegallocContext(lst, ctx)
    lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
    lst = line_rewrite(lst, ctx.post_regalloc_matcher, regalloc_ctx)
    if DEBUG >= 4: print(ctx.asm_str(lst, sink.arg.function_name))
  return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst)),))

def do_estimates(prg:UOp, sink:UOp, lin:UOp) -> UOp|None:
  if sink.arg.estimates is not None: return None
  return prg.replace(src=(sink.replace(arg=replace(sink.arg, estimates=Estimates.from_uops(lin.src, ignore_indexing=True))),)+prg.src[1:])

def do_assemble(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = "\n".join(str(u.arg) for u in lin.src)
  if DEBUG >= 4: print(src)
  binary = ctx.asm(prg, lin)
  return prg.replace(src=prg.src[:2]+(UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

def do_render(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = ctx.render(list(lin.src))
  return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),))

def do_compile(ctx:Renderer, prg:UOp, source:UOp) -> UOp|None:
  if DEBUG >= 4: print(source.arg)
  lib = ctx.compiler.compile_cached(source.arg)
  if DEBUG >= 7: ctx.compiler.disassemble(lib)
  return prg.replace(src=prg.src + (UOp(Ops.BINARY, arg=lib),))

pm_to_program = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"),), name="prg"), do_linearize),
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.LINEAR, name="lin")), name="prg"), do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR, src=UPat(Ops.INS), name="lin")), name="prg"), do_assemble),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR, name="lin")), name="prg"), do_render),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR), UPat(Ops.SOURCE, name="source")), name="prg"), do_compile),
])

@rewrite_group(name=lambda ast,renderer,ret,**_: TracingKey((k:=ret.src[0].arg).name,(k.function_name, ast, ret.key),ret=renderer), replay=True)
@Context(ALLOW_DEVICE_USAGE=0)
def do_to_program(ast:UOp, renderer:Renderer) -> UOp:
  """
  Transform an AST into a compiled PROGRAM. May trigger BEAM search.

  Args:
    ast: The Ops.SINK/Ops.PROGRAM rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.PROGRAM with SINK/LINEAR/SOURCE/BINARY.
  """
  if ast.op is Ops.PROGRAM: prg = ast
  elif ast.op is Ops.SINK:
    assert isinstance(ast.arg, KernelInfo), "requires KernelInfo on arg to to_program"
    full_sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
    prog_info = ProgramInfo.from_sink(full_sink, renderer.target)
    # instruction selection
    if isinstance(renderer, ISARenderer):
      full_sink = graph_rewrite(full_sink, renderer.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre instruction selection", bottom_up=True)
      full_sink = graph_rewrite(full_sink, renderer.isel_matcher, ctx=IselContext(full_sink), name="instruction selection", bottom_up=True)
    prg = UOp(Ops.PROGRAM, src=(full_sink,), arg=prog_info)
  else: raise RuntimeError(f"can't call to_program on {ast.op}")
  if not isinstance(prg.arg, ProgramInfo): prg = prg.replace(arg=ProgramInfo.from_sink(prg.src[0], renderer.target))
  prg = graph_rewrite(prg, pm_to_program, ctx=renderer, name="linearize/render")
  if VIZ: graph_rewrite(prg, PatternMatcher([]), name="View Program")
  return prg

# config affects generated programs and cache keys; context also carries compile-only behavior to workers
to_program_config = (NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32,
                     DEFAULT_FLOAT, DEFAULT_INT, NUM_CPU_THREADS, TC_SELECT, TC_OPT)
to_program_context = (*to_program_config, SPEC, DEBUG)
def to_program_key(ast:UOp, renderer:Renderer) -> tuple:
  return (ast.key, type(renderer), renderer.target, *[x.value for x in to_program_config])

to_program_cache: dict[tuple, UOp] = {}
def to_program(ast:UOp, renderer:Renderer) -> UOp:
  if (prg:=to_program_cache.get(key:=to_program_key(ast, renderer))) is None: to_program_cache[key] = prg = do_to_program(ast, renderer)
  return prg
