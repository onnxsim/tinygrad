from __future__ import annotations
import ctypes, os, mmap, tempfile, pathlib, array, threading, contextlib, sys, subprocess, struct, re
assert sys.platform != 'win32'
from tinygrad.device import BufferSpec, Compiled, Allocator, Compiler, Program, TinyELF, CompileError
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp, GroupOp, AxisType
from tinygrad.helpers import getenv, round_up, mv_address, to_mv, cpu_objdump, system, DEBUG, suppress_finalizing, Target, unwrap, prod
from tinygrad.renderer.cstyle import ClangRenderer, wmma_args, _wmma_name
from tinygrad.codegen.opt import tc
from tinygrad.runtime.autogen import libc, qcom_dsp
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

from tinygrad.uop.ops import PatternMatcher, UPat

HVX_PREFETCH = getenv("HVX_PREFETCH", 2048)
HVX_PREFETCH_HALF = getenv("HVX_PREFETCH_HALF", 128)
# a load whose address moves by more than 256 bytes per iteration of its innermost loop (a reduction down the rows of a
# row-major matrix) prefetches HVX_PREFETCH_STRIDES iterations ahead instead of HVX_PREFETCH bytes (one iteration, for
# a 2 KB row): the MCC LayerNorm statistics read 1 MB at ~1.2 GB/s
HVX_PREFETCH_STRIDES = getenv("HVX_PREFETCH_STRIDES", 4)
# HVX ISA the DSP code is compiled for. v65 (the default) has no HVX float at all. From v68 on, float32 vector math is
# qfloat (qf32): v68/v69 HVX has no IEEE fp32 (the Snapdragon 8+ Gen 1 test phone is v69), and LLVM lowers plain float
# vector arithmetic to qf32 on its own. Note vector int<->float conversion only exists from v73 (vconv_sf_w/w_sf).
HVX_ARCH = getenv("HVX_ARCH", "v65")
HVX_QFLOAT = int(HVX_ARCH.lstrip("v")) >= 68

# ***** qfloat lowering (v68+) *****
# LLVM's qfloat lowering is fast and accurate for adds/subs and for multiplies with an IEEE sf operand (a load or a
# constant): measured bit-identical to converting after every op. The case it gets badly wrong is a multiply of two
# *computed* values: it keeps both in qf32 and emits qf32 x qf32 without renormalizing (21% worst-case relative error
# on ((a-b)*(c-d))*((a+b)*(c+d)) on hardware, vs 4.2% renormalized). For exactly those multiplies, each operand is
# forced to IEEE sf with an empty asm barrier (the NMS kernel's recipe) before the multiply.
def _computed(u:UOp) -> bool:
  # a value LLVM may be holding as qf32 (an ALU result), as opposed to IEEE sf straight from memory or a constant
  while u.op in (Ops.STACK, Ops.CAST, Ops.BITCAST) and len(u.src) >= 1 and all(s is u.src[0] for s in u.src): u = u.src[0]
  return u.op in GroupOp.ALU

def _qf_vec(x:UOp) -> bool:
  return HVX_QFLOAT and x.op is Ops.MUL and x.dtype == dtypes.float32 and x._shape is not None and \
    (n:=x.max_numel()) >= 32 and n % 32 == 0 and all(_computed(s) for s in x.src)

def _qf_helpers(uops:list[UOp], vec_type) -> list[str]:
  widths = sorted({x.max_numel() for x in uops if _qf_vec(x)})
  if not widths: return []
  # everything stays in registers: wider vectors are split into / rebuilt from whole HVX registers with shufflevector
  # (going through `((__hvx_v*)&a)[i]` instead forces a stack round trip per register -- measured 10x slower)
  out = ["typedef float __hvx_f __attribute__((ext_vector_type(32)));",
         "static inline __hvx_f __hvx_mulsf(__hvx_f a, __hvx_f b) { __asm__(\"\" : \"+v\"(a)); __asm__(\"\" : \"+v\"(b)); return a*b; }"]
  def lanes(lo:int, n:int) -> str: return ",".join(str(i) for i in range(lo, lo+n))
  for n in widths:
    t, k = vec_type(n), n // 32
    parts = [f"__hvx_mulsf({'a' if k == 1 else f'__builtin_shufflevector(a, a, {lanes(32*i, 32)})'}, "
             f"{'b' if k == 1 else f'__builtin_shufflevector(b, b, {lanes(32*i, 32)})'})" for i in range(k)]
    while len(parts) > 1:  # concatenate registers pairwise back up to n lanes
      w = 32 * (k // len(parts))
      parts = [f"__builtin_shufflevector({parts[j]}, {parts[j+1]}, {lanes(0, 2*w)})" for j in range(0, len(parts), 2)]
    out.append(f"static inline {t} __hvx_mul_f{n}({t} a, {t} b) {{ return ({t}){parts[0]}; }}")
  return out

# ***** qfloat exp2 / reciprocal (v68+) *****
# tinygrad's own EXP2 decomposition rounds with a float->int conversion and builds 2^q with int->float, and vector int<->float
# conversions only exist from v73 (vconv_sf_w / w_sf), so on v68/v69 every transcendental kernel falls back to scalar code.
# These helpers need no conversion: exp2 rounds with the 1.5*2^23 magic add (n = bits - 0x4B400000, an int subtract on the
# reinterpreted bits; the qf32->sf rounding may be one off, so f = x - (r - magic) stays exact and the polynomial covers
# [-1, 1]), a degree-6 polynomial for 2^f and adds n to the exponent field; the reciprocal is the 0x7EF311C7 bit estimate +
# 3 Newton steps. Every computed x computed multiply goes through IEEE sf first (the "+v" barrier, as _qf_helpers).
# Domain: exp2 input clamped to [-126, 126] (so -inf -> 2^-126, not 0), reciprocal for normal x != 0.
QF_MATH = HVX_QFLOAT and bool(getenv("HVX_QF_MATH", 1))
_QF_EXP2_C = [1.53592089e-4, 1.33926270e-3, 9.61838476e-3, 5.55034727e-2, 2.40226448e-1, 6.93147182e-1, 1.0]

def _hf_exp2_helpers(uops:list[UOp], vtype) -> list[str]:
  widths = sorted({u.max_numel() for u in uops if _hf_exp2_src(u) is not None})
  if not widths: return []
  out = [_HF_EXP2]
  for n in widths:
    t, k = vtype(dtypes.half, n), n // 64
    lanes = [",".join(str(i) for i in range(64*j, 64*j+64)) for j in range(k)]
    parts = [f"__tg_exp2_h({'x' if k == 1 else f'__builtin_shufflevector(x, x, {lanes[j]})'})" for j in range(k)]
    while len(parts) > 1:
      w = 64 * (k // len(parts))
      parts = [f"__builtin_shufflevector({parts[j]}, {parts[j+1]}, {','.join(str(i) for i in range(2*w))})" for j in range(0, len(parts), 2)]
    out.append(f"static inline {t} __tg_exp2_h{n}({t} x) {{ return ({t}){parts[0]}; }}")
  return out

def _qf_math_helpers(uops:list[UOp], vtype) -> list[str]:
  if not QF_MATH: return []
  uses = {(Ops.RECIPROCAL if u.op is Ops.FDIV else u.op, u.max_numel()) for u in uops
          if u.op in (Ops.EXP2, Ops.RECIPROCAL, Ops.FDIV, Ops.SQRT) and u.dtype == dtypes.float32 or _hf_div(u)}
  if not uses: return []
  poly = "".join(f" p = __hvx_mulsf(p, f) + {c}f;" for c in _QF_EXP2_C[1:])
  out = ["typedef float __hvx_f __attribute__((ext_vector_type(32)));",
         "typedef int __hvx_i __attribute__((ext_vector_type(32)));",
         "static inline __hvx_f __hvx_mulsf(__hvx_f a, __hvx_f b) { __asm__(\"\" : \"+v\"(a)); __asm__(\"\" : \"+v\"(b)); return a*b; }",
         "static inline __hvx_f __tg_exp2_v(__hvx_f x) {"
         " x = __builtin_elementwise_min(__builtin_elementwise_max(x, (__hvx_f)(-126.0f)), (__hvx_f)(126.0f));"
         " __hvx_f r = x + 12582912.0f; __asm__(\"\" : \"+v\"(r));"
         " __hvx_i n = __builtin_bit_cast(__hvx_i, r) - 0x4B400000; __hvx_f f = x - (r - 12582912.0f);"
         f" __hvx_f p = {_QF_EXP2_C[0]}f * f + {_QF_EXP2_C[1]}f;" + poly.split(";", 1)[1] +
         " __asm__(\"\" : \"+v\"(p)); return __builtin_bit_cast(__hvx_f, __builtin_bit_cast(__hvx_i, p) + (n << 23)); }",
         "static inline __hvx_f __tg_recip_v(__hvx_f x) {"
         " __hvx_f y = __builtin_bit_cast(__hvx_f, 0x7EF311C7 - __builtin_bit_cast(__hvx_i, x));"
         " for (int k = 0; k < 3; k++) y = __hvx_mulsf(y, 2.0f - __hvx_mulsf(x, y)); return y; }",
         "static inline float __tg_exp2_s(float x) {"
         " x = x < -126.0f ? -126.0f : (x > 126.0f ? 126.0f : x); float r = x + 12582912.0f; int n; __builtin_memcpy(&n, &r, 4);"
         f" n -= 0x4B400000; float f = x - (r - 12582912.0f), p = {_QF_EXP2_C[0]}f;" +
         "".join(f" p = p * f + {c}f;" for c in _QF_EXP2_C[1:]) +
         " int b; __builtin_memcpy(&b, &p, 4); b += n << 23; __builtin_memcpy(&p, &b, 4); return p; }",
         "static inline float __tg_recip_s(float x) {"
         " int b; __builtin_memcpy(&b, &x, 4); b = 0x7EF311C7 - b; float y; __builtin_memcpy(&y, &b, 4);"
         " for (int k = 0; k < 3; k++) y = y * (2.0f - x * y); return y; }",
         # sqrt(x) = x / sqrt(x): the rsqrt bit estimate + 3 Newton steps (0 -> 0; tinygrad decomposes SQRT into a scalar loop)
         "static inline __hvx_f __tg_sqrt_v(__hvx_f x) {"
         " __hvx_f y = __builtin_bit_cast(__hvx_f, 0x5F3759DF - (__builtin_bit_cast(__hvx_i, x) >> 1)), h = __hvx_mulsf(x, (__hvx_f)(0.5f));"
         " for (int k = 0; k < 3; k++) y = __hvx_mulsf(y, 1.5f - __hvx_mulsf(h, __hvx_mulsf(y, y))); return __hvx_mulsf(x, y); }",
         "static inline float __tg_sqrt_s(float x) {"
         " int b; __builtin_memcpy(&b, &x, 4); b = 0x5F3759DF - (b >> 1); float y; __builtin_memcpy(&y, &b, 4);"
         " for (int k = 0; k < 3; k++) y = y * (1.5f - 0.5f * x * y * y); return x * y; }"]
  def lanes(lo:int, n:int) -> str: return ",".join(str(i) for i in range(lo, lo+n))
  for name, op in (("exp2", Ops.EXP2), ("recip", Ops.RECIPROCAL), ("sqrt", Ops.SQRT)):
    widths = sorted({n for o, n in uses if o is op})
    if not widths: continue
    assoc = []
    for n in widths:
      if n == 1:
        assoc.append(f"float: __tg_{name}_s")
        continue
      t = vtype(dtypes.float32, n)
      if n % 32 == 0:  # whole HVX registers, split / rebuilt with shufflevector (in registers)
        k = n // 32
        parts = [f"__tg_{name}_v({'x' if k == 1 else f'__builtin_shufflevector(x, x, {lanes(32*i, 32)})'})" for i in range(k)]
        while len(parts) > 1:
          w = 32 * (k // len(parts))
          parts = [f"__builtin_shufflevector({parts[j]}, {parts[j+1]}, {lanes(0, 2*w)})" for j in range(0, len(parts), 2)]
        out.append(f"static inline {t} __tg_{name}_f{n}({t} x) {{ return ({t}){parts[0]}; }}")
      else:  # other widths: per lane (scalar helper)
        out.append(f"static inline {t} __tg_{name}_f{n}({t} x) {{ {t} y; for (int i = 0; i < {n}; i++) y[i] = __tg_{name}_s(x[i]); return y; }}")
      assoc.append(f"{t}: __tg_{name}_f{n}")
    # the renderer only knows the scalar dtype; the C type picks the width (a function designator, then the call). Variadic:
    # an operand may be a lane constructor, (float128){a, b, ...}, whose commas would split a plain macro argument
    out.append(f"#define __TG_{name.upper()}(...) _Generic((__VA_ARGS__), {', '.join(assoc)})(__VA_ARGS__)")
  return out

# half EXP2 is rewritten to float (_qf_math_half); a 64-lane-multiple vector of that, CAST(half, EXP2(CAST(float, x))), renders as
# an hf helper instead (the hand-written MCC kernel's: 1536.0 magic-number round, degree-4 polynomial in qf16 converted back to
# hf after every op, exponent add; 0 below 2^-14, inf from 2^16): 64 lanes per HVX register instead of 32, and no hf<->sf
# conversions. Relative error ~5e-4 near 0, up to 7e-3 for |x| > 8 (qf16 rounding of n - x), vs ~5e-4 through float
HVX_HF_EXP2 = getenv("HVX_HF_EXP2", 1)
def _splat_const(u:UOp) -> float|None:
  # a constant, possibly cast, possibly splat across a STACK
  if u.op is Ops.STACK and all(s is u.src[0] for s in u.src): u = u.src[0]
  while u.op is Ops.CAST: u = u.src[0]
  return float(u.arg) if u.op is Ops.CONST else None

def _hf_exp2_src(u:UOp) -> tuple[UOp, float|None]|None:
  # CAST(half, EXP2(y)): (x, c) for y = CAST(float, x) * c (exp: times log2(e)) or CAST(float, x) with x half, else (y, None) --
  # a float exponent is rounded to hf first (<= 2^-8 absolute for |y| < 16, so < 0.3% relative; the result is half anyway)
  if not (HVX_HF_EXP2 and QF_MATH) or u.op is not Ops.CAST or u.dtype != dtypes.half or u.max_numel() % 64 != 0: return None
  if (e:=u.src[0]).op is not Ops.EXP2 or e.dtype != dtypes.float32: return None
  a = e.src[0]
  if a.op is Ops.MUL and (i:=next((i for i in (0, 1) if _splat_const(a.src[1-i]) is not None), None)) is not None:
    if a.src[i].op is Ops.CAST and a.src[i].src[0].dtype == dtypes.half: return a.src[i].src[0], _splat_const(a.src[1-i])
  return (a.src[0], None) if a.op is Ops.CAST and a.src[0].dtype == dtypes.half else (a, None)

_HF_EXP2 = r"""typedef __fp16 __hvx_h __attribute__((ext_vector_type(64)));
typedef short __hvx_hs __attribute__((ext_vector_type(64)));
static inline __hvx_h __hvx_hfb(__hvx_h a) { __asm__("" : "+v"(a)); return a; }
static inline __hvx_h __tg_exp2_h(__hvx_h x) {
  const __hvx_h magic = (__hvx_h)(__fp16)1536.0f;
  __hvx_h xc = __builtin_elementwise_min(__builtin_elementwise_max(x, (__hvx_h)(__fp16)-14.0f), (__hvx_h)(__fp16)15.99f);
  __hvx_h r = __hvx_hfb(xc + magic);
  __hvx_hs n = __builtin_bit_cast(__hvx_hs, r) - (__hvx_hs)(short)0x6600;
  __hvx_h g = __hvx_hfb(__hvx_hfb(r - magic) - xc);
  __hvx_h p = __hvx_hfb(__hvx_hfb(g * (__fp16)0.00961813f) + (__fp16)-0.0555041f);
  p = __hvx_hfb(__hvx_hfb(p * g) + (__fp16)0.2402265f);
  p = __hvx_hfb(__hvx_hfb(p * g) + (__fp16)-0.6931472f);
  p = __hvx_hfb(__hvx_hfb(p * g) + (__fp16)1.0f);
  __hvx_hs b = __builtin_bit_cast(__hvx_hs, p) + (n << 10);
  b = x < (__hvx_h)(__fp16)-14.0f ? (__hvx_hs)(short)0 : b;
  b = x >= (__hvx_h)(__fp16)16.0f ? (__hvx_hs)(short)0x7C00 : b;
  return __builtin_bit_cast(__hvx_h, b);
}"""

def _hf_div(u:UOp) -> bool:
  # a vector half division (QF_MATH): a * (1 / b), the reciprocal through the float helper -- clang scalarizes a vector hf
  # division (GELU's x / (1 + 2^t): 190 ms instead of ~4). At render time: as a rewrite, the late decompositions turn it back
  return bool(QF_MATH) and u.op is Ops.FDIV and u.dtype == dtypes.half and u.max_numel() > 1

def _hf_div_render(ctx, x:UOp) -> str|None:
  if not _hf_div(x): return None
  ft, ht = ctx._render_dtype(dtypes.float32, x.max_numel(), AddrSpace.REG), ctx.render_type(x)
  return f"({ctx[x.src[0]]}*__builtin_convertvector(__TG_RECIP(__builtin_convertvector({ctx[x.src[1]]}, {ft})), {ht}))"

def _hf_exp2_render(ctx, x:UOp) -> str|None:
  if (m:=_hf_exp2_src(x)) is None: return None
  arg = ctx[m[0]] if m[0].dtype == dtypes.half else f"__builtin_convertvector({ctx[m[0]]}, {ctx.render_type(x)})"
  if m[1] is not None: arg = f"({arg}*(({ctx.render_type(x)})((__fp16){m[1]!r}f)))"
  return f"__tg_exp2_h{x.max_numel()}({arg})"

def _qf_math_half(x:UOp) -> UOp|None:
  # half EXP2 / RECIPROCAL go through float32 (the helpers are float32; scalar __fp16 can't be a function argument)
  if x.dtype != dtypes.half: return None
  return x.src[0].cast(dtypes.float32).alu(x.op).cast(dtypes.half)

def _lane_slice(x:UOp) -> tuple[UOp, list[int]]|None:
  # STACK(v[k], v[k+1], ..., v[k+n-1]) of one vector v -> (v, [k..k+n-1])
  if len(x.src) < 2 or any(s.op is not Ops.INDEX or len(s.src) != 2 or s.src[0] is not x.src[0].src[0] for s in x.src): return None
  v = x.src[0].src[0]
  if v._shape is None or len(v._shape) != 1: return None
  lanes = [s.src[1].src[0].arg if s.src[1].op is Ops.CAST else s.src[1].arg for s in x.src]
  if not all(isinstance(l, int) for l in lanes) or lanes != list(range(lanes[0], lanes[0]+len(lanes))): return None
  return v, lanes

# NOTE: this just increases readability of the generated code
def _lane_window(ctx, x:UOp) -> str|None:
  # a window of a loaded vector (an HMX epilogue's 32-lane rows of the 128-lane accumulator-array vectors): read it again from
  # memory. As a __builtin_shufflevector at a lane offset that isn't a register boundary, clang v19 builds it lane by lane
  # (vinsert, ~6x the packets); a vector load of the window is one vmem(u). Only if nothing stores to that buffer in between
  if (sl:=_lane_slice(x)) is None or not _loaded_unchanged(ctx, v:=sl[0], x): return None
  return f"(*(({ctx.render_type(x)}*)(({ctx.render_dtype(x.src[0].dtype)}*){ctx[v.src[0]]}+{sl[1][0]})))"

def _inline_vector_load(ctx, u:UOp) -> bool:
  # a vector load used once, later in the same loop body, with no store to its buffer in between: render it at the use. Loads
  # come out of the linearizer early (an HMX epilogue loads all 32 residual rows before the first store), and clang then
  # keeps them all live -- 32 x 64 bytes against 32 HVX registers -- and spills them, rebuilding each lane by lane (vinsert)
  if u.max_numel() == 1 or len(users:=getattr(ctx, "_users", {}).get(u, [])) != 1: return False
  x, pos = users[0], ctx._pos
  return x in pos and not any(pos[u] < i < pos[x] for i in ctx._scopes) and _loaded_unchanged(ctx, u, x)

def _loaded_unchanged(ctx, v:UOp, x:UOp) -> bool:
  # v is a plain load and nothing stores to its buffer between v and its use x (so the memory can be read again at x)
  pos, stores = getattr(ctx, "_pos", {}), getattr(ctx, "_stores", [])
  if v.op is not Ops.LOAD or len(v.src) != 1 or v not in pos or x not in pos: return False
  return not any(pos[v] < i < pos[x] and p is _hmx_param(v.src[0]) for i, p in stores)

def _splat_of_loaded_lane(ctx, x:UOp) -> str|None:
  if len(x.src) < 2 or any(s is not x.src[0] for s in x.src): return None
  s = x.src[0]
  if s.op is not Ops.INDEX or len(s.src) != 2 or s._shape != () or (lane:=_lane(s.src[1])) is None: return None
  if not _loaded_unchanged(ctx, v:=s.src[0], x): return None
  return f"(({ctx.render_type(x)})((({ctx.render_dtype(s.dtype)}*){ctx[v.src[0]]})[{lane}]))"

def _prefetch_distance(ctx, bidx:UOp, itemsize:int) -> int:
  if HVX_PREFETCH_STRIDES <= 0 or len(bidx.src) < 2 or not (pos:=getattr(ctx, "_pos", None)): return HVX_PREFETCH
  idx = bidx.src[1]
  rngs = [r for r in idx.toposort() if r.op is Ops.RANGE and r in pos]
  if not rngs: return HVX_PREFETCH
  inner = max(rngs, key=lambda r: pos[r])
  zero = {r:r.const_like(0) for r in rngs}
  d = (idx.substitute({**zero, inner:inner.const_like(1)}).simplify() - idx.substitute(zero).simplify()).simplify()
  if d.op is not Ops.CONST: return HVX_PREFETCH
  step = int(d.arg) * itemsize
  return HVX_PREFETCH_STRIDES * step if step > 256 else HVX_PREFETCH

def _vec_fmax(ctx, x:UOp) -> str|None:
  return f"__builtin_elementwise_max({ctx[x.src[0]]},{ctx[x.src[1]]})" if HVX_QFLOAT and x.max_numel() > 1 else None

dsp_string = PatternMatcher([
  (UPat(Ops.CAST, dtype=dtypes.half, name="x"), lambda ctx,x: _hf_exp2_render(ctx, x)),
  (UPat(Ops.FDIV, dtype=dtypes.half, name="x"), lambda ctx,x: _hf_div_render(ctx, x)),
  # a float MAX of vectors (see hvx_revectorize): HVX's native sf / hf max. It is IEEE maxNum (a NaN operand yields the other
  # operand) where the scalar form keeps tinygrad's (a<b)?b:a -- they only differ on NaN
  (UPat(Ops.MAX, dtype=(dtypes.float32, dtypes.half), name="x"), lambda ctx,x: _vec_fmax(ctx, x)),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.uint8), name="x"), lambda ctx,x: str(x.val)),
  (UPat(Ops.MUL, dtypes.float32, name="x"), lambda ctx,x:
   f"__hvx_{x.op.name.lower()}_f{x.max_numel()}({', '.join(ctx[s] for s in x.src)})" if _qf_vec(x) else None),
  # a STACK of consecutive lanes of one wider vector (memory_coalescing merged two adjacent loads) is a lane slice: one
  # shufflevector instead of a per-lane constructor; of a loaded vector, a narrower load of the same memory (_lane_window)
  (UPat(Ops.STACK, name="x"), lambda ctx,x: _lane_window(ctx, x)),
  (UPat(Ops.STACK, name="x"), lambda ctx,x: f"(({ctx.render_type(x)})__builtin_shufflevector({ctx[v]}, {ctx[v]}, {','.join(str(l) for l in lanes)}))"
   if (sl:=_lane_slice(x)) is not None and (v:=sl[0]) is not None and (lanes:=sl[1]) else None),
  # a splat of one lane of a loaded vector (memory_coalescing merged per-row loads, e.g. an HMX epilogue's bias, into one
  # vector load): reload that element as a scalar (then one vsplat) -- the lane splat clang v19 builds with vinserts
  (UPat(Ops.STACK, name="x"), lambda ctx,x: _splat_of_loaded_lane(ctx, x)),
  # a STACK of one repeated scalar is a splat, which clang lowers to a single HVX vsplat
  (UPat(Ops.STACK, name="x"), lambda ctx,x: f"(({ctx.render_type(x)})({ctx[x.src[0]]}))"
   if len(x.src) > 1 and all(s is x.src[0] for s in x.src) and x.src[0]._shape == () else None),
  # vector casts must convert per lane; a C cast between ext_vector_types is a bitcast
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_type(x)})"
   if x.max_numel() > 1 else None),
  # software-prefetch ahead of every vector load: streaming kernels on this DSP stall on DDR latency, not ALU. dcfetch is a
  # non-faulting hint, so prefetching past the end of a buffer is harmless. One dcfetch per 128-byte line the load covers
  # (a 128-lane int32 load is four HVX registers / lines); sub-line loads are skipped, a dcfetch per 32-byte load costs more
  # than it hides. HVX_PREFETCH is the distance in bytes (0 = off). A half-line (64-byte) load gets one dcfetch of the next
  # line instead (HVX_PREFETCH_HALF): an HMX epilogue reads its 32 rows of 64 bytes at the row stride, each a DDR miss, and the
  # tile after next reads that next line
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"), lambda ctx,bidx,x:
   "(" + "".join(f"__builtin_HEXAGON_Y2_dcfetch((char*){ctx[bidx]}+{_prefetch_distance(ctx, bidx, x.dtype.itemsize)+o}), "
                 for o in range(0, max(x.max_numel()*x.dtype.itemsize, 1), 128)) + f"{ctx.render_access(bidx)})"
   if HVX_PREFETCH > 0 and x.max_numel()*x.dtype.itemsize >= 128 and bidx.addrspace is AddrSpace.GLOBAL else None),
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"), lambda ctx,bidx,x:
   f"(__builtin_HEXAGON_Y2_dcfetch((char*){ctx[bidx]}+{HVX_PREFETCH_HALF}), {ctx.render_access(bidx)})"
   if HVX_PREFETCH_HALF > 0 and x.max_numel()*x.dtype.itemsize == 64 and bidx.addrspace is AddrSpace.GLOBAL else None),
])

# ***** HVX re-vectorization *****
# devectorizer2 splits every elementwise op into per-lane scalars and memory_coalescing only regroups the loads and
# stores (up to 128 lanes for DSP), so the ALU in between is rendered as `(int128){(a[0]+b[0]),(a[1]+b[1]),...}`, which
# LLVM lowers lane by lane (~10x the instructions of one vector add). This pass rebuilds vector ALU ops bottom-up:
# STACK(op(a_i, b_i) for i) -> op(STACK(a_i), STACK(b_i)); STACK(v[0], ..., v[n-1]) -> v. clang lowers ext_vector_type
# arithmetic straight to HVX under -mhvx. Compares/WHERE are left scalar: C vector compares yield same-width int masks,
# not _Bool vectors, so they'd need mask-dtype plumbing -- MAX (the common select) is native instead.
HVX_VEC_OPS = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.MAX, Ops.CAST}
# with qfloat (v68+) EXP2 / RECIPROCAL / float FDIV are rendered as vector helpers (_qf_math_helpers) instead of being
# decomposed, so they re-vectorize too (added while QF_MATH is on, see _vec_ops)
_QF_VEC_OPS = {Ops.EXP2, Ops.RECIPROCAL, Ops.FDIV, Ops.SQRT}

def _vec_ops() -> set: return HVX_VEC_OPS | _QF_VEC_OPS if QF_MATH else HVX_VEC_OPS

def _lane(u:UOp) -> int|None:
  if u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST and isinstance(u.arg, int) else None

def _vec_column_ok(col:tuple[UOp, ...], depth:int=0) -> bool:
  # a column (the j-th operand of every lane) can become one vector operand if it's a splat, constants, consecutive lanes
  # of one vector, or (recursively) the same vectorizable op in every lane. Anything else -- e.g. a per-lane mix of
  # compare/cast/mul -- stays a scalar constructor: LLVM's Hexagon backend crashes selecting some of those once they're
  # wrapped in a vector op ("Cannot select v2i32 = bitcast (V2Q ...)"), and there's nothing to gain vectorizing them.
  c0 = col[0]
  if c0._shape != () or c0.dtype == dtypes.bool or depth > 32: return False
  if all(c is c0 for c in col) or all(c.op is Ops.CONST for c in col): return True
  if c0.op is Ops.INDEX and len(c0.src) == 2 and c0.src[0]._shape is not None and len(c0.src[0]._shape) == 1:
    # consecutive lanes of one vector, or of several vectors one after another (a row loaded as two 64-byte halves):
    # the vector operand is then that lane constructor, which clang lowers to shuffles
    if any(c.op is not Ops.INDEX or len(c.src) != 2 or c.src[0]._shape is None or len(c.src[0]._shape) != 1 for c in col): return False
    lanes = [_lane(c.src[1]) for c in col]
    if None in lanes: return False
    return all(b is not a or lb == la + 1 for (a, la), (b, lb) in zip(((c.src[0], l) for c, l in zip(col, lanes)),
                                                                       ((c.src[0], l) for c, l in zip(col[1:], lanes[1:]))))
  if c0.op in _vec_ops() and all(c.op is c0.op and c.dtype == c0.dtype and c.arg == c0.arg and len(c.src) == len(c0.src) for c in col):
    return all(_vec_column_ok(tuple(c.src[j] for c in col), depth+1) for j in range(len(c0.src)))
  return False

# pattern functions get a snapshot of their globals when the matcher is first built (deconstruct_function): module flags are
# read through a function so they stay live
def _hvx_qfloat() -> bool: return HVX_QFLOAT

def hvx_revectorize(x:UOp) -> UOp|None:
  srcs, n = x.src, len(x.src)
  if n < 2 or x.dtype == dtypes.void: return None
  s0 = srcs[0]
  # STACK(v[0], v[1], ..., v[n-1]) of a length-n vector is v itself
  if s0.op is Ops.INDEX and len(s0.src) == 2 and s0.src[0]._shape == (n,) and \
     all(s.op is Ops.INDEX and len(s.src) == 2 and s.src[0] is s0.src[0] and _lane(s.src[1]) == i for i,s in enumerate(srcs)):
    return s0.src[0]
  if s0.op not in _vec_ops() or s0.dtype == dtypes.bool or s0._shape != (): return None
  # a scalar float max renders as a statement expression; a vector one (qfloat targets) as HVX's native max (dsp_string)
  if s0.op is Ops.MAX and dtypes.is_float(s0.dtype) and not _hvx_qfloat(): return None
  if any(s.op is not s0.op or s.dtype != s0.dtype or s.arg != s0.arg or len(s.src) != len(s0.src) or s._shape != () for s in srcs): return None
  if not _vec_column_ok(srcs): return None
  return UOp(s0.op, s0.dtype, tuple(UOp.stack(*[s.src[j] for s in srcs]) for j in range(len(s0.src))), s0.arg)

pm_hvx_revectorize = PatternMatcher([(UPat(Ops.STACK, name="x"), hvx_revectorize)])

# HMX=1 adds the V69 HMX (fp16) tensor core in front of the HVX vrmpy ones
def _dsp_tcs(): return (tc.hexagon_hmx + tc.hexagon_hmx_i8 if getenv("HMX") else []) + tc.hexagon_v65

# HMX (V69 matrix unit) tile op for the hexagon_hmx TensorCore: D = rne_fp16(C + A.B), all three 32x32 fp16 tiles in the HMX
# layout IDX(i,j) = 64*(i/2)+2*j+i%2 (the TC's swizzles make tinygrad's fragments exactly that order). C is folded into the
# same accumulation as a second K block against an identity weight tile, so one load pair + one store does the whole op.
# Preconditions are the runtime's (scripts/android/hmx_gemm/hmx_runtime.h in onnxsim): HMX power vote, VTCM + HMX context,
# HVX + HMX lock on this thread, __hmx_vtcm pointing at >= 16 KB of that VTCM inside one 256 KB window, 2 KB aligned, and
# __hmx_gen bumped (nonzero) on every (re)acquire so the tile op re-initializes its identity tile and bias table there.
# -DHMX_REF builds a scalar reference of the same op on the same layout instead (for qemu, which can't run HMX).
_HMX_REF_CONV = r"""/* bit-level fp16 <-> double (the freestanding qemu link has no __extendhfsf2/__truncdfhf2) */
static inline double __hmx_h2d(unsigned short h) {
  int e = (h >> 10) & 31, f = h & 1023; double v = e ? (double)(f | 1024) : (double)f;
  for (int i = 0; i < (e ? e : 1); i++) v *= 2.0;
  v /= 33554432.0;  /* 2^25 = 2^(15 + 10) */
  return (h & 0x8000) ? -v : v;
}
static inline unsigned short __hmx_d2h(double x) {  /* round to nearest even, saturate to inf */
  unsigned short sgn = x < 0 ? 0x8000 : 0; if (x < 0) x = -x;
  if (x >= 65520.0) return sgn | 0x7c00;
  int e = 15; double m = x;
  if (m >= 2048.0 / 1024.0 * 1.0) { while (m >= 2.0 && e < 30) { m /= 2.0; e++; } } else { while (m < 1.0 && e > 1) { m *= 2.0; e--; } }
  if (m < 1.0) e = 0;  /* subnormal: m = x / 2^-14 */
  double q = m * 1024.0; long r = (long)q; double fr = q - (double)r;
  if (fr > 0.5 || (fr == 0.5 && (r & 1))) r++;
  if (e == 0) return sgn | (unsigned short)r;  /* r == 1024 rounds up into the smallest normal, same bits */
  if (r == 2048) { r = 1024; e++; if (e >= 31) return sgn | 0x7c00; }
  return sgn | (unsigned short)((e << 10) | (r - 1024));
}
"""
def _hmx_wmma_helper(name:str, vt:str) -> str:
  return f"""#ifndef HMX_IDX
#define HMX_IDX(i, j) (64 * ((i) / 2) + 2 * (j) + ((i) % 2))
#endif
#ifdef HMX_REF
{_HMX_REF_CONV}static inline {vt} __{name}({vt} a, {vt} b, {vt} c) {{
  unsigned short A[1024], B[1024], C[1024], D[1024];
  __builtin_memcpy(A, &a, 2048); __builtin_memcpy(B, &b, 2048); __builtin_memcpy(C, &c, 2048);
  for (int m = 0; m < 32; m++) for (int n = 0; n < 32; n++) {{
    double s = __hmx_h2d(C[HMX_IDX(m, n)]);
    for (int k = 0; k < 32; k++) s += __hmx_h2d(A[HMX_IDX(m, k)]) * __hmx_h2d(B[HMX_IDX(k, n)]);
    D[HMX_IDX(m, n)] = __hmx_d2h(s);
  }}
  {vt} d; __builtin_memcpy(&d, D, 2048); return d;
}}
#else
extern unsigned char* __hmx_vtcm;
extern unsigned int __hmx_gen;  /* bumped by the runtime whenever __hmx_vtcm is (re)acquired */
static inline {vt} __{name}({vt} a, {vt} b, {vt} c) {{
  unsigned short* v = (unsigned short*)__hmx_vtcm;  /* act [C, A] | weight [I, B] | out | table */
  static unsigned int init = 0;
  if (init != __hmx_gen) {{
    for (int i = 0; i < 1024; i++) v[2048 + i] = 0;
    for (int k = 0; k < 32; k++) v[2048 + HMX_IDX(k, k)] = 0x3c00;  /* fp16 1.0 */
    for (int j = 0; j < 64; j++) ((unsigned int*)(v + 5120))[j] = 0;  /* zero bias */
    init = __hmx_gen;
  }}
  *({vt}*)v = c; *({vt}*)(v + 1024) = a; *({vt}*)(v + 3072) = b;  /* plain vector stores (no libc memcpy on the skel) */
  __asm__ volatile("bias = mxmem(%0)" :: "r"(v + 5120) : "memory");
  __asm__ volatile("{{ activation.hf = mxmem(%0,%1):deep\\n weight.hf = mxmem(%2,%3) }}" :: "r"(v), "r"(4095), "r"(v + 2048), "r"(4095) : "memory");
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(v + 4096), "r"(0) : "memory");
  return *({vt}*)(v + 4096);
}}
#endif"""

# int8 ":cm" tile op (hexagon_hmx_i8): D = C + A.B exact in int32, A u8 64x32 row-major, B s8 32x32 in the W layout, C/D
# int32 64x32 row-major. The plain (non-HMX_ACC) hardware op: four byte-plane stores of the exact accumulator (scales 1,
# 2^-8, 2^-16, 2^-24 in the 64-bit column table, the non-saturating :cm.ub store wraps), HVX byte->word interleave, + C.
def _hmx_i8_wmma_helper(name:str, vt_a:str, vt_b:str, vt_c:str) -> str:
  return f"""#pragma clang diagnostic ignored "-Wunused-variable"
#ifdef HMX_REF
static inline {vt_c} __{name}({vt_a} a, {vt_b} b, {vt_c} c) {{
  unsigned char A[2048]; signed char B[1024]; int C[2048];
  __builtin_memcpy(A, &a, 2048); __builtin_memcpy(B, &b, 1024); __builtin_memcpy(C, &c, 8192);
  for (int m = 0; m < 64; m++) for (int n = 0; n < 32; n++) {{
    int s = C[32 * m + n];
    for (int k = 0; k < 32; k++) s += (int)A[32 * m + k] * (int)B[128 * (k / 4) + 4 * n + k % 4];
    C[32 * m + n] = s;
  }}
  {vt_c} d; __builtin_memcpy(&d, C, 8192); return d;
}}
#else
extern unsigned char* __hmx_vtcm;
static inline {vt_c} __{name}({vt_a} a, {vt_b} b, {vt_c} c) {{
  unsigned char* v = __hmx_vtcm;  /* act @0 | weight @2K | table x4 @4K (256-byte aligned, 64-bit words) | planes @8K */
  unsigned int* t = (unsigned int*)(v + 4096);
  static const unsigned short sc[4] = {{0x6000, 0x4000, 0x2000, 0x0800}};
  for (int p = 0; p < 4; p++) for (int j = 0; j < 32; j++) {{ t[64 * p + j] = sc[p]; t[64 * p + 32 + j] = 0; }}
  *({vt_a}*)v = a; *({vt_b}*)(v + 2048) = b;
  __asm__ volatile("{{{{ activation.ub = mxmem(%0,%1):cm\\n weight.b = mxmem(%2,%3) }}}}" :: "r"(v), "r"(0x7ff), "r"(v + 2048), "r"(0x3ff) : "memory");
  for (int p = 0; p < 4; p++) {{
    __asm__ volatile("bias = mxmem2(%0)" :: "r"(t + 64 * p) : "memory");
    if (p < 3) __asm__ volatile("mxmem(%0,%1):after:retain:cm.ub = acc" :: "r"(v + 8192 + 2048 * p), "r"(0) : "memory");
    else __asm__ volatile("mxmem(%0,%1):after:cm.ub = acc" :: "r"(v + 8192 + 2048 * p), "r"(0) : "memory");
  }}
  const unsigned char* P = v + 8192;
  int D[2048] __attribute__((aligned(128)));
  for (int i = 0; i < 2048; i++) D[i] = (int)((unsigned)P[i] | (unsigned)P[2048 + i] << 8 | (unsigned)P[4096 + i] << 16 | (unsigned)P[6144 + i] << 24);
  return *({vt_c}*)D + c;
}}
#endif"""


# ---- HMX_ACC (default with HMX=1): accumulator kept inside HMX across the reduce loop ----
# The hexagon_hmx WMMA as tinygrad lowers it passes 2 KB tiles by value and round-trips the fp16 accumulator through a
# register array every K block. _hmx_acc_rewrite turns the linearized kernel into what the hand kernel does instead:
#   before the reduce loop   __hmx_begin();                  (bias table, clear state)
#   each K block             pack A, B rows into VTCM (one 128-byte halfword interleave per row pair) + one load pair
#   after the reduce loop    __hmx_out(dst...)               (one :after.hf store, then HVX loads/shuffles into the acc array)
# so each output tile is rounded to fp16 once (exact accumulation, like hmx_block.h) and nothing 2 KB-sized lives on the stack.
# VTCM tile cache slots (2 KB each) for A and B. HMX_VTCM_KB=256 (default): 16 KB + 116 * 2 KB, one 256 KB window. Larger
# (the runtime must then give that much VTCM, 256 KB aligned): A from 64 KB, a multiple of 128 slots, then 128 B slots, and
# every K panel at a stride of _hmx_stride(kt) slots -- no load pair (<= 32 tiles) crosses a 256 KB window (a PD fault), so
# whole weight matrices (fc2: 16 x 64 tiles) stay packed across a call instead of being repacked per output tile
HMX_VTCM_KB = getenv("HMX_VTCM_KB", 256)
_HMX_AO, _HMX_CA, _HMX_CB = (65536, (HMX_VTCM_KB // 2 - 32 - 128) // 128 * 128, 128) if HMX_VTCM_KB > 256 else (16384, 76, 40)
def _hmx_stride(kt:int) -> int:
  # slots per K panel: a power of two (<= 32 tiles) or a multiple of 32, so from a 32-slot aligned base no 32-tile load pair
  # crosses a 128-slot (256 KB) window; the default layout packs panels densely (it is one window)
  if HMX_VTCM_KB <= 256: return kt
  return 1 << (kt - 1).bit_length() if kt <= 32 else round_up(kt, 32)
_HMX_ACC_HELPERS = r"""#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
typedef __fp16 __hmx_h32 __attribute__((ext_vector_type(32)));
typedef __fp16 __hmx_h64 __attribute__((aligned(128),ext_vector_type(64)));
typedef __fp16 __hmx_h128 __attribute__((aligned(128),ext_vector_type(128)));
typedef int __hmx_v __attribute__((vector_size(128)));
typedef int __hmx_vp __attribute__((vector_size(256)));
#ifdef HMX_REF
/* rows r0, r1 (32 fp16 each, 64-byte aligned) -> dst[2j + i] = row_i[j] (one HMX row pair) */
static inline void __hmx_pack2(__fp16* dst, const __fp16* r0, const __fp16* r1) {
  for (int j = 0; j < 32; j++) { dst[2 * j] = r0[j]; dst[2 * j + 1] = r1[j]; }
}
/* rows r0, r1 of two horizontally adjacent tiles (64 fp16 each, 128-byte aligned) -> both tiles' row pair */
static inline void __hmx_pack2x2(__fp16* d0, __fp16* d1, const __fp16* r0, const __fp16* r1) {
  for (int j = 0; j < 32; j++) { d0[2*j] = r0[j]; d0[2*j+1] = r1[j]; d1[2*j] = r0[32+j]; d1[2*j+1] = r1[32+j]; }
}
static inline void __hmx_pack2x2_blk(__fp16* d0, __fp16* d1, const __fp16* r0, const __fp16* r1) {
  for (int q = 0; q < 16; q++) __hmx_pack2x2(d0 + 64 * q, d1 + 64 * q, r0 + 2 * q * (r1 - r0), r0 + (2 * q + 1) * (r1 - r0));
}
#else
/* one halfword vshuff of two full 128-byte rows gives the row pair of both tiles (low half: tile n, high half: n+1) */
static inline void __hmx_pack2x2(__fp16* d0, __fp16* d1, const __fp16* r0, const __fp16* r1) {
  __hmx_vp p = __builtin_HEXAGON_V6_vshuffvdd_128B(*(const __hmx_v*)r1, *(const __hmx_v*)r0, -2);
  *(__hmx_v*)d0 = __builtin_HEXAGON_V6_lo_128B(p);
  *(__hmx_v*)d1 = __builtin_HEXAGON_V6_hi_128B(p);
}
/* one K block (32 rows at a constant row stride from r0) of two adjacent tiles: loads issued in batches of 8 rows before the
 * shuffles and VTCM stores (as 16 separate __hmx_pack2x2 calls the stores may alias the next rows' loads, so every load
 * waits for the previous store) */
static inline void __hmx_pack2x2_blk(__fp16* d0, __fp16* d1, const __fp16* r0, const __fp16* r1) {
  const char* p = (const char*)r0;
  int st = (int)((const char*)r1 - (const char*)r0);
  for (int q0 = 0; q0 < 16; q0 += 4) {
    __hmx_v x0 = *(const __hmx_v*)(p + (2*q0+0) * st), x1 = *(const __hmx_v*)(p + (2*q0+1) * st);
    __hmx_v x2 = *(const __hmx_v*)(p + (2*q0+2) * st), x3 = *(const __hmx_v*)(p + (2*q0+3) * st);
    __hmx_v x4 = *(const __hmx_v*)(p + (2*q0+4) * st), x5 = *(const __hmx_v*)(p + (2*q0+5) * st);
    __hmx_v x6 = *(const __hmx_v*)(p + (2*q0+6) * st), x7 = *(const __hmx_v*)(p + (2*q0+7) * st);
    __hmx_vp a = __builtin_HEXAGON_V6_vshuffvdd_128B(x1, x0, -2), b = __builtin_HEXAGON_V6_vshuffvdd_128B(x3, x2, -2);
    __hmx_vp c = __builtin_HEXAGON_V6_vshuffvdd_128B(x5, x4, -2), d = __builtin_HEXAGON_V6_vshuffvdd_128B(x7, x6, -2);
    __hmx_v* o0 = (__hmx_v*)(d0 + 64 * q0);
    __hmx_v* o1 = (__hmx_v*)(d1 + 64 * q0);
    o0[0] = __builtin_HEXAGON_V6_lo_128B(a); o1[0] = __builtin_HEXAGON_V6_hi_128B(a);
    o0[1] = __builtin_HEXAGON_V6_lo_128B(b); o1[1] = __builtin_HEXAGON_V6_hi_128B(b);
    o0[2] = __builtin_HEXAGON_V6_lo_128B(c); o1[2] = __builtin_HEXAGON_V6_hi_128B(c);
    o0[3] = __builtin_HEXAGON_V6_lo_128B(d); o1[3] = __builtin_HEXAGON_V6_hi_128B(d);
  }
}
/* the same with HVX: an aligned 128-byte load never leaves the row's 128-byte block (rows are 64-byte aligned), vror brings
 * the row to the low half, one halfword vshuff interleaves the pair */
static inline void __hmx_pack2(__fp16* dst, const __fp16* r0, const __fp16* r1) {
  __hmx_v a = __builtin_HEXAGON_V6_vror_128B(*(const __hmx_v*)((unsigned)r0 & ~127u), (int)(unsigned)r0);
  __hmx_v b = __builtin_HEXAGON_V6_vror_128B(*(const __hmx_v*)((unsigned)r1 & ~127u), (int)(unsigned)r1);
  *(__hmx_v*)dst = __builtin_HEXAGON_V6_lo_128B(__builtin_HEXAGON_V6_vshuffvdd_128B(b, a, -2));
}
#endif
#ifndef HMX_IDX
#define HMX_IDX(i, j) (64 * ((i) / 2) + 2 * (j) + ((i) % 2))
#endif
#ifdef HMX_REF
#ifndef __HMX_CONV
#define __HMX_CONV
""" + _HMX_REF_CONV + r"""#endif
static __fp16 __hmx_ra[2][1024] __attribute__((aligned(128))), __hmx_rb[2][1024] __attribute__((aligned(128)));
static __fp16 __hmx_ro[1024] __attribute__((aligned(128)));
static double __hmx_racc[1024];
static int __hmx_t;
static inline __fp16* __hmx_sa(void) { return __hmx_ra[__hmx_t]; }
static inline __fp16* __hmx_sb(void) { return __hmx_rb[__hmx_t]; }
#define __HMX_CA @CA@
#define __HMX_CB @CB@
static __fp16 __hmx_rca[__HMX_CA + __HMX_CB][1024] __attribute__((aligned(128)));  /* one pool, B after A (as in VTCM) */
static inline __fp16* __hmx_ca(int i) { return __hmx_rca[i]; }
static inline __fp16* __hmx_cb(int i) { return __hmx_rca[__HMX_CA + i]; }
static inline void __hmx_begin(void) { for (int i = 0; i < 1024; i++) __hmx_racc[i] = 0.0; }
static inline void __hmx_mac(__fp16* a, __fp16* b) {
  const unsigned short *A = (const unsigned short*)a, *B = (const unsigned short*)b;
  for (int m = 0; m < 32; m++) for (int n = 0; n < 32; n++) {
    double s = 0.0;
    for (int k = 0; k < 32; k++) s += __hmx_h2d(A[HMX_IDX(m, k)]) * __hmx_h2d(B[HMX_IDX(k, n)]);
    __hmx_racc[HMX_IDX(m, n)] += s;
  }
  __hmx_t ^= 1;
}
static inline void __hmx_mac_span(__fp16* a, __fp16* b, int kt) {
  for (int k = 0; k < kt; k++) { __hmx_mac(a + 1024 * k, b + 1024 * k); }
}
static inline __fp16* __hmx_store(void) {
  unsigned short* o = (unsigned short*)__hmx_ro;
  for (int i = 0; i < 1024; i++) { o[i] = __hmx_d2h(__hmx_racc[i]); __hmx_racc[i] = 0.0; }
  return __hmx_ro;
}
static __fp16 __hmx_ro2[1024] __attribute__((aligned(128)));
static inline __fp16* __hmx_store2(void) {
  unsigned short* o = (unsigned short*)__hmx_ro2;
  for (int i = 0; i < 1024; i++) { o[i] = __hmx_d2h(__hmx_racc[i]); __hmx_racc[i] = 0.0; }
  return __hmx_ro2;
}
#else
extern unsigned char* __hmx_vtcm;
extern unsigned int __hmx_gen;
static int __hmx_t;
/* VTCM (2 KB aligned): A stage x2 @0, B stage x2 @4 KB, out @8 KB, bias table @10 KB */
static inline __fp16* __hmx_sa(void) { return (__fp16*)(__hmx_vtcm + 2048 * __hmx_t); }
static inline __fp16* __hmx_sb(void) { return (__fp16*)(__hmx_vtcm + 4096 + 2048 * __hmx_t); }
/* tile caches: A slots from @AO@ bytes, B slots after them (HMX_VTCM_KB: the runtime gives that much, 256 KB aligned) */
#define __HMX_CA @CA@
#define __HMX_CB @CB@
static inline __fp16* __hmx_ca(int i) { return (__fp16*)(__hmx_vtcm + @AO@ + 2048 * i); }
static inline __fp16* __hmx_cb(int i) { return (__fp16*)(__hmx_vtcm + @AO@ + 2048 * (__HMX_CA + i)); }
static inline void __hmx_begin(void) {
  static unsigned int init = 0;
  if (init != __hmx_gen) {
    for (int j = 0; j < 64; j++) ((unsigned int*)(__hmx_vtcm + 10240))[j] = 0;  /* zero bias */
    __asm__ volatile("bias = mxmem(%0)" :: "r"(__hmx_vtcm + 10240) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(__hmx_vtcm + 8192), "r"(0) : "memory");  /* clear the accumulator */
    init = __hmx_gen;
  }
}
static inline void __hmx_mac(__fp16* a, __fp16* b) {
  __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" :: "r"(a), "r"(2047), "r"(b), "r"(2047) : "memory");
  __hmx_t ^= 1;
}
static inline void __hmx_mac_span(__fp16* a, __fp16* b, int kt) {  /* kt consecutive K tiles, <= 32 per load pair */
  for (int k0 = 0; k0 < kt; k0 += 32) {
    int n = kt - k0 < 32 ? kt - k0 : 32;
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" :: "r"(a + 1024 * k0), "r"(n * 2048 - 1),
                     "r"(b + 1024 * k0), "r"(n * 2048 - 1) : "memory");
  }
}
static inline __fp16* __hmx_store(void) {
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(__hmx_vtcm + 8192), "r"(0) : "memory");
  return (__fp16*)(__hmx_vtcm + 8192);
}
/* the second output tile of a horizontal pair (VTCM @12 KB, free between the bias table and the tile caches) */
static inline __fp16* __hmx_store2(void) {
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(__hmx_vtcm + 12288), "r"(0) : "memory");
  return (__fp16*)(__hmx_vtcm + 12288);
}
#endif
/* output tile row pair q (IDX layout: rows 2q, 2q+1 interleaved) -> two 32-element rows at 64-byte aligned pointers */
#ifdef HMX_REF
static inline void __hmx_out2(__fp16* d0, __fp16* d1, const __fp16* o) {
  for (int j = 0; j < 32; j++) { d0[j] = o[2 * j]; d1[j] = o[2 * j + 1]; }
}
static inline void __hmx_deal2(__fp16* d, const __fp16* o) {
  for (int q = 0; q < 2; q++) __hmx_out2(d + 64 * q, d + 64 * q + 32, o + 64 * q);
}
/* row pair q of two horizontally adjacent output tiles (o0: columns 0..31, o1: 32..63) -> two full 64-element rows */
static inline void __hmx_outp(__fp16* d0, __fp16* d1, const __fp16* o0, const __fp16* o1) {
  for (int j = 0; j < 32; j++) { d0[j] = o0[2*j]; d1[j] = o0[2*j+1]; d0[32+j] = o1[2*j]; d1[32+j] = o1[2*j+1]; }
}
#else
static inline void __hmx_store64(__fp16* d, __hmx_v v) {  /* v's low 64 bytes -> d (64-byte aligned): byte-predicated vmem store
                                                             (no read of the destination block; the clang predicate builtins
                                                             assert in this toolchain, so the predicate is set in asm) */
  unsigned base = (unsigned)d & ~127u;
  if ((unsigned)d & 64u) {
    v = __builtin_HEXAGON_V6_vror_128B(v, 64);
    __asm__ volatile("q0 = vsetq(%2)\n if (!q0) vmem(%0+#0) = %1" :: "r"(base), "v"(v), "r"(64) : "q0", "memory");
  } else {
    __asm__ volatile("q0 = vsetq(%2)\n if (q0) vmem(%0+#0) = %1" :: "r"(base), "v"(v), "r"(64) : "q0", "memory");
  }
}
static inline void __hmx_out2(__fp16* d0, __fp16* d1, const __fp16* o) {
  __hmx_v v = __builtin_HEXAGON_V6_vdealh_128B(*(const __hmx_v*)o);  /* even halfwords (row 2q) low, odd (row 2q+1) high */
  __hmx_store64(d0, v);
  __hmx_store64(d1, __builtin_HEXAGON_V6_vror_128B(v, 64));
}
/* row pair q of two horizontally adjacent output tiles -> two full, 128-byte aligned 64-element rows: one vdealh per tile,
 * vmux/vror to combine the halves (the hand kernel's hmx_unpack2), plain aligned stores (no predicate, no half lines) */
static inline void __hmx_outp(__fp16* d0, __fp16* d1, const __fp16* o0, const __fp16* o1) {
  __hmx_v x = __builtin_HEXAGON_V6_vdealh_128B(*(const __hmx_v*)o0), y = __builtin_HEXAGON_V6_vdealh_128B(*(const __hmx_v*)o1);
  __hmx_v yr = __builtin_HEXAGON_V6_vror_128B(y, 64), xr = __builtin_HEXAGON_V6_vror_128B(x, 64);
  __hmx_v r0, r1;
  __asm__ volatile("q0 = vsetq(%4)\n %0 = vmux(q0,%2,%3)\n %1 = vmux(q0,%5,%6)" : "=&v"(r0), "=&v"(r1) : "v"(x), "v"(yr), "r"(64), "v"(xr), "v"(y) : "q0");
  *(__hmx_v*)d0 = r0;
  *(__hmx_v*)d1 = r1;
}
/* output row pairs q, q+1 -> rows 2q..2q+3 in order into a 256-byte accumulator-array vector: one vdealh per register (as
 * a __builtin_shufflevector clang doesn't find that and builds it lane by lane through the stack). The stores are asm: as C
 * stores clang forwards the registers into the epilogue's 32-lane row loads and assembles every row lane by lane (vinsert) */
static inline void __hmx_deal2(__fp16* d, const __fp16* o) {
  __asm__ volatile("vmem(%0+#0) = %1" :: "r"(d), "v"(__builtin_HEXAGON_V6_vdealh_128B(((const __hmx_v*)o)[0])) : "memory");
  __asm__ volatile("vmem(%0+#0) = %1" :: "r"(d + 64), "v"(__builtin_HEXAGON_V6_vdealh_128B(((const __hmx_v*)o)[1])) : "memory");
}
#endif
/* packed tiles of read-only operands are memoized by their first row pointer (same pointer = same tile within one
 * kernel call); __hmx_call_start() invalidates at every call */
static const void* __hmx_tag_a[__HMX_CA];
static const void* __hmx_tag_b[__HMX_CB];
static unsigned char __hmx_rr_a[__HMX_CA], __hmx_rr_b[__HMX_CB];
/* L2-prefetch the next K block of a 32-row operand (32 rows x 64 bytes at the row stride of r0, r1) */
static inline void __hmx_prefetch_next(const __fp16* r0, const __fp16* r1) {
#ifndef HMX_REF
  unsigned stride = (unsigned)((const char*)r1 - (const char*)r0);
  if (stride < 65536u) __builtin_HEXAGON_Y4_l2fetch((void*)((const char*)r0 + 32 * stride), (stride << 16) | (64u << 8) | 32u);
#else
  (void)r0; (void)r1;
#endif
}
/* L2-prefetch a whole K panel of a 32-column operand: rows x 64 bytes at the row stride of r0, r1 (l2fetch height <= 255) */
static inline void __hmx_prefetch_panel(const __fp16* r0, const __fp16* r1, int rows) {
#ifndef HMX_REF
  unsigned stride = (unsigned)((const char*)r1 - (const char*)r0);
  if (stride >= 65536u) return;
  for (int r = 0; r < rows; r += 240)
    __builtin_HEXAGON_Y4_l2fetch((void*)((const char*)r0 + r * stride), (stride << 16) | (64u << 8) | (unsigned)(rows - r < 240 ? rows - r : 240));
#else
  (void)r0; (void)r1; (void)rows;
#endif
}
/* the same for a pair of adjacent 32-column tiles packed together: whole 128-byte row lines. `ahead`: also fetch the next
 * pair's panel (the next 128 bytes of every row), which then streams in while this pair packs (only when the row stride
 * leaves room for it, i.e. the operand is wider than this pair) */
static inline void __hmx_prefetch_panel2(const __fp16* r0, const __fp16* r1, int rows, int ahead) {
#ifndef HMX_REF
  unsigned stride = (unsigned)((const char*)r1 - (const char*)r0);
  if (stride >= 65536u) return;
  if (ahead != 2)  /* 2: this pair's panel was already fetched as the previous pair's "ahead" */
    for (int r = 0; r < rows; r += 240)
      __builtin_HEXAGON_Y4_l2fetch((void*)((const char*)r0 + r * stride), (stride << 16) | (128u << 8) | (unsigned)(rows - r < 240 ? rows - r : 240));
  if (ahead && stride >= 256u)
    for (int r = 0; r < rows; r += 240)
      __builtin_HEXAGON_Y4_l2fetch((void*)((const char*)r0 + 128 + r * stride), (stride << 16) | (128u << 8) | (unsigned)(rows - r < 240 ? rows - r : 240));
#else
  (void)r0; (void)r1; (void)rows; (void)ahead;
#endif
}
/* L2-prefetch the K panel of a 32-row operand (A: 32 rows of K columns): each row's `bytes` contiguous bytes at the row stride
 * of r0, r1 (as 128-byte lines, <= 240 lines per l2fetch). __hmx_prefetch_panel is the 32-column (B) shape; used on A it
 * fetched 32*kt rows -- far past the operand, which faults the PD on the phone once that lands on an unmapped page. */
static inline void __hmx_prefetch_rows(const __fp16* r0, const __fp16* r1, int bytes) {
#ifndef HMX_REF
  unsigned stride = (unsigned)((const char*)r1 - (const char*)r0);
  int lines = (bytes + 127) / 128;
  for (int i = 0; i < 32; i++)
    for (int l = 0; l < lines; l += 240) {
      unsigned h = (unsigned)(lines - l < 240 ? lines - l : 240);
      __builtin_HEXAGON_Y4_l2fetch((void*)((const char*)r0 + i * stride + 128 * l), (128u << 16) | (128u << 8) | h);
    }
#else
  (void)r0; (void)r1; (void)bytes;
#endif
}
static inline void __hmx_call_start(void) {
  for (int i = 0; i < __HMX_CA; i++) { __hmx_tag_a[i] = 0; __hmx_rr_a[i] = 0; }
  for (int i = 0; i < __HMX_CB; i++) { __hmx_tag_b[i] = 0; __hmx_rr_b[i] = 0; }
}
/* returns the cached tile for key k, or 0 after claiming its slot (*slot = where to pack) */
/* set-associative on the K index: set k owns slots [k*ways, k*ways+ways); ways and k are known when the kernel is rendered
 * (ways = slots / K tiles), round-robin replacement */
static inline __fp16* __hmx_lookup(const void** tags, unsigned char* rr, __fp16* (*at)(int), const void* key, int k, int ways,
                                   __fp16** slot) {
  const void** t = tags + k * ways;
  for (int w = 0; w < ways; w++) if (t[w] == key) return at(k * ways + w);
  int v = rr[k]; rr[k] = (unsigned char)(v + 1 == ways ? 0 : v + 1);
  t[v] = key; *slot = at(k * ways + v); return 0;
}
/* ---- int8 ":cm" accumulator in HMX (hexagon_hmx_i8): A crouton = 64 rows x 32 bytes row-major, B block = W(k, n) at byte
 * 128*(k/4) + 4*n + k%4; the exact int32 tile comes out of four non-saturating byte-plane stores ---- */
typedef int __hmx_iv __attribute__((vector_size(128)));
#ifdef HMX_REF
static unsigned char __hmx_i8a[2][2048] __attribute__((aligned(128)));
static signed char __hmx_i8b[2][1024] __attribute__((aligned(128)));
static int __hmx_i8acc[2048] __attribute__((aligned(128))), __hmx_i8out[2048] __attribute__((aligned(128)));
static int __hmx_i8t;
static inline unsigned char* __hmx_i8sa(void) { return __hmx_i8a[__hmx_i8t]; }
static inline signed char* __hmx_i8sb(void) { return __hmx_i8b[__hmx_i8t]; }
static inline void __hmx_i8_begin(void) { for (int i = 0; i < 2048; i++) __hmx_i8acc[i] = 0; }
static inline void __hmx_i8_pack_a4(unsigned char* d, const unsigned char* r0, const unsigned char* r1, const unsigned char* r2, const unsigned char* r3) {
  for (int j = 0; j < 32; j++) { d[j] = r0[j]; d[32 + j] = r1[j]; d[64 + j] = r2[j]; d[96 + j] = r3[j]; }
}
static inline void __hmx_i8_pack_b4(signed char* d, const signed char* r0, const signed char* r1, const signed char* r2, const signed char* r3) {
  for (int n = 0; n < 32; n++) { d[4 * n] = r0[n]; d[4 * n + 1] = r1[n]; d[4 * n + 2] = r2[n]; d[4 * n + 3] = r3[n]; }
}
static inline void __hmx_i8_mac(const void* av, const void* bv) {
  const unsigned char* a = (const unsigned char*)av; const signed char* b = (const signed char*)bv;
  for (int m = 0; m < 64; m++) for (int n = 0; n < 32; n++) {
    int s = 0;
    for (int k = 0; k < 32; k++) s += (int)a[32 * m + k] * (int)b[128 * (k / 4) + 4 * n + k % 4];
    __hmx_i8acc[32 * m + n] += s;
  }
}
static inline const unsigned char* __hmx_i8_store(void) {
  for (int i = 0; i < 2048; i++) { __hmx_i8out[i] = __hmx_i8acc[i]; __hmx_i8acc[i] = 0; }
  return (const unsigned char*)__hmx_i8out;
}
static inline void __hmx_i8_addq(int* acc, const unsigned char* P, int q) {
  for (int i = 0; i < 128; i++) acc[i] += ((const int*)P)[128 * q + i];
}
static inline void __hmx_i8_pack_b4x4(signed char* d0, signed char* d1, signed char* d2, signed char* d3, const signed char* r0,
                                      const signed char* r1, const signed char* r2, const signed char* r3) {
  __hmx_i8_pack_b4(d0, r0, r1, r2, r3); __hmx_i8_pack_b4(d1, r0 + 32, r1 + 32, r2 + 32, r3 + 32);
  __hmx_i8_pack_b4(d2, r0 + 64, r1 + 64, r2 + 64, r3 + 64); __hmx_i8_pack_b4(d3, r0 + 96, r1 + 96, r2 + 96, r3 + 96);
}
static inline void __hmx_i8_pack_a4x4(unsigned char* d0, unsigned char* d1, unsigned char* d2, unsigned char* d3, const unsigned char* r0,
                                      const unsigned char* r1, const unsigned char* r2, const unsigned char* r3) {
  __hmx_i8_pack_a4(d0, r0, r1, r2, r3); __hmx_i8_pack_a4(d1, r0 + 32, r1 + 32, r2 + 32, r3 + 32);
  __hmx_i8_pack_a4(d2, r0 + 64, r1 + 64, r2 + 64, r3 + 64); __hmx_i8_pack_a4(d3, r0 + 96, r1 + 96, r2 + 96, r3 + 96);
}
static int __hmx_i8acc2[2048], __hmx_i8out2[2048];
static inline void __hmx_i8_mac2(const void* av, const void* bv) {
  const unsigned char* a = (const unsigned char*)av; const signed char* b = (const signed char*)bv;
  for (int m = 0; m < 64; m++) for (int n = 0; n < 32; n++) {
    int s0 = 0, s1 = 0;
    for (int k = 0; k < 32; k++) { s0 += (int)a[32 * m + k] * (int)b[128 * (k / 4) + 4 * n + k % 4]; s1 += (int)a[32 * m + k] * (int)b[1024 + 128 * (k / 4) + 4 * n + k % 4]; }
    __hmx_i8acc[32 * m + n] += s0; __hmx_i8acc2[32 * m + n] += s1;
  }
}
static inline const unsigned char* __hmx_i8_store2(unsigned char* P) {
  for (int i = 0; i < 2048; i++) { ((int*)P)[i] = __hmx_i8acc2[i]; __hmx_i8acc2[i] = 0; }
  return __hmx_i8_store();
}
#else
/* VTCM: A stage @0 (2 KB), B stage @2 KB (1 KB), plane tables @4 KB (4 x 256 B), byte planes @8 KB (8 KB); tile caches in
 * the fp16 path's slots (__hmx_ca / __hmx_cb, 2 KB each) from @AO */
static inline unsigned char* __hmx_i8sa(void) { return __hmx_vtcm; }
static inline signed char* __hmx_i8sb(void) { return (signed char*)(__hmx_vtcm + 2048); }
static inline void __hmx_i8_begin(void) {
  static unsigned int init = 0;
  if (init != __hmx_gen) {
    static const unsigned short sc[4] = {0x6000, 0x4000, 0x2000, 0x0800};  /* x1, /2^8, /2^16, /2^24 */
    for (int p = 0; p < 4; p++)
      for (int j = 0; j < 32; j++) { ((unsigned int*)(__hmx_vtcm + 4096))[64 * p + j] = sc[p]; ((unsigned int*)(__hmx_vtcm + 4096))[64 * p + 32 + j] = 0; }
    init = __hmx_gen;
  }
  __asm__ volatile("mxclracc" ::: "memory");
}
typedef int __hmx_vu __attribute__((vector_size(128), aligned(1)));
static inline __hmx_v __hmx_row32(const void* r) {
  /* the 32 bytes at r in lanes 0..31. A row inside its 128-byte line: the aligned line rotated. Otherwise (rows at 8-byte
   * steps, e.g. a stride-2 stem's overlapping windows) an unaligned load -- which reads up to 96 bytes past the row: the
   * runtime keeps 128 bytes of slack after every buffer */
  unsigned o = (unsigned)r & 127u;
  if (__builtin_expect(o <= 96u, 1)) return __builtin_HEXAGON_V6_vror_128B(*(const __hmx_v*)((unsigned)r & ~127u), (int)o);
  return (__hmx_v)*(const __hmx_vu*)r;
}
static inline void __hmx_i8_pack_a4(unsigned char* d, const unsigned char* r0, const unsigned char* r1, const unsigned char* r2, const unsigned char* r3) {
  __hmx_vp x = __builtin_HEXAGON_V6_vshuffvdd_128B(__hmx_row32(r1), __hmx_row32(r0), -32);
  __hmx_vp y = __builtin_HEXAGON_V6_vshuffvdd_128B(__hmx_row32(r3), __hmx_row32(r2), -32);
  *(__hmx_v*)d = __builtin_HEXAGON_V6_lo_128B(__builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_lo_128B(y), __builtin_HEXAGON_V6_lo_128B(x), -64));
}
static inline void __hmx_i8_pack_b4(signed char* d, const signed char* r0, const signed char* r1, const signed char* r2, const signed char* r3) {
  __hmx_vp x = __builtin_HEXAGON_V6_vshuffvdd_128B(__hmx_row32(r1), __hmx_row32(r0), -1);
  __hmx_vp y = __builtin_HEXAGON_V6_vshuffvdd_128B(__hmx_row32(r3), __hmx_row32(r2), -1);
  *(__hmx_v*)d = __builtin_HEXAGON_V6_lo_128B(__builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_lo_128B(y), __builtin_HEXAGON_V6_lo_128B(x), -2));
}
static inline void __hmx_i8_mac(const void* a, const void* b) {
  __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3) }" :: "r"(a), "r"(0x7ff), "r"(b), "r"(0x3ff) : "memory");
}
static inline const unsigned char* __hmx_i8_store(void) {  /* the exact int32 tile as four byte planes (VTCM @8 KB) */
  unsigned char* P = __hmx_vtcm + 8192;
  for (int p = 0; p < 4; p++) {
    __asm__ volatile("bias = mxmem2(%0)" :: "r"(__hmx_vtcm + 4096 + 256 * p) : "memory");
    if (p < 3) __asm__ volatile("mxmem(%0,%1):after:retain:cm.ub = acc" :: "r"(P + 2048 * p), "r"(0) : "memory");
    else __asm__ volatile("mxmem(%0,%1):after:cm.ub = acc" :: "r"(P + 2048 * p), "r"(0) : "memory");
  }
  return P;
}
/* four rows (k..k+3, 128-byte aligned lines) of four horizontally adjacent 32-column tiles -> each tile's 128-byte W-layout
 * row group: byte interleave of the row pairs, then halfword interleave (d0..d3: tiles n..n+3) */
static inline void __hmx_i8_pack_b4x4(signed char* d0, signed char* d1, signed char* d2, signed char* d3, const signed char* r0,
                                      const signed char* r1, const signed char* r2, const signed char* r3) {
  __hmx_vp x = __builtin_HEXAGON_V6_vshuffvdd_128B(*(const __hmx_v*)r1, *(const __hmx_v*)r0, -1);
  __hmx_vp y = __builtin_HEXAGON_V6_vshuffvdd_128B(*(const __hmx_v*)r3, *(const __hmx_v*)r2, -1);
  __hmx_vp lo = __builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_lo_128B(y), __builtin_HEXAGON_V6_lo_128B(x), -2);
  __hmx_vp hi = __builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_hi_128B(y), __builtin_HEXAGON_V6_hi_128B(x), -2);
  *(__hmx_v*)d0 = __builtin_HEXAGON_V6_lo_128B(lo); *(__hmx_v*)d1 = __builtin_HEXAGON_V6_hi_128B(lo);
  *(__hmx_v*)d2 = __builtin_HEXAGON_V6_lo_128B(hi); *(__hmx_v*)d3 = __builtin_HEXAGON_V6_hi_128B(hi);
}
/* four rows (128-byte aligned lines, e.g. 4 NHWC pixels of 128 channels) -> the 128-byte row group of each of four activation
 * tiles (channel blocks 0..3): a 4x4 transpose of 32-byte blocks, two vshuff stages; each line is read once for four tiles */
static inline void __hmx_i8_pack_a4x4(unsigned char* d0, unsigned char* d1, unsigned char* d2, unsigned char* d3, const unsigned char* r0,
                                      const unsigned char* r1, const unsigned char* r2, const unsigned char* r3) {
  __hmx_vp x = __builtin_HEXAGON_V6_vshuffvdd_128B(*(const __hmx_v*)r1, *(const __hmx_v*)r0, -32);
  __hmx_vp y = __builtin_HEXAGON_V6_vshuffvdd_128B(*(const __hmx_v*)r3, *(const __hmx_v*)r2, -32);
  __hmx_vp lo = __builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_lo_128B(y), __builtin_HEXAGON_V6_lo_128B(x), -64);
  __hmx_vp hi = __builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_hi_128B(y), __builtin_HEXAGON_V6_hi_128B(x), -64);
  *(__hmx_v*)d0 = __builtin_HEXAGON_V6_lo_128B(lo); *(__hmx_v*)d1 = __builtin_HEXAGON_V6_hi_128B(lo);
  *(__hmx_v*)d2 = __builtin_HEXAGON_V6_lo_128B(hi); *(__hmx_v*)d3 = __builtin_HEXAGON_V6_hi_128B(hi);
}
/* weight :deep: 64 columns (b = [tile n | tile n+1], 2 KB) into both accumulators */
static inline void __hmx_i8_mac2(const void* a, const void* b) {
  __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3):deep }" :: "r"(a), "r"(0x7ff), "r"(b), "r"(0x7ff) : "memory");
}
/* both accumulators of a :deep pair as byte planes: the first (tile n) @8 KB, the second (n+1) at P */
static inline const unsigned char* __hmx_i8_store2(unsigned char* P) {  /* P: 8 KB of VTCM for the second tile's planes */
  const unsigned char* P0 = __hmx_i8_store();
  for (int p = 0; p < 4; p++) {
    __asm__ volatile("bias = mxmem2(%0)" :: "r"(__hmx_vtcm + 4096 + 256 * p) : "memory");
    if (p < 3) __asm__ volatile("mxmem(%0,%1):after:retain:cm.ub = acc" :: "r"(P + 2048 * p), "r"(0) : "memory");
    else __asm__ volatile("mxmem(%0,%1):after:cm.ub = acc" :: "r"(P + 2048 * p), "r"(0) : "memory");
  }
  return P0;
}
/* acc (int32, 128-byte aligned) += tile rows 4q .. 4q+3 (128 ints) from the byte planes: one byte -> word interleave */
static inline void __hmx_i8_addq(int* acc, const unsigned char* P, int q) {
  const __hmx_v* v = (const __hmx_v*)P;
  __hmx_vp b01 = __builtin_HEXAGON_V6_vshuffvdd_128B(v[16 + q], v[q], -1), b23 = __builtin_HEXAGON_V6_vshuffvdd_128B(v[48 + q], v[32 + q], -1);
  __hmx_vp w0 = __builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_lo_128B(b23), __builtin_HEXAGON_V6_lo_128B(b01), -2);
  __hmx_vp w1 = __builtin_HEXAGON_V6_vshuffvdd_128B(__builtin_HEXAGON_V6_hi_128B(b23), __builtin_HEXAGON_V6_hi_128B(b01), -2);
  __hmx_iv* a = (__hmx_iv*)acc;
  a[0] += (__hmx_iv)__builtin_HEXAGON_V6_lo_128B(w0); a[1] += (__hmx_iv)__builtin_HEXAGON_V6_hi_128B(w0);
  a[2] += (__hmx_iv)__builtin_HEXAGON_V6_lo_128B(w1); a[3] += (__hmx_iv)__builtin_HEXAGON_V6_hi_128B(w1);
}
#endif
/* ---- ORT-exact requantization of 32 int32 accumulators (QLinearConv / a QDQ Conv or MatMul after ORT's QDQ fusion):
 *   y = clamp(rne(fp32(fp32(a + b) * m)) + zy, lo, 255)
 * in integer HVX ops. V69 HVX has no IEEE fp32 (its sf encodings compute qf32 on the phone; hexagon-sim runs them as IEEE, so
 * the simulator can't catch that), so ORT's two fp32 roundings are emulated exactly: fp32(acc) = |acc| rounded to 24
 * significant bits (RNE), the 24 x 24-bit mantissa product with m's mantissa exact in two words, rounded to 24 bits (RNE),
 * then to an integer (RNE) at the combined exponent, saturated. m > 0, normal ---- */
typedef int __hmx_i32x32 __attribute__((ext_vector_type(32)));
typedef float __hmx_f32x32 __attribute__((ext_vector_type(32)));
#ifdef HMX_REF
static inline unsigned char __hmx_rq1s(int a, int b, float m, int zy, int lo) {
  volatile float f = (float)(a + b), v = f * m;
  float l = (float)(lo - zy), h = (float)(255 - zy), c = v < l ? l : v > h ? h : v;
  volatile float t = c + 12582912.0f;
  return (unsigned char)((int)(t - 12582912.0f) + zy);
}
static inline void __hmx_rq1(unsigned char* d, __hmx_i32x32 a, __hmx_i32x32 b, __hmx_f32x32 m, int zy, int lo) {
  for (int i = 0; i < 32; i++) d[i] = __hmx_rq1s(a[i], b[i], m[i], zy, lo);
}
static inline void __hmx_rq4(unsigned char* d0, unsigned char* d1, unsigned char* d2, unsigned char* d3, const int* acc,
                             __hmx_i32x32 b, __hmx_f32x32 m, int zy, int lo) {
  const __hmx_i32x32* a = (const __hmx_i32x32*)acc;
  __hmx_rq1(d0, a[0], b, m, zy, lo); __hmx_rq1(d1, a[1], b, m, zy, lo); __hmx_rq1(d2, a[2], b, m, zy, lo); __hmx_rq1(d3, a[3], b, m, zy, lo);
}
typedef unsigned int __hmx_u32x32 __attribute__((ext_vector_type(32)));
#define __hmx_rq4f(d0, d1, d2, d3, acc, b, m, zy, lo, fl) __hmx_rq4(d0, d1, d2, d3, acc, b, m, zy, lo)
#define __hmx_rq4x(d0, d1, d2, d3, acc, b, m, zy, lo) do {} while (0)
static inline int __hmx_rq_any(__hmx_u32x32 fl) { (void)fl; return 0; }
#else
typedef unsigned int __hmx_u32x32 __attribute__((ext_vector_type(32)));
static inline __hmx_u32x32 __hmx_bitlen(__hmx_u32x32 x) { return (__hmx_u32x32)32 - (__hmx_u32x32)__builtin_HEXAGON_V6_vcl0w_128B((__hmx_v)x); }
static inline __hmx_u32x32 __hmx_rne_shr(__hmx_u32x32 q, __hmx_u32x32 sh) {  /* rne(q / 2^sh), sh in [0, 31] */
  __hmx_u32x32 one = 1, r = q >> sh, rem = q & ((one << sh) - one), half = (one << sh) >> one;
  __hmx_i32x32 up = (sh > 0) & ((rem > half) | ((rem == half) & ((r & one) == one)));
  return r + ((__hmx_u32x32)up & one);
}
/* 32 int32 accumulators (+ bias) and m's fp32 bits -> 32 words y in [lo, 255]; out of line: only for rows with some
 * |acc| > 2^24 */
__attribute__((noinline)) static __hmx_v __hmx_rqw(__hmx_i32x32 acc, __hmx_u32x32 mbits, int zy, int lo) {
  __hmx_u32x32 one = 1, z = 0;
  __hmx_i32x32 neg = acc < 0;
  __hmx_u32x32 u = (__hmx_u32x32)(neg ? -acc : acc), na = __hmx_bitlen(u), sa = na > 24 ? na - 24 : z;
  __hmx_u32x32 ma = __hmx_rne_shr(u, sa), mm = (mbits & 0x7fffff) | 0x800000;           /* fp32(acc) = ma 2^sa, m = mm 2^em */
  __hmx_i32x32 em = (__hmx_i32x32)((mbits >> 23) & 255) - 150;
  __hmx_u32x32 al = ma & 0xfff, ah = ma >> 12, ml = mm & 0xfff, mh = mm >> 12;          /* ma * mm = hw 2^24 + lw, exact */
  __hmx_u32x32 t1 = ah * ml + al * mh, lf = al * ml + ((t1 & 0xfff) << 12);
  __hmx_u32x32 lw = lf & 0xffffff, hw = ah * mh + (t1 >> 12) + (lf >> 24);
  __hmx_u32x32 n = hw > z ? __hmx_bitlen(hw) + 24 : __hmx_bitlen(lw), sp = n > 24 ? n - 24 : z;   /* product -> 24 bits */
  __hmx_u32x32 q = (hw << (24 - sp)) | (lw >> sp), rem = lw & ((one << sp) - one), half = (one << sp) >> one;
  __hmx_i32x32 up = (sp > z) & ((rem > half) | ((rem == half) & ((q & one) == one)));
  __hmx_u32x32 pq = q + ((__hmx_u32x32)up & one);
  __hmx_i32x32 e = (__hmx_i32x32)sa + em + (__hmx_i32x32)sp;                               /* value = pq 2^e */
  __hmx_u32x32 left = (__hmx_u32x32)(e > 8 ? 9 : e), s = (__hmx_u32x32)(-e);
  __hmx_u32x32 yl = pq == z ? z : e > 8 ? (__hmx_u32x32)256 : pq > ((__hmx_u32x32)256 >> left) ? (__hmx_u32x32)256 : pq << left;
  __hmx_u32x32 yr = s >= 32 ? z : __hmx_rne_shr(pq, s & 31);
  __hmx_i32x32 y = (__hmx_i32x32)(e >= 0 ? yl : yr);
  y = neg ? -y : y;
  y = y < lo - zy ? lo - zy : y > 255 - zy ? 255 - zy : y;
  return (__hmx_v)(y + zy);
}
/* the common case, |acc| <= 2^24 (fp32(acc) exact), as fixed point: r = floor(|acc| 2^L * mm7 / 2^32) = floor(|v| 2^F) with
 * |acc| 2^L normalized below 2^31 and mm7 = m's 24-bit mantissa << 7, F = L + emf (emf = -em - 25); y = round half up of
 * r / 2^F. It differs from ORT only where |v| is within truncation (a unit of r) or ORT's double rounding (|v| 2^-24 <= 2^-16)
 * of a .5: those lanes set *flag and the caller redoes the rows with __hmx_rqw */
__attribute__((always_inline)) static inline __hmx_v __hmx_rqwf(__hmx_i32x32 acc, __hmx_u32x32 mm7, __hmx_i32x32 emf, int zy, int lo,
                                                                 __hmx_u32x32* flag) {
  __hmx_u32x32 one = 1, u = (__hmx_u32x32)__builtin_HEXAGON_V6_vabsw_128B((__hmx_v)acc);
  __hmx_u32x32 L = __builtin_elementwise_min((__hmx_u32x32)__builtin_HEXAGON_V6_vcl0w_128B((__hmx_v)u) - one, (__hmx_u32x32)30);
  __hmx_u32x32 a = u << L;
  __hmx_vp p = __builtin_HEXAGON_V6_vmpyowh_64_acc_128B(__builtin_HEXAGON_V6_vmpyewuh_64_128B((__hmx_v)a, (__hmx_v)mm7), (__hmx_v)a, (__hmx_v)mm7);
  __hmx_u32x32 r = (__hmx_u32x32)__builtin_HEXAGON_V6_hi_128B(p);
  __hmx_i32x32 F = (__hmx_i32x32)L + emf;
  __hmx_u32x32 Fc = (__hmx_u32x32)__builtin_elementwise_min(__builtin_elementwise_max(F, (__hmx_i32x32)1), (__hmx_i32x32)31);
  __hmx_u32x32 half = one << (Fc - one), y = (r + half) >> Fc;
  __hmx_u32x32 d = (__hmx_u32x32)__builtin_HEXAGON_V6_vabsw_128B((__hmx_v)((__hmx_i32x32)(r & ((one << Fc) - one)) - (__hmx_i32x32)half));
  __hmx_u32x32 win = (one << (__hmx_u32x32)__builtin_elementwise_max(F - 16, (__hmx_i32x32)0)) + 3;
  *flag |= (__hmx_u32x32)(d <= win) & (__hmx_u32x32)(F > 0);
  y = F <= 0 ? (__hmx_u32x32)256 : y;
  __hmx_i32x32 ys = acc < 0 ? -(__hmx_i32x32)y : (__hmx_i32x32)y;
  ys = __builtin_elementwise_min(__builtin_elementwise_max(ys, (__hmx_i32x32)(lo - zy)), (__hmx_i32x32)(255 - zy));
  return (__hmx_v)(ys + zy);
}
/* nonzero when some lane of x has |x| > 2^24 (x = OR of a ^ (a >> 31) over the rows) or a flag set (flags: 0 or -1) */
static inline int __hmx_any_big(__hmx_i32x32 x, __hmx_u32x32 flags) {
  __hmx_v v = __builtin_HEXAGON_V6_vor_128B(__builtin_HEXAGON_V6_vand_128B((__hmx_v)x, __builtin_HEXAGON_V6_lvsplatw_128B((int)0xff000000)), (__hmx_v)flags);
  for (int r = 64; r >= 4; r >>= 1) v = __builtin_HEXAGON_V6_vor_128B(v, __builtin_HEXAGON_V6_vror_128B(v, r));
  return __builtin_HEXAGON_V6_extractw_128B(v, 0);
}
/* bytes 32j .. 32j+31 of v to d (32-byte aligned): rotated into place in d's 128-byte line, one byte-predicated store */
static inline void __hmx_st32(unsigned char* d, __hmx_v v, int j) {
  unsigned off = (unsigned)d & 127u;
  v = __builtin_HEXAGON_V6_vror_128B(v, (int)((32u * (unsigned)j - off) & 127u));
  __asm__ volatile("q0 = vsetq2(%2)\n q1 = vsetq(%3)\n q0 = and(q0, !q1)\n if (q0) vmem(%0+#0) = %1"
                   :: "r"((unsigned)d & ~127u), "v"(v), "r"(off + 32u), "r"(off) : "q0", "q1", "memory");
}
__attribute__((always_inline)) static inline void __hmx_rq1(unsigned char* d, __hmx_i32x32 a, __hmx_i32x32 b, __hmx_f32x32 m, int zy, int lo) {
  __hmx_i32x32 x = a + b;
  __hmx_v z = __builtin_HEXAGON_V6_vd0_128B();
  __hmx_u32x32 mb = (__hmx_u32x32)m, mm7 = ((mb & 0x7fffff) | 0x800000) << 7, fl = 0;
  __hmx_i32x32 emf = 125 - (__hmx_i32x32)((mb >> 23) & 255);  /* -em - 25, em = exp - 150 */
  __hmx_v y = __hmx_rqwf(x, mm7, emf, zy, lo, &fl);
  if (__builtin_expect(__hmx_any_big(x ^ (x >> 31), fl), 0)) y = __hmx_rqw(x, mb, zy, lo);
  __hmx_st32(d, __builtin_HEXAGON_V6_vpackhub_sat_128B(z, __builtin_HEXAGON_V6_vpackwh_sat_128B(z, y)), 0);
}
/* four rows sharing bias and scale (the four rows at acc, one 128-lane accumulator-array vector): the fixed-point fast path,
 * flags (window lanes, |acc| > 2^24) ORed into *fl -- checked once per output tile (__hmx_rq_any: moving a vector to a scalar
 * stalls ~250 cycles, per group it cost 3x the whole requantization) -- then two word -> halfword packs, one halfword -> byte
 * pack, the rows at bytes 0, 32, 64, 96 */
__attribute__((always_inline)) static inline void __hmx_rq4f(unsigned char* d0, unsigned char* d1, unsigned char* d2, unsigned char* d3,
    const int* acc, __hmx_i32x32 b, __hmx_f32x32 m, int zy, int lo, __hmx_u32x32* fl) {
  const __hmx_i32x32* a = (const __hmx_i32x32*)acc;
  __hmx_u32x32 mb = (__hmx_u32x32)m, mm7 = ((mb & 0x7fffff) | 0x800000) << 7;
  __hmx_i32x32 emf = 125 - (__hmx_i32x32)((mb >> 23) & 255);  /* -em - 25, em = exp - 150 */
  __hmx_i32x32 x0 = a[0] + b, x1 = a[1] + b, x2 = a[2] + b, x3 = a[3] + b;
  __hmx_v y0 = __hmx_rqwf(x0, mm7, emf, zy, lo, fl), y1 = __hmx_rqwf(x1, mm7, emf, zy, lo, fl);
  __hmx_v y2 = __hmx_rqwf(x2, mm7, emf, zy, lo, fl), y3 = __hmx_rqwf(x3, mm7, emf, zy, lo, fl);
  *fl |= (__hmx_u32x32)(((x0 ^ (x0 >> 31)) | (x1 ^ (x1 >> 31)) | (x2 ^ (x2 >> 31)) | (x3 ^ (x3 >> 31))) & (int)0xff000000);
  __hmx_v p = __builtin_HEXAGON_V6_vpackhub_sat_128B(__builtin_HEXAGON_V6_vpackwh_sat_128B(y3, y2), __builtin_HEXAGON_V6_vpackwh_sat_128B(y1, y0));
  __hmx_st32(d0, p, 0); __hmx_st32(d1, p, 1); __hmx_st32(d2, p, 2); __hmx_st32(d3, p, 3);
}
/* the same four rows through the exact emulation (a tile whose fast path flagged something) */
__attribute__((noinline)) static void __hmx_rq4x(unsigned char* d0, unsigned char* d1, unsigned char* d2, unsigned char* d3,
    const int* acc, __hmx_i32x32 b, __hmx_f32x32 m, int zy, int lo) {
  const __hmx_i32x32* a = (const __hmx_i32x32*)acc;
  __hmx_u32x32 mb = (__hmx_u32x32)m;
  __hmx_v y0 = __hmx_rqw(a[0] + b, mb, zy, lo), y1 = __hmx_rqw(a[1] + b, mb, zy, lo), y2 = __hmx_rqw(a[2] + b, mb, zy, lo), y3 = __hmx_rqw(a[3] + b, mb, zy, lo);
  __hmx_v p = __builtin_HEXAGON_V6_vpackhub_sat_128B(__builtin_HEXAGON_V6_vpackwh_sat_128B(y3, y2), __builtin_HEXAGON_V6_vpackwh_sat_128B(y1, y0));
  __hmx_st32(d0, p, 0); __hmx_st32(d1, p, 1); __hmx_st32(d2, p, 2); __hmx_st32(d3, p, 3);
}
static inline int __hmx_rq_any(__hmx_u32x32 fl) {
  __hmx_v v = (__hmx_v)fl;
  for (int r = 64; r >= 4; r >>= 1) v = __builtin_HEXAGON_V6_vor_128B(v, __builtin_HEXAGON_V6_vror_128B(v, r));
  return __builtin_HEXAGON_V6_extractw_128B(v, 0);
}
#endif
""".replace("@CA@", str(_HMX_CA)).replace("@CB@", str(_HMX_CB)).replace("@AO@", str(_HMX_AO))

def _hmx_lane(u:UOp):
  # lane j of a vector value: INDEX(value, CAST(CONST j)) -> (value, j)
  if u.op is not Ops.INDEX or len(u.src) != 2: return None
  i = u.src[1]
  while i.op is Ops.CAST: i = i.src[0]
  return (u.src[0], i.arg) if i.op is Ops.CONST else None

def _hmx_rows(stack:UOp):
  # a 1024-lane operand whose lanes are HMX IDX(i, j) = lane base_i + j of vector value v_i (32/64/128 lanes, base_i a
  # multiple of 32) -> the 32 (v_i, base_i), else None
  if stack.op is not Ops.STACK or len(stack.src) != 1024: return None
  rows: list = [None]*32
  for p, s in enumerate(stack.src):
    if (lj:=_hmx_lane(s)) is None: return None
    i, j = 2*(p//64) + p%2, (p%64)//2
    v, lane = lj
    if v.max_numel() not in (32, 64, 128) or (lane - j) % 32 != 0 or not 0 <= lane - j < v.max_numel(): return None
    if rows[i] is None: rows[i] = (v, lane - j)
    elif rows[i] != (v, lane - j): return None
  return rows

_ILV = ",".join(f"{j},{j+32}" for j in range(32))
def _hmx_param(u:UOp):
  # the PARAM (or register BUFFER) an address expression indexes
  while u.op in (Ops.SHRINK, Ops.INDEX, Ops.AFTER, Ops.CAST, Ops.BITCAST): u = u.src[0]
  return u

def _hmx_const(u:UOp):
  while u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST else None

def _hmx_direct_out(uops, pos, users, at:int, stores):
  # after the reduce loop tinygrad reloads the accumulator array and stores lane permutations of it to the output. If that
  # is all that happens to it, return (pointer uop for each tile row 0..31 as an "(expr+off)"-able uop list, uops to drop)
  buf = _hmx_param(stores[0].src[0])
  elem_lane: dict[int, int] = {}
  for so in stores:
    if (off:=_hmx_const(so.src[0].src[1])) is None or len(so.src[0].src) < 2: return None
    for t, x in enumerate(so.src[1].src): elem_lane[off + t] = _hmx_lane(x)[1]
  loads = [u for u in uops[at+1:] if u.op is Ops.LOAD and _hmx_param(u.src[0]) is buf]
  if not loads: return None
  gone: set[UOp] = set(loads)
  rows: dict[int, UOp] = {}
  for ld in loads:
    if (off:=_hmx_const(ld.src[0].src[1])) is None: return None
    for u in users.get(ld, []):
      if u.op is Ops.STORE and u.src[1] is ld:
        elems, outs = [[(off + t) for t in range(ld.max_numel())]], [u]
      elif _hmx_lane(u) is not None:
        gone.add(u)
        outs = []
        for st in users.get(u, []):
          if st.op is not Ops.STACK: return None
          gone.add(st)
          for g in users.get(st, []):
            if g.op is not Ops.STORE or g.src[1] is not st: return None
            outs.append(g)
        elems = []
        for g in outs:
          el = []
          for x in g.src[1].src:
            if (lj:=_hmx_lane(x)) is None or lj[0].op is not Ops.LOAD or _hmx_param(lj[0].src[0]) is not buf: return None
            el.append(_hmx_const(lj[0].src[0].src[1]) + lj[1])
          elems.append(el)
      else: return None
      for g, el in zip(outs, elems):
        if _hmx_param(g.src[0]).op is not Ops.PARAM or len(el) % 32: return None
        gone.add(g)
        for c in range(len(el) // 32):
          lanes = [elem_lane.get(x) for x in el[32*c:32*c+32]]
          if any(l is None for l in lanes): return None
          i = 2 * (lanes[0] // 64) + lanes[0] % 2
          if lanes != [64*(i//2) + 2*j + i%2 for j in range(32)]: return None
          ptr = g.src[0] if c == 0 else UOp(Ops.CUSTOMI, g.src[0].dtype, (g.src[0],), f"({{0}}+{32*c})")
          if rows.setdefault(i, ptr) is not ptr: return None
  if sorted(rows) != list(range(32)): return None
  return [rows[i] for i in range(32)], gone

def _hmx_rows_i8(stack:UOp, n:int, lane_of):
  # the rows of an int8 operand: lane_of(r, j) = the fragment lane of row r, column j (32 columns) -> [(value, base)] per
  # row when every row is 32 consecutive lanes of one loaded vector, else None
  # a row may also run on from the end of one load into lane 0 of the load right after it in memory (a stride-2 conv's
  # overlapping rows): only the row's start address is used, and the bytes are contiguous
  if stack.op is not Ops.STACK: return None
  rows, nxt = [], {}
  def follows(v:UOp, v2:UOp) -> bool:
    if (v, v2) not in nxt:
      ok = v.src[0].op in (Ops.INDEX, Ops.SHRINK) and v2.src[0].op is v.src[0].op and v.src[0].src[0] is v2.src[0].src[0]
      if ok:
        i0, i1 = v.src[0].src[1], v2.src[0].src[1]
        rng = list({u for u in (*i0.toposort(), *i1.toposort()) if u.op is Ops.RANGE})
        ok = _hmx_const_delta(lambda env: None if (a:=_hmx_eval(i1, env)) is None or (z:=_hmx_eval(i0, env)) is None else a - z,
                              rng) == v.max_numel()
      nxt[(v, v2)] = ok
    return nxt[(v, v2)]
  for r in range(n):
    lj = [_hmx_lane(stack.src[lane_of(r, j)]) for j in range(32)]
    if any(x is None for x in lj): return None
    for (v0, b0), (v1, b1) in zip(lj, lj[1:]):
      if not ((v1 is v0 and b1 == b0 + 1) or (b0 == v0.max_numel() - 1 and b1 == 0 and follows(v0, v1))): return None
    rows.append(lj[0])
  return rows

def _hmx_i8_rewrite(w:UOp, uops, pos, users, drop, before, after, replace, swap_out:list):
  # int8 :cm tile op with the accumulator kept in HMX: per K block pack A (64 rows x 32 B) and B (32 rows x 32 B -> W layout)
  # into VTCM stages and issue one :cm load pair; after the reduce loop four byte-plane stores give the exact int32 tile,
  # which is added into tinygrad's accumulator array (whose per-K-block += of the WMMA lanes is dropped)
  ra = _hmx_rows_i8(w.src[0], 64, lambda m, k: 32 * m + k)
  rb = _hmx_rows_i8(w.src[1], 32, lambda k, n: 128 * (k // 4) + 4 * n + k % 4)
  if ra is None or rb is None: return 21
  ends = [e for e in uops if e.op is Ops.END and len(e.src) > 1 and e.src[1].op is Ops.RANGE and e.src[1].arg[-1] == AxisType.REDUCE
          and pos[e.src[1]] < pos[w] < pos[e]]
  e = min(ends, key=lambda e: pos[e]-pos[e.src[1]]) if ends else None
  # every reduce loop around the tile op (a 3x3 conv: the tap row dy outside the K blocks of dx*C + c): the accumulator is begun
  # before the outermost and stored after it, and a cached tile's K slot is the index over all of them (outermost first)
  eo = max(ends, key=lambda e: pos[e]-pos[e.src[1]]) if ends else None
  reds = [x.src[1] for x in sorted(ends, key=lambda x: pos[x.src[1]])]
  # consumers: lane INDEX -> STACK (128) -> ADD(LOAD acc, STACK) -> STORE acc
  adds: list[UOp] = []
  for lane in users.get(w, []):
    if _hmx_lane(lane) is None: return 23
    for st in users.get(lane, []):
      if st.op is not Ops.STACK: return 24
      for a in users.get(st, []):
        if a.op is not Ops.ADD or a.src[1] is not st or a.src[0].op is not Ops.LOAD: return 25
        if a not in adds: adds.append(a)
  stores = []
  for a in adds:
    so = [u for u in users.get(a, []) if u.op is Ops.STORE and u.src[1] is a]
    if len(so) != 1 or len(users.get(a, [])) != 1: return 26
    lanes = [_hmx_lane(x)[1] for x in a.src[1].src]
    if lanes != list(range(lanes[0], lanes[0] + len(lanes))) or len(lanes) % 32: return 27
    stores.append((so[0], lanes[0], len(lanes)))
  vals = list(dict.fromkeys(v for v, _ in ra + rb))
  if not all(v.op is Ops.LOAD and len(v.src) == 1 for v in vals): return 28
  dead = {w, *w.src[:3], *w.src[0].src, *w.src[1].src, *vals} | {x for x in w.src[2].src if x.op not in (Ops.CONST, Ops.CAST)}
  for a in adds: dead |= {a, a.src[0], a.src[1], *a.src[1].src}
  for so, _, _ in stores: dead.add(so)
  bad = [(d.op, v.op) for d in dead for v in users.get(d, []) if v not in dead and v.op not in (Ops.GROUP, Ops.END)]
  if bad:
    if getenv("HMX_DEBUG"): print("i8 live users of dropped uops:", bad[:6])
    return 29
  drop |= dead
  before.setdefault(pos[eo.src[1]] if eo is not None else pos[w], []).append(UOp(Ops.CUSTOM, dtypes.void, (), "__hmx_i8_begin();"))
  ptr = [f"((const unsigned char*){{{vals.index(v)}}}+{b})" if b else f"((const unsigned char*){{{vals.index(v)}}})" for v, b in ra] + \
        [f"((const signed char*){{{vals.index(v)}}}+{b})" if b else f"((const signed char*){{{vals.index(v)}}})" for v, b in rb]
  pa = "".join(f" __hmx_i8_pack_a4((unsigned char*)_a+{128*q}, {ptr[4*q]}, {ptr[4*q+1]}, {ptr[4*q+2]}, {ptr[4*q+3]});" for q in range(16))
  pb = "".join(f" __hmx_i8_pack_b4((signed char*)_b+{128*q}, {ptr[64+4*q]}, {ptr[65+4*q]}, {ptr[66+4*q]}, {ptr[67+4*q]});" for q in range(8))
  srcs = tuple(v.src[0] for v in vals)
  code = "{{ void* _a = __hmx_i8sa(); void* _b = __hmx_i8sb();" + pa + pb + " __hmx_i8_mac(_a, _b); }}"
  loops = _hmx_tile_loops(uops, pos[w])
  ranges = [u for u in uops if u.op is Ops.RANGE]
  quad = None
  written = {_hmx_param(u.src[0]) for u in uops if u.op is Ops.STORE}
  ro = [all(_hmx_param(v.src[0]) not in written for v, _ in r) for r in (ra, rb)]
  if loops is not None and e is not None and all(ro) and getenv("HMX_I8_CACHE", 1):
    # read-only operand tiles stay in VTCM, in one pool of 2 KB slots (__hmx_ca: A first, then the quad path's spare byte-plane
    # slots, then B): an operand indexed by one output-tile loop is packed on the first iteration of the other, at slot (its
    # tile index)*kt + k. The loop order is chosen for what fits: both orders are planned when they can be interchanged
    kt = prod(int(r.vmax) + 1 for r in reds)
    no = len(reds) - 1  # outer reduce loops, as srcs before the inner one (outer, inner stay the last two srcs)
    kr = f"{{{len(srcs)+no}}}"
    for j in range(no - 1, -1, -1):
      kr = f"({{{len(srcs)+j}}})*{prod(int(r.vmax) + 1 for r in reds[j+1:])}+" + kr
    k0 = " && ".join(f"({{{len(srcs)+j}}})==0" for j in range(no + 1))  # the first K block of the whole reduction
    on, iname = f"{{{len(srcs)+no+1}}}", f"{{{len(srcs)+no+2}}}"
    pool = _HMX_CA + _HMX_CB
    def plan(outer, inner):
      # -> (rank, need_a, need_b, quad): rank 4 quad, 3 both cached, 2 A only, 1 B only, 0 none
      ti = int(inner.vmax) + 1
      def need(r):
        deps = {l for l in (outer, inner) if _hmx_uses(r[0][0].src[0], l)}
        return ti * kt if deps == {inner} else kt if deps in ({outer}, set()) else None
      na, nb = need(ra), need(rb)
      bdeps = {l for l in (outer, inner) if _hmx_uses(rb[0][0].src[0], l)}
      if (na is not None and bdeps == {outer} and int(outer.vmax + 1) % 4 == 0 and na + 4 * ti + 2 * kt <= pool and
          _hmx_quad_rows(rb, outer, ranges) and getenv("HMX_I8_QUAD", 1)): return (4, na, 2 * kt, True)
      if na is not None and nb is not None and na + nb <= pool: return (3, na, nb, False)
      if na is not None and na <= pool: return (2, na, 0, False)
      if nb is not None and nb <= pool: return (1, 0, nb, False)
      return (0, 0, 0, False)
    orders = [(loops[0], loops[1], False)] + ([(loops[1], loops[0], True)] if loops[3] else [])
    # best rank; ties keep the default (the longer loop outermost)
    outer, inner, sw = max(orders, key=lambda o: (plan(o[0], o[1])[0], o[2] == loops[2]))
    swap_out.append(sw)
    rank, na, nb, is_quad = plan(outer, inner)
    ti = int(inner.vmax) + 1
    def slot(r, base, j=0):
      deps = {l for l in (outer, inner) if _hmx_uses(r[0][0].src[0], l)}
      oj = f"+{j}" if j else ""
      if deps == {inner}: return f"__hmx_ca({base}+({iname})*{kt}+({kr}){oj})", f"({on})==0"
      if deps == {outer}: return f"__hmx_ca({base}+({kr}){oj})", f"({iname})==0"
      return f"__hmx_ca({base}+({kr}){oj})", f"({on})==0 && ({iname})==0"
    # A rows reading the same 128-byte lines for K blocks 4j .. 4j+3: filled four slots at a time, each line loaded once
    ki = f"{{{len(srcs)+no}}}"
    qa = bool(getenv("HMX_I8_QUAD_A", 1)) and _hmx_quad_k_rows(ra, e.src[1], ranges)
    def fill_a(a_):
      if not qa: return f" if ({a_[1]}) {{{{{pa} }}}}"
      d = [slot(ra, 0, j)[0] for j in range(4)]
      pq4 = "".join(f" __hmx_i8_pack_a4x4((unsigned char*){d[0]}+{128*q}, (unsigned char*){d[1]}+{128*q}, (unsigned char*){d[2]}+{128*q}, "
                    f"(unsigned char*){d[3]}+{128*q}, {ptr[4*q]}, {ptr[4*q+1]}, {ptr[4*q+2]}, {ptr[4*q+3]});" for q in range(16))
      return f" if ({a_[1]} && ({ki})%4==0) {{{{{pq4} }}}}"
    if is_quad:
      # B: four adjacent N tiles share every 128-byte row line: packed together (one byte + one halfword interleave per four
      # rows) into pair slots [n | n+1] (2 KB), pair p of the quad at slot p*kt + k; even n runs one weight :deep load pair
      # per K block for tiles n and n+1 and stores both accumulators (n+1's byte planes in 8 KB of spare slots, per inner
      # tile), odd n only adds those planes
      sa_ = slot(ra, 0)
      bb = na + 4 * ti
      nm = f"({on})"
      dst = lambda t, g: f"(signed char*)__hmx_ca({bb}+{t//2}*{kt}+({kr}))+{1024*(t%2)+128*g}"
      pq = "".join(f" __hmx_i8_pack_b4x4({dst(0,g)}, {dst(1,g)}, {dst(2,g)}, {dst(3,g)}, {ptr[64+4*g]}, {ptr[65+4*g]}, {ptr[66+4*g]}, {ptr[67+4*g]});"
                   for g in range(8))
      pf = (f" if ({k0}) __hmx_prefetch_panel2((const __fp16*){ptr[64]}, (const __fp16*){ptr[65]}, {32*kt}, "
            f"{'(' + nm + ')+4<' + str(int(outer.vmax)+1) + ' ? ((' + nm + ')==0 ? 1 : 2) : 0' if getenv('HMX_PF_AHEAD', 1) else 0});")
      code = (f"{{{{ void* _a = {sa_[0]};{fill_a(sa_)} if ({nm}%4==0 && ({iname})==0) {{{{{pf}{pq} }}}}"
              f" if ({nm}%2==0) __hmx_i8_mac2(_a, __hmx_ca({bb}+({nm}%4/2)*{kt}+({kr}))); }}}}")
      srcs = srcs + (*reds[:-1], e.src[1], outer, inner)
      quad = (nm, f"(unsigned char*)__hmx_ca({na}+4*({iname}))")
    elif rank >= 1:
      a_ = slot(ra, 0) if rank in (3, 2) else None
      b_ = slot(rb, na) if rank in (3, 1) else None
      ca = f"void* _a = {a_[0]};{fill_a(a_)}" if a_ else f"void* _a = __hmx_i8sa();{pa}"
      cb = f" void* _b = {b_[0]}; if ({b_[1]}) {{{{{pb} }}}}" if b_ else f" void* _b = __hmx_i8sb();{pb}"
      code = "{{ " + ca + cb + " __hmx_i8_mac(_a, _b); }}"
      srcs = srcs + (*reds[:-1], e.src[1], outer, inner)
    if getenv("HMX_DEBUG"): print(f"i8 plan: rank {rank} (quad {is_quad}, quad A {qa}) A {na} B {nb} slots of {pool}, kt {kt}, swap {sw}")
  replace[w] = UOp(Ops.CUSTOM, dtypes.void, srcs, code)
  end_at = pos[eo] if eo is not None else max(pos[so] for so, _, _ in stores)
  body = []
  for k, (_, l0, n) in enumerate(stores):
    if l0 % 128 or n % 128: return 30
    body += [f" __hmx_i8_addq((int*){{{k}}}+{128*j}, _P, {l0//128 + j});" for j in range(n // 128)]
  accs = tuple(a.src[0].src[0] for a in adds)
  if quad is not None:
    nm, p1 = quad
    for k_, v_ in ((f"{{{len(srcs)-2}}}", f"{{{len(accs)}}}"), (f"{{{len(srcs)-1}}}", f"{{{len(accs)+1}}}")):
      nm, p1 = nm.replace(k_, v_), p1.replace(k_, v_)
    after.setdefault(end_at, []).append(UOp(Ops.CUSTOM, dtypes.void, accs + (srcs[-2], srcs[-1]),
      f"{{{{ const unsigned char* _P = ({nm}%2==0) ? __hmx_i8_store2({p1}) : (const unsigned char*)({p1});" + "".join(body) + " }}"))
  else:
    after.setdefault(end_at, []).append(UOp(Ops.CUSTOM, dtypes.void, accs,
                                            "{{ const unsigned char* _P = __hmx_i8_store();" + "".join(body) + " }}"))
  return None

def _hmx_cval(u:UOp):
  # the value of a (casted) constant, else None
  while u.op is Ops.CAST: u = u.src[0]
  if u.op is not Ops.CONST: return None
  try: return float(u.arg)
  except (TypeError, ValueError): return None

def _hmx_cadd(u:UOp):
  # x + c (either order) -> (x, c), else None
  if u.op is not Ops.ADD: return None
  for x, c in (u.src, u.src[::-1]):
    if (cv:=_hmx_cval(c)) is not None and _hmx_cval(x) is None: return x, cv
  return None

def _hmx_rne_of(r:UOp):
  # tinygrad's round() (half to even) as it is linearized -> its operand v, else None:
  #   WHERE((0 < v) != (trunc(h) == h), floor(v + 0.5), ceil(v - 0.5)), h = trunc(v) * 0.5
  #   floor(z) = WHERE(z < trunc(z), trunc(z) - 1, trunc(z)), ceil(z) = WHERE(trunc(z) < z, trunc(z) + 1, trunc(z))
  if r.op is not Ops.WHERE: return None
  c, fl, ce = r.src
  if c.op is not Ops.CMPNE or c.src[0].op is not Ops.CMPLT or _hmx_cval(c.src[0].src[0]) != 0.0: return None
  v, e = c.src[0].src[1], c.src[1]
  if e.op is not Ops.CMPEQ or e.src[0].op is not Ops.TRUNC or e.src[0].src[0] is not e.src[1]: return None
  h = e.src[1]
  if h.op is not Ops.MUL or not any(x.op is Ops.TRUNC and x.src[0] is v and _hmx_cval(y) == 0.5 for x, y in (h.src, h.src[::-1])): return None
  def rounded(w:UOp, off:float, lt_first:bool, step:float) -> bool:
    if w.op is not Ops.WHERE or w.src[1].op is not Ops.ADD or w.src[0].op is not Ops.CMPLT: return False
    t = w.src[2]
    if t.op is not Ops.TRUNC or _hmx_cadd(t.src[0]) != (v, off) or _hmx_cadd(w.src[1]) != (t, step): return False
    return w.src[0].src == ((t.src[0], t) if lt_first else (t, t.src[0]))
  return v if rounded(fl, 0.5, True, -1.0) and rounded(ce, -0.5, False, 1.0) else None

def _hmx_rq_lane(x:UOp):
  # one lane of a QDQ requantization, y = clip(round((float)(p [+ q]) * m) + zy, lo, 255).cast(uint8), as linearized:
  #   WHERE(255 < X, 255, (uchar)X), X = MAX(round(v) [+ zy], lo), v = (float)(p [+ q]) * m (lanes of vector loads)
  # -> ((p, lane), (q, lane) | None, (m, lane), zy, lo) or None
  if x.op is not Ops.WHERE or x.dtype != dtypes.uchar or len(x.src) != 3: return None
  c, hi, cx = x.src
  if c.op is not Ops.CMPLT or _hmx_cval(c.src[0]) != 255.0 or _hmx_cval(hi) != 255.0: return None
  X = c.src[1]
  if cx.op is not Ops.CAST or cx.src[0] is not X or X.op is not Ops.MAX: return None
  y, lo = next(((a, _hmx_cval(b)) for a, b in (X.src, X.src[::-1]) if _hmx_cval(b) is not None), (None, None))
  if y is None: return None
  y, zy = _hmx_cadd(y) or (y, 0.0)
  v = _hmx_rne_of(y)
  if v is None or v.op is not Ops.MUL: return None
  for f, ml in (v.src, v.src[::-1]):
    if f.op is Ops.CAST and f.dtype == dtypes.float and f.src[0].dtype == dtypes.int and (m := _hmx_lane(ml)) is not None: break
  else: return None
  a = f.src[0]
  pq = [_hmx_lane(a)] if a.op is Ops.INDEX else [_hmx_lane(z) for z in a.src] if a.op is Ops.ADD else [None]
  if any(t is None or t[0].op is not Ops.LOAD for t in pq + [m]) or m[0].dtype != dtypes.float: return None
  if zy != int(zy) or lo != int(lo) or not 0 <= zy <= 255 or not -255 <= lo <= 255: return None
  return pq[0], (pq[1] if len(pq) > 1 else None), m, int(zy), int(lo)

def _hmx_rq_rows(uops, users, drop:set, before:dict, replace:dict, pos) -> int:
  # uint8 row stores of 32 lanes that are each an ORT-exact requantization of the same lane of the same vector loads
  # (accumulator, bias, per-column scale): the lanes go; the four rows of one 128-lane accumulator load (same bias, scale,
  # zero point) become one HVX __hmx_rq4, any other row one __hmx_rq1
  rows = []
  for st in uops:
    if st.op is not Ops.STORE or st.src[1].op is not Ops.STACK or st.src[1].dtype != dtypes.uchar or len(st.src[1].src) != 32: continue
    ms = [_hmx_rq_lane(x) for x in st.src[1].src]
    if any(m is None for m in ms) or len({(m[3], m[4]) for m in ms}) != 1: continue
    cols = [tuple(m[i] for m in ms) for i in range(3)]
    if cols[1][0] is None: cols[1] = None
    picks = []
    for c in cols:
      if c is None: picks.append(None); continue
      o, ld = c[0][1], c[0][0]
      if len({t[0] for t in c}) != 1 or [t[1] for t in c] != list(range(o, o + 32)) or o % 32 or len(ld.shape) != 1 or ld.shape[0] % 32: break
      picks.append((ld, o))
    else: rows.append((st, picks, ms[0][3], ms[0][4]))
  def pick(k:int, o:int, n:int) -> str:
    return f"{{{k}}}" if n == 32 else f"__builtin_shufflevector({{{k}}}, {{{k}}}, {', '.join(str(o+i) for i in range(32))})"
  groups: dict[tuple, dict[int, tuple]] = {}
  for r in rows:
    st, (a, b, m), zy, lo = r
    if a[0].shape == (128,) and (b is None or b[1] == 0) and m[1] == 0:
      groups.setdefault((a[0], b and b[0], m[0], zy, lo), {})[a[1] // 32] = r
  done = set()
  quads = []
  for (al, bl, ml, zy, lo), g in groups.items():
    if sorted(g) != [0, 1, 2, 3]: continue
    sts = [g[j][0] for j in range(4)]
    # the accumulator rows go in by address (al's pointer into the accumulator array), not as the loaded 128 lanes
    loads = [al.src[0], ml] + ([bl] if bl is not None else [])
    ptrs = tuple(x.src[0] for x in sts)
    bias = pick(2, 0, bl.shape[0]) if bl is not None else "(__hmx_i32x32)(0)"
    k = len(loads)
    args = f"{{{k}}}, {{{k+1}}}, {{{k+2}}}, {{{k+3}}}, (const int*){{0}}, {bias}, {pick(1, 0, ml.shape[0])}, {zy}, {lo}"
    quads.append((max(sts, key=lambda x: pos[x]), tuple(loads) + ptrs, args))
    drop.update(sts)
    done.update(sts)
  # fast path per group, flags ORed per run of groups between two uses of the same accumulator row (an output tile); one check
  # after the run redoes all of it exactly when anything flagged
  quads.sort(key=lambda q: pos[q[0]])
  runs: list[list] = []
  for q in quads:
    if not runs or any(q[1][0] is r[1][0] for r in runs[-1]): runs.append([q])
    else: runs[-1].append(q)
  ranges = [u for u in uops if u.op is Ops.RANGE]
  def delta(p0:UOp, p1:UOp):
    # element offset of pointer p1 from p0 (same buffer), constant over the loops, else None
    if p0.op not in (Ops.INDEX, Ops.SHRINK) or p1.op is not p0.op or p0.src[0] is not p1.src[0]: return None
    def f(env):
      a, b = _hmx_eval(p1.src[1], env), _hmx_eval(p0.src[1], env)
      return None if a is None or b is None else a - b
    return _hmx_const_delta(f, ranges)
  for run in runs:
    # a regular run -- every group's pointers at one constant offset per group from the first's -- becomes one C loop: 16
    # unrolled copies of the requantization are ~25 KB of code per output tile and ran at half the speed of the loop
    # in accumulator-row order (the store order has group 0 last)
    offs = [delta(run[0][1][0], q[1][0]) for q in run]
    if None not in offs: run = [q for _, q in sorted(zip(offs, run), key=lambda t: t[0])]
    g0 = run[0][1]
    steps = None
    if len(run) > 1 and all(len(q[1]) == len(g0) and all(a is b for a, b in zip(q[1][1:len(g0)-4], g0[1:len(g0)-4])) for q in run):
      pidx = [0] + list(range(len(g0) - 4, len(g0)))  # the accumulator pointer and the four row pointers
      d1 = [delta(g0[i], run[1][1][i]) for i in pidx]
      if None not in d1 and len(set(d1[1:])) == 1 and \
         all(delta(g0[i], q[1][i]) == n * d for n, q in enumerate(run) for i, d in zip(pidx, d1)): steps = (d1[0], d1[1])
    if steps is not None:
      st_last, srcs, args = max((q[0] for q in run), key=lambda x: pos[x]), g0, run[0][2]
      da, dd = steps
      ga = re.sub(r"\{(\d+)\}", lambda mt: f"({{{mt.group(1)}}}+_g*{dd})" if int(mt.group(1)) >= len(g0) - 4 else mt.group(0), args)
      ga = ga.replace("(const int*){0}", f"(const int*)({{0}}+_g*{da})")
      code = (f"__hmx_u32x32 _rqf = (__hmx_u32x32)(0), _rqg[{len(run)}]; for (int _g = 0; _g < {len(run)}; _g++) {{{{ _rqg[_g] = "
              f"(__hmx_u32x32)(0); __hmx_rq4f({ga}, &_rqg[_g]); _rqf |= _rqg[_g]; }}}} if (__builtin_expect(__hmx_rq_any(_rqf), 0)) "
              f"for (int _g = 0; _g < {len(run)}; _g++) if (__hmx_rq_any(_rqg[_g])) __hmx_rq4x({ga});")
      replace[st_last] = UOp(Ops.CUSTOM, dtypes.void, srcs, code)
      drop.discard(st_last)
      continue
    # per group its own flags (_rqg[n]) and the run's OR (_rqf): the rare flagged run then redoes only its flagged groups
    for n, (st, srcs, args) in enumerate(run):
      code = (f"__hmx_u32x32 _rqf = (__hmx_u32x32)(0), _rqg[{len(run)}]; " if n == 0 else "") + \
             f"_rqg[{n}] = (__hmx_u32x32)(0); __hmx_rq4f({args}, &_rqg[{n}]); _rqf |= _rqg[{n}];"
      if n == len(run) - 1:
        base, allsrc, redo = 0, (), []
        for j, (_, s2, a2) in enumerate(run):
          a2 = re.sub(r"\{(\d+)\}", lambda mt: f"{{{int(mt.group(1)) + base}}}", a2)
          redo.append(f"if (__hmx_rq_any(_rqg[{j}])) __hmx_rq4x({a2});"); allsrc += s2; base += len(s2)
        code = re.sub(r"\{(\d+)\}", lambda mt: f"{{{int(mt.group(1)) + base - len(srcs)}}}", code)
        code += " if (__builtin_expect(__hmx_rq_any(_rqf), 0)) {{ " + " ".join(redo) + " }}"
        srcs = allsrc
      replace[st] = UOp(Ops.CUSTOM, dtypes.void, srcs, code)
      drop.discard(st)
  for st, (a, b, m), zy, lo in rows:
    if st in done: continue
    loads = [a[0], m[0]] + ([b[0]] if b is not None else [])
    bias = pick(2, b[1], b[0].shape[0]) if b is not None else "(__hmx_i32x32)(0)"
    replace[st] = UOp(Ops.CUSTOM, dtypes.void, tuple(loads) + (st.src[0],),
                      f"__hmx_rq1({{{len(loads)}}}, {pick(0, a[1], a[0].shape[0])}, {bias}, {pick(1, m[1], m[0].shape[0])}, {zy}, {lo});")
    done.add(st)
  # the lanes' own uops, down to the loads, go when nothing else uses them
  keep_loads = {x for u in replace.values() if u.op is Ops.CUSTOM for x in u.src}
  dead, stack = set(), [x for st in done for x in st.src[1:]]
  while stack:
    u = stack.pop()
    if u in dead or u in keep_loads or u.op in (Ops.CONST, Ops.PARAM, Ops.RANGE) or u not in pos: continue
    dead.add(u); stack.extend(u.src)
  live = {d for d in dead if any(v not in dead and v not in done for v in users.get(d, []))}
  while live:
    more = {x for d in live for x in d.src if x in dead} - live
    dead -= live
    live = more
  drop |= dead
  return len(done)

# ---- ORT's QLinearAdd, exactly: y = clamp(rne(rb*b + (ra*a + fixed)), 0, 255) in separate fp32 operations in that order (MLAS),
# fixed = zy - (ra*za + rb*zb). Spelled as tinygrad float ops it can't be exact on any backend: symbolic reassociates the adds
# ((a*ra + b*rb) + fixed, round()'s +-0.5 folded into the constant), and which input the constant pairs with is lost. So it is a
# custom kernel over 2 KB chunks calling one helper per chunk (onnxsim's hmx_gemm/runner rn_add): HVX fixed point v * 2^F from
# the exact products a * mantissa(ra) (12-bit halves), round half up, lanes within the window of ORT's four fp32 roundings of a
# .5 recomputed with ORT's sequence on the scalar core; one vector -> scalar flag check per chunk
_HMX_QADD_HELPERS = r"""#pragma clang diagnostic ignored "-Wunused-function"
#ifndef __HMX_QADD
#define __HMX_QADD
#ifdef HMX_REF
static void __hmx_qadd_chunk(unsigned char* y, const unsigned char* a, const unsigned char* b, int nvec, float ra, float rb, float fixed,
                             int ah, int al, int bh, int bl, int sa, int sb, int fq, int F, int win) {
  (void)ah; (void)al; (void)bh; (void)bl; (void)sa; (void)sb; (void)fq; (void)F; (void)win;
  for (int i = 0; i < 128 * nvec; i++) {
    volatile float t1 = ra * (float)a[i], t2 = t1 + fixed, t3 = rb * (float)b[i], v = t3 + t2;
    float c = v < 0.0f ? 0.0f : v > 255.0f ? 255.0f : v;
    volatile float r = c + 12582912.0f;
    y[i] = (unsigned char)(int)(r - 12582912.0f);
  }
}
#else
typedef int __hmx_qv __attribute__((vector_size(128)));
typedef int __hmx_qvp __attribute__((vector_size(256)));
/* one word half: v = (a*ma >> sa) + (b*mb >> sb) + fq in 2^-F units from the 12-bit mantissa halves, r = round half up, f = 1
 * where |frac - .5| < win (the sign bit of |frac - half| - win: no vector predicates, their builtins assert in this toolchain) */
__attribute__((always_inline)) static inline __hmx_qv __hmx_qadd_w(__hmx_qv xa, __hmx_qv xl, __hmx_qv xb, __hmx_qv xm, int sa, int sb,
                                                                    __hmx_qv fq, __hmx_qv half, __hmx_qv mask, __hmx_qv win, int F, __hmx_qv* f) {
  __hmx_qv t = __builtin_HEXAGON_V6_vaddw_128B(__builtin_HEXAGON_V6_vaslw_128B(xa, 12 - sa), __builtin_HEXAGON_V6_vlsrw_128B(xl, sa));
  __hmx_qv u = __builtin_HEXAGON_V6_vaddw_128B(__builtin_HEXAGON_V6_vaslw_128B(xb, 12 - sb), __builtin_HEXAGON_V6_vlsrw_128B(xm, sb));
  __hmx_qv v = __builtin_HEXAGON_V6_vaddw_128B(__builtin_HEXAGON_V6_vaddw_128B(t, u), fq);
  __hmx_qv d = __builtin_HEXAGON_V6_vabsw_128B(__builtin_HEXAGON_V6_vsubw_128B(__builtin_HEXAGON_V6_vand_128B(v, mask), half));
  *f = __builtin_HEXAGON_V6_vlsrw_128B(__builtin_HEXAGON_V6_vsubw_128B(d, win), 31);
  return __builtin_HEXAGON_V6_vasrw_128B(__builtin_HEXAGON_V6_vaddw_128B(v, half), F);
}
/* 64 halfword lanes (a, b unpacked): both word halves of the products, straight-line so the two chains interleave */
__attribute__((always_inline)) static inline __hmx_qv __hmx_qadd_half(__hmx_qv va, __hmx_qv vb, int ah, int al, int bh, int bl, int sa, int sb,
                                                                       __hmx_qv fq, __hmx_qv half, __hmx_qv mask, __hmx_qv win, int F, __hmx_qv* fl) {
  __hmx_qvp pah = __builtin_HEXAGON_V6_vmpyh_128B(va, ah), pal = __builtin_HEXAGON_V6_vmpyh_128B(va, al);
  __hmx_qvp pbh = __builtin_HEXAGON_V6_vmpyh_128B(vb, bh), pbl = __builtin_HEXAGON_V6_vmpyh_128B(vb, bl);
  __hmx_qv f0, f1;
  __hmx_qv r0 = __hmx_qadd_w(__builtin_HEXAGON_V6_lo_128B(pah), __builtin_HEXAGON_V6_lo_128B(pal), __builtin_HEXAGON_V6_lo_128B(pbh),
                             __builtin_HEXAGON_V6_lo_128B(pbl), sa, sb, fq, half, mask, win, F, &f0);
  __hmx_qv r1 = __hmx_qadd_w(__builtin_HEXAGON_V6_hi_128B(pah), __builtin_HEXAGON_V6_hi_128B(pal), __builtin_HEXAGON_V6_hi_128B(pbh),
                             __builtin_HEXAGON_V6_hi_128B(pbl), sa, sb, fq, half, mask, win, F, &f1);
  *fl = __builtin_HEXAGON_V6_vsatwh_128B(f1, f0);
  return __builtin_HEXAGON_V6_vsatwh_128B(r1, r0);
}
static void __hmx_qadd_chunk(unsigned char* y, const unsigned char* a, const unsigned char* b, int nvec, float ra, float rb, float fixed,
                             int ah, int al, int bh, int bl, int sa, int sb, int fq, int F, int win) {
  __hmx_qv fl[16] __attribute__((aligned(128))), any = __builtin_HEXAGON_V6_vd0_128B();
  /* the inputs stream from DDR: L2-prefetch the next chunk of both while this one computes (16 lines of 128 bytes) */
  __builtin_HEXAGON_Y4_l2fetch((void*)(a + 2048), (128u << 16) | (128u << 8) | 16u);
  __builtin_HEXAGON_Y4_l2fetch((void*)(b + 2048), (128u << 16) | (128u << 8) | 16u);
  int ahh = (ah << 16) | ah, all = (al << 16) | al, bhh = (bh << 16) | bh, bll = (bl << 16) | bl;
  __hmx_qv fqv = __builtin_HEXAGON_V6_lvsplatw_128B(fq), half = __builtin_HEXAGON_V6_lvsplatw_128B(1 << (F - 1));
  __hmx_qv mask = __builtin_HEXAGON_V6_lvsplatw_128B((1 << F) - 1), winv = __builtin_HEXAGON_V6_lvsplatw_128B(win);
  for (int k = 0; k < nvec; k++) {
    __hmx_qvp ua = __builtin_HEXAGON_V6_vunpackub_128B(((const __hmx_qv*)a)[k]), ub = __builtin_HEXAGON_V6_vunpackub_128B(((const __hmx_qv*)b)[k]);
    __hmx_qv f0, f1;
    __hmx_qv y0 = __hmx_qadd_half(__builtin_HEXAGON_V6_lo_128B(ua), __builtin_HEXAGON_V6_lo_128B(ub), ahh, all, bhh, bll, sa, sb, fqv, half, mask, winv, F, &f0);
    __hmx_qv y1 = __hmx_qadd_half(__builtin_HEXAGON_V6_hi_128B(ua), __builtin_HEXAGON_V6_hi_128B(ub), ahh, all, bhh, bll, sa, sb, fqv, half, mask, winv, F, &f1);
    ((__hmx_qv*)y)[k] = __builtin_HEXAGON_V6_vpackhub_sat_128B(y1, y0);
    fl[k] = __builtin_HEXAGON_V6_vpackhub_sat_128B(f1, f0);
    any = __builtin_HEXAGON_V6_vor_128B(any, fl[k]);
  }
  for (int r = 64; r >= 4; r >>= 1) any = __builtin_HEXAGON_V6_vor_128B(any, __builtin_HEXAGON_V6_vror_128B(any, r));
  if (__builtin_expect(!__builtin_HEXAGON_V6_extractw_128B(any, 0), 1)) return;
  const unsigned long long* fw = (const unsigned long long*)fl;  /* 64-bit words, skipping the (almost always) zero ones */
  for (int w = 0; w < 16 * nvec; w++) {
    unsigned long long bits = fw[w];
    while (bits) {
      int j = __builtin_ctzll(bits) / 8, off = 8 * w + j;
      bits &= ~(0xffull << (8 * j));
      float t1 = __builtin_HEXAGON_F2_sfmpy(ra, __builtin_HEXAGON_F2_conv_w2sf(a[off]));
      float t2 = __builtin_HEXAGON_F2_sfadd(t1, fixed);
      float t3 = __builtin_HEXAGON_F2_sfmpy(rb, __builtin_HEXAGON_F2_conv_w2sf(b[off]));
      int v = __builtin_HEXAGON_F2_conv_sf2w(__builtin_HEXAGON_F2_sfadd(t3, t2));
      y[off] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  }
}
#endif
#endif
"""

def _hmx_qadd_consts(ra:float, rb:float, fixed:float) -> tuple:
  # the fixed-point form of ORT's add (onnxsim hmx_gemm/runner/rn_load.h): v * 2^F = a*ra*2^F + b*rb*2^F + fixed*2^F from
  # 24-bit mantissas in 12-bit halves, and the window: our truncations (< 3 units) + half an ulp of each of ORT's 4 fp32 roundings
  import math
  bound = 255.0 * (ra + rb) + abs(fixed) + 1
  F = min(21, math.floor(math.log2(2.0 ** 30 / bound)))
  (fa, ea), (fb, eb) = math.frexp(ra), math.frexp(rb)
  ma, mb = round(math.ldexp(fa, 24)), round(math.ldexp(fb, 24))
  sa, sb = -(ea - 24) - F, -(eb - 24) - F
  if not (0 <= sa <= 12 and 0 <= sb <= 12): raise ValueError(f"QLinearAdd scale ratios {ra} / {rb} out of the supported range")
  fq = round(math.ldexp(fixed, F))
  ulp = lambda x: math.ldexp(1.0, math.floor(math.log2(x)) - 23)
  win = math.ceil(3.0 + math.ldexp(0.5 * (ulp(255.0 * ra) + ulp(255.0 * rb) + ulp(255.0 * ra + abs(fixed) + 1) + ulp(bound)), F)) + 2
  return (ma >> 12, ma & 4095, mb >> 12, mb & 4095, sa, sb, fq, F, win)

def hmx_qlinear_add(a, b, ra:float, rb:float, fixed:float):
  """ORT's QLinearAdd on uint8 a, b (same shape; ra, rb, fixed as ORT computes them in fp32): exactly
  clamp(rne(rb*b + (ra*a + fixed)), 0, 255), as one DSP kernel over 2 KB chunks. numel % 128 == 0."""
  import struct
  from tinygrad import Tensor
  from tinygrad.uop.ops import KernelInfo
  f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
  ra, rb, fixed = f32(ra), f32(rb), f32(fixed)
  n = a.numel()
  assert a.dtype == b.dtype == dtypes.uint8 and b.numel() == n and n % 128 == 0, "uint8 inputs of the same size, a multiple of 128"
  c = ", ".join(str(x) for x in _hmx_qadd_consts(ra, rb, fixed))
  fl = lambda x: f"{x.hex()}f" if x != 0 else "0.0f"
  nv = n // 128
  def kern(Y, A, B):
    Y, A, B = Y.flatten(), A.flatten(), B.flatten()
    i = UOp.range((n + 2047) // 2048, 0)
    cu = UOp(Ops.CUSTOM, dtypes.void, (Y.index(i * 2048), A.index(i * 2048), B.index(i * 2048), i),
             arg=f"__hmx_qadd_chunk({{0}}, {{1}}, {{2}}, {nv}-16*{{3}} < 16 ? {nv}-16*{{3}} : 16, {fl(ra)}, {fl(rb)}, {fl(fixed)}, {c});")
    return cu.end(i).sink(arg=KernelInfo(name=f"qadd_{n}", opts_to_apply=()))
  y = Tensor.empty(*a.shape, dtype=dtypes.uint8, device=a.device)
  return Tensor.custom_kernel(y, a, b, fxn=kern)[0]

def _hmx_bail(uops, why:int):
  if getenv("HMX_DEBUG"): print(f"hmx_acc rewrite skipped (check {why})")
  return uops, False

def _hmx_acc_rewrite(uops:list[UOp]) -> tuple[list[UOp], bool]:
  pos = {u:i for i,u in enumerate(uops)}
  users: dict[UOp, list[UOp]] = {}
  for u in uops:
    for x in u.src: users.setdefault(x, []).append(u)
  written = {_hmx_param(u.src[0]) for u in uops if u.op is Ops.STORE}
  first = next((k for k,u in enumerate(uops) if u.op is Ops.WMMA), None)
  ranges = [u for u in uops if u.op is Ops.RANGE]
  loops = _hmx_tile_loops(uops, first) if first is not None else None
  call_start = False
  span: dict[UOp, tuple] = {}
  drop: set[UOp] = set()
  before: dict[int, list[UOp]] = {}
  after: dict[int, list[UOp]] = {}
  replace: dict[UOp, UOp] = {}
  swap_out: list[bool] = []  # the int8 path's loop-order choice (planned for what fits in the VTCM tile pool)
  for w in uops:
    if w.op is Ops.WMMA and w.arg[0] == (32, 64, 32) and w.arg[1] == dtypes.uint8 and getenv("HMX_I8", 1):
      r = _hmx_i8_rewrite(w, uops, pos, users, drop, before, after, replace, swap_out)
      if r is not None: return _hmx_bail(uops, r)
      continue
    if w.op is not Ops.WMMA or w.arg[1] != dtypes.half: continue
    ra, rb = _hmx_rows(w.src[0]), _hmx_rows(w.src[1])
    if ra is None or rb is None: return _hmx_bail(uops, 1)
    # the reduce loop: the innermost REDUCE RANGE whose END encloses the WMMA. None when K is a single tile (e.g. an
    # attention score q . k^T with head_dim 32): then the tile op runs once, begun right before the WMMA and stored after
    # its accumulator stores (an enclosing output-tile loop must not be mistaken for the reduction)
    ends = [e for e in uops if e.op is Ops.END and len(e.src) > 1 and e.src[1].op is Ops.RANGE and e.src[1].arg[-1] == AxisType.REDUCE
            and pos[e.src[1]] < pos[w] < pos[e]]
    e = min(ends, key=lambda e: pos[e]-pos[e.src[1]]) if ends else None
    # consumers: lane INDEXes of w -> STACKs -> STOREs into the accumulator array
    stores = []
    for lane in users.get(w, []):
      if _hmx_lane(lane) is None: return _hmx_bail(uops, 3)
      for st in users.get(lane, []):
        if st.op is not Ops.STACK or any((l:=_hmx_lane(x)) is None or l[0] is not w for x in st.src): return _hmx_bail(uops, 4)
        for so in users.get(st, []):
          if so.op is not Ops.STORE or so.src[1] is not st: return _hmx_bail(uops, 5)
          if so not in stores: stores.append(so)
    if not stores: return _hmx_bail(uops, 6)
    # dead after the rewrite: the three 1024-lane operands, their lanes, the accumulator loads feeding C, the old stores
    c_loads = {l[0] for x in w.src[2].src if (l:=_hmx_lane(x)) is not None}
    if w.src[2].op is not Ops.STACK or any(l.op is not Ops.LOAD for l in c_loads): return _hmx_bail(uops, 7)
    dead = {w, *w.src[:3], *w.src[0].src, *w.src[1].src, *w.src[2].src, *c_loads}
    for so in stores: dead |= {so, so.src[1], *so.src[1].src}
    if any(any(v not in dead and v.op not in (Ops.GROUP, Ops.END) for v in users.get(d, [])) for d in dead): return _hmx_bail(uops, 8)
    drop |= dead
    end_at = pos[e] if e is not None else max(pos[so] for so in stores)  # where the reduction is complete
    before.setdefault(pos[e.src[1]] if e is not None else pos[w], []).append(UOp(Ops.CUSTOM, dtypes.void, (), "__hmx_begin();"))
    rows = ra + rb
    vals = list(dict.fromkeys(v for v, _ in rows))
    if all(v.op is Ops.LOAD and len(v.src) == 1 and all(u in dead or u.op in (Ops.GROUP, Ops.END) for u in users.get(v, [])) for v in vals):
      # plain vector loads: pack straight from the row pointers with HVX (the loaded values themselves become dead)
      drop |= set(vals)
      ptr = [f"({{{vals.index(v)}}}+{b})" if b else f"{{{vals.index(v)}}}" for v, b in rows]
      pa = "".join(f" __hmx_pack2(_a+{64*q}, {ptr[2*q]}, {ptr[2*q+1]});" for q in range(16))
      pb = "".join(f" __hmx_pack2(_b+{64*q}, {ptr[32+2*q]}, {ptr[33+2*q]});" for q in range(16))
      srcs = tuple(v.src[0] for v in vals)
      # operands the kernel never writes are memoized in VTCM by their first row pointer (A repeats across the N tiles)
      ro = [all(_hmx_param(v.src[0]) not in written for v, _ in r) for r in (ra, rb)]
      # the reduce loop's variable and trip count; no reduce loop: K index 0, one K tile (the placeholder slot is then
      # filled with an already-rendered uop and never referenced)
      kr, kt = (f"{{{len(srcs)}}}", int(e.src[1].vmax) + 1) if e is not None else ("0", 1)
      def cached(x, n, key, dst, pack):
        if kt > n: return pack  # fewer slots than K tiles: no set per K index, just pack
        return (f" __fp16* _c{dst} = __hmx_lookup(__hmx_tag_{x}, __hmx_rr_{x}, __hmx_c{x}, {key}, {kr}, {min(n // kt, 8)}, &_s);"
                f" if (_c{dst}) _{dst} = _c{dst}; else {{{{ _{dst} = _s;{pack} }}}}")
      if loops is not None:
        # exact, tag-free: an operand's tiles are indexed by the loops its address depends on; pack on the first iteration of
        # the others (outer/inner as rendered, i.e. after the interchange)
        outer, inner = (loops[1], loops[0]) if loops[2] else (loops[0], loops[1])
        ti = int(inner.vmax) + 1
        on, iname = f"{{{len(srcs)+1}}}", f"{{{len(srcs)+2}}}"
        def exact(r, n, x, pack, acc=None, off=0):
          # slots contiguous in K for each tile index, so one spanning load pair can read them after the loop. Returns (code,
          # the panel base expression, slots used); acc / off: the slot accessor and first slot (B in A's pool: "ca", A's use)
          acc, o = acc or f"c{x}", f"{off}+" if off else ""
          deps = {l for l in (outer, inner) if _hmx_uses(r[0][0].src[0], l)}
          st = _hmx_stride(kt)
          if deps == {inner} and st * ti <= n: base, cond, used = f"{o}({iname})*{st}", f"({on})==0", st * ti
          elif deps == {outer} and 2 * st <= n and _hmx_pairable(r, outer, ranges):
            # adjacent outer tiles share every 128-byte row line: pack tiles n, n+1 together on even n (vshuff of full rows)
            p0 = 0 if x == "a" else 32
            pk = "".join(f" __hmx_pack2x2(_{x}+{64*q}, _{x}+{1024*st+64*q}, {ptr[p0+2*q]}, {ptr[p0+2*q+1]});" for q in range(16))
            if _hmx_uniform_rows(r, ranges): pk = f" __hmx_pack2x2_blk(_{x}, _{x}+{1024*st}, {ptr[p0]}, {ptr[p0+1]});"
            pf = (f" if (({kr})==0) __hmx_prefetch_rows({ptr[p0]}, {ptr[p0+1]}, {128*kt});" if x == "a" else
                  f" if (({kr})==0) __hmx_prefetch_panel2({ptr[p0]}, {ptr[p0+1]}, {32*kt}, "
                  f"{'(' + on + ')+2<' + str(int(outer.vmax)+1) + ' ? ((' + on + ')==0 ? 1 : 2) : 0' if getenv('HMX_PF_AHEAD', 1) else 0});")
            return (f" _{x} = __hmx_{acc}({o}({on})%2*{st}+({kr})); if (({iname})==0 && ({on})%2==0) {{{{{pf}{pk} }}}}",
                    f"{o}({on})%2*{st}", 2 * st)
          elif deps == {outer} and kt <= n: base, cond, used = f"{off}", f"({iname})==0", st
          elif not deps and kt <= n: base, cond, used = f"{off}", f"({on})==0 && ({iname})==0", st
          else: return None
          p0, p1 = (ptr[0], ptr[1]) if x == "a" else (ptr[32], ptr[33])
          # the whole K panel, on its first fill: A = 32 rows x K columns, B = K rows x 32 columns
          pf = (f" if (({kr})==0) __hmx_prefetch_rows({p0}, {p1}, {64*kt});" if x == "a" else
                f" if (({kr})==0) __hmx_prefetch_panel({p0}, {p1}, {32*kt});")
          return (f" _{x} = __hmx_{acc}({base}+({kr})); if ({cond}) {{{{{pf}{pack.replace(f' __hmx_prefetch_next({p0}, {p1});', '')} }}}}",
                  base, used)
        eb_acc = "cb"
        if HMX_VTCM_KB > 256 and ro[0] and ro[1]:
          # one pool (the B slots follow A's): A takes what it needs, B the rest from the next 32-slot boundary; if B doesn't
          # fit, back to the fixed split (B then goes through its tag cache)
          ea = exact(ra, _HMX_CA + _HMX_CB, "a", pa)
          eb = exact(rb, _HMX_CA + _HMX_CB - round_up(ea[2], 32), "b", pb, "ca", round_up(ea[2], 32)) if ea else None
          if eb: eb_acc = "ca"
          else: ea, eb = exact(ra, _HMX_CA, "a", pa), exact(rb, _HMX_CB, "b", pb)
        else:
          ea = exact(ra, _HMX_CA, "a", pa) if ro[0] else None
          eb = exact(rb, _HMX_CB, "b", pb) if ro[1] else None
        if ea and eb:
          # K tiles stay in VTCM: no load pair per K block, one spanning pair after the loop (see the output statement)
          span[w] = (ea[1], eb[1], kt, outer, inner, on, iname, eb_acc, _hmx_stride(kt), "%2*" in eb[1], int(outer.vmax) + 1)
          ea, eb = ea[0], eb[0]
        else: ea, eb = ea and ea[0], eb and eb[0]
        pa, pb = ea or (cached("a", _HMX_CA, ptr[0], "a", pa) if ro[0] else pa), eb or (cached("b", _HMX_CB, ptr[32], "b", pb) if ro[1] else pb)
        srcs = srcs + (e.src[1] if e is not None else outer, outer, inner)
      else:
        if ro[0]: pa = cached("a", _HMX_CA, ptr[0], "a", pa)
        if ro[1]: pb = cached("b", _HMX_CB, ptr[32], "b", pb)
        if any(ro) and e is not None: srcs = srcs + (e.src[1],)
      if any(ro): call_start = True
      # the next K block of B (and of A while it's still being packed) is fetched into L2 while this one packs; the exact
      # cache paths prefetch whole panels on their first fill instead (a per-K-block l2fetch there cost ~12% of the call)
      if w not in span: pb = f" __hmx_prefetch_next({ptr[32]}, {ptr[33]});" + pb
      pfa = f" __hmx_prefetch_next({ptr[0]}, {ptr[1]});"
      pa = pfa + pa if not ro[0] else pa.replace("{{ _a = _s;", "{{ _a = _s;" + pfa, 1)
    elif all(v.max_numel() == 32 for v, _ in rows):
      pa = "".join(f" *(__hmx_h64*)(_a+{64*q}) = __builtin_shufflevector({{{2*q}}},{{{2*q+1}}},{_ILV});" for q in range(16))
      pb = "".join(f" *(__hmx_h64*)(_b+{64*q}) = __builtin_shufflevector({{{32+2*q}}},{{{33+2*q}}},{_ILV});" for q in range(16))
      srcs = tuple(v for v, _ in rows)
    else: return _hmx_bail(uops, 10)
    replace[w] = UOp(Ops.CUSTOM, dtypes.void, srcs, "{{ __fp16* _a = __hmx_sa(); __fp16* _b = __hmx_sb(); __fp16* _s = 0; (void)_s;"+pa+pb+
                     (" (void)_a; (void)_b; }}" if w in span else " __hmx_mac(_a, _b); }}"))
    def span_mac(srcs0:tuple) -> tuple[str, tuple]:
      # after the loop: the spanning load pair(s) over all K tiles, rendered with the tile loops' indices
      if w not in span: return "", ()
      ab, bb, kt_, o_, i_, on_, in_, bacc = span[w][:8]
      for k_, v_ in ((on_, "{%d}" % len(srcs0)), (in_, "{%d}" % (len(srcs0)+1))): ab, bb = ab.replace(k_, v_), bb.replace(k_, v_)
      return f" __hmx_mac_span(__hmx_ca({ab}), __hmx_{bacc}({bb}), {kt_});", (o_, i_)
    # after the loop: one store, then each accumulator-array vector = the same lanes of the output tile
    outs = []
    for k, so in enumerate(stores):
      lanes = [l[1] for x in so.src[1].src if (l:=_hmx_lane(x)) is not None]
      if len(lanes) == 128 and all(0 <= l < 1024 for l in lanes) and len({l//128 for l in lanes}) <= 2:
        blks = sorted({l//128 for l in lanes})
        b0, b1 = blks[0], blks[-1]
        if lanes == [b0*128 + 64*q + 2*j + i for q in range(2) for i in range(2) for j in range(32)]:
          # four consecutive rows, in order (the accumulator array is row-major): one vdealh per HVX register
          outs.append(f" __hmx_deal2((__fp16*){{{k}}}, _p+{b0*128});")
          continue
        idx = ",".join(str(l - b0*128 if l//128 == b0 else 128 + l - b1*128) for l in lanes)
        outs.append(f" *(__hmx_h128*){{{k}}} = __builtin_shufflevector(_o[{b0}], _o[{b1}], {idx});")
      else:  # a vector as wide as the store (a single-K-tile op stores straight to the output, 32 lanes per row)
        vt = {32: "__hmx_h32", 64: "__hmx_h64", 128: "__hmx_h128"}.get(len(lanes))
        if vt is None: return _hmx_bail(uops, 11)
        outs.append(f" *({vt}*){{{k}}} = ({vt}){{{{{','.join(f'_p[{l}]' for l in lanes)}}}}};")
    if (direct:=_hmx_direct_out(uops, pos, users, end_at, stores)) is not None:
      # the accumulator array is only copied to an output buffer after the loop: write the tile rows there directly
      rowptr, gone = direct
      drop |= gone
      pairs = "".join(f" __hmx_out2({{{2*q}}}, {{{2*q+1}}}, _p+{64*q});" for q in range(16))
      if getenv("HMX_DEBUG"): print("outpair:", w in span, w in span and span[w][9], w in span and span[w][1], [(p.op, len(p.src)) for p in rowptr[:2]],
                                    w in span and _hmx_ptr_pairable(rowptr, span[w][3], ranges))
      if w in span and span[w][9] and bool(getenv("HMX_OUTPAIR", 1)) and _hmx_ptr_pairable(rowptr, span[w][3], ranges):
        # B tiles n, n+1 are packed together on even n: compute both output tiles there (two spanning load pairs, two
        # stores to VTCM) and write full 128-byte rows; odd n has nothing left to do; an unpaired last tile stores alone
        ab, bb, kt_, o_, i_, on_, in_, bacc, st_, _, ON = span[w]
        n0, n1 = "{%d}" % len(rowptr), "{%d}" % (len(rowptr)+1)
        ab, bb = ab.replace(on_, n0).replace(in_, n1), bb.replace(on_, n0).replace(in_, n1)
        mac0 = f" __hmx_mac_span(__hmx_ca({ab}), __hmx_{bacc}({bb}), {kt_});"
        mac1 = f" __hmx_mac_span(__hmx_ca({ab}), __hmx_{bacc}({bb}+{st_}), {kt_});"
        pp = "".join(f" __hmx_outp({{{2*q}}}, {{{2*q+1}}}, _p+{64*q}, _q+{64*q});" for q in range(16))
        code = (f"{{{{ if (({n0})%2==0 && ({n0})+1<{ON}) {{{{{mac0} const __fp16* _p = __hmx_store();{mac1} const __fp16* _q = __hmx_store2();{pp} }}}}"
                f" else if (({n0})%2==0) {{{{{mac0} const __fp16* _p = __hmx_store();{pairs} }}}} }}}}")
        at = max(pos[g] for g in gone if g.op is Ops.STORE)
        after.setdefault(at, []).extend([u for u in dict.fromkeys(rowptr) if u not in pos] +
                                        [UOp(Ops.CUSTOM, dtypes.void, tuple(rowptr)+(o_, i_), code)])
        continue
      # where the last replaced store was: every row pointer expression is rendered by then
      at = max(pos[g] for g in gone if g.op is Ops.STORE)
      sm, sx = span_mac(tuple(rowptr))
      after.setdefault(at, []).extend([u for u in dict.fromkeys(rowptr) if u not in pos] +
                                      [UOp(Ops.CUSTOM, dtypes.void, tuple(rowptr)+sx, "{{"+sm+" const __fp16* _p = __hmx_store();"+pairs+" }}")])
      continue
    row_of = {}
    for so in stores:  # stores that are exactly the tile's 32 rows (IDX(i, 0..31)): the HVX row-pair output
      lanes = [l[1] for x in so.src[1].src if (l:=_hmx_lane(x)) is not None]
      i = 2 * (lanes[0] // 64) + lanes[0] % 2 if lanes else -1
      if len(lanes) != 32 or lanes != [64*(i//2) + 2*j + i%2 for j in range(32)] or i in row_of: break
      row_of[i] = so.src[0]
    if sorted(row_of) == list(range(32)):
      rowptr = [row_of[i] for i in range(32)]
      pairs = "".join(f" __hmx_out2({{{2*q}}}, {{{2*q+1}}}, _p+{64*q});" for q in range(16))
      sm, sx = span_mac(tuple(rowptr))
      after.setdefault(end_at, []).append(UOp(Ops.CUSTOM, dtypes.void, tuple(rowptr)+sx, "{{"+sm+" const __fp16* _p = __hmx_store();"+pairs+" }}"))
      continue
    sm, sx = span_mac(tuple(so.src[0] for so in stores))
    after.setdefault(end_at, []).append(UOp(Ops.CUSTOM, dtypes.void, tuple(so.src[0] for so in stores)+sx,
      "{{"+sm+" __fp16* _p = __hmx_store(); __hmx_h128* _o = (__hmx_h128*)_p; (void)_o;"+"".join(outs)+" }}"))
  if any(w.op is Ops.WMMA and w.arg[1] == dtypes.uint8 for w in uops) and getenv("HMX_RQ", 1):
    nrq = _hmx_rq_rows(uops, users, drop, before, replace, pos)
    if getenv("HMX_DEBUG"): print(f"hmx requant rows: {nrq}")
  if not replace: return _hmx_bail(uops, 9)
  if call_start: before.setdefault(0, []).insert(0, UOp(Ops.CUSTOM, dtypes.void, (), "__hmx_call_start();"))
  out = []
  for i, u in enumerate(uops):
    out += before.get(i, [])
    if u in replace: out.append(replace[u])
    elif u not in drop: out.append(u)
    out += after.get(i, [])
  if loops is not None and (swap_out[0] if swap_out else loops[2]): out = _hmx_interchange(out, loops[0], loops[1])
  return out, True

_HMX_EVAL = {Ops.ADD: lambda a,b: a+b, Ops.SUB: lambda a,b: a-b, Ops.MUL: lambda a,b: a*b, Ops.SHL: lambda a,b: a<<b,
             Ops.SHR: lambda a,b: a>>b, Ops.AND: lambda a,b: a&b, Ops.OR: lambda a,b: a|b, Ops.XOR: lambda a,b: a^b,
             Ops.CDIV: lambda a,b: int(a/b), Ops.CMOD: lambda a,b: a-b*int(a/b), Ops.MAX: max}
def _hmx_eval(u:UOp, env:dict):
  # integer value of an index expression with the loops in env set, or None
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.RANGE: return env.get(u)
  if u.op is Ops.CAST: return _hmx_eval(u.src[0], env)
  if u.op in _HMX_EVAL and len(u.src) == 2:
    a, b = _hmx_eval(u.src[0], env), _hmx_eval(u.src[1], env)
    return None if a is None or b is None else _HMX_EVAL[u.op](a, b)
  return None

def _hmx_const_delta(f, ranges:list[UOp]):
  # f(env) at a few sample points of the loops -> the one constant it always equals, else None
  vals = set()
  for smp in range(4):
    env = {r: (smp * 7 + 3 * n) % max(1, int(r.vmax)) for n, r in enumerate(ranges)}
    vals.add(f(env))
  return vals.pop() if len(vals) == 1 and None not in vals else None

def _hmx_pairable(rows, outer:UOp, ranges:list[UOp]) -> bool:
  # 32-lane row loads whose row stride is a multiple of 64 elements (128 bytes) and whose next outer tile is the next 32
  # columns: tile n (even) and n+1 then share each aligned 128-byte row line (buffers are 128-byte aligned)
  if any(b or v.max_numel() != 32 or len(v.src[0].src) < 2 for v, b in rows): return False
  i0, i1 = rows[0][0].src[0].src[1], rows[1][0].src[0].src[1]
  def diff(f):
    def g(env):
      a, b = f(env), _hmx_eval(i0, env)
      return None if a is None or b is None else a - b
    return _hmx_const_delta(g, ranges)
  stride, nxt = diff(lambda env: _hmx_eval(i1, env)), diff(lambda env: _hmx_eval(i0, {**env, outer: env[outer] + 1}))
  return stride is not None and stride % 64 == 0 and nxt == 32

def _hmx_quad_rows(rows, n:UOp, ranges:list[UOp]) -> bool:
  # 32-byte row windows whose row stride is a multiple of 128 bytes and whose next N tile (loop n) is the next 32 bytes: tiles
  # 4j .. 4j+3 share each aligned 128-byte row line (buffers are 128-byte aligned)
  if any(b or v.max_numel() != 32 or len(v.src[0].src) < 2 for v, b in rows): return False
  i0, i1 = rows[0][0].src[0].src[1], rows[1][0].src[0].src[1]
  def diff(f):
    def g(env):
      a, b = f(env), _hmx_eval(i0, env)
      return None if a is None or b is None else a - b
    return _hmx_const_delta(g, ranges)
  stride, nxt = diff(lambda env: _hmx_eval(i1, env)), diff(lambda env: _hmx_eval(i0, {**env, n: env[n] + 1}))
  return stride is not None and stride % 128 == 0 and nxt == 32

def _hmx_quad_k_rows(rows, k:UOp, ranges:list[UOp]) -> bool:
  # 32-byte row windows (their own loads) that move 32 bytes on per K block (loop k) and sit 128-byte aligned at k % 4 == 0: K
  # blocks 4j .. 4j+3 read the same aligned 128-byte lines (NHWC activations with 128 | C, K = (.., C) with C innermost)
  if (int(k.vmax) + 1) % 4 or any(b or v.max_numel() != 32 or len(v.src[0].src) < 2 for v, b in rows): return False
  for v, _ in rows:
    ix = v.src[0].src[1]
    nxt = _hmx_const_delta(lambda env: None if (a:=_hmx_eval(ix, {**env, k: 4 * (env[k] // 4) + 1})) is None or
                           (z:=_hmx_eval(ix, {**env, k: 4 * (env[k] // 4)})) is None else a - z, ranges)
    al = _hmx_const_delta(lambda env: None if (a:=_hmx_eval(ix, {**env, k: 4 * (env[k] // 4)})) is None else a % 128, ranges)
    if nxt != 32 or al != 0: return False
  return True

def _hmx_uniform_rows(rows, ranges:list[UOp]) -> bool:
  # the 32 rows of an operand sit at one constant stride from row 0 (so a helper can walk them from two pointers)
  if any(b or len(v.src[0].src) < 2 for v, b in rows): return False
  i0 = rows[0][0].src[0].src[1]
  d = [_hmx_const_delta(lambda env, ii=v.src[0].src[1]: None if (a:=_hmx_eval(ii, env)) is None or (z:=_hmx_eval(i0, env)) is None
                        else a - z, ranges) for v, _ in rows]
  return d[1] is not None and all(x == k * d[1] for k, x in enumerate(d))

def _hmx_ptr_pairable(rowptr:list[UOp], outer:UOp, ranges:list[UOp]) -> bool:
  # output rows at a row stride that is a multiple of 64 elements, the next outer tile 32 columns on: tiles n (even), n+1
  # fill every aligned 128-byte row line together
  if any(p.op not in (Ops.INDEX, Ops.SHRINK) or len(p.src) < 2 for p in rowptr[:2]): return False
  i0, i1 = rowptr[0].src[1], rowptr[1].src[1]
  def diff(f):
    def g(env):
      a, b = f(env), _hmx_eval(i0, env)
      return None if a is None or b is None else a - b
    return _hmx_const_delta(g, ranges)
  stride, nxt = diff(lambda env: _hmx_eval(i1, env)), diff(lambda env: _hmx_eval(i0, {**env, outer: env[outer] + 1}))
  return stride is not None and stride % 64 == 0 and nxt == 32

def _hmx_uses(x:UOp, r:UOp) -> bool:
  # does x's value depend on loop r (data dependence only: a RANGE's own srcs just order it after its enclosing ranges)
  return x is r or any(r in u.src for u in x.toposort(lambda u: u.op is not Ops.RANGE))

def _hmx_tile_loops(uops:list[UOp], at:int):
  # the two innermost output-tile loops open at uops[at] -> (o, i, swap, legal): legal = they can be interchanged (nothing
  # between the loop heads depends on i, nothing between their ENDs), swap = legal and the one with more iterations is inner
  pos = {u:k for k,u in enumerate(uops)}
  end_of = {e.src[1]: e for e in uops if e.op is Ops.END and len(e.src) > 1 and e.src[1].op is Ops.RANGE}
  open_: list[UOp] = []
  for u in uops[:at]:
    if u.op is Ops.RANGE: open_.append(u)
    elif u.op is Ops.END and len(u.src) > 1 and u.src[1] in open_: open_.remove(u.src[1])
  loops = [r for r in open_ if r in end_of and r.arg[-1] != AxisType.REDUCE]
  if len(loops) < 2: return None
  o, i = loops[-2], loops[-1]
  mid = uops[pos[o]+1:pos[i]]
  between_ends = [u for u in uops[pos[end_of[i]]+1:pos[end_of[o]]] if u.op not in (Ops.GROUP, Ops.NOOP)]
  legal = bool(getenv("HMX_INTERCHANGE", 1)) and not between_ends and not any(_hmx_uses(x, i) for x in mid)
  return o, i, legal and o.vmax < i.vmax, legal

def _hmx_interchange(uops:list[UOp], o:UOp, i:UOp) -> list[UOp]:
  pos = {u:k for k,u in enumerate(uops)}
  return uops[:pos[o]] + [i, o] + uops[pos[o]+1:pos[i]] + uops[pos[i]+1:]

class DSPRenderer(ClangRenderer):
  has_threads = False
  def inline_load(self, u:UOp) -> bool: return _inline_vector_load(self, u)
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_typedef = "__attribute__((noinline)) void"
  string_rewrite = dsp_string+ClangRenderer.string_rewrite
  type_map = { **ClangRenderer.type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {**{k:v for k,v in ClangRenderer.code_for_op.items() if k != Ops.SQRT},
                 # native integer max (HVX vmax*); floats keep tinygrad's own (a<b)?b:a semantics, which differ from the
                 # builtin's IEEE maxNum on NaN. The statement expression evaluates each operand once: a plain ternary
                 # repeats both, and since single-use ALU results are inlined, a chain of maxes (argmax) grows exponentially.
                 Ops.MAX: lambda a,b,dtype: f"({{__auto_type _a=({a}); __auto_type _b=({b}); _a<_b?_b:_a;}})" if dtypes.is_float(dtype) else
                   f"__builtin_elementwise_max({a},{b})",
                 }
  # QF_MATH (set per renderer instance, see __init__): float32 a / b renders as a * reciprocal(b) -- v68/v69 HVX has no float
  # divide (LLVM scalarizes a vector one), and a lane stack whose top op is a division would keep the whole expression, EXP2s
  # included, scalar
  qf_code_for_op = {Ops.EXP2: lambda x,dtype: f"__TG_EXP2({x})", Ops.RECIPROCAL: lambda x,dtype: f"__TG_RECIP({x})",
                    Ops.SQRT: lambda x,dtype: f"__TG_SQRT({x})",
                    Ops.FDIV: lambda a,b,dtype: f"({a}*__TG_RECIP({b}))" if dtype == dtypes.float32 else f"({a}/{b})"}
  extra_matcher = (ClangRenderer.extra_matcher + pm_hvx_revectorize) if getenv("HVX_REVEC", 1) else ClangRenderer.extra_matcher
  qf_matcher = PatternMatcher([(UPat((Ops.EXP2, Ops.RECIPROCAL, Ops.SQRT), name="x"), _qf_math_half)])

  def __init__(self, target:Target):
    self.target, self.compiler, self.tensor_cores = target, DSPCompiler(), _dsp_tcs()
    self._qf_math_on()

  def _qf_math_on(self):
    # the qfloat exp2 / reciprocal helpers: EXP2 / RECIPROCAL stay ops (not decomposed) and render as the helpers
    if QF_MATH: self.code_for_op, self.extra_matcher = {**type(self).code_for_op, **self.qf_code_for_op}, self.qf_matcher + type(self).extra_matcher

  # HMX_ACC=0 keeps the plain per-K-block tile op (C round trip, 2 KB values) for comparison
  hmx_acc = bool(getenv("HMX_ACC", 1))
  def render(self, uops:list[UOp]) -> str:
    self._hmx_acc = False
    if self.hmx_acc: uops, self._hmx_acc = _hmx_acc_rewrite(uops)
    # _lane_window: uop positions and (position, buffer) of every store
    self._pos = {u:i for i,u in enumerate(uops)}
    self._stores = [(i, _hmx_param(u.src[0])) for i,u in enumerate(uops) if u.op is Ops.STORE]
    self._users: dict[UOp, list[UOp]] = {}
    for u in uops:
      for s in u.src: self._users.setdefault(s, []).append(u)
    self._scopes = [i for i,u in enumerate(uops) if u.op in (Ops.RANGE, Ops.END, Ops.IF, Ops.ENDIF)]
    src = self.render_kernel(*self._render(uops), uops)
    return src

  # V6_vrmpyub/V6_vrmpybusv (HVX): D(int32x32) = C(int32x32) + dot4(A(u8x4 broadcast scalar), B(u8x128, 32
  # groups of 4)) in one instruction -- no warp/lane cooperation needed (tensor_cores' threads=1), unlike
  # every other backend's WMMA. The accumulator-add form always needs a real HVX_Vector C; a plain (not
  # "_acc") vrmpy variant only exists for the from-zero case, so we always pass C through.
  #
  # KNOWN PERFORMANCE ISSUE: correct but currently slower than plain scalar code on real hardware,
  # because devectorizer2's do_stack_wmma (codegen/__init__.py) unconditionally decomposes every WMMA's
  # accumulator into per-element scalar loads/stores before rendering -- the right behavior for every
  # other backend (each GPU thread only ever holds a few scalar elements of a warp-distributed
  # fragment), but wrong here: Hexagon's "32 elements" is one HVX vector register that a single thread
  # (threads=1, no warp) processes atomically, and it should stay vector-resident across the reduction
  # loop instead of being rebuilt from 32 scalar reads on every accumulate call. A real fix needs the
  # generic devectorizer (or the accumulator's axis-ordering/layout in postrange.py's _apply_tc_opt) to
  # recognize single-thread/vector-native tensor cores as a distinct case -- out of scope here.
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix = list(prefix or [])
    b_dtypes = {_wmma_name(u): u.src[1].dtype for u in uops if u.op is Ops.WMMA}
    for name, _, dtype_in, dtype_out, _, _, upcast_sizes in wmma_args(uops):
      if dtype_in == dtypes.half:
        prefix.append(_hmx_wmma_helper(name, self._render_dtype(dtypes.half, 1024, AddrSpace.REG)))
        continue
      dtype_b = b_dtypes[name]
      if dtype_out == dtypes.int32 and upcast_sizes[2] == 2048:
        prefix.append(_hmx_i8_wmma_helper(name, *(self._render_dtype(dt, sz, AddrSpace.REG) for dt, sz in
                                                  zip([dtype_in, dtype_b, dtype_out], upcast_sizes))))
        continue
      dstr_a, dstr_b, dstr_c = (self._render_dtype(dt, sz, AddrSpace.REG) for dt, sz in
                                 zip([dtype_in, dtype_b, dtype_out], upcast_sizes))
      # u8 x u8: Vx.uw += vrmpy(Vu.ub, Rt.ub) takes A as a scalar. The signed forms have no scalar-A variant with B in the
      # vector, so A (4 bytes) is splat: u8 x s8 = Vx.w += vrmpy(Vu.ub, Vv.b) (vrmpybusv), s8 x s8 = vrmpy(Vu.b, Vv.b) (vrmpybv).
      if dtype_in == dtypes.uint8 and dtype_b == dtypes.uint8: call = "__builtin_HEXAGON_V6_vrmpyub_acc_128B(c, b, a_scalar)"
      else:
        vv = "__builtin_HEXAGON_V6_vrmpybusv_acc_128B" if dtype_in == dtypes.uint8 else "__builtin_HEXAGON_V6_vrmpybv_acc_128B"
        call = f"{vv}(c, __builtin_HEXAGON_V6_lvsplatw_128B(a_scalar), b)"
      prefix.append(f"""static inline {dstr_c} __{name}({dstr_a} a, {dstr_b} b, {dstr_c} c) {{
  unsigned int a_scalar; __builtin_memcpy(&a_scalar, &a, 4);
  return {call};
}}""")
    prefix += _qf_helpers(uops, lambda n: self._render_dtype(dtypes.float32, n, AddrSpace.REG))
    qm = _qf_math_helpers(uops, lambda dt, n: self._render_dtype(dt, n, AddrSpace.REG))
    if qm and not any("typedef float __hvx_f " in p for p in prefix): prefix += qm
    elif qm: prefix += [h for h in qm if not h.startswith(("typedef float __hvx_f", "static inline __hvx_f __hvx_mulsf"))]
    prefix += _hf_exp2_helpers(uops, lambda dt, n: self._render_dtype(dt, n, AddrSpace.REG))
    if getattr(self, '_hmx_acc', False): prefix.append(_HMX_ACC_HELPERS)
    if any(u.op is Ops.CUSTOM and isinstance(u.arg, str) and u.arg.startswith("__hmx_qadd_chunk(") for u in uops): prefix.append(_HMX_QADD_HELPERS)
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  # register arrays get HVX alignment: memory_coalescing merges their accesses into vector loads/stores (see coalesce.py),
  # whose ext_vector_types assume natural alignment up to one 128-byte HVX register
  def render_buffer(self, x:UOp):
    ret = super().render_buffer(x)
    if x.addrspace != AddrSpace.REG or x.max_numel() == 1 or "aligned(128)" in ret: return ret
    return ret[:-1] + " __attribute__((aligned(128)));"

  def _render_defines(self, uops) -> list[str]:
    return ['''/* DSP boilerplate */ struct dcvs_v2_req { int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency;
      _Bool set_dcvs_params; short _pad2; char target_corner; char min_corner; char max_corner; int _pad3[3];};''','int HAP_power_set(void*, void*);',
      'typedef union { struct { void *pv; unsigned int len; } buf; struct { int fd; unsigned int offset; } dma; } remote_arg;',
      'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);', 'int HAP_munmap(void *addr, int len);',
      'unsigned long long HAP_perf_get_time_us(void);'] + super()._render_defines(uops)

  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    msrc = ['int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'struct dcvs_v2_req req = {.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner = 6 /* TURBO */};',
            'HAP_power_set((void*)handle, (void*)&req);']
    msrc += ['if ((sc>>24) != 2) return 0;']
    msrc += [f'{self._render_dtype(b[1][0].dtype) if b[1][0].addrspace == AddrSpace.ALU else "int"} sz_or_val_{i} = '
             f'*({self._render_dtype(b[1][0].dtype) if b[1][0].addrspace == AddrSpace.ALU else "int"}*)((char*)pra[0].buf.pv+{i*8});'
             for i,b in enumerate(bufs)]
    msrc += [f'int off{i} = ((int*)pra[1].buf.pv)[{i}];' for i,b in enumerate(bufs) if b[1][0].addrspace == AddrSpace.GLOBAL]
    msrc += [f'void *buf_{i} = HAP_mmap(0,sz_or_val_{i},3,0,pra[{i+3}].dma.fd,0)+off{i};'
             for i,b in enumerate(bufs) if b[1][0].addrspace == AddrSpace.GLOBAL]
    msrc += ["unsigned long long start = HAP_perf_get_time_us();"]
    fbufs = [(f'buf_{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'sz_or_val_{i}') for i,b in enumerate(bufs)]
    msrc += [f"{function_name}({', '.join(fbufs)});"]
    msrc += ["*(unsigned long long *)(pra[2].buf.pv) = HAP_perf_get_time_us() - start;"]
    msrc += [f'HAP_munmap(buf_{i}, sz_or_val_{i});' for i,b in enumerate(bufs) if b[1][0].addrspace == AddrSpace.GLOBAL]
    msrc += ["return 0; }"]
    return '\n'.join(msrc)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}

def rpc_sc(method=0, ins=0, outs=0, fds=0): return (method << 24) | (ins << 16) | (outs << 8) | fds
def rpc_prep_args(ins=None, outs=None, in_fds=None):
  ins, outs, in_fds = ins or list(), outs or list(), in_fds or list()

  pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs) + len(in_fds)))()
  fds = (ctypes.c_int32 * (len(ins) + len(outs) + len(in_fds)))(*([-1] * (len(ins) + len(outs))), *in_fds)
  attrs = (ctypes.c_uint32 * (len(ins) + len(outs) + len(in_fds)))(*([0] * (len(ins) + len(outs))), *([1] * (len(in_fds))))

  for i, mv in enumerate(ins + outs): pra[i].buf.pv, pra[i].buf.len = ctypes.c_void_p(mv_address(mv) if mv.nbytes > 0 else 0), mv.nbytes
  return pra, fds, attrs, (ins, outs)

class DSPProgram(Program['DSPDevice']):
  def __init__(self, dev:DSPDevice, obj:TinyELF): self.dev, self.lib, self.signature = dev, obj.lib, obj.signature

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    if len(bufs) >= 16: raise RuntimeError(f"Too many buffers to execute: {len(bufs)}")

    pra, fds, attrs, _ = rpc_prep_args(ins=[var_vals_mv:=memoryview(bytearray((len(bufs)+len(vals))*8)), off_mv:=memoryview(bytearray(len(bufs)*4))],
                                       outs=[timer:=memoryview(bytearray(8)).cast('Q')], in_fds=[b.share_info.fd for b in bufs])
    for i,b in enumerate(bufs): struct.pack_into('i', var_vals_mv, i*8, b.size)
    for i,(v,(_,_,dt,_)) in enumerate(zip(vals, self.signature[len(bufs):]), start=len(bufs)): struct.pack_into(unwrap(dt.fmt), var_vals_mv, i*8, v)
    off_mv.cast('I')[:] = array.array('I', tuple(b.offset for b in bufs))
    self.dev.exec_lib(self.lib, rpc_sc(method=2, ins=2, outs=1, fds=len(bufs)), pra, fds, attrs)
    return timer[0] / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info, offset:int=0):
    self.va_addr, self.size, self.share_info, self.offset = va_addr, size, share_info, offset

class DSPAllocator(Allocator['DSPDevice']):
  def _alloc(self, size:int, options:BufferSpec):
    if getenv("MOCKDSP") or getenv("HEXSIM"): fd, share_info, flags = -1, None, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS
    else:
      b = qcom_dsp.ION_IOC_ALLOC(self.dev.ion_fd, len=size, align=0x200, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED)
      fd, flags = (share_info:=qcom_dsp.ION_IOC_SHARE(self.dev.ion_fd, handle=b.handle)).fd, mmap.MAP_SHARED
    return DSPBuffer(libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, flags, fd, 0), size, share_info, offset=0)

  @suppress_finalizing
  def _free(self, opaque:DSPBuffer, options:BufferSpec):
    libc.munmap(opaque.va_addr, opaque.size)
    if opaque.share_info is not None:
      os.close(opaque.share_info.fd)
      qcom_dsp.ION_IOC_FREE(self.dev.ion_fd, handle=opaque.share_info.handle)

  def _as_buffer(self, src:DSPBuffer) -> memoryview: return to_mv(src.va_addr, src.size)
  def _copyin(self, dest:DSPBuffer, src:memoryview): ctypes.memmove(dest.va_addr, mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:DSPBuffer): ctypes.memmove(mv_address(dest), src.va_addr, dest.nbytes)
  def _offset(self, buf, size:int, offset:int): return DSPBuffer(buf.va_addr+offset, size, buf.share_info, buf.offset+offset)

def _find_libgcc() -> str:
  # Every kernel this backend has ever compiled before HEXSIM/float32 support (uint8/int8/int32
  # GEMM/conv/elementwise) never needed a real scalar float divide, so this was never hit: DSPCompiler's
  # `-nostdlib -ffreestanding` build has no libc or compiler-rt at all. Scalar Hexagon float division
  # (`float / float`, and transitively anything sigmoid/softmax-shaped that needs it) is NOT always a
  # native instruction sequence -- it depends on the clang/LLVM version: clang>=19 inlines a native
  # Newton-Raphson sequence (sfrecipa/sffixupn/sffixupd/sfmpy:lib), but clang 15/17 emit a call to
  # `__hexagon_divsf3` instead, which then fails to link (`undefined symbol`) with nothing providing it.
  # Hexagon's own toolchain ships these soft-float routines in libgcc.a (a plain static archive -- only
  # symbols actually referenced get pulled in, so this is a costless no-op for every kernel that doesn't
  # need it, uint8/int8/int32 or otherwise). Best-effort: silently skip if the SDK isn't discoverable,
  # so environments without HEXAGON_TOOLCHAIN/HEXAGON_SDK_ROOT set see no behavior change.
  root = getenv("HEXAGON_TOOLCHAIN", "") or getenv("HEXAGON_SDK_ROOT", "")
  if not root: return ""
  libdir = pathlib.Path(root) / "target" / "hexagon" / "lib"
  if not libdir.is_dir(): return ""
  # No v65-specific archive is shipped in some SDK snapshots (oldest available may be v68+); the
  # Hexagon scalar ISA these soft-float routines target has been stable across v65-v81, so the
  # lowest available version is used as a compatible fallback. Not verified on real v65 hardware here
  # (only under MOCKDSP=1/qemu) -- flagged in the README as a caveat for whoever verifies this next.
  for arch in ["v65", "v66", "v67", "v68", "v69", "v71", "v73", "v75", "v77", "v79", "v81"]:
    candidate = libdir / arch / "libgcc.a"
    if candidate.exists(): return str(candidate)
  return ""

class DSPCompiler(Compiler):
  def __init__(self, mock:bool=False):
    self.mock, compiler_args = mock, f"--target=hexagon -mcpu=hexagon{HVX_ARCH} -fuse-ld=lld -nostdlib -mhvx={HVX_ARCH} -mhvx-length=128b"
    self.libgcc = _find_libgcc()
    # qemu cannot run HMX: MOCKDSP builds the hexagon_hmx TC as its scalar reference on the same tile layout
    if mock: self.args = f"-static -DHMX_REF {compiler_args}"
    else:
      # Generate link script to pass into clang. Aligning all used sections to 4k fixes invoke problem.
      sections = ['text', 'rela.plt', 'rela.dyn', 'plt', 'data', 'bss', 'hash', 'dynamic',
                  'got', 'got.plt', 'dynsym', 'dynstr', 'symtab', 'shstrtab', 'strtab']
      sections_link = '\n'.join([f'.{n} : ALIGN(4096) {{ *(.{n}) }}' for n in sections])
      with tempfile.NamedTemporaryFile(delete=False) as self.link_ld:
        self.link_ld.write(f"SECTIONS {{ . = 0x0; {sections_link}\n /DISCARD/ : {{ *(.note .note.* .gnu.hash .comment) }} }}".encode())
        self.link_ld.flush()

      self.args = f"-shared {compiler_args} -T{self.link_ld.name}"

    super().__init__(None if mock else "compile_dsp")

  def __del__(self):
    if not self.mock: os.unlink(self.link_ld.name)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as f:
      system(f"{getenv('CC','clang')} {self.args} -O2 -Wall -Werror -fno-stack-protector -x c -fPIC " +
             f"-ffreestanding -nostdlib - -o {f.name}" + (f" -x none {self.libgcc}" if self.libgcc else ""), input=src.encode())
      return pathlib.Path(f.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, "llvm-objdump")


class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    if getenv("HEXSIM"): super().__init__(device, DSPAllocator(self), [HexagonSimRenderer], HexagonSimProgram)
    elif getenv("MOCKDSP"): super().__init__(device, DSPAllocator(self), [MockDSPRenderer], MockDSPProgram)
    else:
      self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
      super().__init__(device, DSPAllocator(self), [DSPRenderer], DSPProgram)
      fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
      self.shell_buf = self.allocator.alloc(round_up(fastrpc_shell.nbytes, 0x1000), BufferSpec(nolru=True))
      ctypes.memmove(self.shell_buf.va_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

      self.init_dsp()
      RPCListener(self).start()

  def open_lib(self, lib):
    self.binded_lib, self.binded_lib_off = lib, 0
    fp = "file:///tinylib?entry&_modver=1.0&_dom=cdsp\0"
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [len(fp), 0xff])), memoryview(bytearray(fp.encode()))],
                                 outs=[o1:=memoryview(bytearray(0x8)), o2:=memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=0, ins=2, outs=2), pra=pra)
    if o1.cast('i')[1] < 0: raise RuntimeError(f"Cannot open lib: {o2.tobytes().decode()}")
    return o1.cast('I')[0]

  def close_lib(self, handle):
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [handle, 0xff]))], outs=[memoryview(bytearray(0x8)), memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=1, ins=1, outs=2), pra=pra)

  def exec_lib(self, lib, sc, args, fds, attrs):
    def _exec_lib():
      handle = self.open_lib(lib)
      qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.rpc_fd, fds=fds, attrs=attrs, inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=handle, sc=sc, pra=args))
      self.close_lib(handle)
    try: _exec_lib()
    except (OSError, PermissionError):
      # DSP might ask for a connection reset or just fail with operation not permitted, try to reset connection.
      self.init_dsp()
      try: _exec_lib()
      except (OSError, PermissionError) as e: raise RuntimeError(e)

  def init_dsp(self):
    if hasattr(self, 'rpc_fd'):
      with contextlib.suppress(OSError):
        qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=4, sc=rpc_sc(method=2, ins=0, outs=0)) # pylint: disable=access-member-before-definition
      os.close(self.rpc_fd) # pylint: disable=access-member-before-definition

    self.rpc_fd: int = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)
    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=self.shell_buf.va_addr, filelen=self.shell_buf.size, filefd=self.shell_buf.share_info.fd)
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=3, sc=rpc_sc(method=3, ins=0, outs=0))

class RPCListener(threading.Thread):
  def __init__(self, device:DSPDevice):
    super().__init__()
    self.device, self.daemon = device, True

  def run(self):
    # Setup initial request arguments.
    context, status, TINYFD = 0, 0xffffffff, 0xffff
    req_args, _, _, _ = rpc_prep_args(ins=[msg_send:=memoryview(bytearray(0x10)).cast('I'), out_buf:=memoryview(bytearray(0x10000)).cast('I')],
                                      outs=[msg_recv:=memoryview(bytearray(0x10)).cast('I'), in_buf:=memoryview(bytearray(0x10000)).cast('I')])
    req_args[1].buf.len = 0

    while True:
      # Update message request and send it.
      msg_send[:] = array.array('I', [context, status, req_args[1].buf.len, in_buf.nbytes])

      try: qcom_dsp.FASTRPC_IOCTL_INVOKE(self.device.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args)
      except OSError: continue # retry

      context, inbufs, outbufs = msg_recv[0], ((sc:=msg_recv[2]) >> 16) & 0xff, (msg_recv[2] >> 8) & 0xff

      in_ptr, out_ptr, objs = mv_address(in_buf), mv_address(out_buf), []
      for i in range(inbufs + outbufs):
        obj_ptr = round_up(in_ptr + 4, 8) if i < inbufs else round_up(out_ptr + 4, 8)
        objs.append(to_mv(obj_ptr, obj_size:=to_mv(in_ptr, 4).cast('I')[0]))
        if i < inbufs: in_ptr = obj_ptr + obj_size
        else:
          to_mv(out_ptr, 4).cast('I')[0] = obj_size
          out_ptr = obj_ptr + obj_size
          in_ptr += 4

      in_args, out_args = objs[:inbufs], objs[inbufs:]
      req_args[1].buf.len = out_ptr - mv_address(out_buf)

      status = 0 # reset status, will set if error
      if sc == 0x20200: pass # greating
      elif sc == 0x13050100: # open
        try: out_args[0].cast('I')[0] = TINYFD if (name:=in_args[3].tobytes()[:-1].decode()) == "tinylib" else os.open(name, os.O_RDONLY)
        except OSError: status = 1
      elif sc == 0x3010000:
        if (fd:=in_args[0].cast('I')[0]) != TINYFD: os.close(fd)
      elif sc == 0x9010000: # seek
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          assert in_args[0].cast('I')[2] == qcom_dsp.APPS_STD_SEEK_SET, "Supported only SEEK_SET"
          res, self.device.binded_lib_off = 0, in_args[0].cast('I')[1]
        else: res = os.lseek(fd, in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        status = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          buf = self.device.binded_lib[self.device.binded_lib_off:self.device.binded_lib_off+in_args[0].cast('I')[1]]
          self.device.binded_lib_off += len(buf)
        else: buf = os.read(fd, in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0:2] = array.array('I', [len(buf), int(len(buf) == 0)])
      elif sc == 0x1f020100: # stat
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = qcom_dsp.struct_apps_std_STAT.from_address(mv_address(out_args[0]))
        for f in out_stat._real_fields_: out_stat.__setattr__(f[0], int(getattr(stat, f"st_{f[0]}", 0)))
      elif sc == 0x2010100: # mmap
        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.device.rpc_fd, fd=-1, flags=in_args[0].cast('I')[2], vaddrin=0, size=in_args[0].cast('Q')[3])
        out_args[0].cast('Q')[0:2] = array.array('Q', [0, st.vaddrout])
      else: raise RuntimeError(f"Unknown op: {sc=:X}")

# ***** mock DSP *****

mockdsp_boilerplate = '''/* DSP boilerplate */ static long syscall(long r0, long r1, long r2, long r3, long r4, long r5, long r6) {
long retval; __asm__ volatile("r0 = %1; r1 = %2; r2 = %3; r3 = %4; r4 = %5; r5 = %6; r6 = %7; trap0(#1); %0 = r0" : "=r" (retval)
  : "r" (r0), "r" (r1), "r" (r2), "r" (r3), "r" (r4), "r" (r5), "r" (r6) : "r0", "r1", "r2", "r3", "r4", "r5", "r6"); return retval; }
static int read(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 63); }}
static int write(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 64); }}
static int exit(int ret) {{ return syscall(ret, 0, 0, 0, 0, 0, 93); }}
static unsigned int inscount(void) {{ unsigned int ret; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r" (ret) : : "r0"); return ret; }}
static void *mmap2(void *addr, unsigned int length, int prot, int flags, int fd, unsigned long offset) {{
return (void*)syscall((long)addr, length, prot, flags, fd, offset, 222); }}'''

class MockDSPRenderer(DSPRenderer):
  def __init__(self, target:Target):
    self.target, self.compiler, self.tensor_cores = target, DSPCompiler(mock=True), _dsp_tcs()
    self._qf_math_on()
  def _render_defines(self, uops) -> list[str]: return ClangRenderer._render_defines(self, uops)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # https://gpages.juszkiewicz.com.pl/syscalls-table/syscalls.html
    # control register 21 is HEX_REG_QEMU_INSN_CNT, 0x6a15c000 loads it
    msrc = [mockdsp_boilerplate, 'void _start(void) {']
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL:
        sz = b[1][0].max_numel()*b[1][0].dtype.itemsize
        # for loop for big reads
        msrc.append(f"void *buf{i} = mmap2(0, {sz}, 3, 0x21, -1, 0); for(int rd = 0; rd < {sz}; rd += read(0, buf{i}+rd, {sz}-rd));")
      else:
        msrc.append(f"{self._render_dtype(b[1][0].dtype)} val{i}; read(0, &val{i}, {b[1][0].dtype.itemsize});")
    msrc.append("unsigned int st = inscount();")
    params = [(f'(void*)buf{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'val{i}') for i,b in enumerate(bufs)]
    msrc.append(f"{function_name}({', '.join(params)});")
    msrc.append("unsigned int et = inscount() - st; write(1, &et, sizeof(et));")
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL: msrc.append(f"write(1, buf{i}, {b[1][0].max_numel()*b[1][0].dtype.itemsize});")
    msrc.append('exit(0); }')
    return '\n'.join(msrc)

class MockDSPProgram(Program[DSPDevice]):
  def __init__(self, dev:DSPDevice, obj:TinyELF): self.lib, self.signature = obj.lib, obj.signature
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    with tempfile.NamedTemporaryFile(suffix=".out") as dsp_lib:
      dsp_lib.write(self.lib)
      dsp_lib.flush()
      os.chmod(dsp_lib.name, 0o0777)
      proc = subprocess.run(["qemu-hexagon-static", *(['-strace'] if DEBUG >= 5 else []), dsp_lib.name],
        input=b''.join([bytes(to_mv(x.va_addr, x.size)) for x in bufs] +
                       [struct.pack(unwrap(dt.fmt), x) for x,(_,_,dt,_) in zip(vals, self.signature[len(bufs):])]),
        stdout=subprocess.PIPE, check=True)
    offset = 4
    for x in bufs:
      to_mv(x.va_addr, x.size)[:] = proc.stdout[offset:offset+x.size]
      offset += x.size
    assert offset == len(proc.stdout)
    return struct.unpack("I", proc.stdout[0:4])[0] / 1e9  # pretend it's 1 Ghz, but this is an inscount, not a time

# ***** hexagon-sim DSP (BEAM-search timing via Qualcomm's own instruction-set simulator) *****
#
# MOCKDSP's qemu-hexagon-static path times candidates via QEMU's inscount() pseudo-register --
# a raw instruction count, not remotely cycle-accurate, and blind to Hexagon-specific pipeline,
# vector-unit, or memory-hierarchy effects (see scripts/android/tinygrad_hexagon_bridge/README.md
# in the onnx-simplifier repo's "Removing TVM as a transport dependency"/qemu-vs-hexagon-sim note
# for the motivating case: real-hardware speed for a fixed kernel *shape* flipped between faster
# and slower than a baseline purely from cache/channel-count effects instruction count can't see).
#
# hexagon-sim is Qualcomm's own instruction-set simulator (ships in the Hexagon SDK's
# HEXAGON_Tools/*/Tools/bin/hexagon-sim) with a PMU-derived total-cycle count ("Pcycles=", printed
# once at process exit). Its default mode is a fast functional-only estimate not meaningfully
# better than instruction counting, but its `--timing` mode (used below) is a real
# pipeline/dual-issue/cache-hierarchy model: confirmed empirically (see the README section this
# lands with) to report an ~28x cycle difference between two kernels with the IDENTICAL
# instruction count, differing only in whether their memory access pattern stays cache-resident
# or not -- exactly the class of effect raw instruction counting is structurally blind to.
#
# Reading a cycle-counter register live from inside a standalone-sim binary doesn't work (the
# PCYCLE control register pair reads back 0 in this mode -- confirmed empirically in this
# project's separate hexagon_sim_harness.py work), so cycles are measured the same way that
# harness does: compile the SAME kernel wrapper twice, once calling the kernel body once and once
# calling it twice (REPEAT=1 vs REPEAT=2), run both under hexagon-sim, and take the *difference*
# in each run's total Pcycles. The simulator is deterministic, so this exactly isolates the cost
# of one kernel invocation and cancels the fixed process-startup/tear-down overhead.
#
# Kernel *inputs* are zero-initialized static buffers, not real data copied from the caller's
# DSPBuffers: cycle count for a fixed-control-flow kernel (no data-dependent branches -- true of
# every kernel this project generates, conv/gemm with static loop bounds) doesn't depend on the
# data values, only on the shapes/loop-bounds already baked into the generated source. This lets
# HexagonSimCompiler.compile() do the (expensive, ~1s) real hexagon-clang + hexagon-sim round trip
# once per distinct kernel source and cache it via the normal Compiler.compile_cached() path,
# instead of re-running the simulator on every __call__.

HEXSIM_ARCH = getenv("HEXSIM_ARCH", "v73")  # matches this project's real device (Snapdragon/Hexagon v73), not DSPCompiler's v65 baseline
HEXSIM_CLOCK_HZ = 1_000_000_000  # placeholder nominal clock (Hexagon v73 cDSP is close to 1 GHz) -- only the *relative* ranking of
                                  # returned times matters for BEAM; this scales Pcycles into a plausible-looking float, nothing more.

class HexagonSimRenderer(DSPRenderer):
  def __init__(self, target:Target):
    self.target, self.compiler, self.tensor_cores = target, HexagonSimCompiler(), _dsp_tcs()
    self._qf_math_on()
  def _render_defines(self, uops) -> list[str]: return ClangRenderer._render_defines(self, uops)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # Plain hosted main() (hexagon-sim's standalone-OS mode has real libc) -- no raw trap0 dance
    # needed, unlike MockDSPRenderer's qemu-bare-metal entry. Buffers are static, zero-filled,
    # 128B-aligned (HVX vector width) arrays sized from the UOp shapes the renderer already knows
    # at render time; see the module docstring above for why real data isn't needed here.
    # `write()`-ing one byte of each output buffer at the end forces the compiler to treat the
    # whole kernel body as having an externally-observable side effect -- without this, -O1 sees
    # no consumer of the static buffers this synthetic main() writes and dead-code-eliminates the
    # entire kernel call (confirmed empirically: timings came back as exactly 0 without it).
    msrc = ['#include <unistd.h>', '#ifndef REPEAT\n#define REPEAT 1\n#endif', 'int main(void) {']
    global_idxs = []
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL:
        sz = max(b[1][0].max_numel()*b[1][0].dtype.itemsize, 1)
        msrc.append(f"static unsigned char buf{i}[{sz}] __attribute__((aligned(128)));")
        global_idxs.append(i)
      else:
        msrc.append(f"{self._render_dtype(b[1][0].dtype)} val{i} = 0;")
    params = [(f'(void*)buf{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'val{i}') for i,b in enumerate(bufs)]
    msrc.append(f"for (int r = 0; r < REPEAT; r++) {{ {function_name}({', '.join(params)}); }}")
    for i in global_idxs: msrc.append(f"write(1, buf{i}, 1);")
    msrc.append('return 0; }')
    return '\n'.join(msrc)

def _hexsim_tools_dir() -> pathlib.Path:
  root = getenv("HEXAGON_TOOLS", "") or getenv("HEXAGON_TOOLCHAIN", "")
  if not root: raise RuntimeError("HEXSIM=1 needs HEXAGON_TOOLS (or HEXAGON_TOOLCHAIN) set to a Hexagon SDK Tools/ dir with hexagon-sim")
  path = pathlib.Path(root)
  if not (path/"bin"/"hexagon-clang").exists() or not (path/"bin"/"hexagon-sim").exists():
    raise RuntimeError(f"{path} does not look like a Hexagon Tools dir (missing bin/hexagon-clang or bin/hexagon-sim)")
  return path

def _hexsim_env(tools:pathlib.Path, workdir:pathlib.Path) -> dict[str,str]:
  # hexagon-sim links libncurses.so.5, which modern distros only ship as .so.6 (same ABI for
  # this use) -- symlink a shim dir onto LD_LIBRARY_PATH, matching hexagon_sim_harness.py's fix.
  env = dict(os.environ)
  sim = tools/"bin"/"hexagon-sim"
  probe = subprocess.run(["ldd", str(sim)], capture_output=True, text=True, check=False)
  if "not found" not in probe.stdout: return env
  shim = workdir/"shim"
  shim.mkdir(exist_ok=True)
  for name in ("ncurses", "tinfo"):
    link = shim/f"lib{name}.so.5"
    if link.exists(): continue
    for lib_dir in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
      source = pathlib.Path(lib_dir)/f"lib{name}.so.6"
      if source.exists():
        with contextlib.suppress(FileExistsError): link.symlink_to(source)  # benign race under parallel BEAM workers
        break
  env["LD_LIBRARY_PATH"] = f"{shim}:{env.get('LD_LIBRARY_PATH', '')}"
  return env

class HexagonSimCompiler(Compiler):
  def __init__(self): super().__init__("compile_hexsim")

  def _compile_one(self, tools:pathlib.Path, src:str, repeat:int) -> bytes:
    with tempfile.TemporaryDirectory() as d:
      workdir = pathlib.Path(d)
      (workdir/"k.c").write_text(src)
      elf = workdir/"k.elf"
      cmd = [str(tools/"bin"/"hexagon-clang"), f"-m{HEXSIM_ARCH}", "-mhvx", "-mhvx-length=128B", "-O1",
             f"-DREPEAT={repeat}", str(workdir/"k.c"), "-o", str(elf), "-lm"]
      result = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if result.returncode: raise CompileError(f"hexagon-clang failed:\n{result.stderr}")
      return elf.read_bytes()

  def compile(self, src:str) -> bytes:
    tools = _hexsim_tools_dir()
    elf1, elf2 = self._compile_one(tools, src, 1), self._compile_one(tools, src, 2)
    return struct.pack("<Q", len(elf1)) + elf1 + elf2

class HexagonSimProgram(Program[DSPDevice]):
  def __init__(self, dev:DSPDevice, obj:TinyELF):
    n = struct.unpack("<Q", obj.lib[:8])[0]
    self.elf1, self.elf2 = obj.lib[8:8+n], obj.lib[8+n:]

  def _run_pcycles(self, tools:pathlib.Path, elf_bytes:bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".elf") as f:
      f.write(elf_bytes)
      f.flush()
      os.chmod(f.name, 0o755)
      env = _hexsim_env(tools, pathlib.Path(f.name).parent)
      # --timing enables hexagon-sim's cycle-accurate pipeline/dual-issue/cache-hierarchy model
      # (confirmed to change the reported Pcycles vs. the default fast functional-only mode --
      # see the README section this lands with); the default mode's Pcycles is a coarser,
      # instruction-scheduling-blind estimate much closer to plain instruction counting.
      result = subprocess.run(
        [str(tools/"bin"/"hexagon-sim"), f"-m{HEXSIM_ARCH}", "--timing", "--simulated_returnval", f.name],
        capture_output=True, text=True, env=env, timeout=900, check=False)
      output = result.stdout + result.stderr
      if result.returncode != 0: raise RuntimeError(f"hexagon-sim exit {result.returncode}:\n{output}")
      m = re.search(r"Pcycles=(\d+)", output)
      if m is None: raise RuntimeError(f"hexagon-sim output has no Pcycles= line:\n{output}")
      return int(m.group(1))

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    tools = _hexsim_tools_dir()
    base, twice = self._run_pcycles(tools, self.elf1), self._run_pcycles(tools, self.elf2)
    return max(twice - base, 0) / HEXSIM_CLOCK_HZ
