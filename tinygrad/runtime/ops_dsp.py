from __future__ import annotations
import ctypes, os, mmap, tempfile, pathlib, array, threading, contextlib, sys, subprocess, struct, re
assert sys.platform != 'win32'
from tinygrad.device import BufferSpec, Compiled, Allocator, Compiler, Program, TinyELF, CompileError
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp, GroupOp, AxisType
from tinygrad.helpers import getenv, round_up, mv_address, to_mv, cpu_objdump, system, DEBUG, suppress_finalizing, Target, unwrap
from tinygrad.renderer.cstyle import ClangRenderer, wmma_args, _wmma_name
from tinygrad.codegen.opt import tc
from tinygrad.runtime.autogen import libc, qcom_dsp
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

from tinygrad.uop.ops import PatternMatcher, UPat

HVX_PREFETCH = getenv("HVX_PREFETCH", 2048)
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

def _lane_slice(x:UOp) -> tuple[UOp, list[int]]|None:
  # STACK(v[k], v[k+1], ..., v[k+n-1]) of one vector v -> (v, [k..k+n-1])
  if len(x.src) < 2 or any(s.op is not Ops.INDEX or len(s.src) != 2 or s.src[0] is not x.src[0].src[0] for s in x.src): return None
  v = x.src[0].src[0]
  if v._shape is None or len(v._shape) != 1: return None
  lanes = [s.src[1].src[0].arg if s.src[1].op is Ops.CAST else s.src[1].arg for s in x.src]
  if not all(isinstance(l, int) for l in lanes) or lanes != list(range(lanes[0], lanes[0]+len(lanes))): return None
  return v, lanes

# NOTE: this just increases readability of the generated code
dsp_string = PatternMatcher([
  (UPat(Ops.CONST, (dtypes.int8, dtypes.uint8), name="x"), lambda ctx,x: str(x.val)),
  (UPat(Ops.MUL, dtypes.float32, name="x"), lambda ctx,x:
   f"__hvx_{x.op.name.lower()}_f{x.max_numel()}({', '.join(ctx[s] for s in x.src)})" if _qf_vec(x) else None),
  # a STACK of consecutive lanes of one wider vector (memory_coalescing merged two adjacent loads) is a lane slice: one
  # shufflevector instead of a per-lane constructor
  (UPat(Ops.STACK, name="x"), lambda ctx,x: f"(({ctx.render_type(x)})__builtin_shufflevector({ctx[v]}, {ctx[v]}, {','.join(str(l) for l in lanes)}))"
   if (sl:=_lane_slice(x)) is not None and (v:=sl[0]) is not None and (lanes:=sl[1]) else None),
  # a STACK of one repeated scalar is a splat, which clang lowers to a single HVX vsplat
  (UPat(Ops.STACK, name="x"), lambda ctx,x: f"(({ctx.render_type(x)})({ctx[x.src[0]]}))"
   if len(x.src) > 1 and all(s is x.src[0] for s in x.src) and x.src[0]._shape == () else None),
  # vector casts must convert per lane; a C cast between ext_vector_types is a bitcast
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_type(x)})"
   if x.max_numel() > 1 else None),
  # software-prefetch ahead of every vector load: streaming kernels on this DSP stall on DDR latency, not ALU. dcfetch is a
  # non-faulting hint, so prefetching past the end of a buffer is harmless. One dcfetch per 128-byte line the load covers
  # (a 128-lane int32 load is four HVX registers / lines); sub-line loads are skipped, a dcfetch per 32-byte load costs more
  # than it hides. HVX_PREFETCH is the distance in bytes (0 = off).
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"), lambda ctx,bidx,x:
   "(" + "".join(f"__builtin_HEXAGON_Y2_dcfetch((char*){ctx[bidx]}+{HVX_PREFETCH+o}), "
                 for o in range(0, max(x.max_numel()*x.dtype.itemsize, 1), 128)) + f"{ctx.render_access(bidx)})"
   if HVX_PREFETCH > 0 and x.max_numel()*x.dtype.itemsize >= 128 and bidx.addrspace is AddrSpace.GLOBAL else None),
])

# ***** HVX re-vectorization *****
# devectorizer2 splits every elementwise op into per-lane scalars and memory_coalescing only regroups the loads and
# stores (up to 128 lanes for DSP), so the ALU in between is rendered as `(int128){(a[0]+b[0]),(a[1]+b[1]),...}`, which
# LLVM lowers lane by lane (~10x the instructions of one vector add). This pass rebuilds vector ALU ops bottom-up:
# STACK(op(a_i, b_i) for i) -> op(STACK(a_i), STACK(b_i)); STACK(v[0], ..., v[n-1]) -> v. clang lowers ext_vector_type
# arithmetic straight to HVX under -mhvx. Compares/WHERE are left scalar: C vector compares yield same-width int masks,
# not _Bool vectors, so they'd need mask-dtype plumbing -- MAX (the common select) is native instead.
HVX_VEC_OPS = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.MAX, Ops.CAST}

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
    lanes = [_lane(c.src[1]) if c.op is Ops.INDEX and len(c.src) == 2 and c.src[0] is c0.src[0] else None for c in col]
    return None not in lanes and lanes == list(range(lanes[0], lanes[0]+len(lanes)))
  if c0.op in HVX_VEC_OPS and all(c.op is c0.op and c.dtype == c0.dtype and c.arg == c0.arg and len(c.src) == len(c0.src) for c in col):
    return all(_vec_column_ok(tuple(c.src[j] for c in col), depth+1) for j in range(len(c0.src)))
  return False

def hvx_revectorize(x:UOp) -> UOp|None:
  srcs, n = x.src, len(x.src)
  if n < 2 or x.dtype == dtypes.void: return None
  s0 = srcs[0]
  # STACK(v[0], v[1], ..., v[n-1]) of a length-n vector is v itself
  if s0.op is Ops.INDEX and len(s0.src) == 2 and s0.src[0]._shape == (n,) and \
     all(s.op is Ops.INDEX and len(s.src) == 2 and s.src[0] is s0.src[0] and _lane(s.src[1]) == i for i,s in enumerate(srcs)):
    return s0.src[0]
  if s0.op not in HVX_VEC_OPS or s0.dtype == dtypes.bool or s0._shape != (): return None
  if s0.op is Ops.MAX and dtypes.is_float(s0.dtype): return None  # float max renders as a scalar statement expression
  if any(s.op is not s0.op or s.dtype != s0.dtype or s.arg != s0.arg or len(s.src) != len(s0.src) or s._shape != () for s in srcs): return None
  if not _vec_column_ok(srcs): return None
  return UOp(s0.op, s0.dtype, tuple(UOp.stack(*[s.src[j] for s in srcs]) for j in range(len(s0.src))), s0.arg)

pm_hvx_revectorize = PatternMatcher([(UPat(Ops.STACK, name="x"), hvx_revectorize)])

# HMX=1 adds the V69 HMX (fp16) tensor core in front of the HVX vrmpy ones
def _dsp_tcs(): return (tc.hexagon_hmx if getenv("HMX") else []) + tc.hexagon_v65

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


# ---- HMX_ACC (default with HMX=1): accumulator kept inside HMX across the reduce loop ----
# The hexagon_hmx WMMA as tinygrad lowers it passes 2 KB tiles by value and round-trips the fp16 accumulator through a
# register array every K block. _hmx_acc_rewrite turns the linearized kernel into what the hand kernel does instead:
#   before the reduce loop   __hmx_begin();                  (bias table, clear state)
#   each K block             pack A, B rows into VTCM (one 128-byte halfword interleave per row pair) + one load pair
#   after the reduce loop    __hmx_out(dst...)               (one :after.hf store, then HVX loads/shuffles into the acc array)
# so each output tile is rounded to fp16 once (exact accumulation, like hmx_block.h) and nothing 2 KB-sized lives on the stack.
_HMX_CA, _HMX_CB = 80, 32  # VTCM tile cache slots (2 KB each) for A and B
_HMX_ACC_HELPERS = r"""#pragma clang diagnostic ignored "-Wunused-function"
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
#else
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
static __fp16 __hmx_rca[__HMX_CA][1024] __attribute__((aligned(128))), __hmx_rcb[__HMX_CB][1024] __attribute__((aligned(128)));
static inline __fp16* __hmx_ca(int i) { return __hmx_rca[i]; }
static inline __fp16* __hmx_cb(int i) { return __hmx_rcb[i]; }
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
#else
extern unsigned char* __hmx_vtcm;
extern unsigned int __hmx_gen;
static int __hmx_t;
/* VTCM (2 KB aligned): A stage x2 @0, B stage x2 @4 KB, out @8 KB, bias table @10 KB */
static inline __fp16* __hmx_sa(void) { return (__fp16*)(__hmx_vtcm + 2048 * __hmx_t); }
static inline __fp16* __hmx_sb(void) { return (__fp16*)(__hmx_vtcm + 4096 + 2048 * __hmx_t); }
/* tile caches: A 80 slots @16 KB, B 32 slots @176 KB (the runtime gives 256 KB inside one 256 KB window) */
#define __HMX_CA @CA@
#define __HMX_CB @CB@
static inline __fp16* __hmx_ca(int i) { return (__fp16*)(__hmx_vtcm + 16384 + 2048 * i); }
static inline __fp16* __hmx_cb(int i) { return (__fp16*)(__hmx_vtcm + 180224 + 2048 * i); }
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
#endif
/* output tile row pair q (IDX layout: rows 2q, 2q+1 interleaved) -> two 32-element rows at 64-byte aligned pointers */
#ifdef HMX_REF
static inline void __hmx_out2(__fp16* d0, __fp16* d1, const __fp16* o) {
  for (int j = 0; j < 32; j++) { d0[j] = o[2 * j]; d1[j] = o[2 * j + 1]; }
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
}""".replace("@CA@", str(_HMX_CA)).replace("@CB@", str(_HMX_CB))

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

def _hmx_direct_out(uops, pos, users, e, stores):
  # after the reduce loop tinygrad reloads the accumulator array and stores lane permutations of it to the output. If that
  # is all that happens to it, return (pointer uop for each tile row 0..31 as an "(expr+off)"-able uop list, uops to drop)
  buf = _hmx_param(stores[0].src[0])
  elem_lane: dict[int, int] = {}
  for so in stores:
    if (off:=_hmx_const(so.src[0].src[1])) is None or len(so.src[0].src) < 2: return None
    for t, x in enumerate(so.src[1].src): elem_lane[off + t] = _hmx_lane(x)[1]
  loads = [u for u in uops[pos[e]+1:] if u.op is Ops.LOAD and _hmx_param(u.src[0]) is buf]
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
  loops = _hmx_tile_loops(uops, first) if first is not None else None
  call_start = False
  span: dict[UOp, tuple] = {}
  drop: set[UOp] = set()
  before: dict[int, list[UOp]] = {}
  after: dict[int, list[UOp]] = {}
  replace: dict[UOp, UOp] = {}
  for w in uops:
    if w.op is not Ops.WMMA or w.arg[1] != dtypes.half: continue
    ra, rb = _hmx_rows(w.src[0]), _hmx_rows(w.src[1])
    if ra is None or rb is None: return _hmx_bail(uops, 1)
    # the reduce loop: the innermost RANGE whose END encloses the WMMA
    ends = [e for e in uops if e.op is Ops.END and len(e.src) > 1 and e.src[1].op is Ops.RANGE
            and pos[e.src[1]] < pos[w] < pos[e]]
    if not ends: return _hmx_bail(uops, 2)
    e = min(ends, key=lambda e: pos[e]-pos[e.src[1]])
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
    before.setdefault(pos[e.src[1]], []).append(UOp(Ops.CUSTOM, dtypes.void, (), "__hmx_begin();"))
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
      kr, kt = f"{{{len(srcs)}}}", int(e.src[1].vmax) + 1  # the reduce loop's variable and trip count
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
        def exact(r, n, x, pack):
          # slots contiguous in K for each tile index, so one spanning load pair can read them after the loop
          deps = {l for l in (outer, inner) if _hmx_uses(r[0][0].src[0], l)}
          if deps == {inner} and kt * ti <= n: base, cond = f"({iname})*{kt}", f"({on})==0"
          elif deps == {outer} and kt <= n: base, cond = "0", f"({iname})==0"
          elif not deps and kt <= n: base, cond = "0", f"({on})==0 && ({iname})==0"
          else: return None
          return f" _{x} = __hmx_c{x}({base}+({kr})); if ({cond}) {{{{{pack} }}}}", base
        ea = exact(ra, _HMX_CA, "a", pa) if ro[0] else None
        eb = exact(rb, _HMX_CB, "b", pb) if ro[1] else None
        if ea and eb:
          # K tiles stay in VTCM: no load pair per K block, one spanning pair after the loop (see the output statement)
          span[w] = (ea[1], eb[1], kt, outer, inner, on, iname)
          ea, eb = ea[0], eb[0]
        else: ea, eb = ea and ea[0], eb and eb[0]
        pa, pb = ea or (cached("a", _HMX_CA, ptr[0], "a", pa) if ro[0] else pa), eb or (cached("b", _HMX_CB, ptr[32], "b", pb) if ro[1] else pb)
        srcs = srcs + (e.src[1], outer, inner)
      else:
        if ro[0]: pa = cached("a", _HMX_CA, ptr[0], "a", pa)
        if ro[1]: pb = cached("b", _HMX_CB, ptr[32], "b", pb)
        if any(ro): srcs = srcs + (e.src[1],)
      if any(ro): call_start = True
      # the next K block of B (and of A while it's still being packed) is fetched into L2 while this one packs
      pb = f" __hmx_prefetch_next({ptr[32]}, {ptr[33]});" + pb
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
      ab, bb, kt_, o_, i_, on_, in_ = span[w]
      for k_, v_ in ((on_, "{%d}" % len(srcs0)), (in_, "{%d}" % (len(srcs0)+1))): ab, bb = ab.replace(k_, v_), bb.replace(k_, v_)
      return f" __hmx_mac_span(__hmx_ca({ab}), __hmx_cb({bb}), {kt_});", (o_, i_)
    # after the loop: one store, then each accumulator-array vector = the same lanes of the output tile
    outs = []
    for k, so in enumerate(stores):
      lanes = [l[1] for x in so.src[1].src if (l:=_hmx_lane(x)) is not None]
      if len(lanes) == 128 and all(0 <= l < 1024 for l in lanes) and len({l//128 for l in lanes}) <= 2:
        blks = sorted({l//128 for l in lanes})
        b0, b1 = blks[0], blks[-1]
        idx = ",".join(str(l - b0*128 if l//128 == b0 else 128 + l - b1*128) for l in lanes)
        outs.append(f" *(__hmx_h128*){{{k}}} = __builtin_shufflevector(_o[{b0}], _o[{b1}], {idx});")
      else:
        outs.append(f" *(__hmx_h128*){{{k}}} = (__hmx_h128){{{{{','.join(f'_p[{l}]' for l in lanes)}}}}};")
    if (direct:=_hmx_direct_out(uops, pos, users, e, stores)) is not None:
      # the accumulator array is only copied to an output buffer after the loop: write the tile rows there directly
      rowptr, gone = direct
      drop |= gone
      pairs = "".join(f" __hmx_out2({{{2*q}}}, {{{2*q+1}}}, _p+{64*q});" for q in range(16))
      # where the last replaced store was: every row pointer expression is rendered by then
      at = max(pos[g] for g in gone if g.op is Ops.STORE)
      sm, sx = span_mac(tuple(rowptr))
      after.setdefault(at, []).extend([u for u in dict.fromkeys(rowptr) if u not in pos] +
                                      [UOp(Ops.CUSTOM, dtypes.void, tuple(rowptr)+sx, "{{"+sm+" const __fp16* _p = __hmx_store();"+pairs+" }}")])
      continue
    sm, sx = span_mac(tuple(so.src[0] for so in stores))
    after.setdefault(pos[e], []).append(UOp(Ops.CUSTOM, dtypes.void, tuple(so.src[0] for so in stores)+sx,
      "{{"+sm+" __fp16* _p = __hmx_store(); __hmx_h128* _o = (__hmx_h128*)_p; (void)_o;"+"".join(outs)+" }}"))
  if not replace: return _hmx_bail(uops, 9)
  if call_start: before.setdefault(0, []).insert(0, UOp(Ops.CUSTOM, dtypes.void, (), "__hmx_call_start();"))
  out = []
  for i, u in enumerate(uops):
    out += before.get(i, [])
    if u in replace: out.append(replace[u])
    elif u not in drop: out.append(u)
    out += after.get(i, [])
  if loops is not None and loops[2]: out = _hmx_interchange(out, loops[0], loops[1])
  return out, True

def _hmx_uses(x:UOp, r:UOp) -> bool:
  # does x's value depend on loop r (data dependence only: a RANGE's own srcs just order it after its enclosing ranges)
  return x is r or any(r in u.src for u in x.toposort(lambda u: u.op is not Ops.RANGE))

def _hmx_tile_loops(uops:list[UOp], at:int):
  # the two innermost output-tile loops open at uops[at] -> (o, i, swap): swap = the one with more iterations should be
  # outermost and they can be interchanged (nothing between the loop heads depends on i, nothing between their ENDs)
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
  swap = bool(getenv("HMX_INTERCHANGE", 1)) and o.vmax < i.vmax and not between_ends and not any(_hmx_uses(x, i) for x in mid)
  return o, i, swap

def _hmx_interchange(uops:list[UOp], o:UOp, i:UOp) -> list[UOp]:
  pos = {u:k for k,u in enumerate(uops)}
  return uops[:pos[o]] + [i, o] + uops[pos[o]+1:pos[i]] + uops[pos[i]+1:]

class DSPRenderer(ClangRenderer):
  has_threads = False
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_typedef = "__attribute__((noinline)) void"
  string_rewrite = dsp_string+ClangRenderer.string_rewrite
  type_map = { **ClangRenderer.type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {**{k:v for k,v in ClangRenderer.code_for_op.items() if k != Ops.SQRT},
                 # native integer max (HVX vmax*); floats keep tinygrad's own (a<b)?b:a semantics, which differ from the
                 # builtin's IEEE maxNum on NaN. The statement expression evaluates each operand once: a plain ternary
                 # repeats both, and since single-use ALU results are inlined, a chain of maxes (argmax) grows exponentially.
                 Ops.MAX: lambda a,b,dtype: f"({{__auto_type _a=({a}); __auto_type _b=({b}); _a<_b?_b:_a;}})" if dtypes.is_float(dtype) else
                   f"__builtin_elementwise_max({a},{b})"}
  extra_matcher = (ClangRenderer.extra_matcher + pm_hvx_revectorize) if getenv("HVX_REVEC", 1) else ClangRenderer.extra_matcher

  def __init__(self, target:Target): self.target, self.compiler, self.tensor_cores = target, DSPCompiler(), _dsp_tcs()

  # HMX_ACC=0 keeps the plain per-K-block tile op (C round trip, 2 KB values) for comparison
  hmx_acc = bool(getenv("HMX_ACC", 1))
  def render(self, uops:list[UOp]) -> str:
    self._hmx_acc = False
    if self.hmx_acc: uops, self._hmx_acc = _hmx_acc_rewrite(uops)
    return self.render_kernel(*self._render(uops), uops)

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
    if getattr(self, '_hmx_acc', False): prefix.append(_HMX_ACC_HELPERS)
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
  def __init__(self, target:Target): self.target, self.compiler, self.tensor_cores = target, DSPCompiler(mock=True), _dsp_tcs()
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
  def __init__(self, target:Target): self.target, self.compiler, self.tensor_cores = target, HexagonSimCompiler(), _dsp_tcs()
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
