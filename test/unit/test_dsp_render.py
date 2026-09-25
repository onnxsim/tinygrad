import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, Context
from tinygrad.codegen import to_program
from tinygrad.runtime import ops_dsp
from tinygrad.runtime.ops_dsp import MockDSPRenderer
from tinygrad.codegen.opt import tc

class _NoCompile:
  def compile_cached(self, src:str) -> bytes: return b""
  def compile(self, src:str) -> bytes: return b""

def dsp_source(t:Tensor, tensor_cores=None) -> str:
  ast = t.schedule_linear().src[-1].src[0]
  ren = MockDSPRenderer(Target(device="DSP"))
  if tensor_cores is not None: ren.tensor_cores = tensor_cores
  ren.compiler = _NoCompile()  # render only: the point is the source text, and a bad source shouldn't cost a compile
  return to_program(ast, ren).src[2].arg

def max_chain(n:int, dtype) -> Tensor:
  x = Tensor.empty(4096, dtype=dtype)
  for _ in range(n): x = x.maximum(Tensor.empty(4096, dtype=dtype))
  return x

class TestDSPRender(unittest.TestCase):
  def test_float_max_chain_source_is_linear(self):
    # a float MAX rendered as a ternary names each operand twice, and single-use ALU results are inlined into their
    # consumer, so a chain of n maxes grew as 2**n in source text (>6 GB host memory for one test_ops case)
    for dt in (dtypes.float32, dtypes.int32):
      with self.subTest(dtype=dt):
        s8, s16 = len(dsp_source(max_chain(8, dt))), len(dsp_source(max_chain(16, dt)))
        self.assertLess(s16, 3 * s8, f"{dt}: source grew {s8} -> {s16} chars for 8 -> 16 chained maxes")

  def test_reduce_upcasts_contiguous_output_axis(self):
    # a per-element dot product (nothing broadcasts): the 128 contiguous outputs become one vector accumulator instead of the
    # reduce being unrolled over scalar gathers
    a, b = Tensor.empty(32, 1024, dtype=dtypes.half), Tensor.empty(32, 1024, dtype=dtypes.half)
    src = dsp_source((a.float() * b.float()).sum(axis=0))
    self.assertIn("float128", src)

  def test_strided_reduce_prefetches_rows_ahead(self):
    # a reduction down the rows of a row-major (512, 1024) half matrix: the load moves 2 KB per reduce step, so it prefetches
    # HVX_PREFETCH_STRIDES (4) rows ahead, not HVX_PREFETCH bytes (one row)
    src = dsp_source(Tensor.empty(512, 1024, dtype=dtypes.half).float().sum(0))
    self.assertIn("+8192)", src)

  def test_int_add_is_one_vector_op(self):
    src = dsp_source(Tensor.empty(4096, dtype=dtypes.int32) + Tensor.empty(4096, dtype=dtypes.int32))
    self.assertRegex(src, r"\*\(\(int128\*\)\(\(data1_4096\+alu0\)\)\)\)\+\(")  # one int128 + int128 (loads rendered inline)
    self.assertNotIn("val0[1]", src)  # no per-lane constructor

class TestDSPQfloat(unittest.TestCase):
  # HVX_ARCH>=v68 turns on qfloat lowering; flip the module flag directly so this runs without a v68 toolchain
  def setUp(self): self.prev, ops_dsp.HVX_QFLOAT = ops_dsp.HVX_QFLOAT, True
  def tearDown(self): ops_dsp.HVX_QFLOAT = self.prev

  def test_product_of_computed_values_is_renormalized(self):
    # qf32 x qf32 without renormalizing is the badly inaccurate case (21% worst-case relative error on hardware):
    # both operands of such a multiply must go through the sf barrier helper
    a, b, c, d = (Tensor.empty(4096) for _ in range(4))
    src = dsp_source((a - b) * (c + d))
    self.assertIn("__hvx_mul_f", src)
    self.assertIn('__asm__("" : "+v"', src)

  def test_plain_arithmetic_is_left_to_llvm(self):
    # a multiply with a loaded operand, and adds of products, are accurate as LLVM lowers them: no helper
    a, b, w = Tensor.empty(4096), Tensor.empty(4096), Tensor.empty(4096)
    src = dsp_source(a * w + b * w)
    self.assertNotIn("__hvx_", src)

  def test_float_max_is_a_vector_op(self):
    # a row max (softmax) stays scalar as the (a<b)?b:a statement expression; on qfloat targets it is HVX's vmax
    src = dsp_source(Tensor.empty(48, 2048, dtype=dtypes.half).max(axis=0))
    self.assertIn("__builtin_elementwise_max(", src)

  def test_off_below_v68(self):
    ops_dsp.HVX_QFLOAT = False
    a, b, c, d = (Tensor.empty(2048) for _ in range(4))
    self.assertNotIn("__hvx_", dsp_source((a - b) * (c + d)))

class TestDSPQfMath(unittest.TestCase):
  # QF_MATH (v68+): EXP2 / RECIPROCAL are not decomposed (tinygrad's decomposition needs vector int<->float conversion, v73+)
  # but rendered as conversion-free HVX helpers, and float division as a * reciprocal(b), so they vectorize
  def setUp(self): self.prev, ops_dsp.QF_MATH = ops_dsp.QF_MATH, True
  def tearDown(self): ops_dsp.QF_MATH = self.prev

  def test_exp_is_a_vector_helper(self):
    src = dsp_source(Tensor.empty(4096).exp())
    self.assertIn("__TG_EXP2(", src)
    self.assertIn("__tg_exp2_v(", src)  # a whole-HVX-register width is used
    self.assertIn("0x4B400000", src)    # magic-number rounding, no float->int conversion

  def test_sqrt_is_a_vector_helper(self):
    src = dsp_source(Tensor.empty(4096).sqrt())
    self.assertIn("__TG_SQRT(", src)
    self.assertIn("0x5F3759DF", src)

  def test_half_exp_is_an_hf_helper(self):
    # a half exp (float EXP2 cast back to half) at a 64-lane multiple: the hf helper, 64 lanes per register
    src = dsp_source(Tensor.empty(48, 1024, dtype=dtypes.half).exp())
    self.assertIn("__tg_exp2_h128(", src)
    self.assertIn("static inline __hvx_h __tg_exp2_h(", src)

  def test_half_division_is_reciprocal(self):
    # a vector hf division (scalarized by clang) renders as a * (1 / b), the reciprocal through the float helper
    src = dsp_source(Tensor.empty(48, 1024, dtype=dtypes.half) / Tensor.empty(48, 1024, dtype=dtypes.half))
    self.assertIn("__TG_RECIP(__builtin_convertvector(", src)
    self.assertNotRegex(src, r"val\d+/")

  def test_helper_macro_takes_a_lane_constructor(self):
    # an operand built from two half-width loads is a lane constructor with commas: the dispatch macro must be variadic
    src = dsp_source(Tensor.empty(4096).exp())
    self.assertIn("#define __TG_EXP2(...)", src)

  def test_division_is_reciprocal(self):
    src = dsp_source(Tensor.empty(4096) / (Tensor.empty(4096) + 1))
    self.assertIn("__TG_RECIP(", src)
    self.assertNotIn(")/(", src)

  def test_off(self):
    ops_dsp.QF_MATH = False
    self.assertNotIn("__TG_EXP2", dsp_source(Tensor.empty(2048).exp()))

class TestDSPVrmpyGemv(unittest.TestCase):
  # a W8A8 decode GEMV: one uint8 activation row times int8 weights, int32 accumulation -- exactly vrmpybusv (u8 x s8 dot4)
  K, N = 576, 1536

  def test_rowmajor_gemv_uses_vrmpybusv(self):
    # M=1 (no range of its own on the activation side) and mixed u8 x s8 dtypes both used to keep the TC from matching
    src = dsp_source(Tensor.empty(1, self.K, dtype=dtypes.uint8).matmul(Tensor.empty(self.K, self.N, dtype=dtypes.int8), dtype=dtypes.int32))
    self.assertIn("__builtin_HEXAGON_V6_vrmpybusv_acc_128B", src)

  def test_packed_gemv_is_one_weight_vector_per_vrmpy(self):
    # over Wp[N/32][K/4][32][4] each vrmpy's weights are one contiguous 128-byte load, and the 32-lane int32 accumulator
    # stays one HVX register (an aligned register array accessed as a vector, not 32 scalar loads/stores per step)
    x = Tensor.empty(1, self.K, dtype=dtypes.uint8).reshape(1, self.K//4, 1, 4)
    wp = Tensor.empty(self.N//32, self.K//4, 32, 4, dtype=dtypes.int8)
    with Context(TC_OPT=1): src = dsp_source((x * wp).sum((1, 3), dtype=dtypes.int32).reshape(self.N))
    self.assertIn("__builtin_HEXAGON_V6_vrmpybusv_acc_128B", src)
    self.assertIn("*((signed_char128*)((data2_", src)
    self.assertNotIn("signed_char32 ", src)
    self.assertIn("__attribute__((aligned(128)))", src)
    self.assertIn("*((int32*)((buf0+0)))", src)
    self.assertNotIn("*(buf0+1)", src)

class TestDSPHmx(unittest.TestCase):
  # the V69 HMX fp16 TensorCore: whole 32x32 tiles in the HMX tile layout. By default the renderer keeps the accumulator
  # inside HMX across the reduce loop (rows packed straight into VTCM with HVX, one load pair per K block, one store per
  # output tile); hmx_acc=False keeps the plain tile op. MOCKDSP builds the scalar reference (-DHMX_REF) of either.
  def src(self, M, K, N, dtype=dtypes.half, acc=True):
    t = Tensor.empty(M, K, dtype=dtype).matmul(Tensor.empty(K, N, dtype=dtype), dtype=dtype)
    old, MockDSPRenderer.hmx_acc = MockDSPRenderer.hmx_acc, acc
    try: return dsp_source(t, tc.hexagon_hmx + tc.hexagon_v65)
    finally: MockDSPRenderer.hmx_acc = old

  def test_half_matmul_keeps_accumulator_in_hmx(self):
    src = self.src(64, 64, 64)
    kernel = src[src.index("__attribute__((noinline)) void"):]
    self.assertIn("activation.hf = mxmem(%0,%1):deep", src)
    self.assertIn("weight.hf = mxmem(%2,%3)", src)
    self.assertIn("mxmem(%0,%1):after.hf = acc", src)
    self.assertIn("#ifdef HMX_REF", src)
    # begin before the reduce loop; inside it only packing into loop-indexed VTCM slots; after it one spanning load pair
    # over all K tiles, then the store
    loop = kernel.index("for (int Ridx0")
    self.assertLess(kernel.index("__hmx_begin();"), loop)
    self.assertIn("= __hmx_ca(", kernel)
    self.assertNotIn("__hmx_mac(_a, _b)", kernel)
    self.assertEqual(kernel.count("__hmx_mac_span("), 1)
    self.assertLess(loop, kernel.index("__hmx_mac_span("))
    self.assertLess(kernel.index("__hmx_mac_span("), kernel.index("__hmx_store()"))
    # no 2 KB tile values, no per-K accumulator round trip
    self.assertNotIn("__WMMA_32_32_32_half_half(", kernel)
    self.assertNotIn("__fp161024", kernel)

  def test_rows_packed_with_hvx(self):
    kernel = self.src(64, 64, 64).split("__attribute__((noinline)) void", 1)[1]
    self.assertEqual(kernel.count("__hmx_pack2("), 32)  # 16 row pairs of A + 16 of B per K block

  def test_interchange_and_paired_b(self):
    # N has more tiles than M: the N-tile loop goes outermost, A stays in loop-indexed VTCM slots, and B tiles n, n+1 are
    # packed together from full 128-byte row lines on even n
    kernel = self.src(128, 576, 256).split("__attribute__((noinline)) void", 1)[1]
    self.assertLess(kernel.index("for (int Lidx2"), kernel.index("for (int Lidx1"))
    self.assertIn("_a = __hmx_ca((Lidx1)*18+(Ridx0)); if ((Lidx2)==0)", kernel)
    self.assertIn("_b = __hmx_cb((Lidx2)%2*18+(Ridx0)); if ((Lidx1)==0 && (Lidx2)%2==0)", kernel)
    # uniform row strides: one batched pack of the whole pair (16 row pairs of each tile), not 16 calls
    self.assertEqual(kernel.count("__hmx_pack2x2_blk("), 1)
    # B pairs stream from DDR: the next pair's 128-byte panel is L2-prefetched while this one packs
    self.assertIn("(Lidx2)+2<8 ? ((Lidx2)==0 ? 1 : 2) : 0", kernel)

  def test_output_pairing(self):
    # adjacent N tiles n, n+1 (paired B) are computed together on even n: two spanning load pairs, both accumulators stored
    # (n+1 at the second store slot), full 64-column output rows written from the pair; odd n does no work
    kernel = self.src(128, 576, 384).split("__attribute__((noinline)) void", 1)[1]
    self.assertIn("__hmx_mac_span(__hmx_ca((Lidx1)*18), __hmx_cb((Lidx2)%2*18), 18);", kernel)
    self.assertIn("__hmx_mac_span(__hmx_ca((Lidx1)*18), __hmx_cb((Lidx2)%2*18+18), 18);", kernel)  # tile n+1
    self.assertIn("__hmx_store2()", kernel)
    self.assertIn("__hmx_outp(", kernel)

  def test_large_vtcm_pool(self):
    # HMX_VTCM_KB > 256: A and B share one pool of loop-indexed slots, B after A's from a 32-slot boundary, K panels at a
    # window-safe stride (18 K tiles -> 32 slots), so B (inner-indexed after the interchange) no longer goes through the
    # tag cache
    old = ops_dsp.HMX_VTCM_KB, ops_dsp._HMX_CA, ops_dsp._HMX_CB
    ops_dsp.HMX_VTCM_KB, ops_dsp._HMX_CA, ops_dsp._HMX_CB = 4096, 1792, 128
    try: kernel = self.src(384, 576, 64).split("__attribute__((noinline)) void", 1)[1]
    finally: ops_dsp.HMX_VTCM_KB, ops_dsp._HMX_CA, ops_dsp._HMX_CB = old
    self.assertIn("_b = __hmx_ca(32+(Lidx2)*32+(Ridx0))", kernel)  # after A (18 slots, rounded to 32), 32 per panel
    self.assertNotIn("__hmx_lookup(", kernel)
    self.assertEqual(kernel.count("__hmx_mac_span("), 1)

  def test_epilogue_single_m_tile(self):
    # an epilogue (bias + residual) on an HMX output with a single M tile: the accumulator array follows the output's memory
    # order (expanded axes sorted by store stride, not toposort -- that came out column-major, every element a lane gather),
    # its vectors are one vdealh per register, rows are read back as 32-lane loads and the bias as a scalar reload + splat
    a, b = Tensor.empty(32, 224, dtype=dtypes.half), Tensor.empty(224, 512, dtype=dtypes.half)
    bias, res = Tensor.empty(32, 1, dtype=dtypes.half), Tensor.empty(32, 512, dtype=dtypes.half)
    src = dsp_source(a.matmul(b, dtype=dtypes.half) + bias + res, tc.hexagon_hmx + tc.hexagon_v65)
    kernel = src[src.index("__attribute__((noinline)) void"):]
    self.assertEqual(kernel.count("__hmx_deal2("), 8)
    self.assertNotIn("_p[1]", kernel)  # no per-lane accumulator gather
    self.assertIn("(*((__fp1632*)((__fp16*)(buf0+", kernel)
    self.assertNotIn("__builtin_shufflevector(val", kernel)
    self.assertRegex(kernel, r"\(\(__fp1632\)\(\(\(__fp16\*\)\(data\d+_32\+?[^)]*\)\)\[\d+\]\)\)")

  def test_plain_tile_op(self):
    src = self.src(64, 96, 64, acc=False)  # a shape not rendered above (to_program caches by AST)
    self.assertIn("__WMMA_32_32_32_half_half(", src)
    self.assertIn("*(__fp161024*)(v + 1024) = a;", src)
    self.assertNotIn("__builtin_memcpy(v", src)

  def test_single_k_tile_matmul(self):
    # K = 32 (an attention score q . k^T with head_dim 32): no reduce loop, so the tile op is begun right before it and stored
    # right after -- an enclosing output-tile loop is not the reduction -- and the output rows go out with the row-pair store
    kernel = self.src(64, 32, 96).split("__attribute__((noinline)) void", 1)[1]
    self.assertNotIn("for (int Ridx", kernel)
    self.assertEqual(kernel.count("__hmx_begin();"), 1)
    self.assertLess(kernel.index("__hmx_begin();"), kernel.index("__hmx_store()"))
    self.assertIn("__hmx_out2(", kernel)
    self.assertNotIn("*(__hmx_h128*)", kernel)  # no 128-lane store over 32-element rows

  def test_a_panel_prefetch_is_rows(self):
    # A's K panel is 32 rows of K columns: __hmx_prefetch_rows; the 32*kt-row panel prefetch is B's shape only (on A it read
    # far past the operand and faulted the PD on the phone)
    kernel = self.src(64, 2048, 64).split("__attribute__((noinline)) void", 1)[1]
    for call in kernel.split("__hmx_prefetch_panel(")[1:]:
      self.assertNotIn("data1", call.split(")")[0])  # data1 = A

  def test_float_matmul_is_not_hmx(self):
    self.assertNotIn("mxmem", self.src(64, 64, 64, dtypes.float32))

class TestDSPHmxI8(unittest.TestCase):
  # the V69 HMX int8 TensorCore (uint8 activations x int8 weights -> int32): ":cm" activation tiles (64 rows x 32 bytes), weight
  # tiles 32 x 32 with four K rows per 32-bit column group, the exact int32 accumulator read back as four byte planes
  def src(self, M, K, N):
    t = Tensor.empty(M, K, dtype=dtypes.uint8).matmul(Tensor.empty(K, N, dtype=dtypes.int8), dtype=dtypes.int32)
    return dsp_source(t, tc.hexagon_hmx_i8 + tc.hexagon_hmx + tc.hexagon_v65)

  def test_i8_matmul_is_hmx_cm(self):
    src = self.src(128, 256, 256)
    kernel = src[src.index("__attribute__((noinline)) void"):]
    self.assertIn("activation.ub = mxmem(%0,%1):cm", src)
    self.assertIn(":deep", src)
    self.assertIn("#ifdef HMX_REF", src)
    self.assertEqual(kernel.count("__hmx_i8_begin();"), 1)
    self.assertNotIn("__WMMA_", kernel)
    # A: packed once per M tile into loop-indexed VTCM slots
    self.assertEqual(kernel.count("__hmx_i8_pack_a4("), 16)
    self.assertIn("__hmx_ca((Lidx1)*8+(Ridx0))", kernel)

  def test_i8_quad_b_deep_and_planes(self):
    # four adjacent N tiles share each 128-byte weight row line: packed together on n%4==0, one :deep weight load pair per K
    # block on even n drives both accumulators; n+1's byte planes go to spare A slots and are summed into its output on odd n
    kernel = self.src(128, 256, 256).split("__attribute__((noinline)) void", 1)[1]
    self.assertEqual(kernel.count("__hmx_i8_pack_b4x4("), 8)
    self.assertIn("if ((Lidx2)%2==0) __hmx_i8_mac2(_a, __hmx_cb(((Lidx2)%4/2)*8+(Ridx0)));", kernel)
    self.assertIn("__hmx_i8_store2((unsigned char*)__hmx_ca(16+4*(Lidx1)))", kernel)
    self.assertEqual(kernel.count("__hmx_i8_addq("), 16)
    self.assertIn("((Lidx2))+4<8 ? (((Lidx2))==0 ? 1 : 2) : 0", kernel)

  def test_i8_needs_signed_weights(self):
    # uint8 x uint8 has no HMX :cm form here: not the int8 TensorCore
    t = Tensor.empty(64, 64, dtype=dtypes.uint8).matmul(Tensor.empty(64, 64, dtype=dtypes.uint8), dtype=dtypes.int32)
    self.assertNotIn("__hmx_i8_", dsp_source(t, tc.hexagon_hmx_i8 + tc.hexagon_hmx + tc.hexagon_v65))

if __name__ == '__main__':
  unittest.main()
