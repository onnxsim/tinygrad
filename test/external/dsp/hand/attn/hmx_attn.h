/* the HMX side of the hand attention kernels (nn/onnx.py _hmx_attn_dsp); needs attn_softmax.h before it */
extern unsigned char* __hmx_vtcm;
static inline void __ha_hmx_init(unsigned char* vt) {
  for (int j = 0; j < 64; j++) ((unsigned*)vt)[j] = 0;
  __asm__ volatile("bias = mxmem(%0)" :: "r"(vt) : "memory");
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(vt + 2048), "r"(0) : "memory");  /* clear the accumulator */
}
/* head h's k^T tiles (weight: rows d, columns keys 32j..; row pair r = rows 2r, 2r+1 at column 32j interleaved) from k^T rows of
   N keys (unpadded: the last tile's columns from N on are whatever follows, and the softmax masks those keys out) and v tiles
   (weight: rows keys, columns d; keys 2r, 2r+1 are one aligned 128-byte vector, shuffled; v zero-padded to Np rows) */
static inline void __ha_pack_kv(__fp16* ktt, __fp16* vtt, const __fp16* kh, const __fp16* vh, int N, int Np) {
  for (int j = 0; j < Np / 32; j++)
    for (int r = 0; r < 16; r++) {
      int n = N - 32 * j < 32 ? N - 32 * j : 32;
      ((HVX_Vector*)(ktt + 1024 * j))[r] =
          Q6_V_lo_W(Q6_W_vshuff_VVR(__ha_ldu(kh + (2 * r + 1) * N + 32 * j, n), __ha_ldu(kh + 2 * r * N + 32 * j, n), -2));
      ((HVX_Vector*)(vtt + 1024 * j))[r] = Q6_Vh_vshuff_Vh(((const HVX_Vector*)(vh + 1024 * j))[r]);
    }
}
/* P @ v for one 32-query block (P: nt activation tiles at st, v: at vtt) -> out rows m0.. (< M) of a (M, 32) float matrix */
static inline void __ha_pv(float* out, __fp16* st, __fp16* vtt, __fp16* ot, int nt, int m0, int M) {
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00), k1_128 = __ha_ssplat(1.f / 128.f);
  for (int k0 = 0; k0 < nt; k0 += 32) {
    int n = nt - k0 < 32 ? nt - k0 : 32;
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" :: "r"(st + 1024 * k0), "r"(n * 2048 - 1),
                     "r"(vtt + 1024 * k0), "r"(n * 2048 - 1) : "memory");
  }
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(ot), "r"(0) : "memory");
  for (int r = 0; r < 16 && m0 + 2 * r < M; r++) {
    HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(((HVX_Vector*)ot)[r], one);  /* lo: row 2r, hi: row 2r+1 */
    float* o = out + (long)(m0 + 2 * r) * 32;
    *(HVX_UVector*)o = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), k1_128));
    if (m0 + 2 * r + 1 < M) *(HVX_UVector*)(o + 32) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), k1_128));
  }
}
/* S = q k^T for one 32-query block (q rows m0.. of a (Mp, 32) matrix) -> nt tiles at st */
static inline void __ha_s(__fp16* st, __fp16* qt, __fp16* ktt, const __fp16* qh, int nt, int m0) {
  for (int r = 0; r < 16; r++) ((HVX_Vector*)qt)[r] = Q6_Vh_vshuff_Vh(((const HVX_Vector*)(qh + 32 * m0))[r]);
  for (int j = 0; j < nt; j++) {
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" :: "r"(qt), "r"(2047), "r"(ktt + 1024 * j), "r"(2047) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(st + 1024 * j), "r"(0) : "memory");
  }
}
/* fused attention on the HMX, one call: out (H, M, 32) float = softmax(q k^T) v per head, from q (H, Mp, 32), kt (H, 32, N), v
   (H, Np, 32) half, q and v zero-padded to whole 32-row tiles (Mp, Np = N rounded up to 32, Np <= 1024), the keys from N on
   masked out. Per head the k^T and v tiles are packed into VTCM once; per 32-query block S = q k^T goes to VTCM tiles (HMX), the softmax
   runs on them in place (HVX), and those tiles are P @ v's activation tiles as they are (one :deep span). Nothing between the two
   products leaves VTCM. VTCM (one 256 KB window): bias table @0, out tile @2 KB, q tile @4 KB, k^T @8 KB, v @72 KB, S/P @136 KB */
static void __hmx_attn(float* out, const __fp16* q, const __fp16* kt, const __fp16* v, int H, int M, int Mp, int N, int Np) {
  unsigned char* vt = __hmx_vtcm;
  __fp16 *ot = (__fp16*)(vt + 2048), *qt = (__fp16*)(vt + 4096), *ktt = (__fp16*)(vt + 8192), *vtt = (__fp16*)(vt + 73728),
         *st = (__fp16*)(vt + 139264);
  const int nt = Np / 32;
  __ha_hmx_init(vt);
  for (int h = 0; h < H; h++) {
    __ha_pack_kv(ktt, vtt, kt + (long)h * 32 * N, v + (long)h * Np * 32, N, Np);
    for (int m0 = 0; m0 < M; m0 += 32) {
      __ha_s(st, qt, ktt, q + (long)h * Mp * 32, nt, m0);
      __ha_softmax_rows(st, nt, N - 32 * (nt - 1), 0, 16);
      __ha_pv(out + (long)h * M * 32, st, vtt, ot, nt, m0, M);
    }
  }
}
/* the split form, per head h, for the softmax on the HVX thread pool (__attn_sm_vtcm) between two HMX calls: S of every query
   block into VTCM (block b at 256 KB + 64 KB * b: no :deep span crosses a 256 KB window), tok[0] = the VTCM address. `dep`: the
   previous head's output, read by nothing, so this call runs after it (the VTCM is reused) */
static void __hmx_attn_s(int* tok, const __fp16* q, const __fp16* kt, const __fp16* v, const float* dep, int h, int M, int Mp, int N, int Np) {
  unsigned char* vt = __hmx_vtcm;
  (void)dep;
  __ha_hmx_init(vt);
  __ha_pack_kv((__fp16*)(vt + 8192), (__fp16*)(vt + 73728), kt + (long)h * 32 * N, v + (long)h * Np * 32, N, Np);
  for (int m0 = 0; m0 < M; m0 += 32)
    __ha_s((__fp16*)(vt + 262144 + 65536 * (m0 / 32)), (__fp16*)(vt + 4096), (__fp16*)(vt + 8192), q + (long)h * Mp * 32, Np / 32, m0);
  tok[0] = (int)vt;
}
/* ...and P @ v of every block -> out (M, 32) float, this head's */
static void __hmx_attn_pv(float* out, const int* tok, int M, int Np) {
  unsigned char* vt = (unsigned char*)tok[0];
  __ha_hmx_init(vt);
  for (int m0 = 0; m0 < M; m0 += 32)
    __ha_pv(out, (__fp16*)(vt + 262144 + 65536 * (m0 / 32)), (__fp16*)(vt + 73728), (__fp16*)(vt + 2048), Np / 32, m0, M);
}
