#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
extern unsigned char* __hmx_vtcm;
#ifndef HA_W
#define HA_W 2
#endif
#define __HA_H(q) Q6_Vhf_equals_Vqf16(q)
static inline HVX_Vector __ha_hsplat(float f) { union { __fp16 h; short s; } u; u.h = (__fp16)f; return Q6_Vh_vsplat_R(u.s); }
static inline HVX_Vector __ha_ssplat(float f) { union { float f; int i; } u; u.f = f; return Q6_V_vsplat_R(u.i); }
/* 1 / x (sf, x >= 1): the bit-trick estimate + 3 Newton steps in qf32 */
static inline HVX_Vector __ha_recip(HVX_Vector x) {
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C7), x), two = __ha_ssplat(2.f);
  for (int k = 0; k < 3; k++) y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, y))))));
  return y;
}
/* 64 halfwords from p (64-byte aligned, inside one 128-byte block with its row) in the low half: an aligned load + vror never
   reads past the row's 128-byte block */
static inline HVX_Vector __ha_ld64(const __fp16* p) {
  return Q6_V_vror_VR(*(const HVX_Vector*)((unsigned)p & ~127u), (int)(unsigned)p);
}
/* fused attention on the HMX, one head at a time: out (H, M, 32) float = softmax(q k^T) v per head, from q (H, Mp, 32),
   kt (H, 32, Np), v (H, Np, 32) half, zero-padded to whole 32-row / 32-column tiles (Mp, Np multiples of 32, Np <= 1024), the
   keys from N on masked out. Per head the k^T and v tiles are packed into VTCM once; per 32-query block S = q k^T goes to VTCM
   tiles (HMX), the softmax runs on them in place (HVX, tile layout: a vector is one row pair, rows 2r / 2r+1 on the even / odd
   lanes), and those tiles are P @ v's activation tiles as they are (one :deep span). Nothing between the two products leaves
   VTCM. VTCM (one 256 KB window): bias table @0, out tile @2 KB, q tile @4 KB, k^T tiles @8 KB, v @72 KB, S/P @136 KB */
static void __hmx_attn(float* out, const __fp16* q, const __fp16* kt, const __fp16* v, int H, int M, int Mp, int N, int Np) {
  unsigned char* vt = __hmx_vtcm;
  __fp16 *ot = (__fp16*)(vt + 2048), *qt = (__fp16*)(vt + 4096), *ktt = (__fp16*)(vt + 8192), *vtt = (__fp16*)(vt + 73728),
         *st = (__fp16*)(vt + 139264);
  const int nt = Np / 32, valid = N - 32 * (nt - 1);
  for (int j = 0; j < 64; j++) ((unsigned*)vt)[j] = 0;
  __asm__ volatile("bias = mxmem(%0)" :: "r"(vt) : "memory");
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(ot), "r"(0) : "memory");  /* clear the accumulator */
  const HVX_Vector lo = __ha_hsplat(-13.f), l2e = __ha_hsplat(1.44269504f),
                   seven = __ha_hsplat(7.f), magic = Q6_Vh_vsplat_R(0x6600), k128 = __ha_ssplat(128.f), k1_128 = __ha_ssplat(1.f / 128.f), c4 = __ha_hsplat(0.00961813f), c3 = __ha_hsplat(-0.0555041f),
                   c2 = __ha_hsplat(0.2402265f), c1 = __ha_hsplat(-0.6931472f), c0 = __ha_hsplat(1.f), one = Q6_Vh_vsplat_R(0x3C00),
                   zero = Q6_V_vzero(), neg = Q6_Vh_vsplat_R((short)0xF753);
  const HVX_VectorPred keymask = Q6_Q_vsetq_R(4 * valid);  /* the last key tile's valid columns: lanes < 2 * valid */
  for (int h = 0; h < H; h++) {
    const __fp16 *qh = q + (long)h * Mp * 32, *kh = kt + (long)h * 32 * Np, *vh = v + (long)h * Np * 32;
    /* k^T tile j (weight: rows d, columns keys 32j..): row pair r = rows 2r, 2r+1 at column 32j interleaved */
    for (int j = 0; j < nt; j++)
      for (int r = 0; r < 16; r++)
        ((HVX_Vector*)(ktt + 1024 * j))[r] =
            Q6_V_lo_W(Q6_W_vshuff_VVR(__ha_ld64(kh + (2 * r + 1) * Np + 32 * j), __ha_ld64(kh + 2 * r * Np + 32 * j), -2));
    /* v tile j (weight: rows keys 32j.., columns d): keys 2r, 2r+1 are one aligned 128-byte vector, shuffled */
    for (int j = 0; j < nt; j++)
      for (int r = 0; r < 16; r++) ((HVX_Vector*)(vtt + 1024 * j))[r] = Q6_Vh_vshuff_Vh(((const HVX_Vector*)(vh + 1024 * j))[r]);
    for (int m0 = 0; m0 < M; m0 += 32) {
      for (int r = 0; r < 16; r++) ((HVX_Vector*)qt)[r] = Q6_Vh_vshuff_Vh(((const HVX_Vector*)(qh + 32 * m0))[r]);
      for (int j = 0; j < nt; j++) {
        __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" :: "r"(qt), "r"(2047), "r"(ktt + 1024 * j), "r"(2047) : "memory");
        __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(st + 1024 * j), "r"(0) : "memory");
      }
      /* softmax per row pair over the nt tiles, in place: e = 2^((x - max) log2(e) + 7) in hf (2^n by the 1536.0 magic
         number, 2^-g by a degree-4 polynomial kept in qf16), lanes below 2^-13 zeroed (the masked keys among them), HA_W tiles in
         lockstep; the row sums in qf32, reduced and inverted in vectors */
      for (int r = 0; r < 16; r++) {
        HVX_Vector* S = (HVX_Vector*)st + r;  /* tile j's row pair r: S[16 * j] */
        if (valid < 32) S[16 * (nt - 1)] = Q6_V_vmux_QVV(keymask, S[16 * (nt - 1)], neg);
        HVX_Vector m = S[0];
        for (int j = 1; j < nt; j++) m = Q6_Vhf_vmax_VhfVhf(m, S[16 * j]);
        for (int sh = 64; sh >= 4; sh >>= 1) m = Q6_Vhf_vmax_VhfVhf(m, Q6_V_vror_VR(m, sh));  /* same-parity lanes: each row */
        HVX_Vector a0 = zero, a1 = zero;
        for (int j0 = 0; j0 < nt; j0 += HA_W) {
          HVX_Vector x[HA_W], n[HA_W], g[HA_W], p[HA_W];
          HVX_VectorPred keep[HA_W];
          int c = nt - j0 < HA_W ? nt - j0 : HA_W;
          for (int i = 0; i < HA_W; i++) {
            HVX_Vector sv = i < c ? S[16 * (j0 + i)] : m;
            x[i] = __HA_H(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vsub_VhfVhf(sv, m), l2e), seven));  /* finite: x - max >= -30000 */
            keep[i] = Q6_Q_vcmp_gt_VhfVhf(x[i], lo);
            x[i] = Q6_Vhf_vmax_VhfVhf(x[i], lo);
          }
          for (int i = 0; i < HA_W; i++) {
            HVX_Vector rr = __HA_H(Q6_Vqf16_vadd_VhfVhf(x[i], magic));
            n[i] = Q6_Vh_vsub_VhVh(rr, magic);
            g[i] = Q6_Vqf16_vsub_Vqf16Vhf(Q6_Vqf16_vsub_VhfVhf(rr, magic), x[i]);  /* qf16 */
          }
          for (int i = 0; i < HA_W; i++) p[i] = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(g[i], c4), c3);
          for (int i = 0; i < HA_W; i++) p[i] = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(p[i], g[i]), c2);
          for (int i = 0; i < HA_W; i++) p[i] = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(p[i], g[i]), c1);
          for (int i = 0; i < HA_W; i++) p[i] = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(p[i], g[i]), c0);
          for (int i = 0; i < c; i++) {
            HVX_Vector e = Q6_V_vmux_QVV(keep[i], Q6_Vh_vadd_VhVh(__HA_H(p[i]), Q6_Vh_vasl_VhR(n[i], 10)), zero);
            S[16 * (j0 + i)] = e;
            HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(e, one);  /* lo: row 2r's 32 lanes, hi: row 2r+1's */
            a0 = Q6_Vqf32_vadd_Vqf32Vqf32(a0, Q6_V_lo_W(w)), a1 = Q6_Vqf32_vadd_Vqf32Vqf32(a1, Q6_V_hi_W(w));
          }
        }
        for (int sh = 64; sh >= 4; sh >>= 1)
          a0 = Q6_Vqf32_vadd_Vqf32Vqf32(a0, Q6_V_vror_VR(a0, sh)), a1 = Q6_Vqf32_vadd_Vqf32Vqf32(a1, Q6_V_vror_VR(a1, sh));
        /* e <= 2^7 and sum >= 2^7: P * 128 = e * (128 / sum) stays normal in hf and sums to 128 per row, so P @ v's fp16 output
           can't overflow; the 1 / 128 is the output conversion's. The per-row factors interleaved by the qf32 -> hf narrowing */
        HVX_Vector f0 = Q6_Vqf32_vmpy_VsfVsf(__ha_recip(Q6_Vsf_equals_Vqf32(a0)), k128), f1 = Q6_Vqf32_vmpy_VsfVsf(__ha_recip(Q6_Vsf_equals_Vqf32(a1)), k128);
        HVX_Vector f = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(f1, f0));
        for (int j = 0; j < nt; j++) S[16 * j] = __HA_H(Q6_Vqf16_vmpy_VhfVhf(S[16 * j], f));
      }
      for (int k0 = 0; k0 < nt; k0 += 32) {
        int n = nt - k0 < 32 ? nt - k0 : 32;
        __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" :: "r"(st + 1024 * k0), "r"(n * 2048 - 1),
                         "r"(vtt + 1024 * k0), "r"(n * 2048 - 1) : "memory");
      }
      __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(ot), "r"(0) : "memory");
      for (int r = 0; r < 16 && m0 + 2 * r < M; r++) {
        HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(((HVX_Vector*)ot)[r], one);
        float* o = out + ((long)h * M + m0 + 2 * r) * 32;
        *(HVX_UVector*)o = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), k1_128));
        if (m0 + 2 * r + 1 < M)
          *(HVX_UVector*)(o + 32) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), k1_128));
      }
    }
  }
}
