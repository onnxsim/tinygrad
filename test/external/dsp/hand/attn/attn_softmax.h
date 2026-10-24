/* softmax rows of attention scores held in VTCM in the HMX tile layout (no HMX here: also run from the HVX thread pool) */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
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
/* n (1..64) halfwords from p, any alignment, in the low lanes (the rest: whatever follows in memory): aligned loads of only
   the 128-byte blocks holding those n, joined by valign -- unlike an unaligned vector load it never touches a block with
   nothing of them (the end of a buffer) */
static inline HVX_Vector __ha_ldu(const __fp16* p, int n) {
  unsigned a = (unsigned)p & ~127u, e = ((unsigned)p + 2 * n - 1) & ~127u;
  HVX_Vector v0 = *(const HVX_Vector*)a, v1 = e != a ? *(const HVX_Vector*)(a + 128) : v0;
  return Q6_V_valign_VVR(v1, v0, (int)(unsigned)p);
}
/* softmax over the nt tiles (32 keys each) of one 32-query block st, row pairs [r0, r1), in place: e = 2^((x - max) log2(e) + 7) in hf
   (2^n by the 1536.0 magic number, 2^-g by a degree-4 polynomial kept in qf16), lanes below 2^-13 zeroed (the keys from `valid`
   in the last tile among them), HA_W tiles in lockstep; then P * 128 = e * (128 / row sum) (sums in qf32): it stays normal in
   hf and sums to 128 per row, so P @ v's fp16 output can't overflow (the 1 / 128 is the caller's). Tile layout: a vector is one
   row pair, rows 2r / 2r+1 on the even / odd lanes */
static void __ha_softmax_rows(__fp16* st, int nt, int valid, int r0, int r1) {
  const HVX_Vector lo = __ha_hsplat(-13.f), l2e = __ha_hsplat(1.44269504f), seven = __ha_hsplat(7.f), magic = Q6_Vh_vsplat_R(0x6600),
                   k128 = __ha_ssplat(128.f), c4 = __ha_hsplat(0.00961813f), c3 = __ha_hsplat(-0.0555041f), c2 = __ha_hsplat(0.2402265f),
                   c1 = __ha_hsplat(-0.6931472f), c0 = __ha_hsplat(1.f), one = Q6_Vh_vsplat_R(0x3C00), zero = Q6_V_vzero(),
                   neg = Q6_Vh_vsplat_R((short)0xF753);
  const HVX_VectorPred keymask = Q6_Q_vsetq_R(4 * valid);  /* the last key tile's valid columns: lanes < 2 * valid */
  for (int r = r0; r < r1; r++) {
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
}
/* the pool's share of a head's softmax (nn/onnx.py _hmx_attn_dsp, split form): nb query blocks, block b's nt tiles at VTCM + 256 KB +
   64 KB * b (the address from tok_in, which the HMX call that computed S wrote there; no HMX reference may appear in this file: the graph
   emitter runs a kernel mentioning one under the HMX lock, off the pool), this thread's slice of the nb * 16 row pairs; tok_out = tok_in */
static void __attn_sm_vtcm(int* tok_out, const int* tok_in, int t, int nthr, int nb, int nt, int valid) {
  unsigned char* vt = (unsigned char*)tok_in[0];
  int total = nb * 16, u0 = total * t / nthr, u1 = total * (t + 1) / nthr;
  for (int u = u0; u < u1;) {
    int b = u / 16, r = u % 16, e = (b + 1) * 16 < u1 ? 16 : u1 - b * 16;
    __ha_softmax_rows((__fp16*)(vt + 262144 + 65536 * b), nt, valid, r, e);
    u = b * 16 + e;
  }
  if (t == 0) tok_out[0] = tok_in[0];
}
