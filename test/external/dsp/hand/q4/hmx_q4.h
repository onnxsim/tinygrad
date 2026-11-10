/* Group-wise int4 weight GEMM on the HMX: C (M, N) fp16 = A (M, K) fp16 @ W, W = int4 codes (the PLaMo 3 layout: K rows in
 * groups of 128 with one fp16 scale per (group, column)). Self-contained (the HMX helpers are copies under __q4_ names), so it
 * can be concatenated into a generated kernel's prelude (nn/onnx.py _matmul_q4) or compiled on its own (sim_q4.c).
 *
 * Operands: A fp16 row-major (M, K), M <= 32; codes tile-major uint8 (N/32, K/32, 512): tile (nb, kb) holds 512 bytes, byte j =
 * code(halfword 128*(j/128) + (b%2 ? 64 : 0) + b/2) in the low nibble and code(+512) in the high nibble, b = j%128, codes
 * q + 8 in 0..15 (the layout of hmx_gemm_client.c mode 5); sc: fp16 (N/32, K/128, 64): lane h of (nb, g) = scale(group g,
 * column 32 nb + h/2), so one 128-byte vector per (nb, g). C fp16 row-major (M, N).
 *
 * Per 32-column block, for each group of 4 K tiles: the 4 weight tiles are expanded from the codes into VTCM (vlut16, whose
 * table entry e is read from halfword 2e), one HMX MAC over the group's K tiles gives an fp16 partial tile, and the partial is
 * multiplied by the group's scales and added into an fp32 (qf32) accumulator. One fp16 rounding of the sum at the end.
 * The scale is per group, so it cannot be applied once after the whole K reduction (the per-column variant measured 9363
 * elements past fp16 rounding at 2560 x 2560, this one none on hexagon-sim).
 * VTCM (from __hmx_vtcm): table @0, partial tile @2 KB, weight tiles @4 KB (8 KB), A tiles @256 KB (K/32 tiles, 2 KB each).
 * A group's 8 KB span never crosses a 256 KB window, so K up to 10240 works with HMX_VTCM_KB >= 1024 (PLaMo's down projection).
 * The caller sets up the HMX power and context. */
#ifndef HMX_Q4_H
#define HMX_Q4_H
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#ifdef HMX_REF
/* the qemu build only has to compile this (the kernel runs on hexagon-sim or the phone): no VTCM there */
static inline unsigned char* __q4_vtcm(void) { return (unsigned char*)0; }
#else
extern unsigned char* __hmx_vtcm;
static inline unsigned char* __q4_vtcm(void) { return __hmx_vtcm; }
#endif
#define __Q4_TILE_BYTES 2048
#define __Q4_IDX(i, j) (64 * ((i) / 2) + 2 * (j) + ((i) % 2))
typedef HVX_Vector __attribute__((aligned(1))) __q4_uvec;
/* DDR -> L2 prefetch of n bytes (l2fetch boxes of 128-byte rows) */
static inline void __q4_prefetch(const void* p, size_t n) {
  const unsigned char* b = (const unsigned char*)p;
  for (size_t off = 0; off < n; off += 255 * 128) {
    size_t rows = (n - off + 127) / 128;
    if (rows > 255) rows = 255;
    unsigned rt = (128u << 16) | (128u << 8) | (unsigned)rows;
    __asm__ volatile("l2fetch(%0, %1)" ::"r"(b + off), "r"(rt));
  }
}
#ifdef HMX_REF
/* the qemu build (MOCKDSP's -DHMX_REF) has no HMX: it only has to compile; the kernel runs on hexagon-sim or the phone */
static inline void __q4_set_table(const void* t) { (void)t; }
static inline void __q4_mac(const void* a, const void* w, int ktiles) { (void)a; (void)w; (void)ktiles; }
static inline void __q4_store(void* c) { (void)c; }
#else
static inline void __q4_set_table(const void* t) { __asm__ volatile("bias = mxmem(%0)" ::"r"(t) : "memory"); }
static inline void __q4_mac(const void* a, const void* w, int ktiles) {
  /* ktiles <= 32 (one fp16 load pair) */
  int lim = ktiles * __Q4_TILE_BYTES - 1;
  __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" ::"r"(a), "r"(lim), "r"(w), "r"(lim)
                   : "memory");
}
static inline void __q4_store(void* c) { __asm__ volatile("mxmem(%0,%1):after.hf = acc" ::"r"(c), "r"(0) : "memory"); }
#endif
/* A (M, K) fp16 row-major -> the K/32 activation tiles at out (tile j at out + 1024 j halfwords), rows >= M zero.
 * The next row pair is prefetched while this one packs (the rows are read from DDR once, 2 KB per tile row). */
static inline void __q4_pack_a(const uint16_t* A, int M, int K, uint16_t* out) {
  int kt = K / 32;
  HVX_Vector z = Q6_V_vzero();
  for (int p = 0; p < 16; p++) {
    if (2 * p + 2 < M) __q4_prefetch(A + (size_t)(2 * p + 2) * K, (size_t)K * 2);
    if (2 * p + 3 < M) __q4_prefetch(A + (size_t)(2 * p + 3) * K, (size_t)K * 2);
    int r0 = 2 * p, r1 = r0 + 1;
    const uint16_t* a0 = r0 < M ? A + (size_t)r0 * K : NULL;
    const uint16_t* a1 = r1 < M ? A + (size_t)r1 * K : NULL;
    for (int j = 0; j < kt; j += 2) {
      HVX_Vector v0 = a0 ? *(const __q4_uvec*)(a0 + 32 * j) : z, v1 = a1 ? *(const __q4_uvec*)(a1 + 32 * j) : z;
      HVX_VectorPair s = Q6_W_vshuff_VVR(v1, v0, -2);
      *(HVX_Vector*)(out + (size_t)j * 1024 + 64 * p) = Q6_V_lo_W(s);
      if (j + 1 < kt) *(HVX_Vector*)(out + (size_t)(j + 1) * 1024 + 64 * p) = Q6_V_hi_W(s);
    }
  }
}
static const unsigned short __q4_lut_bits[16] = {0xC800, 0xC700, 0xC600, 0xC500, 0xC400, 0xC200, 0xC000, 0xBC00,
                                                 0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
/* 512 code bytes -> one 32x32 fp16 weight tile (q values, -8..7) in HMX tile layout */
static inline void __q4_expand_tile(__fp16* dst, const unsigned char* src, HVX_Vector lut) {
  HVX_Vector m4 = Q6_Vb_vsplat_R(0x0F);
  for (int m = 0; m < 4; m++) {
    HVX_Vector v = *(const __q4_uvec*)(src + 128 * m);
    HVX_VectorPair pl = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v, m4), lut, 0);
    HVX_VectorPair ph = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 4), m4), lut, 0);
    __fp16* lo = dst + 128 * m;
    __fp16* hi = dst + 512 + 128 * m;
    *(HVX_Vector*)lo = Q6_V_lo_W(pl);
    *(HVX_Vector*)(lo + 64) = Q6_V_hi_W(pl);
    *(HVX_Vector*)hi = Q6_V_lo_W(ph);
    *(HVX_Vector*)(hi + 64) = Q6_V_hi_W(ph);
  }
}
/* 4 weight tiles of one group (2 KB of codes) in one pass: all 16 code vectors are loaded first, then the nibble splits
 * and vlut16s, so the independent chains overlap (the one-tile loop of __q4_expand_tile was ~2x slower on V69). */
static inline void __q4_expand4(__fp16* W, const unsigned char* src, HVX_Vector lut) {
  const HVX_Vector m4 = Q6_Vb_vsplat_R(0x0F);
  HVX_Vector v0 = *(const __q4_uvec*)(src + 0),    v1 = *(const __q4_uvec*)(src + 128),  v2 = *(const __q4_uvec*)(src + 256),  v3 = *(const __q4_uvec*)(src + 384);
  HVX_Vector v4 = *(const __q4_uvec*)(src + 512),  v5 = *(const __q4_uvec*)(src + 640),  v6 = *(const __q4_uvec*)(src + 768),  v7 = *(const __q4_uvec*)(src + 896);
  HVX_Vector v8 = *(const __q4_uvec*)(src + 1024), v9 = *(const __q4_uvec*)(src + 1152), va = *(const __q4_uvec*)(src + 1280), vb = *(const __q4_uvec*)(src + 1408);
  HVX_Vector vc = *(const __q4_uvec*)(src + 1536), vd = *(const __q4_uvec*)(src + 1664), ve = *(const __q4_uvec*)(src + 1792), vf = *(const __q4_uvec*)(src + 1920);
  HVX_VectorPair p0 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v0, m4), lut, 0), q0 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v0, 4), m4), lut, 0);
  HVX_VectorPair p1 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v1, m4), lut, 0), q1 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v1, 4), m4), lut, 0);
  HVX_VectorPair p2 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v2, m4), lut, 0), q2 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v2, 4), m4), lut, 0);
  HVX_VectorPair p3 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v3, m4), lut, 0), q3 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v3, 4), m4), lut, 0);
  HVX_VectorPair p4 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v4, m4), lut, 0), q4 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v4, 4), m4), lut, 0);
  HVX_VectorPair p5 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v5, m4), lut, 0), q5 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v5, 4), m4), lut, 0);
  HVX_VectorPair p6 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v6, m4), lut, 0), q6 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v6, 4), m4), lut, 0);
  HVX_VectorPair p7 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v7, m4), lut, 0), q7 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v7, 4), m4), lut, 0);
  HVX_VectorPair p8 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v8, m4), lut, 0), q8 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v8, 4), m4), lut, 0);
  HVX_VectorPair p9 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(v9, m4), lut, 0), q9 = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v9, 4), m4), lut, 0);
  HVX_VectorPair pa = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(va, m4), lut, 0), qa = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(va, 4), m4), lut, 0);
  HVX_VectorPair pb = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(vb, m4), lut, 0), qb = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vb, 4), m4), lut, 0);
  HVX_VectorPair pc = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(vc, m4), lut, 0), qc = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vc, 4), m4), lut, 0);
  HVX_VectorPair pd = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(vd, m4), lut, 0), qd = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vd, 4), m4), lut, 0);
  HVX_VectorPair pe = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(ve, m4), lut, 0), qe = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(ve, 4), m4), lut, 0);
  HVX_VectorPair pf = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(vf, m4), lut, 0), qf = Q6_Wh_vlut16_VbVhR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vf, 4), m4), lut, 0);
#define __Q4_ST(T, m, P, Q) { __fp16* lo = (T) + 128 * (m); __fp16* hi = (T) + 512 + 128 * (m); \
    *(HVX_Vector*)lo = Q6_V_lo_W(P); *(HVX_Vector*)(lo + 64) = Q6_V_hi_W(P); \
    *(HVX_Vector*)hi = Q6_V_lo_W(Q); *(HVX_Vector*)(hi + 64) = Q6_V_hi_W(Q); }
  __Q4_ST(W, 0, p0, q0) __Q4_ST(W, 1, p1, q1) __Q4_ST(W, 2, p2, q2) __Q4_ST(W, 3, p3, q3)
  __Q4_ST(W + 1024, 0, p4, q4) __Q4_ST(W + 1024, 1, p5, q5) __Q4_ST(W + 1024, 2, p6, q6) __Q4_ST(W + 1024, 3, p7, q7)
  __Q4_ST(W + 2048, 0, p8, q8) __Q4_ST(W + 2048, 1, p9, q9) __Q4_ST(W + 2048, 2, pa, qa) __Q4_ST(W + 2048, 3, pb, qb)
  __Q4_ST(W + 3072, 0, pc, qc) __Q4_ST(W + 3072, 1, pd, qd) __Q4_ST(W + 3072, 2, pe, qe) __Q4_ST(W + 3072, 3, pf, qf)
#undef __Q4_ST
}
/* Groups of 4 K tiles: expand the 4 weight tiles into VTCM, one HMX MAC gives the group's fp16 partial tile, stored to its
 * own VTCM slot. After the K loop the slots are multiplied by their group scales and summed, two passes of 8 output vectors
 * with the 16 qf32 accumulators in registers (an array would be kept in memory and serialize the loop). The codes (2 KB) and
 * scales (128 B) of a group are prefetched __Q4_PF groups ahead; the stream is contiguous across column blocks. Summation
 * order per output element is the same as the per-group loop, so the result is bit-identical to it. */
#define __Q4_PF 16
static void __hmx_q4_gemm(__fp16* C, const __fp16* A, const unsigned char* codes, const __fp16* sc, int M, int K, int N) {
  unsigned char* vt = __q4_vtcm();
  __fp16* tbl = (__fp16*)(vt + 0);
  __fp16* W = (__fp16*)(vt + 4096);
  __fp16* PS = (__fp16*)(vt + 16384);                                 /* K/128 partial tiles, 2 KB each */
  __fp16* OUT = (__fp16*)(vt + 16384 + (size_t)(K / 128) * 2048);     /* the 16 output vectors of a column block */
  __fp16* At = (__fp16*)(vt + 262144);
  const int kt = K / 32, ng = K / 128, nt = N / 32;
  const size_t total_groups = (size_t)nt * ng;
  for (int j = 0; j < 64; j++) ((unsigned*)tbl)[j] = 0;  /* zero bias */
  __q4_pack_a((const uint16_t*)A, M, K, (uint16_t*)At);
  unsigned short lv[64] __attribute__((aligned(128)));
  for (int i = 0; i < 64; i++) lv[i] = (i % 2 == 0 && i / 2 < 16) ? __q4_lut_bits[i / 2] : 0;
  HVX_Vector lut = *(HVX_Vector*)lv;
  for (int nb = 0; nb < nt; nb++) {
    for (int g = 0; g < ng; g++) {
      size_t G = (size_t)nb * ng + g;
      if (G + __Q4_PF < total_groups) {
        __q4_prefetch(codes + (G + __Q4_PF) * 2048, 2048);
        __q4_prefetch(sc + (G + __Q4_PF) * 64, 128);
      }
      __q4_expand4(W, codes + ((size_t)nb * kt + 4 * g) * 512, lut);
      __q4_set_table(tbl);
      __q4_mac(At + 4 * g * 1024, W, 4);
      __q4_store(PS + (size_t)g * 1024);
    }
    for (int hf = 0; hf < 2; hf++) {
    HVX_Vector a0l = Q6_V_vzero(), a0h = Q6_V_vzero();
    HVX_Vector a1l = Q6_V_vzero(), a1h = Q6_V_vzero();
    HVX_Vector a2l = Q6_V_vzero(), a2h = Q6_V_vzero();
    HVX_Vector a3l = Q6_V_vzero(), a3h = Q6_V_vzero();
    HVX_Vector a4l = Q6_V_vzero(), a4h = Q6_V_vzero();
    HVX_Vector a5l = Q6_V_vzero(), a5h = Q6_V_vzero();
    HVX_Vector a6l = Q6_V_vzero(), a6h = Q6_V_vzero();
    HVX_Vector a7l = Q6_V_vzero(), a7h = Q6_V_vzero();
      for (int g = 0; g < ng; g++) {
        HVX_Vector s = *(const HVX_Vector*)(sc + ((size_t)nb * ng + g) * 64);
        const HVX_Vector* pg = (const HVX_Vector*)(PS + (size_t)g * 1024) + 8 * hf;
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[0], s); a0l = Q6_Vqf32_vadd_Vqf32Vqf32(a0l, Q6_V_lo_W(w)); a0h = Q6_Vqf32_vadd_Vqf32Vqf32(a0h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[1], s); a1l = Q6_Vqf32_vadd_Vqf32Vqf32(a1l, Q6_V_lo_W(w)); a1h = Q6_Vqf32_vadd_Vqf32Vqf32(a1h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[2], s); a2l = Q6_Vqf32_vadd_Vqf32Vqf32(a2l, Q6_V_lo_W(w)); a2h = Q6_Vqf32_vadd_Vqf32Vqf32(a2h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[3], s); a3l = Q6_Vqf32_vadd_Vqf32Vqf32(a3l, Q6_V_lo_W(w)); a3h = Q6_Vqf32_vadd_Vqf32Vqf32(a3h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[4], s); a4l = Q6_Vqf32_vadd_Vqf32Vqf32(a4l, Q6_V_lo_W(w)); a4h = Q6_Vqf32_vadd_Vqf32Vqf32(a4h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[5], s); a5l = Q6_Vqf32_vadd_Vqf32Vqf32(a5l, Q6_V_lo_W(w)); a5h = Q6_Vqf32_vadd_Vqf32Vqf32(a5h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[6], s); a6l = Q6_Vqf32_vadd_Vqf32Vqf32(a6l, Q6_V_lo_W(w)); a6h = Q6_Vqf32_vadd_Vqf32Vqf32(a6h, Q6_V_hi_W(w)); }
      { HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pg[7], s); a7l = Q6_Vqf32_vadd_Vqf32Vqf32(a7l, Q6_V_lo_W(w)); a7h = Q6_Vqf32_vadd_Vqf32Vqf32(a7h, Q6_V_hi_W(w)); }
      }
    ((HVX_Vector*)OUT)[8 * hf + 0] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a0h, a0l));
    ((HVX_Vector*)OUT)[8 * hf + 1] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1h, a1l));
    ((HVX_Vector*)OUT)[8 * hf + 2] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a2h, a2l));
    ((HVX_Vector*)OUT)[8 * hf + 3] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a3h, a3l));
    ((HVX_Vector*)OUT)[8 * hf + 4] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a4h, a4l));
    ((HVX_Vector*)OUT)[8 * hf + 5] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a5h, a5l));
    ((HVX_Vector*)OUT)[8 * hf + 6] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a6h, a6l));
    ((HVX_Vector*)OUT)[8 * hf + 7] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a7h, a7l));
    }
    /* OUT vector v holds rows 2v (even halfwords) and 2v+1 (odd) of this column block, interleaved. vdeal splits them into
     * the two rows; each row's 32 columns go out as 16 aligned 32-bit words (C rows start 4-byte aligned) */
    for (int v = 0; v < 16; v++) {
      HVX_Vector x = ((const HVX_Vector*)OUT)[v];
      HVX_VectorPair d = Q6_W_vdeal_VVR(x, x, -2);
      unsigned short t0[64] __attribute__((aligned(128))), t1[64] __attribute__((aligned(128)));
      *(HVX_Vector*)t0 = Q6_V_lo_W(d);
      *(HVX_Vector*)t1 = Q6_V_hi_W(d);
      uint32_t* c0 = (uint32_t*)(C + (size_t)(2 * v) * N + 32 * nb);
      const uint32_t* w0 = (const uint32_t*)t0;
      for (int k = 0; k < 16; k++) c0[k] = w0[k];
      if (2 * v + 1 < M) {
        uint32_t* c1 = (uint32_t*)(C + (size_t)(2 * v + 1) * N + 32 * nb);
        const uint32_t* w1 = (const uint32_t*)t1;
        for (int k = 0; k < 16; k++) c1[k] = w1[k];
      }
    }
  }
}
#endif /* HMX_Q4_H */
