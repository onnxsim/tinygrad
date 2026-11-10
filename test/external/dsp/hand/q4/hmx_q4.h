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
/* A (M, K) fp16 row-major -> the K/32 activation tiles at out (tile j at out + 1024 j halfwords), rows >= M zero */
static inline void __q4_pack_a(const uint16_t* A, int M, int K, uint16_t* out) {
  int kt = K / 32;
  HVX_Vector z = Q6_V_vzero();
  for (int p = 0; p < 16; p++) {
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
static void __hmx_q4_gemm(__fp16* C, const __fp16* A, const unsigned char* codes, const __fp16* sc, int M, int K, int N) {
  unsigned char* vt = __q4_vtcm();
  __fp16* tbl = (__fp16*)(vt + 0);
  __fp16* P = (__fp16*)(vt + 2048);
  __fp16* W = (__fp16*)(vt + 4096);
  __fp16* At = (__fp16*)(vt + 262144);
  const int kt = K / 32, ng = K / 128, nt = N / 32;
  for (int j = 0; j < 64; j++) ((unsigned*)tbl)[j] = 0;  /* zero bias */
  __q4_pack_a((const uint16_t*)A, M, K, (uint16_t*)At);
  unsigned short lv[64] __attribute__((aligned(128)));
  for (int i = 0; i < 64; i++) lv[i] = (i % 2 == 0 && i / 2 < 16) ? __q4_lut_bits[i / 2] : 0;
  HVX_Vector lut = *(HVX_Vector*)lv;
  /* the codes stream from DDR (a column block is ng * 4 * 512 bytes, contiguous): prefetched two blocks ahead */
  __q4_prefetch(codes, (size_t)2 * kt * 512 < (size_t)nt * kt * 512 ? (size_t)2 * kt * 512 : (size_t)nt * kt * 512);
  for (int nb = 0; nb < nt; nb++) {
    if (nb + 2 < nt) __q4_prefetch(codes + (size_t)(nb + 2) * kt * 512, (size_t)kt * 512);
    HVX_Vector acc[32];
    for (int v = 0; v < 32; v++) acc[v] = Q6_V_vzero();
    for (int g = 0; g < ng; g++) {
      for (int j = 0; j < 4; j++) __q4_expand_tile(W + 1024 * j, codes + ((size_t)nb * kt + 4 * g + j) * 512, lut);
      HVX_Vector s = *(const HVX_Vector*)(sc + ((size_t)nb * ng + g) * 64);
      __q4_set_table(tbl);
      __q4_mac(At + 4 * g * 1024, W, 4);
      __q4_store(P);
      for (int v = 0; v < 16; v++) {
        HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector*)P)[v], s);
        acc[2 * v] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[2 * v], Q6_V_lo_W(w));
        acc[2 * v + 1] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[2 * v + 1], Q6_V_hi_W(w));
      }
    }
    for (int v = 0; v < 16; v++) ((HVX_Vector*)P)[v] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(acc[2 * v + 1], acc[2 * v]));
    /* P is the interleaved output tile (row 2v + h%2, column h/2 of this block): deal its even and odd halfwords into the two
     * rows of the pair (vdeal), then copy each row's 32 columns as 16 aligned 32-bit words (C rows start 4-byte aligned) */
    for (int v = 0; v < 16; v++) {
      HVX_Vector x = ((const HVX_Vector*)P)[v];
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
