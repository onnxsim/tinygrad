/* hexagon-sim check of __hmx_q4_gemm (hmx_q4.h) vs a double reference on the dequantized weights:
 * sim_q4 M K N   (M <= 32, K % 128 == 0, N % 32 == 0, K <= 4096). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hmx_q4.h"
static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
static float h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static uint16_t f2h(float f) { __fp16 x = (__fp16)f; uint16_t h; memcpy(&h, &x, 2); return h; }
static int idx(int i, int j) { return __Q4_IDX(i, j); }
unsigned char* __hmx_vtcm;
int main(int argc, char** argv) {
  int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
  if (K % 128 || N % 32 || K > 4096 || M > 32) return 2;
  __hmx_vtcm = (unsigned char*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  int ng = K / 128, kt = K / 32, nt = N / 32;
  uint16_t* A = malloc((size_t)M * K * 2);
  float* W = malloc((size_t)K * N * sizeof(float));
  float* Wd = malloc((size_t)K * N * sizeof(float));  /* the dequantized weights (s * q, fp16) */
  uint8_t* codes = malloc((size_t)nt * kt * 512);
  uint16_t* sc = aligned_alloc(128, (size_t)nt * ng * 64 * 2);  /* HVX vector loads need 128-byte alignment */
  uint16_t* C = malloc((size_t)M * N * 2);
  uint32_t seed = 7;
#define RND() (seed = seed * 1103515245u + 12345u, ((seed >> 8) & 0xFFFF) / 65535.0f * 2.f - 1.f)
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = f2h(RND());
  for (size_t i = 0; i < (size_t)K * N; i++) W[i] = RND() * 0.05f;
  /* per (group, column): s = fp16(max|W| / 7); q = clamp(round(W / s), -8, 7); Wd = fp16(s * q). W keeps q for the packing. */
  for (int g = 0; g < ng; g++)
    for (int n = 0; n < N; n++) {
      float mx = 0;
      for (int k = 128 * g; k < 128 * g + 128; k++) if (fabsf(W[(size_t)k * N + n]) > mx) mx = fabsf(W[(size_t)k * N + n]);
      uint16_t s16 = f2h(mx > 0 ? mx / 7 : 1.f);
      float s = h2f(s16);
      for (int k = 128 * g; k < 128 * g + 128; k++) {
        int q = (int)lrintf(W[(size_t)k * N + n] / s);
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        Wd[(size_t)k * N + n] = h2f(f2h(q * s));
        W[(size_t)k * N + n] = (float)q;
      }
      int nb = n / 32, c = n % 32;
      for (int h = 2 * c; h < 2 * c + 2; h++) sc[((size_t)nb * ng + g) * 64 + h] = s16;
    }
  for (int nb = 0; nb < nt; nb++)
    for (int kb = 0; kb < kt; kb++) {
      int tl[1024];
      for (int k = 0; k < 32; k++)
        for (int c = 0; c < 32; c++) tl[idx(k, c)] = (int)W[(size_t)(kb * 32 + k) * N + nb * 32 + c] + 8;
      unsigned char* out = codes + ((size_t)nb * kt + kb) * 512;
      for (int j = 0; j < 512; j++) {
        int m = j / 128, b = j % 128;
        int hl = 128 * m + (b % 2 == 0 ? b / 2 : 64 + b / 2);
        out[j] = (unsigned char)((tl[hl] & 15) | ((tl[hl + 512] & 15) << 4));
      }
    }
  extern unsigned long long hexagon_sim_read_pcycles(void);
  unsigned long long c0, c1;
  c0 = hexagon_sim_read_pcycles();
  __hmx_q4_gemm((__fp16*)C, (const __fp16*)A, codes, (const __fp16*)sc, M, K, N);
  c1 = hexagon_sim_read_pcycles();
  double maxe = 0, maxr = 0; int bad = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < K; k++) s += (double)h2f(A[(size_t)i * K + k]) * Wd[(size_t)k * N + j];
      double got = h2f(C[(size_t)i * N + j]), e = fabs(got - s), tol = fabs(s) / 512 + 2e-3;
      if (e > maxe) maxe = e;
      if (fabs(s) > maxr) maxr = fabs(s);
      if (e > tol) { if (bad < 4) printf("bad %d,%d got %g want %g\n", i, j, got, s); bad++; }
    }
  printf("hmx_q4 %dx%dx%d: max abs err %.3g (max |ref| %.3g), %d beyond tolerance (pcycle counter %llu)\n", M, K, N, maxe,
         maxr, bad, c1 - c0);
  return bad ? 3 : 0;
}
