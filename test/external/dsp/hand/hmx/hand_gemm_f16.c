/* Hand oracle driver: onnxsim hmx_gemm's fp16 GEMM (hmx_gemm.h: row-major A, W prepacked once on the host, DDR-fed -- pack A,
 * stream the weight tiles, unpack C) on hexagon-sim. Reads a.bin [M,K] and w.bin [K,N] (fp16 bits), writes c.bin [M,N];
 * prints the pcycles of one timed call after a warm-up (the host weight prepack excluded).  hand_gemm_f16 M K N */
#include <stdio.h>
#include <stdlib.h>
#include "hmx_gemm.h"
extern unsigned long long hexagon_sim_read_pcycles(void);
static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
static void* rd(const char* p, size_t n) { FILE* f = fopen(p, "rb"); void* b = malloc(n); if (!f || fread(b, 1, n, f) != n) { printf("read %s\n", p); exit(2); } fclose(f); return b; }
int main(int argc, char** argv) {
  int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16); unsigned vs = cfg(0x3c) * 1024;
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  uint16_t *A = rd("a.bin", (size_t)M * K * 2), *W = rd("w.bin", (size_t)K * N * 2);
  uint16_t *Wp = malloc((size_t)K * N * 2), *C = malloc((size_t)M * N * 2);
  hmx_pack_w_f16(W, K, N, Wp);
  hmx_gemm_f16(A, Wp, NULL, C, M, K, N, v, vs);
  unsigned long long t0 = hexagon_sim_read_pcycles();
  int rc = hmx_gemm_f16(A, Wp, NULL, C, M, K, N, v, vs);
  printf("pcycles %llu rc %d\n", hexagon_sim_read_pcycles() - t0, rc);
  FILE* f = fopen("c.bin", "wb"); fwrite(C, 2, (size_t)M * N, f); fclose(f);
  return 0;
}
