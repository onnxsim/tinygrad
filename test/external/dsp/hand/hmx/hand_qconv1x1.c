/* Hand oracle driver: the QDQ-exact 1x1 conv of onnxsim's hmx_gemm (hmx_qconv.h, QC_EXACT) on one case dir (qc_case.h:
 * meta.txt, x.bin, w.bin, bq.bin, sw.bin, ref.bin), hexagon-sim. Writes y.bin ([M, N] uint8) and prints the pcycles of one
 * timed call (after a warm-up), packing excluded (inputs as the kernel keeps them in VTCM).  hand_qconv1x1 <case dir> */
#include <stdio.h>
#include "qc_case.h"
extern unsigned long long hexagon_sim_read_pcycles(void);
static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
int main(int argc, char** argv) {
  qc_case_t c;
  if (argc < 2 || qc_load_case(argv[1], &c)) { printf("bad case\n"); return 2; }
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  int mt = (c.M + 63) / 64, kt = c.K / 32, nt = c.N / 32;
  size_t off = 0;
  uint8_t* X = v + off; off += (size_t)mt * kt * 2048;
  uint8_t* Y = v + off; off += (size_t)mt * nt * 2048;
  uint8_t* W = v + off; off += (size_t)c.K * c.N;
  off = (off + 255) & ~(size_t)255;
  qc_blk_t* B = (qc_blk_t*)(v + off); off += sizeof(qc_blk_t) * nt;
  qc_hdr_t* H = (qc_hdr_t*)(v + off); off += sizeof(qc_hdr_t);
  off = (off + 2047) & ~(size_t)2047;
  uint8_t* S = v + off; off += 4 * 2048;
  memcpy(W, c.wp, (size_t)c.K * c.N);
  memcpy(B, c.blk, sizeof(qc_blk_t) * nt);
  *H = c.hdr;
  for (int mb = 0; mb < mt; mb++) hmx_pack_a_u8cm(c.x, c.M, c.K, mb * 64, X + (size_t)mb * kt * 2048);
  qc_conv1x1(X, Y, W, B, H, mt, kt, QC_EXACT, S);
  unsigned long long t0 = hexagon_sim_read_pcycles();
  qc_conv1x1(X, Y, W, B, H, mt, kt, QC_EXACT, S);
  printf("pcycles %llu\n", hexagon_sim_read_pcycles() - t0);
  uint8_t* y = malloc((size_t)c.M * c.N);
  hmx_unpack_rows_u8cm(Y, y, c.M, c.N);
  FILE* f = fopen("y.bin", "wb"); fwrite(y, 1, (size_t)c.M * c.N, f); fclose(f);
  return 0;
}
