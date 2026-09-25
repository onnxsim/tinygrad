/* Hand oracle driver: the QDQ-exact 3x3 conv (stride 1 or 2, pad 1) of onnxsim's hmx_gemm (hmx_qconv3.h, QC_EXACT) on one
 * case dir (qc_case.h format with H W 3 stride), hexagon-sim. Timed (one call after a warm-up): the shifted copies / phase
 * split, the straddle stitch and the conv, from the flat-packed input in VTCM; writes y.bin ([Ho*Wo, N] uint8). */
#include <stdio.h>
#include "qc_case.h"
extern unsigned long long hexagon_sim_read_pcycles(void);
static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
static uint8_t* valloc_(uint8_t* v, size_t* off, size_t n) { *off = (*off + 2047) & ~(size_t)2047; uint8_t* p = v + *off; *off += n; return p; }
int main(int argc, char** argv) {
  qc_case_t c;
  if (argc < 2 || qc_load_case(argv[1], &c) || c.k != 3) { printf("bad case\n"); return 2; }
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  int kt = c.K / 32, nt = c.N / 32;
  qc_geom_t gi = qc_geom(c.H, c.W), go = qc_geom(c.Ho, c.Wo);
  size_t off = 0;
  uint8_t* X = valloc_(v, &off, qc_geom_bytes(&gi, kt));
  uint8_t* Y = valloc_(v, &off, qc_geom_bytes(&go, nt));
  uint8_t* W = valloc_(v, &off, (size_t)9 * c.K * c.N);
  qc_blk_t* B = (qc_blk_t*)valloc_(v, &off, sizeof(qc_blk_t) * nt);
  qc_hdr_t* H = (qc_hdr_t*)valloc_(v, &off, sizeof(qc_hdr_t));
  uint8_t* S = valloc_(v, &off, 4 * 2048);
  uint8_t *t0 = valloc_(v, &off, qc_geom_bytes(&go, kt)), *t1 = valloc_(v, &off, qc_geom_bytes(&go, kt));
  uint8_t* ph[4];
  for (int i = 0; i < 4; i++) ph[i] = c.stride == 2 ? valloc_(v, &off, qc_geom_bytes(&go, kt)) : NULL;
  uint8_t* side = valloc_(v, &off, 64 * 2048);
  uint8_t* xf = malloc(qc_geom_bytes(&gi, kt));
  qc_flat_pack(c.x, c.H, c.W, c.K, c.zx, &gi, xf);
  memcpy(X, xf, qc_geom_bytes(&gi, kt));
  memcpy(W, c.wp, (size_t)9 * c.K * c.N);
  memcpy(B, c.blk, sizeof(qc_blk_t) * nt);
  *H = c.hdr;
  uint32_t* atab = malloc(sizeof(uint32_t) * qc_geom_nob(&go) * 9 * kt);
  unsigned long long t = 0;
  for (int it = 0; it < 2; it++) {
    unsigned long long ts = hexagon_sim_read_pcycles();
    qc_taps_t tp;
    if (c.stride == 1) {
      qc_shift_copies(X, t0, t1, gi.nblk, kt, c.zx);
      tp = qc_taps_s1(X, t0, t1);
    } else {
      qc_phase_split(X, &gi, ph, &go, kt, c.zx);
      qc_shift_copies(ph[1], t0, Y, go.nblk, kt, c.zx);
      qc_shift_copies(ph[3], t1, Y, go.nblk, kt, c.zx);
      tp = qc_taps_s2(ph, t0, t1);
    }
    qc_stitch_t st[64];
    int ns = qc_conv3x3_plan(&tp, &go, kt, atab, st, side, 64);
    if (ns < 0) { printf("side buffer too small\n"); return 2; }
    qc_conv3x3_stitch(st, ns, kt);
    qc_conv3x3(atab, &go, Y, W, B, H, kt, QC_EXACT, S);
    t = hexagon_sim_read_pcycles() - ts;
  }
  printf("pcycles %llu\n", t);
  uint8_t* y = malloc((size_t)c.Mo * c.N);
  qc_flat_unpack(Y, &go, c.N, y);
  FILE* f = fopen("y.bin", "wb"); fwrite(y, 1, (size_t)c.Mo * c.N, f); fclose(f);
  return 0;
}
