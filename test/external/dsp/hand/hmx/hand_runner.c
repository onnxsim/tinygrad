/* Hand oracle driver: onnxsim hmx_gemm's whole QDQ graph runner (runner/rn_load.h + rn_exec.h: HMX convs, HVX QLinearAdd +
 * MaxPool, padding maintenance, VTCM planning) on a qdq_graph.py program, hexagon-sim, QC_EXACT. Prints the pcycles of one
 * timed inference (after a warm-up), writes y.bin (the output, NCHW uint8).  hand_runner <program dir> <input.bin> */
#include <stdio.h>
#include "runner/rn_load.h"
#include "runner/rn_exec.h"
extern unsigned long long hexagon_sim_read_pcycles(void);
static uint8_t* rd(const char* p, size_t n) {
  FILE* f = fopen(p, "rb"); uint8_t* b = malloc(n);
  if (!f || fread(b, 1, n, f) != n) { printf("cannot read %s\n", p); exit(2); }
  fclose(f); return b;
}
static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
int main(int argc, char** argv) {
  static rn_build_t B;
  if (argc < 3 || rn_build(&B, argv[1])) { printf("build failed: %s\n", B.err); return 2; }
  rn_model_t* m = &B.m;
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16); unsigned vs = cfg(0x3c) * 1024;
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  if (m->vtcm_bytes > vs) { printf("VTCM plan %u > %u\n", m->vtcm_bytes, vs); return 2; }
  const rn_tensor_t *ti = &m->t[m->input], *to = &m->t[m->output];
  size_t xl = (size_t)ti->h * ti->w * ti->c, yl = (size_t)to->c * to->h * to->w, fl;
  uint8_t *x = rd(argv[2], xl), *y = malloc(yl);
  uint8_t* xf = rn_pack_input(m, x, &fl);
  static rn_ctx_t c;
  c.m = m, c.blob = B.blob, c.vtcm = v;
  if (rn_plan(&c)) { printf("plan failed\n"); return 2; }
  rn_run(&c, xf, QC_EXACT);
  unsigned long long t0 = hexagon_sim_read_pcycles();
  rn_run(&c, xf, QC_EXACT);
  printf("pcycles %llu\n", hexagon_sim_read_pcycles() - t0);
  rn_unpack_output(m, v + to->off, y);
  FILE* f = fopen("y.bin", "wb"); fwrite(y, 1, yl, f); fclose(f);
  return 0;
}
