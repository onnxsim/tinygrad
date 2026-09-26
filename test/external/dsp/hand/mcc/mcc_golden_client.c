/* mcc_golden_client <uri> <case dir> <rt> <nt> [iters] [steps mask, default 7 = all]
 *
 * Runs mb_hvx.h's steps on the case's tile-layout inputs (mcc_case.py) and writes the phone's outputs back
 * into the case dir as gold_h.bin / gold_spv.bin / gold_pv.bin / gold_gelu.bin (fp16 tiles), plus
 * gold_times.txt. Re-run with the same inputs and the bytes must be identical: the qfloat rounding is
 * data-independent, so a diff here means a real difference (different inputs, a changed kernel, or a
 * different power corner).
 *
 * The case also holds the float64 references (ref_h.bin, ref_s.bin, ref_pv.bin) and the untouched inputs
 * (x.bin, s.bin, q.bin, k.bin, g.bin, ln.bin), so a capture can be sanity-checked against float64 without
 * the phone: test_mcc.py does exactly that. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mcc_golden_rpc.h"
#include "remote.h"

#define STEP_LN 1
#define STEP_SM 2
#define STEP_GELU 4

static void* slurp(const char* dir, const char* name, size_t* n) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) exit(1);
  fclose(f);
  return b;
}
static void spit(const char* dir, const char* name, const void* b, size_t n) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "wb");
  if (!f) { printf("cannot write %s\n", p); exit(1); }
  if (fwrite(b, 1, n, f) != n) exit(1);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s <uri> <case dir> <rt> <nt> [iters] [steps]\n", argv[0]); return 2; }
  const char* dir = argv[2];
  int rt = atoi(argv[3]), nt = atoi(argv[4]), it = argc > 5 ? atoi(argv[5]) : 3, steps = argc > 6 ? atoi(argv[6]) : 7;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = mcc_golden_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  int prc = 0;
  mcc_golden_rpc_perf_vote(h, 3, &prc);
  size_t nx, ns, nq, nk, ng, nln, oh, ospv;
  unsigned short *x = slurp(dir, "x.bin", &nx), *s = slurp(dir, "s.bin", &ns), *q = slurp(dir, "q.bin", &nq),
                  *k = slurp(dir, "k.bin", &nk), *g = slurp(dir, "g.bin", &ng), *ln = slurp(dir, "ln.bin", &nln);
  unsigned short* ho = malloc((size_t)rt * 16 * 2048);
  unsigned short* so = malloc((size_t)rt * 8 * 2048);
  unsigned short* go = malloc((size_t)nt * 2048);
  unsigned short* po = malloc((size_t)rt * 2048);
  uint64 t[4];
  int codes[1];
  /* sequence<uint16> lengths are element counts, and every file is 2 B per element, so a file's byte
   * count is its element count */
  rc = mcc_golden_rpc_run(h, steps, rt, nt, it, x, (int)nx, s, (int)ns, q, (int)nq, k, (int)nk, g, (int)ng, ln, (int)nln,
                          ho, (int)((size_t)rt * 16 * 2048), so, (int)((size_t)rt * 8 * 2048), go, (int)((size_t)nt * 2048),
                          po, (int)((size_t)rt * 2048), t, 4, codes, 1);
  printf("vote %d rc %d | steps %d rt %d nt %d iters %d: %.3f us per call"
         " (layernorm %llu, softmax %llu, gelu %llu pcycles)\n",
         prc, rc, steps, rt, nt, it, (double)t[0], t[1], t[2], t[3]);
  if (rc) return 1;
  if (steps & STEP_LN) spit(dir, "gold_h.bin", ho, (size_t)rt * 16 * 2048);
  if (steps & STEP_SM) {
    spit(dir, "gold_spv.bin", so, (size_t)rt * 8 * 2048);
    spit(dir, "gold_pv.bin", po, (size_t)rt * 2048);
  }
  if (steps & STEP_GELU) spit(dir, "gold_gelu.bin", go, (size_t)nt * 2048);
  {
    char p[512];
    snprintf(p, sizeof p, "%s/gold_times.txt", dir);
    FILE* f = fopen(p, "w");
    fprintf(f, "steps %d rt %d nt %d iters %d\nus_per_call %.3f\npcycles_layernorm %llu\npcycles_softmax %llu\npcycles_gelu %llu\n", steps,
            rt, nt, it, (double)t[0], t[1], t[2], t[3]);
    fclose(f);
  }
  printf("wrote gold_h.bin (%d B) / gold_spv.bin (%d B) / gold_pv.bin (%d B) / gold_gelu.bin (%d B) / gold_times.txt in %s\n",
         (steps & STEP_LN) ? rt * 16 * 2048 : 0, (steps & STEP_SM) ? rt * 8 * 2048 : 0,
         (steps & STEP_SM) ? rt * 2048 : 0, (steps & STEP_GELU) ? nt * 2048 : 0, dir);
  mcc_golden_rpc_close(h);
  return 0;
}
