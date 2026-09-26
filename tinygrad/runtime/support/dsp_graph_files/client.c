/* Android client for a dsp_graph skel (tg_hmx_rpc_run: a = the graph input, b = the constants blob, c = the output): runs
 * it once, compares c with the expected output, then times `iters` inferences on the DSP.
 * With "prof" as the 4th argument it also prints the per-call breakdown (t[1..], us per inference,
 * per graph.h's G_PROF order) - the device's own answer, where hexagon-sim can only give a different
 * machine's. That costs a HAP_perf_get_time_us per call, so the default stays timing-only.
 *   client <uri> <case dir: a.bin b.bin ref.bin> [iters|"iters prof"] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "remote.h"
#include "tg_hmx_rpc.h"
static unsigned char* rd(const char* dir, const char* nm, long* n) {
  char p[512]; snprintf(p, sizeof p, "%s/%s", dir, nm);
  FILE* f = fopen(p, "rb"); if (!f) { printf("no %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char* b = malloc(*n); if (fread(b, 1, *n, f) != (size_t)*n) exit(1); fclose(f); return b;
}
#define MAXP 256
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s uri case_dir [iters [prof]]\n", argv[0]); return 2; }
  int iters = argc > 3 ? atoi(argv[3]) : 10;
  int prof = argc > 4 && !strcmp(argv[4], "prof");
  int tlen = prof ? 1 + MAXP : 4;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
  remote_handle64 h; int rc = tg_hmx_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  long na, nb, nr;
  unsigned char *a = rd(argv[2], "a.bin", &na), *b = rd(argv[2], "b.bin", &nb), *ref = rd(argv[2], "ref.bin", &nr), *c = malloc(nr);
  unsigned long long t[1 + MAXP]; int codes[8];
  rc = tg_hmx_rpc_run(h, 1, a, na, b, nb, c, nr, t, tlen, codes, 8);
  long bad = 0; for (long i = 0; i < nr; i++) bad += c[i] != ref[i];
  printf("rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d heap %d KB; %ld/%ld mismatches\n", rc, codes[0], codes[1],
         codes[2], codes[3], codes[4], codes[5], codes[6], bad, nr);
  rc = tg_hmx_rpc_run(h, iters, a, na, b, nb, c, nr, t, tlen, codes, 8);
  printf("%.1f us/inference (%d iters) %s\n", (double)t[0] / iters, iters, bad ? "FAIL" : "PASS");
  if (prof) {
    unsigned long long sum = 0;
    for (int i = 1; i < tlen; i++) sum += t[i];
    printf("prof %llu us in %d calls (%.1f%% of %.1f us)\n", sum, tlen - 1,
           (double)sum * 100 / (double)t[0], (double)t[0] / iters);
    for (int i = 1; i < tlen; i++) if (t[i] > 0) printf("  call %3d %8.1f us %5.1f%%\n", i - 1, (double)t[i], (double)t[i] * 100 / (double)sum);
  }
  tg_hmx_rpc_close(h);
  return bad != 0;
}
