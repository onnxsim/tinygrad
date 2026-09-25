/* Android client for a dsp_graph skel (tg_hmx_rpc_run: a = the graph input, b = the constants blob, c = the output): runs
 * it once, compares c with the expected output, then times `iters` inferences on the DSP.
 *   client <uri> <case dir: a.bin b.bin ref.bin> [iters] */
#include <stdio.h>
#include <stdlib.h>
#include "remote.h"
#include "tg_hmx_rpc.h"
static unsigned char* rd(const char* dir, const char* nm, long* n) {
  char p[512]; snprintf(p, sizeof p, "%s/%s", dir, nm);
  FILE* f = fopen(p, "rb"); if (!f) { printf("no %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char* b = malloc(*n); if (fread(b, 1, *n, f) != (size_t)*n) exit(1); fclose(f); return b;
}
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s uri case_dir [iters]\n", argv[0]); return 2; }
  int iters = argc > 3 ? atoi(argv[3]) : 10;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
  remote_handle64 h; int rc = tg_hmx_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  long na, nb, nr;
  unsigned char *a = rd(argv[2], "a.bin", &na), *b = rd(argv[2], "b.bin", &nb), *ref = rd(argv[2], "ref.bin", &nr), *c = malloc(nr);
  unsigned long long t[4]; int codes[8];
  rc = tg_hmx_rpc_run(h, 1, a, na, b, nb, c, nr, t, 4, codes, 8);
  long bad = 0; for (long i = 0; i < nr; i++) bad += c[i] != ref[i];
  printf("rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d heap %d KB; %ld/%ld mismatches\n", rc, codes[0], codes[1],
         codes[2], codes[3], codes[4], codes[5], codes[6], bad, nr);
  rc = tg_hmx_rpc_run(h, iters, a, na, b, nb, c, nr, t, 4, codes, 8);
  printf("%.1f us/inference (%d iters) %s\n", (double)t[0] / iters, iters, bad ? "FAIL" : "PASS");
  tg_hmx_rpc_close(h);
  return bad != 0;
}
