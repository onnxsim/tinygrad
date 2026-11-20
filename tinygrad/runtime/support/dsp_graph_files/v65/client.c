/* Android client for a dsp_graph_v65.py skel: uploads blob.bin in chunks, runs the graph once on input.bin and compares the
 * output with ref.bin byte for byte, then times `iters` inferences.
 *   client <uri> <dir: blob.bin input.bin ref.bin> [iters] [threads] [batch] [prof]
 * batch > 0 splits the graph into RPCs of that many calls (the input goes with the first, the output comes with the last); prof
 * prints the slowest calls, measured on the DSP. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "tg_graph.h"
static unsigned char* rd(const char* dir, const char* nm, long* n) {
  char p[512]; snprintf(p, sizeof p, "%s/%s", dir, nm);
  FILE* f = fopen(p, "rb"); if (!f) { printf("no %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char* b = malloc(*n ? *n : 1); if (fread(b, 1, *n, f) != (size_t)*n) exit(1); fclose(f); return b;
}
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
#define CHUNK (8 << 20)
#define MAXCALLS 4096
static unsigned long long g_call_us[MAXCALLS];

static int run_graph(remote_handle64 h, int ncalls, int threads, int batch, const unsigned char* in, long nin, unsigned char* out,
                     long nout, int prof) {
  static unsigned long long t[1 + MAXCALLS];
  if (batch <= 0) batch = ncalls;
  for (int s = 0; s < ncalls; s += batch) {
    int n = s + batch > ncalls ? ncalls - s : batch;
    int rc = tg_graph_run(h, s, n, threads, 0, s == 0 ? in : NULL, s == 0 ? nin : 0, s + n == ncalls ? out : NULL,
                          s + n == ncalls ? nout : 0, t, prof ? 1 + n : 1);
    if (rc) { printf("run [%d, %d) failed %d (0x%x)\n", s, s + n, rc, rc); return rc; }
    if (prof) for (int i = 0; i < n && s + i < MAXCALLS; i++) g_call_us[s + i] += t[1 + i];
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s uri dir [iters] [threads] [batch] [prof]\n", argv[0]); return 2; }
  int iters = argc > 3 ? atoi(argv[3]) : 5, threads = argc > 4 ? atoi(argv[4]) : 1, batch = argc > 5 ? atoi(argv[5]) : 0;
  int prof = argc > 6 && !strcmp(argv[6], "prof");
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
  remote_handle64 h; int rc = tg_graph_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  long nb, ni, nr;
  unsigned char *blob = rd(argv[2], "blob.bin", &nb), *in = rd(argv[2], "input.bin", &ni), *ref = rd(argv[2], "ref.bin", &nr);
  unsigned char* out = malloc(nr);
  double t0 = now_ms();
  for (long off = 0; off < nb || off == 0; off += CHUNK) {
    long n = nb - off < CHUNK ? nb - off : CHUNK;
    if ((rc = tg_graph_load(h, (int)off, (int)nb, blob + off, (int)n))) { printf("load at %ld failed %d\n", off, rc); return 1; }
    if (nb == 0) break;
  }
  printf("loaded %ld weight bytes in %.1f ms\n", nb, now_ms() - t0);
  /* the number of calls is not in the client: ask for more than any graph has and let the skel's bound check answer */
  int ncalls = 0;
  for (int lo = 1, hi = MAXCALLS; lo <= hi; ) {
    int mid = (lo + hi) / 2; unsigned long long t;
    if (tg_graph_run(h, mid, 0, threads, 0, NULL, 0, NULL, 0, &t, 1) == 0) ncalls = mid, lo = mid + 1; else hi = mid - 1;
  }
  printf("graph has %d calls\n", ncalls);
  memset(g_call_us, 0, sizeof g_call_us);
  t0 = now_ms();
  if ((rc = run_graph(h, ncalls, threads, batch, in, ni, out, nr, prof))) return 1;
  double first = now_ms() - t0;
  long bad = 0; for (long i = 0; i < nr; i++) bad += out[i] != ref[i];
  printf("first run %.2f ms, %ld/%ld output bytes differ from the reference: %s\n", first, bad, nr, bad ? "FAIL" : "bit-exact");
  if (prof) memset(g_call_us, 0, sizeof g_call_us);
  double best = 1e30, sum = 0;
  for (int it = 0; it < iters; it++) {
    t0 = now_ms();
    if ((rc = run_graph(h, ncalls, threads, batch, in, ni, out, nr, prof))) return 1;
    double ms = now_ms() - t0; sum += ms; if (ms < best) best = ms;
  }
  if (iters) printf("%d iters, %d threads: best %.2f ms, mean %.2f ms per inference (RPC-inclusive)\n", iters, threads, best, sum / iters);
  if (prof && iters) {
    unsigned long long tot = 0; for (int i = 0; i < ncalls; i++) tot += g_call_us[i];
    int top = getenv("PROF_TOP") ? atoi(getenv("PROF_TOP")) : 15;  /* PROF_TOP=N lists the N slowest calls */
    for (int k = 0; k < top; k++) {
      int m = -1; for (int i = 0; i < ncalls; i++) if (g_call_us[i] && (m < 0 || g_call_us[i] > g_call_us[m])) m = i;
      if (m < 0) break;
      printf("  call %4d %9.1f us %5.1f%%\n", m, (double)g_call_us[m] / iters, 100.0 * g_call_us[m] / tot);
      g_call_us[m] = 0;
    }
  }
  tg_graph_close(h);
  return bad != 0;
}
