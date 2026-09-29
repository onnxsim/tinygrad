/* FastRPC skel around a dsp_graph_v65.py program (k<n>.c + graph.h). The weights (blob.bin) are uploaded once by load() and read
 * in place; every other region is allocated once and zeroed. run() executes a batch of calls on a graph thread with a large stack
 * (generated kernels can need 20+ KB of frame; FastRPC's own thread has 16 KB). A DSP_THREADS kernel's core_id slices go to a
 * pool: the graph thread and threads-1 helpers, woken by a qurt barrier. From v65 on QuRT context-switches HVX itself, so every
 * participant can take an HVX context without deadlocking on the two a v65 cDSP has. */
#include <stdlib.h>
#include <string.h>
#include "qurt.h"
#include "AEEStdErr.h"
#include "tg_graph.h"
extern unsigned long long HAP_perf_get_time_us(void);

static void g_parallel(void (*f)(unsigned char**, int), unsigned char** R, int n);
#define G_PARALLEL(f, R, n) g_parallel(f, R, n)
#include "graph.h"

static unsigned char* g_blob;
#ifdef G_VTCM_BYTES
/* a program with vgather kernels (DSP_V65_VGATHER): the constant 128 KB tables they look up are copied into VTCM once, after the
 * weights are in, and the kernels read them there (G_VTAB(slot) is null without VTCM: they run their scalar loops). vgather reads
 * only VTCM and a region must sit in one page, so the tables need a 128 KB-aligned base. */
#include "HAP_compute_res.h"
unsigned char* g_vtcm;
static unsigned g_vtcm_ctx;
static int g_vtab_ok;
static void vtcm_release(void) { if (g_vtcm) { HAP_compute_res_release(g_vtcm_ctx); g_vtcm = 0; g_vtcm_ctx = 0; g_vtab_ok = 0; } }
static void vtab_setup(void) {
  if (!g_vtcm) {
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, (G_VTCM_BYTES + 0xFFFF) & ~0xFFFFu, 0, 0);
    if (!(g_vtcm_ctx = HAP_compute_res_acquire(&attr, 100000))) return;
    void* p = 0; unsigned n = 0;
    HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &p, &n);
    if (!p || n < G_VTCM_BYTES || ((uintptr_t)p & (G_VTAB_BYTES - 1))) { HAP_compute_res_release(g_vtcm_ctx); g_vtcm_ctx = 0; return; }
    g_vtcm = (unsigned char*)p;
  }
  if (g_vtab_ok) return;
  for (int s = 0; s < G_NVTAB; s++) memcpy(g_vtcm + (size_t)s * G_VTAB_BYTES, g_blob + G_VTAB_BLOB[s], G_VTAB_BYTES);
  g_vtab_ok = 1;
}
#define G_VTCM_SETUP() vtab_setup()
#define G_VTCM_RELEASE() vtcm_release()
#define G_VTCM_RELOAD() (g_vtab_ok = 0)
#else
#define G_VTCM_SETUP()
#define G_VTCM_RELEASE()
#define G_VTCM_RELOAD()
#endif

static char* g_blob_raw;  /* no memalign in the DSP libc: aligned by hand */
static int g_blob_total, g_blob_loaded;
static unsigned char* g_R[G_NREG];
static char* g_raw;

/* ----- thread pool ----- */
#define POOL_MAX 8
#define HELPER_STACK (96 * 1024)
static char g_hstack[POOL_MAX][HELPER_STACK] __attribute__((aligned(128)));
static qurt_thread_t g_htid[POOL_MAX];
static int g_pool_n = 1;  /* participants, the graph thread included */
static qurt_barrier_t g_go, g_done;
static void (*volatile g_fn)(unsigned char**, int);
static unsigned char** volatile g_fR;
static volatile int g_fn_n, g_quit;

static void helper(void* arg) {
  int me = (int)(uintptr_t)arg;
  qurt_hvx_lock(QURT_HVX_MODE_128B);
  for (;;) {
    qurt_barrier_wait(&g_go);
    if (g_quit) break;
    for (int c = me; c < g_fn_n; c += g_pool_n) g_fn(g_fR, c);
    qurt_barrier_wait(&g_done);
  }
  qurt_hvx_unlock();
  qurt_thread_exit(0);
}

static void g_parallel(void (*f)(unsigned char**, int), unsigned char** R, int n) {
  if (g_pool_n <= 1) { for (int c = 0; c < n; c++) f(R, c); return; }
  g_fn = f, g_fR = R, g_fn_n = n;
  qurt_barrier_wait(&g_go);
  for (int c = 0; c < n; c += g_pool_n) f(R, c);
  qurt_barrier_wait(&g_done);
}

static void pool_stop(void) {
  if (g_pool_n <= 1) return;
  g_quit = 1;
  qurt_barrier_wait(&g_go);
  for (int i = 1; i < g_pool_n; i++) { int st; qurt_thread_join(g_htid[i], &st); }
  qurt_barrier_destroy(&g_go); qurt_barrier_destroy(&g_done);
  g_pool_n = 1, g_quit = 0;
}

static int pool_start(int n) {
  if (n < 1) n = 1;
  if (n > POOL_MAX) n = POOL_MAX;
  if (n == g_pool_n) return 0;
  pool_stop();
  if (n == 1) return 0;
  qurt_barrier_init(&g_go, n); qurt_barrier_init(&g_done, n);
  g_pool_n = n;
  for (int i = 1; i < n; i++) {
    qurt_thread_attr_t ta; qurt_thread_attr_init(&ta);
    qurt_thread_attr_set_stack_addr(&ta, g_hstack[i]); qurt_thread_attr_set_stack_size(&ta, HELPER_STACK);
    qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
    if (qurt_thread_create(&g_htid[i], &ta, helper, (void*)(uintptr_t)i)) return -1;
  }
  return 0;
}

/* ----- FastRPC methods ----- */
#ifdef G_PERF_VOTE
#include "HAP_power.h"
/* G_PERF_VOTE (DSP_V65_PERF_VOTE): 1 = HVX power up + DCVS performance mode, 2 = also pin the TURBO corner, 3 = pin the MAX corner.
 * A light FastRPC client otherwise runs at whatever DCVS picks: a PMU probe measured ~360 MHz effective and driving ranged from 144 to
 * 192 ms between identical runs. Pinned (driving DSP ms / DM): none 144-192 / 72-96, TURBO 147-149 / 76.4, MAX 139.2-139.7 / 71.6.
 * The vote is held while any program is open and released when the last one closes. */
static int g_perf_refs;
static char g_perf_ctx;
static void perf_vote(void) {
  if (g_perf_refs++) return;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX, req.hvx.power_up = 1;
  HAP_power_set(&g_perf_ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2, d.dcvs_v2.dcvs_enable = 0, d.dcvs_v2.set_dcvs_params = 1, d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  if (G_PERF_VOTE >= 2) {
    int c = G_PERF_VOTE >= 3 ? HAP_DCVS_VCORNER_MAX : HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.target_corner = c;
    d.dcvs_v2.dcvs_params.min_corner = c;
    d.dcvs_v2.dcvs_params.max_corner = c;
  }
  HAP_power_set(&g_perf_ctx, &d);
}
static void perf_release(void) { if (g_perf_refs > 0 && --g_perf_refs == 0) HAP_power_destroy(&g_perf_ctx); }
#define G_PERF() perf_vote()
#define G_PERF_RELEASE() perf_release()
#else
#define G_PERF()
#define G_PERF_RELEASE()
#endif
int tg_graph_open(const char* uri, remote_handle64* h) { G_PERF(); *h = (remote_handle64)(uintptr_t)malloc(1); return 0; }
int tg_graph_close(remote_handle64 h) { G_VTCM_RELEASE(); G_PERF_RELEASE(); free((void*)(uintptr_t)h); return 0; }

int tg_graph_load(remote_handle64 h, int offset, int total, const uint8* chunk, int chunkLen) {
  if (total != G_BLOB_BYTES) return AEE_EBADPARM;
  if (offset == 0) {
    if (g_blob && g_blob_total != total) { free(g_blob_raw); g_blob = 0; }
    if (!g_blob && (g_blob_raw = malloc(total + 128))) g_blob = (unsigned char*)(((uintptr_t)g_blob_raw + 127) & ~(uintptr_t)127);
    if (!g_blob) return AEE_ENOMEMORY;
    g_blob_total = total, g_blob_loaded = 0;
    G_VTCM_RELOAD();
  }
  if (offset != g_blob_loaded || offset + chunkLen > total) return AEE_EBADPARM;
  memcpy(g_blob + offset, chunk, chunkLen);
  g_blob_loaded += chunkLen;
  return 0;
}

static int regions(void) {
  if (g_raw) return 0;
  size_t total = 128;
  for (int i = 0; i < G_NREG; i++) if (G_REG_BLOB[i] < 0) total += G_REG_BYTES[i];
  if (!(g_raw = malloc(total))) return -1;
  memset(g_raw, 0, total);
  unsigned char* p = (unsigned char*)(((uintptr_t)g_raw + 127) & ~(uintptr_t)127);
  for (int i = 0; i < G_NREG; i++) {
    if (G_REG_BLOB[i] >= 0) g_R[i] = g_blob + G_REG_BLOB[i];
    else g_R[i] = p, p += G_REG_BYTES[i];
  }
  return 0;
}

typedef struct { int start, count, threads, rc; uint64* t; int tLen; } job_t;
#define GRAPH_STACK (256 * 1024)
static char g_gstack[GRAPH_STACK] __attribute__((aligned(128)));

static void graph_thread(void* p) {
  job_t* j = (job_t*)p;
  qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->rc = pool_start(j->threads);
  if (j->rc == 0) {
    unsigned long long t0 = HAP_perf_get_time_us(), last = t0;
    for (int i = 0; i < j->count; i++) {
      g_call(j->start + i, g_R);
      if (1 + i < j->tLen) { unsigned long long now = HAP_perf_get_time_us(); j->t[1 + i] = now - last; last = now; }
    }
    j->t[0] = HAP_perf_get_time_us() - t0;
  }
  qurt_hvx_unlock();
  qurt_thread_exit(0);
}

int tg_graph_run(remote_handle64 h, int start, int count, int threads, const uint8* in, int inLen, uint8* out, int outLen,
                 uint64* t, int tLen) {
  if (!g_blob || g_blob_loaded != G_BLOB_BYTES) return AEE_EBADSTATE;
  if (start < 0 || count < 0 || start + count > G_NCALLS || tLen < 1) return AEE_EBADPARM;
  if (regions()) return AEE_ENOMEMORY;
  G_VTCM_SETUP();
  memset(t, 0, tLen * sizeof(uint64));
  if (start == 0 && inLen > 0) {
    int off = 0;
    for (int i = 0; i < G_NIN; i++) {
      if (off + (int)G_IN_BYTES[i] > inLen) return AEE_EBADPARM;
      memcpy(g_R[G_IN_REG[i]], in + off, G_IN_BYTES[i]);
      off += (G_IN_BYTES[i] + 127) & ~127u;
    }
  }
  job_t j = {start, count, threads, 0, t, tLen};
  qurt_thread_attr_t ta; qurt_thread_attr_init(&ta);
  qurt_thread_attr_set_stack_addr(&ta, g_gstack); qurt_thread_attr_set_stack_size(&ta, GRAPH_STACK);
  qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
  qurt_thread_t tid; int st;
  if (qurt_thread_create(&tid, &ta, graph_thread, &j)) return AEE_EFAILED;
  qurt_thread_join(tid, &st);
  if (j.rc) return AEE_EFAILED;
  if (start + count == G_NCALLS && outLen > 0) memcpy(out, g_R[G_OUT_REG] + G_OUT_OFF, outLen < G_OUT_BYTES ? outLen : G_OUT_BYTES);
  return 0;
}
