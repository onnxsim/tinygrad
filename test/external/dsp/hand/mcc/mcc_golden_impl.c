/* FastRPC skel that runs mb_hvx.h's three steps (LayerNorm, base-2 softmax + self term, GELU) on fixed
 * tile-layout inputs and returns the bytes the phone computes for them. test_mcc.py compares tinygrad's
 * lowering of the same steps against those bytes.
 *
 * The inputs are laid out exactly as mcc_block.h's mb_layout does for the Q the case was exported at, so
 * every step sees the pointers and strides it sees inside the block. No HMX op runs here, so no
 * compute_res / VTCM acquire is needed (mcc_hmx_impl.c acquires VTCM only because it calls HMX): the tile
 * blocks are one DSP-heap allocation, aligned to MB_TB so the 2 KB tiles and 128 B vectors are aligned.
 * The HMX work that produces S, q and k in the real block is absent -- mcc_case.py packs those from the
 * float64 reference as fixed fp16 tiles -- but it does not change the steps' arithmetic. */
#include <stdlib.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "mcc_golden_rpc.h"
#include "qurt.h"
#define MB_NOW() qurt_get_core_pcycles()
#include "mcc_block.h" /* mb_hvx.h's mbv_* come in with it, under __HVX__ */

extern unsigned long long HAP_perf_get_time_us(void);

#define MG_STEP_LN 1
#define MG_STEP_SM 2
#define MG_STEP_GELU 4
#define MG_NPH 3 /* times: [0] us per call, [1..3] pcycles of layernorm / softmax / gelu */

typedef struct {
  uint8_t *x, *h, *s, *q, *k, *pv, *g;
  uint8_t *sg, *gg; /* the pristine inputs of the two in-place steps (see mcc_golden_rpc_run) */
  size_t xb, hb, sb, qkb, pvb, gb, lnb;
  int rt, nt;
} mg_blk;

/* One allocation for the layout the steps see, plus a second one holding the pristine copy of the two
 * in-place steps' inputs: the restore at the end of a repeat reads from there, never from the buffer the
 * step just overwrote (which is what a naive `memcpy(b.s, s, ...)` off the incoming argument does -- the
 * step has already eaten it, so the restore is a copy of the result and the next repeat degenerates).
 * The order inside the layout allocation is the same relative order as mcc_block.h's mb_layout (X, H,
 * S, q, k, p_self, then the GELU input, which in the block is the MLP's hid scratch). */
static int mg_layout(mg_blk* b, int rt, int nt) {
  const size_t tb = MB_TB;
  b->xb = (size_t)rt * MB_KT * tb, b->hb = b->xb, b->sb = (size_t)rt * 8 * tb, b->qkb = (size_t)rt * tb,
  b->pvb = (size_t)rt * tb, b->gb = (size_t)nt * tb, b->lnb = 2 * MB_D * 2;
  void* p = NULL;
  if (posix_memalign(&p, MB_TB, b->xb + b->hb + b->sb + b->qkb + b->qkb + b->pvb + b->gb + b->sb + b->gb))
    return AEE_ENOMEMORY;
  b->x = (uint8_t*)p;
  b->h = b->x + b->xb, b->s = b->h + b->hb, b->q = b->s + b->sb, b->k = b->q + b->qkb, b->pv = b->k + b->qkb,
  b->g = b->pv + b->pvb;
  b->rt = rt, b->nt = nt;
  /* the pristine inputs: one S (+ the padding the block's ts6 applies) and one G tile per step */
  b->sg = b->g + b->gb, b->gg = b->sg + b->sb;
  return 0;
}

int mcc_golden_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return 0;
}
int mcc_golden_rpc_close(remote_handle64 h) {
  free((void*)(uintptr_t)h);
  return 0;
}

int mcc_golden_rpc_perf_vote(remote_handle64 h, int flags, int* rc) {
  void* ctx = (void*)mcc_golden_rpc_perf_vote;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = 0;
  d.dcvs_v2.set_dcvs_params = 1;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  if (flags & 1) {
    d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  }
  int r2 = HAP_power_set(ctx, &d);
  *rc = r1 * 1000 + r2;
  return 0;
}

int mcc_golden_rpc_run(remote_handle64 h, int steps, int rt, int nt, int it, const unsigned short* x, int xLen,
                       const unsigned short* s, int sLen, const unsigned short* q, int qLen, const unsigned short* k,
                       int kLen, const unsigned short* g, int gLen, const unsigned short* ln, int lnLen,
                       unsigned short* hout, int hLen, unsigned short* spv, int spvLen, unsigned short* gout,
                       int goutLen, unsigned short* pvout, int pvoutLen, unsigned long long* t, int tLen, int* codes,
                       int codesLen) {
  if (rt <= 0 || rt > 1024 || nt <= 0 || nt > 64 || it < 1 || tLen < 1 + MG_NPH || codesLen < 1) return AEE_EBADPARM;
  mg_blk b;
  int rc = mg_layout(&b, rt, nt);
  if (rc) return rc;
  if ((size_t)xLen != b.xb || (size_t)sLen != b.sb || (size_t)qLen != b.qkb || (size_t)kLen != b.qkb ||
      (size_t)gLen != b.gb || (size_t)lnLen != b.lnb || (size_t)hLen != b.hb || (size_t)spvLen != b.sb ||
      (size_t)goutLen != b.gb || (size_t)pvoutLen != b.pvb) {
    /* sizes are sequence<uint16> element counts; a mismatch here is a host/client bug, not a kernel one,
     * so report which lengths are off instead of just AEE_EBADPARM */
    codes[0] = (int)(b.xb - xLen) | ((int)(b.sb - sLen) << 10) | ((int)(b.qkb - qLen) << 20) | ((int)(b.gb - gLen) << 5);
    return AEE_EBADPARM;
  }
  memset(t, 0, tLen * sizeof *t);
  memset(codes, 0, codesLen * sizeof *codes);
  memcpy(b.x, x, b.xb);
  memcpy(b.h, x, b.hb); /* H aliases X, the way the block reuses the same VTCM: mbv_layernorm is
                        * out-of-place, so the next repeat still sees the original scores in X */
  /* The padded score columns (197..223 of the 224, i.e. j >= 5 of S tile 6) are already -65504 in the
   * case file, exactly what mcc_block.h's ts6 column table makes the HMX GEMM write, so the skel must NOT
   * re-apply that bias: the tile is row-pair interleaved, so "half the tile" is not a contiguous run, and
   * laying -65504 into a full 256 B vector instead overwrites vector 0 -- the 32 scores of row pair 0 in
   * columns 192..196 -- which then shifts that row pair's max and scales its whole P row. */
  memcpy(b.s, s, b.sb);
  memcpy(b.q, q, b.qkb);
  memcpy(b.k, k, b.qkb);
  memcpy(b.g, g, b.gb);
  /* the pristine copies of the two in-place steps' inputs: the restore at the end of a repeat reads
   * from these, not from the argument the step has already overwritten */
  memcpy(b.sg, b.s, b.sb);
  memcpy(b.gg, b.g, b.gb);
  unsigned long long t0 = 0, pl[MG_NPH] = {0, 0, 0};
  for (int i = 0; i < it; i++) {
    const unsigned long long a = HAP_perf_get_time_us(), c = MB_NOW();
    if (steps & MG_STEP_LN) mbv_layernorm(b.x, b.h, b.rt, ln, 0, 1);
    pl[0] = MB_NOW() - c;
    const unsigned long long c1 = MB_NOW();
    if (steps & MG_STEP_SM) mbv_softmax(b.s, b.q, b.k, b.pv, b.rt, 0, 1);
    pl[1] = MB_NOW() - c1;
    const unsigned long long c2 = MB_NOW();
    for (int n = 0; n < b.nt; n++) mbv_tile_gelu(b.g + (size_t)n * MB_TB);
    pl[2] = MB_NOW() - c2;
    t0 += HAP_perf_get_time_us() - a;
    /* Both in-place steps leave their result in the buffer they read, and the goldens are the results,
     * so the copy out happens *before* the restore -- otherwise the restore wipes them. */
    if (steps & MG_STEP_SM) {
      memcpy(spv, b.s, b.sb);
      memcpy(pvout, b.pv, b.pvb);
      memcpy(b.s, b.sg, b.sb);
    }
    if (steps & MG_STEP_GELU) {
      memcpy(gout, b.g, b.gb);
      memcpy(b.g, b.gg, b.gb);
    }
  }
  t[0] = t0 / it;
  for (int i = 0; i < MG_NPH; i++) t[1 + i] = pl[i];
  if (steps & MG_STEP_LN) memcpy(hout, b.h, b.hb);
  free(b.x);
  return 0;
}
