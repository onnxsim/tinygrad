/* Oracle driver for pd_kernel.h (RPN proposal decode, one FPN level).
 * argv: params anchors idx deltas nchw_u8 | out_ref out_fast | A k H W
 * out_ref : the reference path (per box, real divisions, table anchors, fp32 deltas)
 * out_fast: the shipped fast path (blocked, LUT deltas straight from the backbone's uint8 NCHW map, grid anchors) */
#include "pd_kernel.h"
#include "qemu_rt.h"
int oracle_main(int argc, char** argv) {
  if (argc != 12) { qrt_puts("usage: params anchors idx deltas nchw out_ref out_fast A k H W\n"); return 2; }
  int A = qrt_atoi(argv[8]), k = qrt_atoi(argv[9]), H = qrt_atoi(argv[10]), W = qrt_atoi(argv[11]);
  pd_params* P = qrt_load(argv[1], sizeof(pd_params));
  float* an = qrt_load(argv[2], 16L * A);
  int32_t* idx = qrt_load(argv[3], 4L * k);
  float* dl = qrt_load(argv[4], 16L * A);
  uint8_t* nq = qrt_load(argv[5], 12L * H * W);
  float* out = qrt_map(16L * k);
  pd_prep* Q = qrt_map(sizeof(pd_prep));
  unsigned t0 = qrt_inscount();
  pd_decode(an, idx, k, dl, 0, H, W, P, 0, out);
  unsigned t1 = qrt_inscount();
  qrt_insns("reference", t1 - t0);
  qrt_store(argv[6], out, 16L * k);
  t0 = qrt_inscount();
  pd_prepare(P, Q);
  pd_prepare_anchors(an, H, W, Q);
  t1 = qrt_inscount();
  qrt_insns("prepare", t1 - t0);
  t0 = qrt_inscount();
  pd_decode(an, idx, k, 0, nq, H, W, P, Q, out);
  t1 = qrt_inscount();
  qrt_insns("fast", t1 - t0);
  qrt_store(argv[7], out, 16L * k);
  return 0;
}
