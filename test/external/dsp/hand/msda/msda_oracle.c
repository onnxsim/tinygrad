/* Oracle driver for msda_kernel.h (multi-scale deformable attention, the contract in onnx-simplifier's
 * scripts/android/msda_hvx/README.md). argv: shape value value_u8 vscale vzp loc ref attw vis | out
 * shape: msda_shape.h's packed int32 array (vis flag included); unused buffers are passed as 4-byte dummies.
 * Built for v65: msda_run takes the scalar body (the HVX body is V68+ qfloat, which qemu 8.2 can't decode). */
#include "qemu_rt.h"
#include "msda_shape.h"
int oracle_main(int argc, char** argv) {
  if (argc != 11) { qrt_puts("usage: shape value value_u8 vscale vzp loc ref attw vis out\n"); return 2; }
  int32_t* sh = qrt_load(argv[1], 4 * (MSDA_SHAPE_HDR + 3 * MSDA_MAX_L));
  msda_args_t A;
  memset(&A, 0, sizeof A);
  int has_vis = msda_shape_unpack(sh, MSDA_SHAPE_LEN(sh[1] > 0 && sh[1] <= MSDA_MAX_L ? sh[1] : 1), &A);
  if (has_vis < 0 || msda_check(&A)) { qrt_puts("bad shape\n"); return 2; }
  long nval = msda_n_value(&A) > 0 ? msda_n_value(&A) : 1;
  A.value = qrt_load(argv[2], A.vdtype == MSDA_F32 ? 4 * nval : 4);
  A.value_u8 = qrt_load(argv[3], A.vdtype == MSDA_U8 ? nval : 4);
  A.vscale = qrt_load(argv[4], 4L * A.NV);
  A.vzp = qrt_load(argv[5], 4L * A.NV);
  A.loc = qrt_load(argv[6], 4 * msda_n_loc(&A));
  A.ref = qrt_load(argv[7], A.mode == MSDA_LOC ? 4 : 4 * msda_n_ref(&A));
  A.attw = qrt_load(argv[8], 4 * msda_n_attw(&A));
  A.vis = has_vis ? qrt_load(argv[9], msda_n_vis(&A)) : 0;
  A.out = qrt_map(4 * msda_n_out(&A));
  unsigned t0 = qrt_inscount();
  msda_run(&A, 0, A.Q);
  unsigned t1 = qrt_inscount();
  qrt_insns("msda_scalar", t1 - t0);
  qrt_store(argv[10], A.out, 4 * msda_n_out(&A));
  return 0;
}
