/* Oracle driver for layout_kernels.h (fp32 NCHW -> NHWC per image, the FPN maps' conversion for RoiAlign).
 * argv: in | out | C HW   (in gets 31 floats of slack: the HVX kernel over-reads the last pixel block) */
#include "qemu_rt.h"
#include "layout_kernels.h"
int oracle_main(int argc, char** argv) {
  if (argc != 5) { qrt_puts("usage: in out C HW\n"); return 2; }
  int C = qrt_atoi(argv[3]), HW = qrt_atoi(argv[4]);
  float* in = qrt_load(argv[1], 4L * C * HW + 128);
  float *out = qrt_map(4L * C * HW), *ref = qrt_map(4L * C * HW);
  unsigned t0 = qrt_inscount();
  transpose_chw_hwc_scalar(in, ref, C, HW, 0, HW);
  unsigned t1 = qrt_inscount();
  qrt_insns("scalar", t1 - t0);
  t0 = qrt_inscount();
  transpose_chw_hwc_hvx(in, out, C, HW, 0, HW, 0);
  t1 = qrt_inscount();
  qrt_insns("hvx", t1 - t0);
  if (memcmp(out, ref, 4L * C * HW)) { qrt_puts("scalar/hvx disagree\n"); return 1; }
  qrt_store(argv[2], out, 4L * C * HW);
  return 0;
}
