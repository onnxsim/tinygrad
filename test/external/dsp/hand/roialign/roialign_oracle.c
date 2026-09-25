/* Oracle driver for roialign_kernel.h (fp32 RoiAlign, opset < 16, avg, channels-last (H, W, C) -> (R, OH, OW, C)).
 * argv: feat rois | out | H W C R OH OW sr inv_scale   (spatial_scale = 1 / inv_scale, like roialign_qemu.c) */
#include "qemu_rt.h"
#include "roialign_kernel.h"
int oracle_main(int argc, char** argv) {
  if (argc != 12) { qrt_puts("usage: feat rois out H W C R OH OW sr inv_scale\n"); return 2; }
  int H = qrt_atoi(argv[4]), W = qrt_atoi(argv[5]), C = qrt_atoi(argv[6]), R = qrt_atoi(argv[7]);
  int OH = qrt_atoi(argv[8]), OW = qrt_atoi(argv[9]), sr = qrt_atoi(argv[10]);
  float scale = 1.0f / (float)qrt_atoi(argv[11]);
  long m = (long)R * OH * OW * C;
  float *feat = qrt_load(argv[1], 4L * H * W * C), *rois = qrt_load(argv[2], 16L * R), *out = qrt_map(4 * m);
  unsigned t0 = qrt_inscount();
  roialign_hwc(feat, H, W, C, rois, R, OH, OW, sr, scale, out);
  unsigned t1 = qrt_inscount();
  qrt_insns("roialign_hwc", t1 - t0);
  qrt_store(argv[3], out, 4 * m);
  return 0;
}
