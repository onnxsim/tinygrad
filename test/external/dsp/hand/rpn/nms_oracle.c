/* Oracle driver for nms_kernel.h (ORT NonMaxSuppression, one batch/class, center_point_box=0).
 * argv: boxes scores | out_sel out_count out_suppress | n max_out thr_bits
 * thr_bits: the IoU threshold's float32 bit pattern (no float parser here).
 * out_sel: the kept box indices in selection order (n int32, -1 past the count); out_count: 1 int32.
 * out_suppress: n*n uint8, [i][j] = SuppressByIOU(box i, box j) -- the pairwise test nms_scalar/nms_hvx decide with.
 * Built without HVX: nms_hvx's intrinsic path uses qfloat (V68+), which qemu 8.2 can't decode; its portable
 * nms_chunk runs instead -- the same keep decisions by construction, checked here against nms_scalar. */
#include <stddef.h>
#include "qemu_rt.h"
#include "nms_kernel.h"
int oracle_main(int argc, char** argv) {
  if (argc != 9) { qrt_puts("usage: boxes scores out_sel out_count out_suppress n max_out thr_bits\n"); return 2; }
  int n = qrt_atoi(argv[6]), max_out = qrt_atoi(argv[7]);
  union { int i; float f; } thr = {qrt_atoi(argv[8])};
  float *boxes = qrt_load(argv[1], 16L * n), *scores = qrt_load(argv[2], 4L * n);
  int *sel = qrt_map(4L * n + 4), *sel2 = qrt_map(4L * n + 4), *order = qrt_map(4L * n + 4), *tmp = qrt_map(4L * n + 4);
  float* soa = qrt_map(4L * NMS_SOA * ((n + 63) & ~63) + 128);
  unsigned t0 = qrt_inscount();
  int ns = nms_scalar(boxes, scores, n, thr.f, max_out, sel, order, tmp);
  unsigned t1 = qrt_inscount();
  qrt_insns("scalar", t1 - t0);
  t0 = qrt_inscount();
  int ns2 = nms_hvx(boxes, scores, n, thr.f, max_out, sel2, order, tmp, soa);
  t1 = qrt_inscount();
  qrt_insns("hvx_portable", t1 - t0);
  if (ns != ns2 || memcmp(sel, sel2, 4L * ns)) { qrt_puts("nms_scalar/nms_hvx disagree\n"); return 1; }
  for (int i = ns; i < n; i++) sel[i] = -1;
  qrt_store(argv[3], sel, 4L * n);
  qrt_store(argv[4], &ns, 4);
  unsigned char* s = qrt_map((long)n * n);
  t0 = qrt_inscount();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) s[(long)i * n + j] = (unsigned char)nms_suppress_exact(boxes + 4 * i, boxes + 4 * j, thr.f);
  t1 = qrt_inscount();
  qrt_insns("suppress_matrix", t1 - t0);
  qrt_store(argv[5], s, (long)n * n);
  return 0;
}
