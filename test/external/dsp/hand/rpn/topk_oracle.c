/* Oracle driver for topk_kernel.h (exact 1-D TopK, largest, sorted, ORT's tie order).
 * argv: x | out_vals out_idx | n k
 * Runs the shipped HVX variant (2: vec-rot) and the scalar collect (0); both must agree, the vec-rot output is
 * written. Variant 3 (vec-reduce_or-mask) is a known qemu 8.2 HVX miscompile reproducer, not run here. */
#include <stddef.h>
#include "qemu_rt.h"
#include "topk_kernel.h"
int oracle_main(int argc, char** argv) {
  if (argc != 6) { qrt_puts("usage: x out_vals out_idx n k\n"); return 2; }
  int n = qrt_atoi(argv[4]), k = qrt_atoi(argv[5]);
  float* x = qrt_load(argv[1], 4L * n);
  uint32_t* scr = qrt_map(4L * TK_SCRATCH_WORDS(n));
  float *ov = qrt_map(4L * k + 4), *sv = qrt_map(4L * k + 4);
  int64_t *oi = qrt_map(8L * k + 8), *si = qrt_map(8L * k + 8);
  unsigned t0 = qrt_inscount();
  int c = topk_desc(x, n, k, sv, si, scr, 0);
  unsigned t1 = qrt_inscount();
  qrt_insns("scalar", t1 - t0);
  t0 = qrt_inscount();
  int c2 = topk_desc(x, n, k, ov, oi, scr, 2);
  t1 = qrt_inscount();
  qrt_insns("vec-rot", t1 - t0);
  if (c < 0 || c != c2 || memcmp(ov, sv, 4L * k) || memcmp(oi, si, 8L * k)) { qrt_puts("scalar/vec-rot disagree\n"); return 1; }
  qrt_store(argv[2], ov, 4L * k);
  qrt_store(argv[3], oi, 8L * k);
  return 0;
}
