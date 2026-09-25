/* Oracle driver for roialign_u8_kernel.h (uint8 NHWC FPN map -> quantized RoiAlign rows, one level).
 * argv: map rois | out | H W C R OH OW sr z_in z_out mult shift scale_bits
 * scale_bits: the level's spatial_scale as float32 bits; mult/shift from ru8_requant_params (host side).
 * Built with 128-byte HVX: the shipped integer HVX body (vzxt + vmpyacc), which qemu 8.2 decodes. */
#include "qemu_rt.h"
#include "roialign_u8_kernel.h"
int oracle_main(int argc, char** argv) {
  if (argc != 16) { qrt_puts("usage: map rois out H W C R OH OW sr z_in z_out mult shift scale_bits\n"); return 2; }
  int H = qrt_atoi(argv[4]), W = qrt_atoi(argv[5]), C = qrt_atoi(argv[6]), R = qrt_atoi(argv[7]);
  int OH = qrt_atoi(argv[8]), OW = qrt_atoi(argv[9]), sr = qrt_atoi(argv[10]), z_out = qrt_atoi(argv[12]);
  union { int i; float f; } ss = {qrt_atoi(argv[15])};
  ru8_level_t L = {qrt_load(argv[1], (long)H * W * C), H, W, ss.f, qrt_atoi(argv[11]), qrt_atoi(argv[13]), qrt_atoi(argv[14])};
  float* rois = qrt_load(argv[2], 16L * R);
  long row = (long)OH * OW * C;
  uint8_t* out = qrt_map(R * row);
  unsigned t0 = qrt_inscount();
  for (int r = 0; r < R; r++) ru8_roi(&L, C, rois + 4 * r, OH, OW, sr, z_out, 0, out + r * row);
  unsigned t1 = qrt_inscount();
  qrt_insns("roialign_u8", t1 - t0);
  qrt_store(argv[3], out, R * row);
  return 0;
}
