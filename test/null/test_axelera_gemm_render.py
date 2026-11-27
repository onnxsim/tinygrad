"""AxelerScript renderer for the tiled int8 GEMM (tinygrad/renderer/axelera.py). No device needed.

Fixtures are the text scripts/axelera/pipe_unroll.py emits for OVERLAP=1, OUTSLICE=1, NSLOTS=2.
"""
import pathlib, unittest
from tinygrad.renderer.axelera import render_gemm, schedule_tiled_gemm, tile_plan

FIX = pathlib.Path(__file__).parent

class TestAxelerascriptGemm(unittest.TestCase):
  def test_tile_plan_alternates_slots_and_covers_rows(self):
    plan = tile_plan(1024, 8)
    self.assertEqual([t.slot for t in plan], [0, 1, 0, 1, 0, 1, 0, 1])
    self.assertEqual((plan[0].row0, plan[-1].row1), (0, 1024))
    self.assertTrue(all(a.row1 == b.row0 for a, b in zip(plan, plan[1:])))

  def test_tinygrad_fuses_tiles_into_one_kernel(self):
    # tinygrad's scheduler has no notion of tile slots or waits: the whole tiled graph is one compute kernel.
    # copy-call counts differ across tinygrad versions, so only the compute-kernel count is asserted
    compute, _ = schedule_tiled_gemm(1024, 512, 512, 8)
    self.assertEqual(compute, 1)

  def test_render_matches_pipe_unroll_1024x512x512_nt8(self):
    expected = (FIX / "axelera_gemm_1024x512x512_nt8.axs").read_text()
    self.assertEqual(render_gemm(1024, 512, 512, 8), expected)

  def test_render_matches_pipe_unroll_2048x512x512_nt16(self):
    expected = (FIX / "axelera_gemm_2048x512x512_nt16.axs").read_text()
    self.assertEqual(render_gemm(2048, 512, 512, 16), expected)

if __name__ == "__main__":
  unittest.main()
