# Why the HMX-on-hexagon-sim oracle tests were quarantined

**Update 2026-09-26: root-caused, and it is not what the quarantine note said.**

The quarantined tests do not crash because of a bug in the generated kernel. They are
*correct but enormous*, and the simulator cannot finish them inside the test's time budget.

## What the tests actually do

`test_hand_hmx_qconv.py::TestHandHmxQconv1x1` **passes** (295 s, bit-exact). So do the two
1x1 cases. Only the 3x3 family and the whole-graph tests fail, and the failure is not a crash in
the kernel:

- The captured 3x3 kernel run on its own: `Total: Insns=254,065,225 Pcycles=321,647,532`,
  **97 s and 150 MB peak**, and it **completes with exit 0**.
- Under pytest it aborts, because the process is being torn down while the simulation is still
  running (the abort lands in Python's `subprocess` frame, in `_sim`, not in the simulator).

So "hexagon-sim aborts (SIGABRT) on tinygrad's captured kernel" was the wrong diagnosis, and the
quarantine note said so much was unknown. What is actually wrong is the *shape*: the test asks
for a 16x16x64 -> 64 stride-1 conv, and gets a kernel with

```
for (Lidx3 = 0; Lidx3 < 320; Lidx3++)      // r64(16 * (16+2)) output pixels
  for (Lidx4 = 0; Lidx4 < 2;   Lidx4++)     // N / 32 output tiles
    for (Ridx0 = 0; Ridx0 < 3;  Ridx0++)    // dy
      for (Ridx1 = 0; Ridx1 < 48; Ridx1++)  // dx * K, FLATTENED into one axis
```

`Ridx1 < 48` is `3 taps x 16 K-blocks` collapsed into a single serial axis. The working
ResNet-18 3x3 convs (`k81` and siblings, `r_16_..._3_3_16`) keep them separate:

```
for (Ridx0 = 0; Ridx0 < 3; Ridx0++)   // dy
  for (Ridx1 = 0; Ridx1 < 3; Ridx1++) // dx
    for (Ridx2 = 0; Ridx2 < 16; Ridx2++)  // K, tiled
```

The flattened form still does 144 reduce iterations, but with no separate dx/dy axes the
renderer cannot keep the weight tiles resident across taps the way the nested form does - and
that is worth a lot, given the VTCM tile cache is carrying 6.9 ms on the real phone
(see PHONE_BISECT.md).

The test case is 14.3x larger in MAC count than the ResNet stem (`320*2*3*48 = 92160` against
`202*2*16 = 6464`), which is why it is the one that falls off the simulator.

## What this means for the tests

They should not be quarantined for a kernel bug - there is no kernel bug. Two things are worth
doing, in order:

1. **Make the oracle cases the size the kernels are actually meant for.** A 16x16 conv is
   degenerate for a tile-based kernel: 320 output pixels is five 64-row panels, and the padding
   ring (`Wp = W + 2 = 18`) costs more than the image. Real shapes - the 32x64 and 64x64 cases the
   HMX README quotes - exercise the same code at a size the simulator can run.
2. **Then lift the quarantine** for the 3x3 and whole-graph tests, and let the renderer's dx/dy
   nesting be checked by a case that can actually complete.

Until then the quarantine stays, but the reason is now "this case does not finish in the
simulator", not "this kernel is broken".
