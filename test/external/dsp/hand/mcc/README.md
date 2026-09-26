# MCC decoder steps as a tinygrad oracle -- in progress

## What this is meant to be

`test_mcc.py` was a `pytest.mark.xfail(strict=True, run=False)` placeholder. Its own docstring
spelled out what would make it real: capture golden outputs **on the phone**, commit them, and
compare tinygrad's lowering of the same step against those bytes under the family's contract
(fp16-level: the decoder's occupancy logits within 0.045 of float64).

The blocker was stated plainly: *"this pass had no phone access."* The phone (Xiaomi 12S,
`239dbd8f`) is available now.

## What is here

- `mb_hvx.h`, `mcc_block.h`, `mcc_decoder.h` -- the three hand-written kernel headers, moved here
  from onnxsim's `scripts/android/mcc_hmx`. Third chunk of the hand-kernel consolidation.
- `mcc_golden_{client,impl}.c` + `mcc_golden_rpc.idl` -- a FastRPC driver that runs `mb_hvx.h`'s
  steps on the device and writes the output tiles out. This is what the missing phone run needed.
- `build.sh` / `run.sh` -- build and phone-run drivers. **Use the phone lock**:
  `PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh ...`, and keep to
  `/data/local/tmp/<branch>/` on the device.
- `goldens/mcc_r4/` -- 832 KB of real captures: inputs (`x.bin`, `k.bin`, `s.bin`, `q.bin`,
  `g.bin`, `ln.bin`), the hand kernel's outputs (`gold_h.bin`, `gold_spv.bin`, `gold_gelu.bin`),
  the float64 references (`ref_*.bin`), and `gold_times.txt` (pcycles) plus `case.json`.
- `mcc_case.py` / `tg_mcc.py` -- the case generator and tinygrad's lowering of the same steps.

## Current state: NOT a working oracle yet

`test_mcc.py` is still the original `xfail`. The comparison harness was written and run during
this work and is **not** finished:

| step | state |
|---|---|
| LayerNorm | was passing against the phone bytes in the last run |
| base-2 softmax + self term | fails |
| GELU | fails |

Two faults were identified and **not** fixed when the work ran out of turns:

1. the skel's in-place restore wiped the result before it was copied out, and
2. GELU's output was never returned.

A later pass also flagged a padding issue in the softmax reference and a `Tensor(Tensor)`
misuse on the tinygrad side. All of that is unfinished.

**Do not treat this as verified.** The goldens are genuine device output, so the *comparison* is
honest -- but the work on top of it is not done, and the committed test still skips rather than
pretending otherwise. Re-run the three steps before relying on any of it.

## The constraint that shaped it

These are V69 **qfloat** kernels: every step computes in qf16/qf32 and the rounding is the
phone's. qemu 8.2 cannot decode qfloat at all, and hexagon-sim runs V69's IEEE-named `.sf` vector
ops as IEEE fp32 where the hardware computes qf32. That is not a theoretical concern -- the fork's
exact-requant oracles were bit-exact on the simulator and 94% wrong on the phone for exactly this
reason. So the goldens have to come from the device, and they did; the tolerance must not be
widened to make anything pass.
