# Why the HMX-on-hexagon-sim oracle tests are quarantined — what is actually known

Supersedes the "not root-caused" note. Two separate findings, one of which corrects the other.

## 1. The kernel is fine. The test body is fine. pytest is what breaks it.

The 3x3 oracle was quarantined as *"hexagon-sim aborts (SIGABRT) on tinygrad's captured
kernel"*. That is not what happens. Established by reproduction:

- The captured 3x3 kernel on its own: `Insns=254,065,225 Pcycles=321,647,532`, **97 s, 150 MB
  peak, exit 0.** It completes.
- `run_captured` in isolation: **284,167 pcycles in 12.7 s**, correct output.
- The **whole `_run` body** — hand driver build + sim, then the tinygrad kernel + sim — in one
  process: **hand 10.8 s, tinygrad 12.8 s, exit 0.** Including inside a `TemporaryDirectory`,
  which is what the test uses.
- The **same body under pytest**: aborts. The abort lands in Python's `subprocess` frame inside
  `hexsim._sim` (line 95/102 → `run_captured`), i.e. the parent is waiting on the simulator when
  the process dies.

So the sequence is not the problem, the shape is not the problem, and the kernel is not the
problem. Something about running that sequence under pytest's process kills the simulator child.
Not yet identified — it needs a core dump of the *parent* (apport is currently swallowing them:
`core_pattern` is `|/usr/share/apport/apport ...`, and no core file is left behind).

**What would settle it:** set `kernel.core_pattern` to a plain path for one run and get a
backtrace of the parent at the `_sim` frame, or run the failing test under `gdb --args python -m
pytest ...`. I did not do this, so I am not claiming a cause.

## 2. The 16x16 case is also genuinely oversized — worth fixing regardless

`test_qconv3x3_s1` asks for 16x16x64 -> 64 stride 1. The kernel it produces is

```
for (Lidx3 = 0; Lidx3 < 320; Lidx3++)      // r64(16 * (16+2)) -- the padding ring is 2 of every 18 columns
  for (Lidx4 = 0; Lidx4 < 2;   Lidx4++)     // N/32 output tiles
    for (Ridx0 = 0; Ridx0 < 3;  Ridx0++)    // dy
      for (Ridx1 = 0; Ridx1 < 48; Ridx1++)  // dx * K, FLATTENED
```

`Ridx1 < 48` is `3 taps x 16 K-blocks` in one serial axis. The ResNet-18 3x3 convs that *work*
(`k81`, `r_16_..._3_3_16`) keep them separate — `Ridx0<3, Ridx1<3, Ridx2<16` — so the renderer
can keep weight tiles resident across taps. The flattened form does the same 144 reduce
iterations without that, which is why it is ~14x the stem in MAC count (92160 against 6464) and
why it is the case that strains the simulator.

Changing the cases to 32x32 does make them much cheaper standalone (12.7 s against 97 s for the
16x16 3x3), so it is worth doing on its own merits. It does **not** fix the pytest abort — the
32x32 case aborts under pytest too, in exactly the same place.

## What to do, in order

1. **Get the parent's core dump** and find why the simulator child dies under pytest. That is the
   actual bug, and it is the only thing standing between these five tests and being real.
2. Meanwhile, move the 3x3 oracle cases to 32x32 (`ORACLE_QUARANTINE.md` has the reasoning).
3. Then lift the quarantine on the 3x3 and whole-graph tests and keep the 1x1 ones running -
   **those already pass**, bit-exact, in 295 s, and the quarantine on them is wrong.

## The 1x1 family: passes here, aborts on the CI runner

`TestHandHmxQconv1x1` **passes on this machine** - 2 passed in 411 s, and under gdb, and with
either python. On the GitHub runner it aborts in `_sim` after ~5 minutes, in the same
`subprocess.run` frame as the 3x3 family:

    test_hand_hmx_gemm.py::TestHandHmxGemmF16::test_gemm_f16_small SKIPPED [ 90%]
    16:14:17 Fatal Python error: Aborted
      File "test/external/dsp/hand/hexsim.py", line 31 in _sim
      File "test/external/dsp/hand/hexsim.py", line 95 in run_captured
      File "test/external/dsp/hand/test_hand_hmx_qconv.py", line 47 in _run

It was un-quarantined on the strength of the local passes, and that turned the hexsim job red.
Re-quarantined: a test that only passes on the developer's machine is not a test the job can
trust, and leaving it in trades a reliably red job for a flaky one.

So the picture is now: everything runs and passes here (49 passed, 11 skipped, 642 s for the
hexsim tier's exact command), and the CI runner cannot complete `run_captured` for the HMX
kernels at all. That is a runner-environment problem, not a code one, and it is worth its own
investigation: the two data points to collect are the simulator child's exit status and its
stderr on the runner, and whether the abort is OOM (the 1x1 tile is 150 MB peak locally) rather
than anything to do with pytest.

## 2026-09-27: the diagnostics, and a correction to "runner-only"

`_sim` and `_cc` no longer use `check=True`. Both now report the child's return code, the **signal
name** if it was killed, its stderr, the tail of its stdout, and the exact command line. With
`check=True` pytest reported only `CalledProcessError`, which is why every one of these runs looked
like the same undifferentiated abort.

Two things that surfaced immediately.

**1. The abort is not runner-only, and it is not deterministic.** Lifting the quarantine on
`test_hand_hmx_qconv.py` locally:

| run | result |
|---|---|
| to a file | `2 passed, 2 skipped in 438 s`, **exit 0** |
| under a pty (`script -qec`, which is what the runner gives pytest) | `2 passed, 2 skipped in 427 s`, **exit 0** |
| to a pipe (`| grep`) | `2 passed, 2 skipped`, then **`Fatal Python error: Aborted (core dumped)`** |

So the same tests, the same tree, pass twice and abort once. That rules out the framing this
document previously used - "passes here, aborts on the runner" implies a difference *between* the
machines, and there does not seem to be one. It is more likely a race or an at-exit fault whose
timing depends on how long the ~7-minute run leaves the process in a given state.

**2. When it aborts, the tests have already passed.** The `Fatal Python error` frame is in
`pytest`'s own `wrap_session` at interpreter shutdown, well after `2 passed` is printed, and
`ulimit -c` is `unlimited` here, so the message is `Aborted (core dumped)`. This is a *parent*
abort at exit, not a child abort mid-test. That is consistent with the earlier observation that
"the abort lands in Python's `subprocess` frame" and with the suggestion that the fault is at
`hexagon-sim` teardown rather than during any kernel.

**Core dumps are unavailable either way.** `core_pattern` is
`|/usr/share/apport/apport -p%p -s%s -c%c -d%d -P%P -u%u -g%g -F%F -- %E` and apport is not
installed, so the kernel hands the core to a handler that is not there. `_core_dumps_enabled()`
reads `core_pattern` and says so in the failure message, so the next runner failure will say
directly that there is nothing to inspect rather than leaving it to be rediscovered.

**What this changes about the plan.** Item 1 ("get the parent's core dump") is still the right next
step, but it needs a runner where `core_pattern` is a plain path - neither the GitHub runner nor
this machine can produce one. Until then, the most likely cheap discriminators are: run the same
test under `gdb --args python -m pytest ...` (a core is not needed to catch SIGABRT), and
bisect on *what the parent was doing when it exited* - the 1x1 tile is 150 MB peak locally, so OOM
at teardown is still live and the runner may simply have less headroom than this box.

**The 1x1 quarantine is still correct.** Two local passes are not a test the job can trust, and
this is now known to be intermittent rather than environment-specific - which is a reason to fix
the fault, not a reason to lift the quarantine.

