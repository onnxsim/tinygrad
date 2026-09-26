# tinygrad reads DEV/MOCKDSP at import: every hand-kernel oracle compares against tinygrad's DSP backend (MOCKDSP=1), so
# set them before any test module imports tinygrad.
import os, sys, pathlib

os.environ.setdefault("DEV", "DSP")
os.environ.setdefault("MOCKDSP", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import clang  # noqa: E402

# DSPCompiler invokes $CC: it has to be a Hexagon-capable clang. The qemu families (HVX_ARCH < v68) also compile tinygrad's
# kernels with -ffp-contract=off, like the hand kernels and ORT: clang's default contracts a*b + c into an FMA, a different
# rounding (a RoiAlign's 4-tap sums differ in 5% of outputs by an ulp).
if int(os.environ.get("HVX_ARCH", "v65").lstrip("v")) < 68:
  # and without tinygrad's float reassociation ((x + c) + y -> (x + y) + c, (x * c) * y -> (x * y) * c): the hand kernels'
  # contracts are ORT's exact float op order
  os.environ.setdefault("FLOAT_REASSOC", "0")
  cc = os.environ.get("CC") or clang()
  if cc is not None and "-ffp-contract" not in cc: os.environ["CC"] = f"{cc} -ffp-contract=off"
else:
  if clang() is not None: os.environ.setdefault("CC", clang())
  # The qemu (v65) families don't run under the HMX families' HVX_ARCH=v69: don't collect them at all (instead of a
  # skip per test), so an HMX run reports only HMX results. `mcc` is the exception: its test_mcc.py holds no
  # tinygrad-built kernel (it is a phone-golden oracle, mb_hvx.h's own steps against bytes captured off the
  # device), so it runs on whatever host device it names itself and is the one test here that an HMX run can
  # add. Everything else in the list compiles for HVX and belongs to the v65 tier.
