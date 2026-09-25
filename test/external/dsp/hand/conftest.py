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
  cc = os.environ.get("CC") or clang()
  if cc is not None and "-ffp-contract" not in cc: os.environ["CC"] = f"{cc} -ffp-contract=off"
elif clang() is not None: os.environ.setdefault("CC", clang())
