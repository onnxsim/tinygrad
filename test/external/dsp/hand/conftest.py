# tinygrad reads DEV/MOCKDSP at import: every hand-kernel oracle compares against tinygrad's DSP backend under
# qemu-hexagon (MOCKDSP=1), so set them before any test module imports tinygrad.
import os, sys, pathlib

os.environ.setdefault("DEV", "DSP")
os.environ.setdefault("MOCKDSP", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import clang  # noqa: E402

# DSPCompiler invokes $CC: it has to be the Hexagon-capable clang the harness found
if clang() is not None: os.environ.setdefault("CC", clang())
