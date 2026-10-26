"""Step 0: check that the machine can run the pipeline.

Run this first. It says in plain words what is missing, instead of letting a
long extraction or training job fail an hour in. Every check runs even if an
earlier one fails, so you see all problems at once.
"""

import importlib
import shutil
import sys

import _common  # noqa: F401  (adds src/ to the path)

ok = True


def check(name, fn, required=True):
    global ok
    try:
        msg = fn()
        print(f"  [ok]   {name}{': ' + msg if msg else ''}")
    except Exception as e:
        if required:
            ok = False
        print(f"  [{'FAIL' if required else 'warn'}] {name}: {e}")


def need_python():
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is needed")


def torch_check():
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU (fine for tests and the demo; training needs a GPU)"
    return f"{torch.__version__}, {gpu}"


def nemo_check():
    import nemo
    from nemo.collections.asr.models import EncDecFrameClassificationModel, EncDecSpeakerLabelModel  # noqa: F401

    return nemo.__version__


print("Python", sys.version.split()[0])
check("python >= 3.10", need_python)
for mod in ["numpy", "scipy", "soundfile", "yaml", "matplotlib", "pandas"]:
    check(mod, lambda m=mod: getattr(importlib.import_module(m), "__version__", ""))
check("torch", torch_check)
check("ctxdiar package (src/)", lambda: importlib.import_module("ctxdiar").__version__)
check("nemo_toolkit[asr] (MarbleNet, TitaNet; needed from step 3 on)", nemo_check, required=False)
check("pyannote.metrics (optional DER cross-check)", lambda: importlib.import_module("pyannote.metrics").__version__, required=False)
check("ffmpeg or sox (optional, for converting audio)",
      lambda: shutil.which("ffmpeg") or shutil.which("sox") or (_ for _ in ()).throw(RuntimeError("not found; only needed if your audio is not 16 kHz mono wav")),
      required=False)

print("\nAll required checks passed." if ok else "\nFix the [FAIL] lines above (README, 'Installation').")
sys.exit(0 if ok else 1)
