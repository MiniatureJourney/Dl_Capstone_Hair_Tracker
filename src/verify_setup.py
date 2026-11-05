"""
verify_setup.py
---------------
Checks that all required libraries are installed correctly
and that the GPU (RTX 5060) is accessible via PyTorch CUDA.
Run this after installing packages to confirm the environment is ready.
"""

import sys
import importlib
import os

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REQUIRED = [
    "torch", "torchvision", "cv2", "PIL", "numpy",
    "albumentations", "torchmetrics", "customtkinter",
    "tqdm", "matplotlib", "pandas", "skimage", "fpdf",
]

print("=" * 60)
print("  DL Capstone -- Environment Verification")
print("=" * 60)

# Check Python
print(f"\n[Python] {sys.version}")

# Check libraries
print("\n[Libraries]")
all_ok = True
for lib in REQUIRED:
    try:
        mod = importlib.import_module(lib)
        ver = getattr(mod, "__version__", "installed")
        print(f"  OK   {lib:<22} {ver}")
    except ImportError:
        print(f"  FAIL {lib:<22} NOT INSTALLED")
        all_ok = False

# Check PyTorch + CUDA
print("\n[GPU / CUDA]")
try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total      : {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA version    : {torch.version.cuda}")
        # Quick tensor test on GPU
        x = torch.randn(256, 256).cuda()
        y = x @ x.T
        print(f"  GPU compute test: PASSED (256x256 matmul on GPU)")
    else:
        print("  WARNING: CUDA not available. Training will use CPU (very slow).")
        print("  -> Re-run install_cuda.bat and check NVIDIA drivers.")
        all_ok = False
except Exception as e:
    print(f"  FAIL: PyTorch check failed: {e}")
    all_ok = False

# Summary
print("\n" + "=" * 60)
if all_ok:
    print("  ALL CHECKS PASSED -- Ready for Phase 2 (Dataset)")
else:
    print("  SOME CHECKS FAILED -- Fix the issues above first")
print("=" * 60)
