"""
GPU Configuration for Windows + CUDA.

Run: python gpu_config.py --check  to verify GPU setup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def check_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(
                "   VRAM: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            print("[WARN] CUDA not available - running on CPU")
            print("   Install GPU version: pip install -r requirements-gpu.txt")
    except ImportError:
        print("[WARN] PyTorch not installed")


def _add_windows_dll_dir(path: Path) -> None:
    """Make native DLL dependencies visible on Windows/Python 3.8+."""
    if sys.platform != "win32" or not path.exists():
        return
    try:
        os.add_dll_directory(str(path))
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def configure_gpu_for_project() -> bool:
    """Call this before loading any models."""
    try:
        import torch

        # llama-cpp-python CUDA wheels depend on CUDA/PyTorch DLLs. Make the
        # torch DLL directory visible before any llama_cpp import happens.
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        _add_windows_dll_dir(torch_lib)

        if torch.cuda.is_available():
            # Enable TF32 acceleration on Ampere GPUs (RTX 30xx/40xx) and newer.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Flash Attention / scaled-dot-product attention if available.
            torch.backends.cuda.enable_flash_sdp(True)
            return True
    except ImportError:
        pass
    return False


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_gpu()
    else:
        check_gpu()
