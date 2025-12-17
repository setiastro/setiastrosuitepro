from __future__ import annotations
import contextlib
import os

# Resolve a single "torch-like" object for the whole app
# Try your preferred/backed build order first.
_TORCH = None
_err = None

# If you vendor or rename your build, try that FIRST (example):
# try:
#     import mybundled.torch as torch
#     _TORCH = torch
# except Exception as e:
#     _err = e

if _TORCH is None:
    try:
        import torch  # system/packaged torch
        _TORCH = torch
    except Exception as e:
        _err = e

# Optional: DirectML fallback on Windows (comment out if not needed)
if _TORCH is None:
    try:
        import torch_directml as torch  # pip install torch-directml
        _TORCH = torch
    except Exception:
        pass

def has_torch() -> bool:
    return _TORCH is not None

def torch_module():
    """Return the torch module or None."""
    return _TORCH

def pick_device():
    """Pick best available device. Returns None if no torch."""
    if _TORCH is None:
        return None
    try:
        if hasattr(_TORCH, "cuda") and _TORCH.cuda.is_available():
            return _TORCH.device("cuda")
    except Exception:
        pass
    try:
        mps = getattr(getattr(_TORCH, "backends", None), "mps", None)
        if mps and getattr(mps, "is_available", lambda: False)():
            return _TORCH.device("mps")
    except Exception:
        pass
    return _TORCH.device("cpu")

def no_grad_decorator():
    """
    Returns a decorator:
     • If torch exists, returns torch.no_grad()
     • Else, identity decorator
    """
    if _TORCH and hasattr(_TORCH, "no_grad"):
        return _TORCH.no_grad()
    def _identity(fn): return fn
    return _identity

def inference_ctx():
    """
    Returns a context manager for inference if available (torch.inference_mode),
    else a no-op context manager.
    """
    if _TORCH and hasattr(_TORCH, "inference_mode"):
        return _TORCH.inference_mode()
    return contextlib.nullcontext()

def free_torch_memory():
    """Best-effort GPU memory cleanup."""
    if _TORCH is None:
        return
    try:
        if hasattr(_TORCH, "cuda") and hasattr(_TORCH.cuda, "empty_cache"):
            _TORCH.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(_TORCH, "mps") and hasattr(_TORCH.mps, "empty_cache"):
            _TORCH.mps.empty_cache()
    except Exception:
        pass
