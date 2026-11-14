#saspro.free_torch_memory.py

def trim_process_memory():
    """Clear Qt pixmap cache, run GC, and ask glibc to return pages (Linux).

    Call after dropping large numpy/QPixmap references so RSS can shrink
    and freed arenas become reusable by the process.
    """
    try:
        from PyQt6.QtGui import QPixmapCache
        QPixmapCache.clear()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import sys
        if sys.platform.startswith("linux"):
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _free_torch_memory():
    """
    Try to return GPU memory to the driver. Works for CUDA, MPS (Apple) and XPU.
    No-op if torch is unavailable. Always runs gc at the end.
    """
    try:
        import gc
        try:
            import torch
        except Exception:
            gc.collect()
            return

        # sync any in-flight work so frees actually happen
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        # free per-backend caches
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # collect stale IPC handles (helps after many tensors)
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:
            pass

        # (DirectML has no public empty_cache; rely on GC)
        gc.collect()
    except Exception:
        pass

    try:
        trim_process_memory()
    except Exception:
        pass
