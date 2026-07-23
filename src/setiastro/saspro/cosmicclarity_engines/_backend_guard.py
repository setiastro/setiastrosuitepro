# src/setiastro/saspro/cosmicclarity_engines/_backend_guard.py
#
# SetiAstro Suite Pro
# Copyright (c) Franklin Marek / www.setiastro.com
#
# Shared GPU backend guard for the Cosmic Clarity engines.
#
# A D3D12 device removal is permanent and PROCESS-WIDE. If the denoise engine
# loses the adapter, the sharpen engine must not then load models onto it.
# That is the entire reason this state lives here and not inside any one engine.
from __future__ import annotations

import os

__all__ = [
    "DeviceLostError", "EngineCancelled",
    "safe_exception_str",
    "is_device_lost_error", "is_backend_unusable_error", "is_recoverable_batch_error",
    "device_type", "dml_disabled", "disable_dml", "reset_dml_disabled",
    "prefer_ort_dml", "tick", "evict_models", "iter_batch_sizes",
    "cap_chunk_size_for_device",
]


# Deliberately private. Always read through dml_disabled() — a
# `from _backend_guard import _DML_DISABLED` would copy the value at import
# time and never observe the update.
_DML_DISABLED = False
_DML_DISABLED_REASON = ""


class DeviceLostError(RuntimeError):
    """The GPU device was removed or hung. Nothing on it is recoverable."""


class EngineCancelled(RuntimeError):
    """User cancelled via a progress callback. Must never be swallowed."""


def safe_exception_str(err: BaseException) -> str:
    """str() an exception without dying on non-UTF-8 driver text.
    Japanese and Dutch Windows both emit these."""
    try:
        return f"{type(err).__name__}: {err}"
    except Exception:
        pass
    try:
        return f"{type(err).__name__}: {str(err).encode('utf-8', errors='replace').decode('utf-8')}"
    except Exception:
        return type(err).__name__


# Unambiguous device loss. A hit here means the adapter is gone for the life of
# the process, and DirectML gets poisoned for every engine.
_HARD_DEVICE_LOST = (
    "887a0005",   # DXGI_ERROR_DEVICE_REMOVED
    "887a0006",   # DXGI_ERROR_DEVICE_HUNG
    "887a0007",   # DXGI_ERROR_DEVICE_RESET
    "887a0020",   # DXGI_ERROR_DRIVER_INTERNAL_ERROR
    "getdeviceremovedreason",
    "will not respond to more commands",   # the AMD TDR text
    "re-create the device",
    "device removed",
    "device lost",
    "device hung",
    "device reset",
    "gpu-apparaat",     # nl-NL
    "onderbroken",      # nl-NL
)

# Ambiguous. These appear on genuine device loss, but ALSO on ordinary OOM and
# on unsupported-op dispatch failures. Enough to abandon the GPU path for this
# run — not enough to disable DirectML for the whole session.
_SOFT_BACKEND_FAIL = (
    "dmlexecutionprovider",
    "executionprovider.cpp",
    "privateuseone",
    "private use",
    "dxgi",
    "887a0001",   # DXGI_ERROR_INVALID_CALL
)

_MEMORY_PRESSURE = (
    "out of memory", "insufficient memory", "bad allocation",
    "allocation", "alloc", "oom",
)


def is_device_lost_error(err: BaseException) -> bool:
    """Hard device loss only. Triggers session-wide DirectML poisoning."""
    if isinstance(err, DeviceLostError):
        return True
    if isinstance(err, EngineCancelled):
        return False
    s = safe_exception_str(err).lower()
    return any(n in s for n in _HARD_DEVICE_LOST)


def is_backend_unusable_error(err: BaseException) -> bool:
    """Soft failure — this backend can't complete the work, but the device is
    probably still alive. Fall back to CPU without poisoning DirectML."""
    if isinstance(err, (DeviceLostError, EngineCancelled)):
        return False
    s = safe_exception_str(err).lower()
    if any(n in s for n in _MEMORY_PRESSURE):
        return False          # that's a batch-size problem, let the retry loop own it
    return any(n in s for n in _SOFT_BACKEND_FAIL)


def is_recoverable_batch_error(err: BaseException) -> bool:
    """True only if shrinking the batch could plausibly help."""
    if isinstance(err, (DeviceLostError, EngineCancelled)):
        return False
    if is_device_lost_error(err):
        return False          # a smaller batch cannot resurrect a removed device
    msg = safe_exception_str(err).lower()
    needles = (
        "out of memory", "cuda", "cudnn", "cublas",
        "directml", "dml", "mps",
        "allocation", "alloc", "resource",
        "execution provider",
        "insufficient memory", "bad allocation", "memory",
        "unknown error",      # torch-directml's catch-all when it runs dry
    )
    return any(n in msg for n in needles)


def device_type(device) -> str:
    """Normalize a device to a short tag. torch-directml reports
    'privateuseone'; the ONNX bundles carry the string 'DirectML'."""
    t = getattr(device, "type", None) or str(device)
    t = str(t).strip().lower()
    if t == "privateuseone" or t.startswith("directml"):
        return "dml"
    return t


def dml_disabled() -> bool:
    return bool(_DML_DISABLED)


def disable_dml(reason: str = "", status_cb=print) -> None:
    global _DML_DISABLED, _DML_DISABLED_REASON
    if _DML_DISABLED:
        return
    _DML_DISABLED = True
    _DML_DISABLED_REASON = str(reason)
    try:
        status_cb(f"[CC] DirectML disabled for this session. {reason}")
    except Exception:
        pass


def reset_dml_disabled() -> None:
    """Manual troubleshooting only. A removed D3D12 device does not come back
    within the same process — this exists for the case where the flag was set
    by a misclassified error, not for real recovery."""
    global _DML_DISABLED, _DML_DISABLED_REASON
    _DML_DISABLED = False
    _DML_DISABLED_REASON = ""


def prefer_ort_dml() -> bool:
    """Set SASPRO_CC_PREFER_ORT_DML=1 to route DirectML through ONNX Runtime
    instead of torch-directml. ORT's DML EP has better op coverage and is
    markedly more robust on AMD. Zero-risk A/B for a user hitting device hangs."""
    return os.environ.get("SASPRO_CC_PREFER_ORT_DML", "").strip().lower() in ("1", "true", "yes", "on")


def tick(progress_cb, done: int, total: int, stage: str | None = None) -> None:
    """Report progress and honour cancellation.

    The old inline pattern raised RuntimeError("Cancelled.") inside a
    `try: ... except Exception: pass`, or inside a batch-retry `except`, so a
    cancel was either discarded or laundered through the batch machinery.
    Here only genuine callback errors are ignored."""
    if progress_cb is None:
        return
    try:
        if stage is None:
            cont = progress_cb(int(done), int(total))
        else:
            try:
                cont = progress_cb(int(done), int(total), str(stage))
            except TypeError:
                cont = progress_cb(int(done), int(total))
    except EngineCancelled:
        raise
    except Exception:
        return
    if cont is False:
        raise EngineCancelled("Cancelled.")


_MODEL_ATTRS = (
    "mono", "color",                                  # denoise
    "stellar", "ns1", "ns2", "ns4", "ns8", "ns_cond", # sharpen
    "correct", "correct_v2",
    "mono_model", "color_model",                      # darkstar
    "detection_model1", "detection_model2",           # satellite
    "removal_model", "tfm",
    "torch",
)


def evict_models(cache: dict, models, *, tag: str = "CC", status_cb=print) -> None:
    """Drop a dead bundle from an engine's cache and release its references.

    After device removal the parameter tensors live on a device that no longer
    exists. Freeing them is never guaranteed safe — but doing it HERE, at a
    controlled point before we allocate the CPU models, beats letting a random
    GC or interpreter shutdown do it inside the DML allocator."""
    if models is None:
        return
    try:
        dead = [k for k, v in cache.items() if v is models]
        for k in dead:
            cache.pop(k, None)
        status_cb(f"[{tag}] Evicted {len(dead)} cached bundle(s) for the lost device.")
    except Exception:
        pass

    # Satellite caches plain dicts rather than a dataclass bundle.
    if isinstance(models, dict):
        for attr in _MODEL_ATTRS:
            try:
                if attr in models:
                    models[attr] = None
            except Exception:
                pass
    else:
        for attr in _MODEL_ATTRS:
            try:
                if hasattr(models, attr):
                    setattr(models, attr, None)
            except Exception:
                pass

    # Highest-risk line in this module: a native fault would land here if
    # torch-directml cannot free a removed device's allocations. Deleting the
    # call does not remove the risk, it only defers it to an arbitrary GC.
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def iter_batch_sizes(initial: int):
    bs = max(1, int(initial))
    yielded = set()
    while bs >= 1:
        if bs not in yielded:
            yielded.add(bs)
            yield bs
        if bs == 1:
            break
        bs = max(1, bs // 2)


def cap_chunk_size_for_device(device, chunk_size: int, *, max_dml: int = 256) -> int:
    """DirectML runs against the Windows TDR watchdog: one dispatch past
    TdrDelay and the D3D12 device is lost permanently. This is a LATENCY
    budget, not a VRAM budget — a 16GB card does not help."""
    if device_type(device) != "dml":
        return int(chunk_size)
    return int(min(int(chunk_size), int(max_dml)))