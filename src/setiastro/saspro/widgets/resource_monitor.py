# src/setiastro/saspro/widgets/resource_monitor.py
"""
System resource monitor widget for SASpro.

Architecture:
  ResourceBackend  — QObject exposed to QML; owns the polling logic.
  GPUWorker        — QThread that composes 1..N GPUBackend probes.
  GPUBackend       — Abstract vendor/OS-specific GPU sampler; concrete impls:
      * NvmlBackend         — NVIDIA via pynvml (native, no subprocess)
      * NvidiaSmiBackend    — NVIDIA fallback if pynvml unavailable
      * RocmBackend         — AMD on Linux via rocm-smi
      * MacOSBackend        — macOS via ioreg IOAccelerator (sudoless)
      * WindowsBackend      — Windows via PowerShell + CIM (last resort)

Each backend returns a GPUSample with util%, VRAM used/total, and device name
where available. The worker combines samples across all successful backends
(max util, first-non-None mem/name) so multi-GPU machines report sensibly.
"""
from __future__ import annotations

import os
import re
import sys
import time
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import psutil

from PyQt6.QtCore import (
    Qt, QTimer, QObject, QSettings, QSize, QRectF,
    QVariantAnimation, QEasingCurve,
    pyqtProperty, pyqtSignal, QThread,
)
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QDockWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy, QApplication,
)

from setiastro.saspro.memory_utils import get_memory_usage_mb


# ─── GPU data model ────────────────────────────────────────────────────────────

@dataclass
class GPUSample:
    """One point-in-time GPU snapshot from a single backend.

    Any field may be None if that backend can't provide it.
    """
    util_pct: Optional[float] = None      # 0..100
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    name: Optional[str] = None
    temp_c: Optional[float] = None        # reserved for future use


# ─── GPU backends ──────────────────────────────────────────────────────────────

class GPUBackend:
    """Abstract vendor/OS-specific GPU sampler."""
    name: str = "unknown"

    def probe(self) -> bool:
        """Return True if this backend can operate on the current system.

        MUST NOT raise. May do first-time init (open handles, cache header,
        etc.). Backends that fail probe are dropped from the worker.
        """
        return False

    def sample(self) -> Optional[GPUSample]:
        """Return a fresh sample, or None if this call failed.

        SHOULD NOT raise. Callers treat None as "no data this tick" and keep
        the previous value on screen.
        """
        return None

    def close(self) -> None:
        """Release handles. Optional."""
        return None


class NvmlBackend(GPUBackend):
    """NVIDIA via pynvml — native calls, no subprocess spawn per poll.

    Preferred for NVIDIA GPUs. Requires `pip install nvidia-ml-py` (aka
    pynvml). Falls through to NvidiaSmiBackend if unavailable.
    """
    name = "nvidia-nvml"

    def __init__(self):
        self._pynvml = None
        self._device = None
        self._device_name: Optional[str] = None
        self._mem_total_mb: Optional[float] = None

    def probe(self) -> bool:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            # For multi-GPU, take index 0 as the "primary". A future
            # enhancement could enumerate all devices and combine.
            self._device = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(self._device)
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode("utf-8", errors="ignore")
            self._device_name = str(raw_name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._device)
            self._mem_total_mb = mem.total / (1024.0 * 1024.0)
            return True
        except Exception:
            return False

    def sample(self) -> Optional[GPUSample]:
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._device)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._device)
            return GPUSample(
                util_pct=float(util.gpu),
                mem_used_mb=mem.used / (1024.0 * 1024.0),
                mem_total_mb=self._mem_total_mb,
                name=self._device_name,
            )
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        except Exception:
            pass


class NvidiaSmiBackend(GPUBackend):
    """Fallback for boxes without pynvml. Spawns nvidia-smi per poll.

    Kept because pynvml is optional (some users install SASpro without an
    NVIDIA-specific extra), and Optimus laptops with the dGPU powered off
    can have nvidia-smi available while nvmlInit fails.
    """
    name = "nvidia-smi"

    def __init__(self):
        self._device_name: Optional[str] = None

    @staticmethod
    def _startupinfo_hidden():
        if os.name != "nt":
            return None
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return si

    def probe(self) -> bool:
        if not shutil.which("nvidia-smi"):
            return False
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                startupinfo=self._startupinfo_hidden(),
                timeout=2.0, stderr=subprocess.DEVNULL,
            )
            self._device_name = out.decode("utf-8", "ignore").strip().split("\n")[0]
            return True
        except Exception:
            return False

    def sample(self) -> Optional[GPUSample]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                startupinfo=self._startupinfo_hidden(),
                timeout=1.0, stderr=subprocess.DEVNULL,
            )
            row = out.decode("utf-8", "ignore").strip().split("\n")[0]
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 3:
                return None
            return GPUSample(
                util_pct=float(parts[0]),
                mem_used_mb=float(parts[1]),
                mem_total_mb=float(parts[2]),
                name=self._device_name,
            )
        except Exception:
            return None


class RocmBackend(GPUBackend):
    """AMD GPU on Linux via rocm-smi.

    Header phrasing shifts across ROCm releases ("GPU use (%)",
    "GPU_use_(%)", "GPU use %"), so we resolve column indices by substring
    match on the first successful call and cache them.
    """
    name = "rocm"

    def __init__(self):
        self._use_col: Optional[int] = None
        self._mem_used_col: Optional[int] = None
        self._mem_total_col: Optional[int] = None
        self._broken = False  # set if rocm-smi disappears mid-session

    def probe(self) -> bool:
        # rocm-smi doesn't exist on Windows or macOS; skip probe there.
        if os.name == "nt" or sys.platform == "darwin":
            return False
        return shutil.which("rocm-smi") is not None

    def _resolve_columns(self, header: list[str]) -> None:
        for i, h in enumerate(header):
            hl = h.lower()
            if self._use_col is None and "use" in hl and "%" in hl:
                self._use_col = i
            if self._mem_used_col is None and "used" in hl and ("mem" in hl or "vram" in hl):
                self._mem_used_col = i
            if self._mem_total_col is None and "total" in hl and ("mem" in hl or "vram" in hl):
                self._mem_total_col = i

    def sample(self) -> Optional[GPUSample]:
        if self._broken:
            return None
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
                timeout=1.5, stderr=subprocess.DEVNULL,
            )
            lines = [l for l in out.decode("utf-8", "ignore").strip().splitlines()
                     if l.strip()]
            if len(lines) < 2:
                return None
            header = [h.strip().strip('"') for h in lines[0].split(",")]
            self._resolve_columns(header)

            utils: list[float] = []
            mem_used_bytes: Optional[float] = None
            mem_total_bytes: Optional[float] = None
            for row in lines[1:]:
                cols = [c.strip().strip('"') for c in row.split(",")]
                if self._use_col is not None and self._use_col < len(cols):
                    try:
                        utils.append(float(cols[self._use_col]))
                    except ValueError:
                        pass
                if self._mem_used_col is not None and self._mem_used_col < len(cols):
                    try:
                        mu = float(cols[self._mem_used_col])
                        if mem_used_bytes is None or mu > mem_used_bytes:
                            mem_used_bytes = mu
                    except ValueError:
                        pass
                if self._mem_total_col is not None and self._mem_total_col < len(cols):
                    try:
                        mt = float(cols[self._mem_total_col])
                        if mem_total_bytes is None or mt > mem_total_bytes:
                            mem_total_bytes = mt
                    except ValueError:
                        pass
            return GPUSample(
                util_pct=(max(utils) if utils else None),
                mem_used_mb=(mem_used_bytes / (1024.0 * 1024.0)) if mem_used_bytes else None,
                mem_total_mb=(mem_total_bytes / (1024.0 * 1024.0)) if mem_total_bytes else None,
            )
        except FileNotFoundError:
            self._broken = True
            return None
        except Exception:
            return None


class MacOSBackend(GPUBackend):
    """macOS GPU via ioreg IOAccelerator — works sudoless on Apple Silicon
    (AGXAcceleratorGxx), Intel Macs (IntelAccelerator), and dGPU Macs (AMD
    subclasses). All GPU drivers inherit from IOAccelerator.

    QUIRK: "Device Utilization %" is an AVERAGE since the counter was last
    read by any process. The first read after boot can span hours and is
    meaningless — we discard it. Subsequent reads reflect ~1s of activity.
    """
    name = "macos-ioreg"

    def __init__(self):
        self._first_read_done = False
        self._device_name: Optional[str] = None

    def probe(self) -> bool:
        return sys.platform == "darwin"

    def sample(self) -> Optional[GPUSample]:
        try:
            out = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                timeout=2.0, stderr=subprocess.DEVNULL,
            )
            text = out.decode("utf-8", errors="ignore")

            # Utilization — multi-GPU systems have multiple IOAccelerator
            # entries; take the max.
            utils: list[float] = []
            for m in re.finditer(r'"Device Utilization %"\s*=\s*(\d+(?:\.\d+)?)', text):
                try:
                    utils.append(float(m.group(1)))
                except ValueError:
                    pass
            util = max(utils) if utils else 0.0

            if not self._first_read_done:
                self._first_read_done = True
                util = 0.0

            # Extract the driver class name (AGXAcceleratorG14, IntelAccelerator, ...)
            # from the "+-o <name> <class Cls, id ...>" header line.
            if self._device_name is None:
                m = re.search(r'\+-o\s+\S+\s+<class\s+([^,>\s]+)', text)
                if m:
                    self._device_name = m.group(1).strip()

            # NOTE: on Apple Silicon, memory is unified with system RAM, so
            # IOAccelerator's memory counters don't map to a "VRAM budget"
            # the way NVIDIA/AMD dGPUs do. Skip mem reporting on macOS.
            return GPUSample(util_pct=util, name=self._device_name)
        except Exception:
            return None


class WindowsBackend(GPUBackend):
    """Windows GPU utilization via PowerShell + Win32_PerfFormattedData_...

    Filters to 3D and Compute engines only. The original implementation
    took max over ALL engines including VideoDecode/VideoEncode, which
    made the gauge pin to 100% during any video playback even at zero
    compute load. This matches what Task Manager's "GPU - 3D" shows.

    PowerShell startup is expensive (~500ms), so throttle to 1.5s and cache.
    A future upgrade could go direct via ctypes+pdh.dll to eliminate the
    subprocess entirely.
    """
    name = "windows-cim"

    def __init__(self):
        self._last_poll = 0.0
        self._cached: Optional[GPUSample] = None

    def probe(self) -> bool:
        return os.name == "nt"

    @staticmethod
    def _startupinfo_hidden():
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return si

    def sample(self) -> Optional[GPUSample]:
        now = time.monotonic()
        if self._cached is not None and (now - self._last_poll) < 1.5:
            return self._cached
        self._last_poll = now
        try:
            cmd = [
                "powershell.exe",
                "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                (
                    "$x = Get-CimInstance "
                    "  Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine "
                    "  -ErrorAction SilentlyContinue "
                    "  | Where-Object { "
                    "      $_.Name -like '*engtype_3D*' -or "
                    "      $_.Name -like '*engtype_Compute*' "
                    "    }; "
                    "if (-not $x) { 0 } else { "
                    "  $m = ($x | Measure-Object "
                    "      -Property UtilizationPercentage -Maximum).Maximum; "
                    "  if ($m) { [math]::Round([double]$m, 1) } else { 0 } "
                    "}"
                ),
            ]
            out = subprocess.check_output(
                cmd,
                startupinfo=self._startupinfo_hidden(),
                timeout=2.5,
                stderr=subprocess.DEVNULL,
            )
            val_str = out.decode("utf-8", errors="ignore").strip()
            val = float(val_str.replace(",", ".")) if val_str else 0.0
            self._cached = GPUSample(util_pct=val)
            return self._cached
        except Exception:
            # keep last known instead of flapping to 0
            return self._cached


# ─── Backend construction ─────────────────────────────────────────────────────

def _build_gpu_backends() -> list[GPUBackend]:
    """Probe once at startup and return the ordered list of usable backends.

    Order controls which backend "wins" for metadata (name/mem) when
    multiple backends produce util values: earlier entries take precedence.
    NvidiaSmi is skipped if Nvml already succeeded (would double-count
    NVIDIA utilization when both sampled).
    """
    ordered: list[GPUBackend] = [
        NvmlBackend(),          # NVIDIA — best data if pynvml installed
        NvidiaSmiBackend(),     # NVIDIA fallback
        RocmBackend(),          # AMD on Linux
        MacOSBackend(),         # macOS (all vendors, unified path)
        WindowsBackend(),       # Windows CIM — heaviest, least data
    ]
    result: list[GPUBackend] = []
    have_nvidia = False
    for b in ordered:
        if not b.probe():
            continue
        if b.name.startswith("nvidia"):
            if have_nvidia:
                continue
            have_nvidia = True
        result.append(b)
    return result


# ─── Worker thread ─────────────────────────────────────────────────────────────

class GPUWorker(QThread):
    """Polls the composed GPU backends and emits combined GPUSample results."""
    resultReady = pyqtSignal(object)  # GPUSample

    def __init__(self, backends: list[GPUBackend], parent=None):
        super().__init__(parent)
        self._backends = backends
        self._last_emit = 0.0
        self._last_util: Optional[float] = None
        self._active = True

    def _combine(self) -> GPUSample:
        util_vals: list[float] = []
        mem_used: Optional[float] = None
        mem_total: Optional[float] = None
        name: Optional[str] = None
        temp: Optional[float] = None

        for b in self._backends:
            s = b.sample()
            if s is None:
                continue
            if s.util_pct is not None:
                util_vals.append(s.util_pct)
            if mem_used is None and s.mem_used_mb is not None:
                mem_used = s.mem_used_mb
            if mem_total is None and s.mem_total_mb is not None:
                mem_total = s.mem_total_mb
            if name is None and s.name:
                name = s.name
            if temp is None and s.temp_c is not None:
                temp = s.temp_c

        return GPUSample(
            util_pct=(max(util_vals) if util_vals else 0.0),
            mem_used_mb=mem_used,
            mem_total_mb=mem_total,
            name=name,
            temp_c=temp,
        )

    def run(self):
        # 2Hz polling — plenty for a UI gauge. Emit only when util changed
        # by >= 1% or at least every 500ms so downstream signals aren't
        # noisy but the gauge still animates smoothly with the QML Behavior.
        while not self.isInterruptionRequested():
            try:
                if not self._active:
                    # Paused (panel hidden): don't spawn GPU probes off-screen.
                    self.msleep(200)
                    continue
                sample = self._combine()
                now = time.monotonic()
                if (self._last_util is None
                        or abs(sample.util_pct - self._last_util) >= 1.0
                        or (now - self._last_emit) >= 0.5):
                    self._last_emit = now
                    self._last_util = sample.util_pct
                    self.resultReady.emit(sample)
                self.msleep(500)
            except Exception:
                self.msleep(1000)

    def set_active(self, active: bool):
        """Pause/resume sampling without tearing the thread down (used when the
        panel is hidden, so we stop spawning GPU subprocesses off-screen)."""
        self._active = bool(active)

    def close_backends(self):
        for b in self._backends:
            try:
                b.close()
            except Exception:
                pass


# ─── ResourceBackend (QML context object) ─────────────────────────────────────

class ResourceBackend(QObject):
    """Backend logic exposed to the QML Resource Monitor.

    Reports SYSTEM utilization (not per-app) for CPU, RAM, and GPU, plus
    the current process's RSS as `appRamString`.
    """
    cpuChanged     = pyqtSignal()
    ramChanged     = pyqtSignal()
    gpuChanged     = pyqtSignal()
    gpuInfoChanged = pyqtSignal()
    appRamChanged  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # gauge values
        self._cpu: float = 0.0
        self._ram: float = 0.0
        self._gpu: float = 0.0

        # gpu extras (for tooltip)
        self._gpu_mem_used_mb: float = 0.0
        self._gpu_mem_total_mb: float = 0.0
        self._gpu_name: str = ""

        # ram extras (for tooltip)
        self._ram_used_gb: float = 0.0
        self._ram_total_gb: float = 0.0

        # app ram (this process)
        self._app_ram_str: str = "0 MB"

        # Prime psutil CPU baselines (IMPORTANT on Windows — first call is
        # a meaningless 0 that only establishes the baseline).
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(percpu=True, interval=None)
        except Exception:
            pass

        self._cpu_ema: Optional[float] = None
        self._last_cpu_times = None

        # GPU worker — probes and starts backends
        self._gpu_worker = GPUWorker(_build_gpu_backends(), self)
        self._gpu_worker.resultReady.connect(self._on_gpu_measured)
        self._gpu_worker.start()

        # CPU / RAM timer — 500ms matches GPU cadence
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start()

    # ─── slots ────────────────────────────────────────────────────────────────

    def _on_gpu_measured(self, sample: GPUSample):
        info_before = (self._gpu_mem_used_mb, self._gpu_mem_total_mb, self._gpu_name)
        self._gpu = float(sample.util_pct or 0.0)
        self._gpu_mem_used_mb = float(sample.mem_used_mb or 0.0)
        self._gpu_mem_total_mb = float(sample.mem_total_mb or 0.0)
        self._gpu_name = sample.name or ""
        self.gpuChanged.emit()
        info_after = (self._gpu_mem_used_mb, self._gpu_mem_total_mb, self._gpu_name)
        if info_before != info_after:
            self.gpuInfoChanged.emit()

    # ─── properties ────────────────────────────────────────────────────────────

    @pyqtProperty(float, notify=cpuChanged)
    def cpuUsage(self) -> float:
        return float(self._cpu)

    @pyqtProperty(float, notify=ramChanged)
    def ramUsage(self) -> float:
        return float(self._ram)

    @pyqtProperty(float, notify=gpuChanged)
    def gpuUsage(self) -> float:
        return float(self._gpu)

    @pyqtProperty(str, notify=appRamChanged)
    def appRamString(self) -> str:
        return self._app_ram_str

    @pyqtProperty(str, notify=ramChanged)
    def ramString(self) -> str:
        """Used/total system RAM in GB, for the tooltip."""
        if self._ram_total_gb > 0:
            return f"{self._ram_used_gb:.1f} / {self._ram_total_gb:.1f} GB"
        return ""

    @pyqtProperty(str, notify=gpuInfoChanged)
    def gpuName(self) -> str:
        return self._gpu_name or "GPU"

    @pyqtProperty(str, notify=gpuInfoChanged)
    def gpuMemString(self) -> str:
        """Used/total VRAM for the tooltip; empty if backend doesn't report."""
        if self._gpu_mem_total_mb > 0:
            used_gb = self._gpu_mem_used_mb / 1024.0
            total_gb = self._gpu_mem_total_mb / 1024.0
            return f"{used_gb:.1f} / {total_gb:.1f} GB"
        return ""

    # ─── polling ──────────────────────────────────────────────────────────────

    def _read_system_cpu_percent(self) -> float:
        """SYSTEM-wide CPU % via cpu_times() deltas — robust even if other
        code in the process also calls psutil.cpu_percent()."""
        try:
            cur = psutil.cpu_times(percpu=True)
            if not cur:
                return 0.0

            if self._last_cpu_times is None:
                self._last_cpu_times = cur
                return float(self._cpu)

            prev = self._last_cpu_times
            self._last_cpu_times = cur

            usages: list[float] = []
            for t0, t1 in zip(prev, cur):
                total0 = float(sum(t0))
                total1 = float(sum(t1))
                dt_total = total1 - total0
                if dt_total <= 1e-9:
                    continue
                idle0 = float(getattr(t0, "idle", 0.0) + getattr(t0, "iowait", 0.0))
                idle1 = float(getattr(t1, "idle", 0.0) + getattr(t1, "iowait", 0.0))
                dt_idle = idle1 - idle0
                busy = 1.0 - (dt_idle / dt_total)
                usages.append(busy)

            if not usages:
                return float(self._cpu)

            avg = (sum(usages) / len(usages)) * 100.0
            return max(0.0, min(100.0, avg))
        except Exception:
            return float(self._cpu)

    def _update_stats(self):
        # CPU — with light EMA smoothing so gauge feels like Task Manager
        cpu = self._read_system_cpu_percent()
        if self._cpu_ema is None:
            self._cpu_ema = cpu
        else:
            a = 0.25   # smoothing factor (0=no update, 1=no smoothing)
            self._cpu_ema = (1.0 - a) * self._cpu_ema + a * cpu
        self._cpu = float(self._cpu_ema)

        # System RAM
        try:
            vm = psutil.virtual_memory()
            self._ram = float(vm.percent)
            self._ram_total_gb = float(vm.total) / (1024.0 ** 3)
            self._ram_used_gb = float(vm.total - vm.available) / (1024.0 ** 3)
        except Exception:
            self._ram = 0.0

        # App RAM (this process)
        try:
            mb = float(get_memory_usage_mb())
            self._app_ram_str = f"{int(mb)} MB"
        except Exception:
            self._app_ram_str = "? MB"

        self.cpuChanged.emit()
        self.ramChanged.emit()
        self.appRamChanged.emit()

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def set_active(self, active: bool):
        """Pause or resume polling when the panel is hidden/shown. Unlike
        stop(), this is reversible and keeps the GPU worker thread alive."""
        active = bool(active)
        try:
            if hasattr(self, "_timer"):
                if active and not self._timer.isActive():
                    self._timer.start()
                elif not active and self._timer.isActive():
                    self._timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_gpu_worker"):
                self._gpu_worker.set_active(active)
        except Exception:
            pass

    def stop(self):
        """Explicitly stop background threads. Called from the widget's
        closeEvent — do NOT rely on __del__ (which fires during interpreter
        shutdown and used to cause 'QThread destroyed while running' warnings)."""
        try:
            if hasattr(self, "_timer") and self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass

        if hasattr(self, "_gpu_worker") and self._gpu_worker.isRunning():
            self._gpu_worker.requestInterruption()
            self._gpu_worker.quit()
            self._gpu_worker.wait(1000)
            try:
                self._gpu_worker.close_backends()
            except Exception:
                pass


# ─── Native gauges (replaces the former QML overlay) ───────────────────────────
#
# The monitor now renders with QPainter inside a normal utility dock, matching
# the rest of SASpro instead of a floating QML HUD. Track/label colours follow
# the widget palette so it reads in both light and dark themes; the per-gauge
# accents (CPU green/amber/red, RAM blue, GPU purple) are carried over from the
# old design. Displayed values ease toward each new sample so the arcs glide
# between polls (the old QML `Behavior on value`, done here with QVariantAnimation).

_TRACK_ALPHA = 90   # translucent grey ring track, theme-independent


class MiniGauge(QWidget):
    """A ring gauge: dim full-circle track + coloured value arc from 12 o'clock,
    clockwise, with a centred percentage label. Pure view — the owner pushes
    values in via setValue()."""

    def __init__(self, accent="#00C851", parent=None):
        super().__init__(parent)
        self._accent = QColor(accent)
        self._value = 0.0     # currently displayed (animated) value
        self._target = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(400)
        self._anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._anim.valueChanged.connect(self._on_anim)

    def sizeHint(self) -> QSize:
        return QSize(46, 46)

    def minimumSizeHint(self) -> QSize:
        return QSize(40, 40)

    def setAccent(self, color) -> None:
        c = QColor(color)
        if c != self._accent:
            self._accent = c
            self.update()

    def setValue(self, v: float) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        v = max(0.0, min(100.0, v))
        if abs(v - self._target) < 0.05:
            return
        self._target = v
        self._anim.stop()
        self._anim.setStartValue(float(self._value))
        self._anim.setEndValue(v)
        self._anim.start()

    def _on_anim(self, val) -> None:
        try:
            self._value = float(val)
        except (TypeError, ValueError):
            return
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        side = min(self.width(), self.height())
        pen_w = max(3.0, side * 0.09)
        margin = pen_w / 2.0 + 1.0
        rect = QRectF(
            (self.width() - side) / 2.0 + margin,
            (self.height() - side) / 2.0 + margin,
            side - 2.0 * margin,
            side - 2.0 * margin,
        )

        pen = QPen(QColor(128, 128, 128, _TRACK_ALPHA), pen_w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)   # background track (full circle)

        # Value arc — start at 12 o'clock (90°), sweep clockwise (negative span).
        # Hidden below ~0.5% so the round cap doesn't leave a stray dot at zero.
        if self._value > 0.5:
            pen.setColor(self._accent)
            p.setPen(pen)
            p.drawArc(rect, 90 * 16, int(-self._value / 100.0 * 360.0 * 16))

        p.setPen(self.palette().color(self.foregroundRole()))
        f = QFont(self.font())
        f.setBold(True)
        f.setPixelSize(max(9, int(side * 0.26)))
        p.setFont(f)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                   f"{int(round(self._value))}%")
        p.end()


class _GaugeColumn(QWidget):
    """A gauge with a dim caption underneath (the old CPU/RAM/GPU stacks)."""

    def __init__(self, caption: str, accent: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.gauge = MiniGauge(accent, self)
        lay.addWidget(self.gauge, 0, Qt.AlignmentFlag.AlignHCenter)

        self.caption = QLabel(caption, self)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        cf = QFont(self.caption.font())
        cf.setPixelSize(9)
        self.caption.setFont(cf)
        self.caption.setEnabled(False)   # dim, theme-aware (was "#aaa")
        lay.addWidget(self.caption)


class ResourceMonitorPanel(QWidget):
    """CPU / RAM / GPU gauges backed by ResourceBackend. Owns the backend and
    pauses its polling while the panel is hidden."""

    ACCENT_CPU = "#00C851"
    ACCENT_RAM = "#33b5e5"
    ACCENT_GPU = "#aa66cc"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ResourceMonitorPanel")

        self._backend = ResourceBackend(self)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(16)
        row.addStretch(1)
        self._cpu = _GaugeColumn(self.tr("CPU"), self.ACCENT_CPU, self)
        self._ram = _GaugeColumn(self.tr("RAM"), self.ACCENT_RAM, self)
        self._gpu = _GaugeColumn(self.tr("GPU"), self.ACCENT_GPU, self)
        for col in (self._cpu, self._ram, self._gpu):
            row.addWidget(col)
        row.addStretch(1)

        b = self._backend
        b.cpuChanged.connect(self._refresh_cpu)
        b.ramChanged.connect(self._refresh_ram)
        b.appRamChanged.connect(self._refresh_ram)    # app-RAM feeds RAM tooltip
        b.gpuChanged.connect(self._refresh_gpu)
        b.gpuInfoChanged.connect(self._refresh_gpu)   # name/VRAM tooltip

        # Stop cleanly on app teardown even if no closeEvent reaches the dock.
        try:
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._backend.stop)
        except Exception:
            pass

        # Prime from whatever the backend already holds.
        self._refresh_cpu()
        self._refresh_ram()
        self._refresh_gpu()

    # ── signal handlers ───────────────────────────────────────────────────────
    def _refresh_cpu(self) -> None:
        v = float(self._backend.cpuUsage)
        self._cpu.gauge.setValue(v)
        self._cpu.gauge.setAccent(
            "#ff4444" if v > 80 else ("#ffbb33" if v > 50 else self.ACCENT_CPU))
        self._cpu.gauge.setToolTip(f"CPU: {int(round(v))}%")

    def _refresh_ram(self) -> None:
        v = float(self._backend.ramUsage)
        self._ram.gauge.setValue(v)
        rs = self._backend.ramString
        if rs:
            self._ram.gauge.setToolTip(
                f"RAM: {rs}  (this app: {self._backend.appRamString})")
        else:
            self._ram.gauge.setToolTip(f"RAM: {int(round(v))}%")

    def _refresh_gpu(self) -> None:
        v = float(self._backend.gpuUsage)
        self._gpu.gauge.setValue(v)
        tip = f"{self._backend.gpuName}: {int(round(v))}%"
        mem = self._backend.gpuMemString
        if mem:
            tip += f"\nVRAM: {mem}"
        self._gpu.gauge.setToolTip(tip)

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def showEvent(self, e):
        self._backend.set_active(True)
        super().showEvent(e)

    def hideEvent(self, e):
        self._backend.set_active(False)
        super().hideEvent(e)

    def stop(self) -> None:
        try:
            self._backend.stop()
        except Exception:
            pass

    @property
    def backend(self) -> "ResourceBackend":
        return self._backend


class ResourceMonitorDock(QDockWidget):
    """System resource monitor as a standard SASpro utility panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("System Monitor"))
        self.setObjectName("ResourceMonitorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.panel = ResourceMonitorPanel(self)
        self.setWidget(self.panel)

    def closeEvent(self, e):
        try:
            self.panel.stop()
        except Exception:
            pass
        super().closeEvent(e)

    @property
    def backend(self) -> "ResourceBackend":
        return self.panel.backend