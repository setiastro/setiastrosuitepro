# src/setiastro/saspro/widgets/resource_monitor.py
from __future__ import annotations

import os
import time
import subprocess
import numpy as np

import psutil

from PyQt6.QtCore import Qt, QUrl, QTimer, QObject, pyqtProperty, pyqtSignal, QThread
from PyQt6.QtQuickWidgets import QQuickWidget

from setiastro.saspro.memory_utils import get_memory_usage_mb
from setiastro.saspro.resources import _get_base_path


class GPUWorker(QThread):
    resultReady = pyqtSignal(float)

    def __init__(self, has_nvidia: bool, parent=None):
        super().__init__(parent)
        self._has_nvidia = has_nvidia

        # cache + throttle (Windows PowerShell is expensive)
        self._last_win_poll = 0.0
        self._cached_win_val = 0.0

        self._last_emit = 0.0
        self._last_emitted_val = None

    def _startupinfo_hidden(self):
        if os.name != "nt":
            return None
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return si

    def _get_windows_gpu_load(self) -> float:
        if os.name != "nt":
            return 0.0

        now = time.monotonic()

        # THROTTLE: run this at most once every 1.5 seconds
        if (now - self._last_win_poll) < 1.5:
            return self._cached_win_val

        self._last_win_poll = now

        try:
            cmd = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                (
                    "$x = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine "
                    "-ErrorAction SilentlyContinue; "
                    "if (-not $x) { 0 } else { "
                    "  $m = ($x | Measure-Object -Property UtilizationPercentage -Maximum).Maximum; "
                    "  if ($m) { [math]::Round([double]$m, 1) } else { 0 } "
                    "}"
                ),
            ]

            out = subprocess.check_output(
                cmd,
                startupinfo=self._startupinfo_hidden(),
                timeout=2.0,
                stderr=subprocess.DEVNULL,
            )
            val_str = out.decode("utf-8", errors="ignore").strip()

            val = float(val_str.replace(",", ".")) if val_str else 0.0
            self._cached_win_val = val
            return val
        except Exception:
            # keep last known value instead of spamming 0.0
            return self._cached_win_val

    def _get_gpu_load(self) -> float:
        nv_val = 0.0
        win_val = 0.0

        # NVIDIA (fast)
        if self._has_nvidia:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    startupinfo=self._startupinfo_hidden(),
                    timeout=0.6,
                    stderr=subprocess.DEVNULL,
                )
                line = out.decode("utf-8", errors="ignore").strip().split("\n")[0]
                nv_val = float(line)
            except Exception:
                pass

        # Windows integrated (slow, throttled)
        if os.name == "nt":
            win_val = self._get_windows_gpu_load()

        return max(nv_val, win_val)

    def run(self):
        while not self.isInterruptionRequested():
            try:
                val = self._get_gpu_load()

                # emit only if changed enough OR periodically
                now = time.monotonic()
                if (
                    self._last_emitted_val is None
                    or abs(val - self._last_emitted_val) >= 1.0
                    or (now - self._last_emit) >= 0.5
                ):
                    self._last_emit = now
                    self._last_emitted_val = val
                    self.resultReady.emit(val)

                self.msleep(250)
            except Exception:
                self.msleep(1000)


class ResourceBackend(QObject):
    """Backend logic for the QML Resource Monitor (SYSTEM usage, not app usage)."""

    cpuChanged = pyqtSignal()
    ramChanged = pyqtSignal()
    gpuChanged = pyqtSignal()
    appRamChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._cpu = 0.0          # system CPU %
        self._ram = 0.0          # system RAM %
        self._gpu = 0.0          # GPU %
        self._app_ram_val = 0.0
        self._app_ram_str = "0 MB"

        # ---- Prime psutil CPU baselines (IMPORTANT on Windows) ----
        # First call returns a meaningless 0.0 (or weird) because it establishes the baseline.
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(percpu=True, interval=None)
        except Exception:
            pass

        # Optional smoothing so gauge feels like Task Manager
        self._cpu_ema = None  # exponential moving average
        self._last_cpu_times = None
        self._last_cpu_sample_t = 0.0
        # Check if nvidia-smi is reachable once
        has_nvidia = False
        try:
            import shutil
            if shutil.which("nvidia-smi"):
                has_nvidia = True
        except Exception:
            pass

        # Start Background GPU Worker
        self._gpu_worker = GPUWorker(has_nvidia, self)
        self._gpu_worker.resultReady.connect(self._on_gpu_measured)
        self._gpu_worker.start()

        # Timer for CPU/RAM updates (250ms)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start()

    def _on_gpu_measured(self, val: float):
        self._gpu = float(val)
        self.gpuChanged.emit()

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

    def _read_system_cpu_percent(self) -> float:
        """
        Return SYSTEM-wide CPU utilization as 0..100 using cpu_times() deltas.
        This is robust even if other code calls psutil.cpu_percent().
        """
        try:
            now = time.monotonic()
            cur = psutil.cpu_times(percpu=True)

            if not cur:
                return 0.0

            # first sample: store and return 0 (or keep last)
            if self._last_cpu_times is None:
                self._last_cpu_times = cur
                self._last_cpu_sample_t = now
                return float(self._cpu)  # keep whatever we had

            prev = self._last_cpu_times
            self._last_cpu_times = cur
            self._last_cpu_sample_t = now

            # usage per logical CPU
            usages = []
            for t0, t1 in zip(prev, cur):
                # sum all fields to get total time
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

            return float(np.clip((sum(usages) / len(usages)) * 100.0, 0.0, 100.0))
        except Exception:
            return float(self._cpu)


    def _update_stats(self):
        # 1) SYSTEM CPU
        cpu = self._read_system_cpu_percent()

        # light smoothing (keeps spikes but reduces jitter)
        if self._cpu_ema is None:
            self._cpu_ema = cpu
        else:
            a = 0.25  # smoothing factor (0.0=no update, 1.0=no smoothing)
            self._cpu_ema = (1.0 - a) * self._cpu_ema + a * cpu
        self._cpu = float(self._cpu_ema)

        # 2) SYSTEM RAM
        try:
            vm = psutil.virtual_memory()
            self._ram = float(vm.percent)
        except Exception:
            self._ram = 0.0

        # 3) APP RAM (your process)
        try:
            mb = float(get_memory_usage_mb())
            self._app_ram_val = mb
            self._app_ram_str = f"{int(mb)} MB"
        except Exception:
            self._app_ram_str = "? MB"

        self.cpuChanged.emit()
        self.ramChanged.emit()
        self.appRamChanged.emit()

    def stop(self):
        """Explicitly stop background threads."""
        try:
            if hasattr(self, "_timer") and self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass

        if hasattr(self, "_gpu_worker") and self._gpu_worker.isRunning():
            self._gpu_worker.requestInterruption()
            self._gpu_worker.quit()
            self._gpu_worker.wait(1000)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


class SystemMonitorWidget(QQuickWidget):
    """
    The QQuickWidget hosting the QML content.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setClearColor(Qt.GlobalColor.transparent)

        # Connect Backend
        self.backend = ResourceBackend(self)
        self.rootContext().setContextProperty("backend", self.backend)

        # Load QML
        qml_path = os.path.join(_get_base_path(), "qml", "ResourceMonitor.qml")
        self.setSource(QUrl.fromLocalFile(qml_path))

    def closeEvent(self, e):
        # make sure worker threads stop when widget closes
        try:
            if hasattr(self, "backend") and self.backend is not None:
                self.backend.stop()
        except Exception:
            pass
        super().closeEvent(e)

    # --- Drag & Drop Support ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Wayland-friendly: ask compositor to move the window
            wh = self.windowHandle()
            if wh is not None:
                try:
                    wh.startSystemMove()
                    event.accept()
                    return
                except Exception:
                    pass

            # Fallback (Windows/X11): manual move tracking
            self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if hasattr(self, "_drag_start_pos"):
                self.move(event.globalPosition().toPoint() - self._drag_start_pos)
                event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            from PyQt6.QtCore import QSettings
            settings = QSettings("SetiAstro", "SetiAstroSuitePro")
            pos = self.pos()
            settings.setValue("ui/resource_monitor_pos_x", pos.x())
            settings.setValue("ui/resource_monitor_pos_y", pos.y())
            event.accept()
        super().mouseReleaseEvent(event)
