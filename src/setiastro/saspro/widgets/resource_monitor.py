# src/setiastro/saspro/widgets/resource_monitor.py
from __future__ import annotations
import os
import psutil
from PyQt6.QtCore import Qt, QUrl, QTimer, QObject, pyqtProperty, pyqtSignal, QThread
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame
from PyQt6.QtQuickWidgets import QQuickWidget

from setiastro.saspro.memory_utils import get_memory_usage_mb
from setiastro.saspro.resources import _get_base_path

class GPUWorker(QThread):
    """Background worker to monitor GPU without blocking the UI."""
    resultReady = pyqtSignal(float)

    def __init__(self, has_nvidia: bool, parent=None):
        super().__init__(parent)
        self._has_nvidia = has_nvidia
        self._last_val = 0.0

    def _get_windows_gpu_load(self) -> float:
        if os.name != 'nt':
            return 0.0
        try:
            import subprocess
            # Aggregation logic to match Task Manager:
            # 1. Group by unique engine (Adapter LUID + Engine Type/Index).
            # 2. Sum utilization of all processes sharing that engine.
            # 3. Take the Maximum of these sums as the overall GPU load.
            # Aggregation logic to match Task Manager:
            # 1. Group by unique engine (Adapter LUID + Engine Type/Index).
            # 2. Sum utilization of all processes sharing that engine.
            # 3. Take the Maximum of these sums as the overall GPU load.
            cmd = (
                "powershell -NoProfile -ExecutionPolicy Bypass -Command \""
                "$groups = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -ErrorAction SilentlyContinue | "
                "Group-Object -Property { $_.Name -replace '^pid_\\d+_', '' }; "
                "$res_list = $groups | ForEach-Object { ($_.Group | Measure-Object -Property UtilizationPercentage -Sum).Sum }; "
                "$max_val = ($res_list | Measure-Object -Maximum).Maximum; "
                "if ($max_val) { [math]::Round($max_val, 1) } else { 0 }\""
            )
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0 
            
            out = subprocess.check_output(cmd, startupinfo=startupinfo, timeout=5.0)
            val_str = out.decode("utf-8").strip()
            
            if not val_str: return 0.0
            return float(val_str.replace(",", "."))
        except Exception:
            return 0.0

    def _get_gpu_load(self) -> float:
        nv_val = 0.0
        win_val = 0.0
        
        # 1. Check NVIDIA (Discrete)
        if self._has_nvidia:
            try:
                import subprocess
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0 

                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    startupinfo=startupinfo,
                    timeout=0.6
                )
                line = out.decode("utf-8").strip().split('\n')[0]
                nv_val = float(line)
            except Exception:
                pass 
        
        # 2. Check Universal (Integrated)
        if os.name == 'nt':
            win_val = self._get_windows_gpu_load()
            
        return max(nv_val, win_val)

    def run(self):
        while not self.isInterruptionRequested():
            try:
                val = self._get_gpu_load()
                self.resultReady.emit(val)
                # Sleep between measurements. 250ms as requested.
                # Note: PowerShell queries might take longer than 250ms, 
                # but this loop will run as fast as the hardware allows without blocking UI.
                self.msleep(250)
            except Exception:
                self.msleep(1000) # Error backoff on failure

class ResourceBackend(QObject):
    """Backend logic for the QML Resource Monitor."""
    
    cpuChanged = pyqtSignal()
    ramChanged = pyqtSignal()
    gpuChanged = pyqtSignal()
    appRamChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cpu = 0.0
        self._ram = 0.0
        self._gpu = 0.0
        self._app_ram_val = 0.0
        self._app_ram_str = "0 MB"
        
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

        # Timer for CPU/RAM updates (250ms as requested)
        self._timer = QTimer(self)
        self._timer.setInterval(250) 
        self._timer.timeout.connect(self._update_stats)
        self._timer.start()

    def _on_gpu_measured(self, val: float):
        self._gpu = val
        self.gpuChanged.emit()

    @pyqtProperty(float, notify=cpuChanged)
    def cpuUsage(self):
        return self._cpu

    @pyqtProperty(float, notify=ramChanged)
    def ramUsage(self):
        return self._ram

    @pyqtProperty(float, notify=gpuChanged)
    def gpuUsage(self):
        return self._gpu

    @pyqtProperty(str, notify=appRamChanged)
    def appRamString(self):
        return self._app_ram_str

    def _update_stats(self):
        # 1. CPU
        try:
            self._cpu = psutil.cpu_percent(interval=None)
        except Exception:
            self._cpu = 0.0
        
        # 2. System RAM
        try:
            vm = psutil.virtual_memory()
            self._ram = vm.percent
        except Exception:
            self._ram = 0.0

        # 3. App RAM
        try:
            mb = get_memory_usage_mb()
            self._app_ram_val = mb
            self._app_ram_str = f"{int(mb)} MB"
        except Exception:
            self._app_ram_str = "? MB"

        self.cpuChanged.emit()
        self.ramChanged.emit()
        self.appRamChanged.emit()

    def stop(self):
        """Explicitly stop background threads."""
        if hasattr(self, "_gpu_worker") and self._gpu_worker.isRunning():
            self._gpu_worker.requestInterruption()
            self._gpu_worker.quit()
            self._gpu_worker.wait(1000)

    def __del__(self):
        self.stop()


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
        
        # We need to manually wire property updates because we are binding to root properties in QML
        # Actually, simpler pattern: QML file reads from an object we inject.
        # Let's adjust QML slightly to bind to `backend.cpuUsage` etc. if we can,
        # OR we leave QML as having properties and we set them from Python.
        #
        # Better approach for Py+QML: 
        # Inject `backend` into context, modify QML to use `backend.cpuUsage`.
        # But since I already wrote QML with root properties, I will just set them directly 
        # or update the QML file. Updating QML is cleaner.
        #
        # For now, let's keep QML independent and binding via setProperty? 
        # No, properly: context property is best.
        #
        # Let's re-write the QML loading part to use a safer 'initialProperties' approach or just signal/slots.
        #
        # EASIEST: QML binds to `root.cpuUsage`. Python sets `root.cpuUsage`.
        
        self.backend.cpuChanged.connect(self._push_data_to_qml)
        self.backend.ramChanged.connect(self._push_data_to_qml)
        self.backend.gpuChanged.connect(self._push_data_to_qml)
        self.backend.appRamChanged.connect(self._push_data_to_qml)
        
        # Load QML
        qml_path = os.path.join(_get_base_path(), "qml", "ResourceMonitor.qml")
        self.setSource(QUrl.fromLocalFile(qml_path))

    def _push_data_to_qml(self):
        root = self.rootObject()
        if root:
            root.setProperty("cpuUsage", self.backend.cpuUsage)
            root.setProperty("ramUsage", self.backend.ramUsage)
            root.setProperty("gpuUsage", self.backend.gpuUsage)
            root.setProperty("appRamString", self.backend.appRamString)

    # --- Drag & Drop Support ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if hasattr(self, "_drag_start_pos"):
                self.move(event.globalPosition().toPoint() - self._drag_start_pos)
                event.accept()
        else:
            super().mouseMoveEvent(event)
