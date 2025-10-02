# pro/accel_workers.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from pro.accel_installer import ensure_torch_installed

class AccelInstallWorker(QObject):
    progress = pyqtSignal(str)        # emitted from worker thread; GUI updates must connect with QueuedConnection
    finished = pyqtSignal(bool, str)  # (ok, message)

    def __init__(self, prefer_gpu: bool = True):
        super().__init__()
        self.prefer_gpu = prefer_gpu

    def _log(self, s: str):
        # Never touch widgets here; just emit text
        self.progress.emit(s)

    def run(self):
        # pure backend work; no QWidget/QMessageBox etc. in this method
        ok, msg = ensure_torch_installed(self.prefer_gpu, self._log)

        # honor cancellation if requested
        if QThread.currentThread().isInterruptionRequested():
            self.finished.emit(False, "Canceled.")
            return

        if ok:
            self.finished.emit(True, "PyTorch installed and ready.")
        else:
            self.finished.emit(False, msg or "Installation failed.")
