# saspro/accel_workers.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from setiastro.saspro.accel_installer import ensure_torch_installed

class AccelInstallWorker(QObject):
    progress = pyqtSignal(str)        # emitted from worker thread
    finished = pyqtSignal(bool, str)  # (ok, message)

    def __init__(self, prefer_gpu: bool = True, preferred_backend: str = "auto"):
        super().__init__()
        self.prefer_gpu = bool(prefer_gpu)
        self.preferred_backend = (preferred_backend or "auto").lower()

    def run(self):
        ok, msg = ensure_torch_installed(
            prefer_gpu=self.prefer_gpu,
            preferred_backend=self.preferred_backend,
            log_cb=self.progress.emit
        )

        if QThread.currentThread().isInterruptionRequested():
            self.finished.emit(False, "Canceled.")
            return

        if ok:
            self.finished.emit(True, "PyTorch installed and ready.")
        else:
            self.finished.emit(False, msg or "Installation failed.")
