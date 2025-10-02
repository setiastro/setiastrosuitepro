# pro/accel_workers.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal
from pro.accel_installer import ensure_torch_installed

class AccelInstallWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, prefer_gpu: bool = True):
        super().__init__()
        self.prefer_gpu = prefer_gpu

    def _log(self, s: str): self.progress.emit(s)

    def run(self):
        ok, msg = ensure_torch_installed(self.prefer_gpu, self._log)
        if ok:
            self.finished.emit(True, "PyTorch installed and ready.")
        else:
            self.finished.emit(False, msg or "Installation failed.")
