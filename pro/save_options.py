# pro/save_options.py
from __future__ import annotations
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt

def _normalize_ext(ext: str) -> str:
    e = ext.lower().lstrip(".")
    if e == "jpeg": return "jpg"
    if e == "tiff": return "tif"
    if e in ("fit", "fits"): return e
    return e

# Allowed bit depths per output format (what your saver actually supports)
_BIT_DEPTHS = {
    "png":  ["8-bit"],
    "jpg":  ["8-bit"],
    "fits": ["32-bit floating point"],     # your saver writes float32 for FITS
    "fit":  ["32-bit floating point"],
    "tif":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "xisf": ["16-bit", "32-bit unsigned", "32-bit floating point"],
}

class SaveOptionsDialog(QDialog):
    def __init__(self, parent, target_ext: str, current_bit_depth: str | None):
        super().__init__(parent)
        self.setWindowTitle("Save Options")
        self.setModal(True)

        self._ext = _normalize_ext(target_ext)
        allowed = _BIT_DEPTHS.get(self._ext, ["32-bit floating point"])

        self.combo = QComboBox(self)
        self.combo.addItems(allowed)
        if current_bit_depth in allowed:
            self.combo.setCurrentText(current_bit_depth)

        lbl = QLabel(f"Choose bit depth for *.{self._ext}* export:")
        lbl.setWordWrap(True)

        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)

        lay = QVBoxLayout(self)
        lay.addWidget(lbl)
        lay.addWidget(self.combo)
        lay.addStretch(1)
        lay.addLayout(row)

    def selected_bit_depth(self) -> str:
        return self.combo.currentText()
