# pro/save_options.py
from __future__ import annotations
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt

from pro.file_utils import _normalize_ext

# Allowed bit depths per output format (what your saver actually supports)
_BIT_DEPTHS = {
    "png":  ["8-bit"],
    "jpg":  ["8-bit"],
    "fits": ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "fit":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "tif":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "xisf": ["16-bit", "32-bit unsigned", "32-bit floating point"],
}

class SaveOptionsDialog(QDialog):
    def __init__(self, parent, target_ext: str, current_bit_depth: str | None):
        super().__init__(parent)
        self.setWindowTitle("Save Options")
        self.setModal(True)

        # Normalize extension aggressively so it matches _BIT_DEPTHS keys
        raw_ext = (target_ext or "").lower().strip()

        # If it's like ".fits" or "image.fits" just keep the part after last dot
        if "." in raw_ext:
            raw_ext = raw_ext.split(".")[-1]

        # Handle common synonyms / compressed variants
        if raw_ext in ("fit", "fits", "fz", "fits.gz", "fit.gz"):
            self._ext = "fits"
        elif raw_ext in ("tif", "tiff"):
            self._ext = "tif"
        elif raw_ext in ("jpg", "jpeg"):
            self._ext = "jpg"
        else:
            # Fallback – already lowercase, no leading dot
            self._ext = raw_ext

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
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(lbl)
        lay.addWidget(self.combo)
        lay.addStretch(1)
        lay.addLayout(row)

    def selected_bit_depth(self) -> str:
        return self.combo.currentText()

