# pro/save_options.py
from __future__ import annotations
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QSpinBox, QWidget
from PyQt6.QtCore import Qt


from setiastro.saspro.file_utils import _normalize_ext

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
    def __init__(
        self,
        parent,
        target_ext: str,
        current_bit_depth: str | None,
        current_jpeg_quality: int | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Save Options"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self.jpeg_quality_spin = None

        # -----------------------------
        # Normalize extension FIRST
        # -----------------------------
        raw_ext = (target_ext or "").lower().strip()
        if "." in raw_ext:
            raw_ext = raw_ext.split(".")[-1]

        if raw_ext in ("fit", "fits", "fz", "fits.gz", "fit.gz"):
            self._ext = "fits"
        elif raw_ext in ("tif", "tiff"):
            self._ext = "tif"
        elif raw_ext in ("jpg", "jpeg"):
            self._ext = "jpg"
        else:
            self._ext = raw_ext

        allowed = _BIT_DEPTHS.get(self._ext, ["32-bit floating point"])

        # -----------------------------
        # Build layout
        # -----------------------------
        lay = QVBoxLayout(self)

        lbl = QLabel(self.tr("Choose bit depth for export:"))
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self.combo = QComboBox(self)
        self.combo.addItems(allowed)
        if current_bit_depth in allowed:
            self.combo.setCurrentText(current_bit_depth)
        lay.addWidget(self.combo)

        # -----------------------------
        # JPEG quality (only for jpg)
        # -----------------------------
        if self._ext == "jpg":
            qlbl = QLabel(self.tr("JPEG quality (1–100):"))
            qlbl.setWordWrap(True)
            lay.addWidget(qlbl)

            self.jpeg_quality_spin = QSpinBox(self)
            self.jpeg_quality_spin.setRange(1, 100)
            default_q = int(current_jpeg_quality) if current_jpeg_quality is not None else 95
            self.jpeg_quality_spin.setValue(max(1, min(100, default_q)))
            lay.addWidget(self.jpeg_quality_spin)

        # -----------------------------
        # Buttons
        # -----------------------------
        btn_ok = QPushButton(self.tr("OK"))
        btn_cancel = QPushButton(self.tr("Cancel"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)

        lay.addStretch(1)
        lay.addLayout(row)


    def selected_bit_depth(self) -> str:
        return self.combo.currentText()

    def selected_jpeg_quality(self) -> int | None:
        if self.jpeg_quality_spin is None:
            return None
        return int(self.jpeg_quality_spin.value())
