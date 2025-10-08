# pro/debayer.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDialogButtonBox,
    QGroupBox, QMessageBox, QProgressBar
)

# fast kernels you already have
try:
    from legacy.numba_utils import debayer_fits_fast
except Exception as e:  # very unlikely in your env
    debayer_fits_fast = None

# -------- helpers ------------------------------------------------------------

_VALID = {"RGGB", "BGGR", "GRBG", "GBRG"}

def _detect_bayer_from_header(doc) -> Optional[str]:
    """
    Best-effort read of a Bayer pattern from the document header/metadata.
    Returns 'RGGB'/'BGGR'/'GRBG'/'GBRG' or None if not found.
    """
    hdr = getattr(doc, "header", None)
    probe = {}

    # 1) FITS-like header (astropy Header behaves like dict, case-insensitive)
    if hdr is not None:
        try:
            for k in hdr.keys():
                probe[k.upper()] = str(hdr.get(k))
        except Exception:
            pass

    # 2) Generic metadata dicts some loaders keep
    meta = getattr(doc, "meta", None)
    if isinstance(meta, dict):
        for k, v in list(meta.items()):
            probe[str(k).upper()] = str(v)

    # Common key names seen in cameras/stackers
    keys = [
        "BAYERPAT", "BAYERPATN", "BAYER_PATTERN", "BAYERPATTERN",
        "CFAPATTERN", "CFA_PATTERN", "PATTERN", "COLORTYPE", "COLORFILTERARRAY"
    ]
    for k in keys:
        val = probe.get(k)
        if not val:
            continue
        s = str(val).upper()
        # Often "RGGB" appears embedded, sometimes with spaces/commas
        for pat in _VALID:
            if pat in s:
                return pat
        # Some tools write e.g. "R G G B" or "RED,GREEN,GREEN,BLUE"
        s = s.replace(",", "").replace(" ", "")
        if s in _VALID:
            return s
    return None


def _apply_result_to_doc(dm, doc, rgb: np.ndarray, step_name: str = "Debayer"):
    """
    Robustly hand the new image back to your doc manager with an undo step.
    Tries a few known helper names to stay compatible with your stack.
    """
    # Prefer explicit doc-targeting apply fns
    for name in (
        "apply_numpy_result_to_document",
        "apply_numpy_result",
        "apply_array_to_document",
        "apply_numpy_image_to_document",
        "apply_edit_to_doc",
    ):
        fn = getattr(dm, name, None)
        if callable(fn):
            try:
                fn(doc, rgb, step_name)
                return
            except TypeError:
                pass

    # Older APIs: apply to active
    fn = getattr(dm, "apply_edit_to_active", None)
    if callable(fn):
        fn(rgb, step_name)
        return

    # Fallback: direct swap (last resort)
    if hasattr(doc, "replace_image") and callable(getattr(doc, "replace_image")):
        doc.replace_image(rgb, step_name)
    else:
        doc.image = rgb  # no undo metadata, but at least it works


# -------- worker -------------------------------------------------------------

class _DebayerWorker(QThread):
    progress = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray, str)  # (rgb, used_pattern)

    def __init__(self, mono: np.ndarray, pattern: str):
        super().__init__()
        self.mono = mono
        self.pattern = pattern

    def run(self):
        try:
            if debayer_fits_fast is None:
                raise RuntimeError("Numba debayer kernels not available.")

            img = self.mono
            if img.ndim != 2:
                raise ValueError("Debayer expects a single-channel (mosaic) image.")

            if self.pattern not in _VALID:
                raise ValueError(f"Unsupported pattern: {self.pattern}")

            self.progress.emit(5, f"Debayering ({self.pattern}) …")
            # Keep dtype/scale; kernels preserve dtype
            rgb = debayer_fits_fast(img, self.pattern)
            self.progress.emit(96, "Finalizing …")

            # Make sure output is float32/float64 or same dtype as source
            # (your pipeline generally works in float; keep as-is)
            self.finished.emit(rgb, self.pattern)
        except Exception as e:
            self.failed.emit(str(e))


# -------- dialog -------------------------------------------------------------

class DebayerDialog(QDialog):
    """
    One-shot debayer UI for the active view. Uses your numba kernels.
    If the image is already RGB, will warn and exit.
    """
    def __init__(self, parent, doc_manager, active_doc):
        super().__init__(parent)
        self.setWindowTitle("Debayer")
        self.dm = doc_manager
        self.doc = active_doc
        self.worker: Optional[_DebayerWorker] = None

        img = getattr(active_doc, "image", None)
        if img is None:
            raise RuntimeError("No image in active document.")
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            QMessageBox.information(self, "Debayer", "Image already has 3 channels.")
            self.setEnabled(False)
            self.close()
            return
        if arr.ndim != 2:
            QMessageBox.warning(self, "Debayer", "Only single-channel mosaics can be debayered.")
            self.setEnabled(False)
            self.close()
            return

        self._src = arr

        v = QVBoxLayout(self)

        # pattern selection
        detected = _detect_bayer_from_header(active_doc)
        gb = QGroupBox("Bayer pattern", self)
        h = QHBoxLayout(gb)
        self.combo_pattern = QComboBox(self)
        self.combo_pattern.addItems([
            "Auto (from header)",
            "RGGB", "BGGR", "GRBG", "GBRG",
        ])
        self.combo_pattern.setCurrentIndex(0)
        self.lbl_detect = QLabel(f"Detected: {detected or '(unknown)'}")
        h.addWidget(self.combo_pattern, 1)
        h.addWidget(self.lbl_detect)
        v.addWidget(gb)

        # method (kept simple; your kernel is edge-aware fast)
        gbm = QGroupBox("Method", self)
        hm = QHBoxLayout(gbm)
        self.combo_method = QComboBox(self)
        self.combo_method.addItems(["Edge-aware (fast)"])
        self.combo_method.setCurrentIndex(0)
        hm.addWidget(self.combo_method)
        hm.addStretch(1)
        v.addWidget(gbm)

        # progress + buttons
        self.status = QLabel("")
        self.bar = QProgressBar(self); self.bar.setRange(0, 100)
        v.addWidget(self.status)
        v.addWidget(self.bar)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._go)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        # store detection
        self._detected_pattern = detected

    def _chosen_pattern(self) -> str:
        txt = self.combo_pattern.currentText()
        if txt.startswith("Auto"):
            return self._detected_pattern or "RGGB"  # fallback if unknown
        return txt

    def _go(self):
        pat = self._chosen_pattern()
        if pat not in _VALID:
            QMessageBox.warning(self, "Debayer", "Unknown pattern (auto-detect failed). Choose a pattern explicitly.")
            return

        self.status.setText(f"Debayering as {pat} …")
        self.bar.setValue(0)
        self.worker = _DebayerWorker(self._src, pat)
        self.worker.progress.connect(self._on_prog)
        self.worker.failed.connect(self._on_fail)
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    def _on_prog(self, p: int, msg: str):
        self.bar.setValue(p); self.status.setText(msg)

    def _on_fail(self, err: str):
        QMessageBox.critical(self, "Debayer", err)
        self.status.setText("Failed.")

    def _on_done(self, rgb: np.ndarray, used_pattern: str):
        # Hand back to doc manager with an undo step name
        _apply_result_to_doc(self.dm, self.doc, rgb, step_name=f"Debayer ({used_pattern})")
        self.status.setText("Done.")
        self.accept()


# -------- headless (shortcut / DnD) -----------------------------------------

def apply_debayer_preset_to_doc(dm, doc, preset: dict) -> Tuple[str, np.ndarray]:
    """
    Headless entry point used by shortcuts DnD:
      preset = { "pattern": "auto|RGGB|BGGR|GRBG|GBRG" }
    Returns (used_pattern, rgb_array).
    """
    if getattr(doc, "image", None) is None:
        raise RuntimeError("No image in document.")
    mono = np.asarray(doc.image)
    if mono.ndim != 2:
        raise RuntimeError("Debayer expects a single-channel (mosaic) image.")

    want = str(preset.get("pattern", "auto")).upper()
    if want == "AUTO":
        pat = _detect_bayer_from_header(doc) or "RGGB"
    else:
        pat = want
    if pat not in _VALID:
        raise ValueError(f"Unsupported Bayer pattern: {pat}")

    if debayer_fits_fast is None:
        raise RuntimeError("Numba debayer kernels not available.")

    rgb = debayer_fits_fast(mono, pat)
    _apply_result_to_doc(dm, doc, rgb, step_name=f"Debayer ({pat})")
    return pat, rgb
