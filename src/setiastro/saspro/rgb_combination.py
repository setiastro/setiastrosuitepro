# pro/rgb_combination.py
from __future__ import annotations
import os
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QGroupBox, QFileDialog, QComboBox, QButtonGroup, QMessageBox
)
from PyQt6.QtCore import Qt

try:
    from setiastro.saspro.legacy.image_manager import load_image
except Exception:
    load_image = None


def _to_f01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32)
    elif a.dtype.kind == "f" and a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.size:
        m = float(np.nanmax(a))
        if m > 1.0 and np.isfinite(m):
            a = a / m
    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def _mono_from_any(arr: np.ndarray) -> np.ndarray:
    a = _to_f01(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] >= 1:
        return a[:, :, 0]
    raise ValueError("Expected mono (HxW) or RGB (HxWx3) image.")


class RGBCombinationDialogPro(QDialog):
    """
    Combine R/G/B from open views or files and ALWAYS create a new document.
    """
    def __init__(self, parent, list_open_docs_fn=None, doc_manager=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("RGB Combination"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._list_open_docs = list_open_docs_fn or (lambda: [])
        self._docman = doc_manager

        # file-mode state
        self.r_path = None
        self.g_path = None
        self.b_path = None

        # ── mode choose
        self.load_files_radio = QRadioButton(self.tr("Load Individual Files"))
        self.use_views_radio  = QRadioButton(self.tr("Use Open Views"))
        self.use_views_radio.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.load_files_radio)
        self.mode_group.addButton(self.use_views_radio)

        mode_box = QGroupBox(self.tr("Select RGB Combination Mode"))
        ml = QVBoxLayout(mode_box)
        ml.addWidget(self.use_views_radio)
        ml.addWidget(self.load_files_radio)

        # ── file mode widgets
        self.r_label = QLabel(self.tr("Red: Not selected"))
        self.g_label = QLabel(self.tr("Green: Not selected"))
        self.b_label = QLabel(self.tr("Blue: Not selected"))
        self.btn_load_r = QPushButton(self.tr("Load Red…"));   self.btn_load_r.clicked.connect(lambda: self._pick_file("R"))
        self.btn_load_g = QPushButton(self.tr("Load Green…")); self.btn_load_g.clicked.connect(lambda: self._pick_file("G"))
        self.btn_load_b = QPushButton(self.tr("Load Blue…"));  self.btn_load_b.clicked.connect(lambda: self._pick_file("B"))

        file_box = QGroupBox(self.tr("Files"))
        fl = QVBoxLayout(file_box)
        for lab, btn in [(self.r_label, self.btn_load_r),
                         (self.g_label, self.btn_load_g),
                         (self.b_label, self.btn_load_b)]:
            row = QHBoxLayout(); row.addWidget(lab); row.addStretch(); row.addWidget(btn)
            fl.addLayout(row)

        # ── open-views widgets
        views_box = QGroupBox(self.tr("Select Open Views for R / G / B"))
        vl = QHBoxLayout(views_box)
        self.cmb_r = QComboBox(); self.cmb_g = QComboBox(); self.cmb_b = QComboBox()
        vl.addLayout(self._labeled(self.tr("Red:"),   self.cmb_r))
        vl.addLayout(self._labeled(self.tr("Green:"), self.cmb_g))
        vl.addLayout(self._labeled(self.tr("Blue:"),  self.cmb_b))

        # ── buttons
        self.btn_combine = QPushButton(self.tr("Combine"))
        self.btn_cancel  = QPushButton(self.tr("Cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_combine.clicked.connect(self._combine)

        btns = QHBoxLayout()
        btns.addStretch(); btns.addWidget(self.btn_combine); btns.addWidget(self.btn_cancel)

        # ── layout
        L = QVBoxLayout(self)
        L.addWidget(mode_box)
        L.addWidget(views_box)
        L.addWidget(file_box)
        L.addLayout(btns)

        self.mode_group.buttonClicked.connect(self._refresh_mode)
        self._populate_views()
        self._refresh_mode()

    # --- helpers
    def _labeled(self, text, w):
        box = QVBoxLayout()
        box.addWidget(QLabel(text))
        box.addWidget(w)
        return box

    def _populate_views(self):
        self.cmb_r.clear(); self.cmb_g.clear(); self.cmb_b.clear()
        for title, doc in self._list_open_docs():
            self.cmb_r.addItem(title, doc)
            self.cmb_g.addItem(title, doc)
            self.cmb_b.addItem(title, doc)

    def _refresh_mode(self):
        use_views = self.use_views_radio.isChecked()
        for w in (self.cmb_r, self.cmb_g, self.cmb_b):
            w.setEnabled(use_views)
        for w in (self.r_label, self.g_label, self.b_label, self.btn_load_r, self.btn_load_g, self.btn_load_b):
            w.setEnabled(not use_views)
        if not load_image:
            self.load_files_radio.setEnabled(False)
            self.use_views_radio.setChecked(True)

    def _pick_file(self, which):
        fp, _ = QFileDialog.getOpenFileName(
            self, f"Select {'Red' if which=='R' else 'Green' if which=='G' else 'Blue'} Image", "",
            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg);;All Files (*)"
        )
        if not fp: return
        base = os.path.basename(fp)
        if which == "R":
            self.r_path = fp; self.r_label.setText(f"Red: {base}")
        elif which == "G":
            self.g_path = fp; self.g_label.setText(f"Green: {base}")
        else:
            self.b_path = fp; self.b_label.setText(f"Blue: {base}")

    def _load_file_mono(self, path: str) -> np.ndarray:
        if load_image is None:
            raise RuntimeError("File loading not available.")
        arr, _, _, _ = load_image(path)
        if arr is None:
            raise RuntimeError(f"Failed to load: {path}")
        return _mono_from_any(arr)

    # --- combine → ALWAYS new document
    def _combine(self):
        try:
            if self.use_views_radio.isChecked():
                rdoc = self.cmb_r.currentData()
                gdoc = self.cmb_g.currentData()
                bdoc = self.cmb_b.currentData()
                if not (rdoc and gdoc and bdoc):
                    QMessageBox.information(self, "RGB Combination", "Please select three views.")
                    return
                r = _mono_from_any(getattr(rdoc, "image", None))
                g = _mono_from_any(getattr(gdoc, "image", None))
                b = _mono_from_any(getattr(bdoc, "image", None))
                # propose a title from selections
                title = "RGB_Combined"
            else:
                if not (self.r_path and self.g_path and self.b_path):
                    QMessageBox.information(self, "RGB Combination", "Please choose three files.")
                    return
                r = self._load_file_mono(self.r_path)
                g = self._load_file_mono(self.g_path)
                b = self._load_file_mono(self.b_path)
                title = f"RGB_{os.path.splitext(os.path.basename(self.r_path))[0]}"

            if r.shape != g.shape or r.shape != b.shape:
                raise ValueError("All three images must have identical dimensions.")

            rgb = np.stack([r, g, b], axis=2).astype(np.float32, copy=False)
            rgb = np.clip(rgb, 0.0, 1.0)

            if not self._docman:
                raise RuntimeError("No document manager available.")
            # create new document every time
            if hasattr(self._docman, "open_array"):
                newdoc = self._docman.open_array(rgb, metadata={"step_name": "RGB Combination"}, title=title)
            elif hasattr(self._docman, "open_numpy"):
                newdoc = self._docman.open_numpy(rgb, metadata={"step_name": "RGB Combination"}, title=title)
            else:
                newdoc = self._docman.create_document(image=rgb, metadata={"step_name": "RGB Combination"}, name=title)

            if hasattr(self.parent(), "_spawn_subwindow_for"):
                self.parent()._spawn_subwindow_for(newdoc)

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "RGB Combination", f"Failed to combine:\n{e}")
