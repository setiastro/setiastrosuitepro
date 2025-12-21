# Sample SASpro script
# UI with two dropdowns listing open views by their CURRENT window titles.
# Averages the two selected documents and opens a new document.

from __future__ import annotations

SCRIPT_NAME  = "Average Two Documents (UI Sample)"
SCRIPT_GROUP = "Samples"

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QMessageBox
)


class AverageTwoDocsDialog(QDialog):
    def __init__(self, ctx):
        super().__init__(parent=ctx.app)
        self.ctx = ctx
        self.setWindowTitle("Average Two Documents")
        self.resize(520, 180)

        self._title_to_doc = {}

        root = QVBoxLayout(self)

        # Row A
        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("Document A:"))
        self.combo_a = QComboBox()
        row_a.addWidget(self.combo_a, 1)
        root.addLayout(row_a)

        # Row B
        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Document B:"))
        self.combo_b = QComboBox()
        row_b.addWidget(self.combo_b, 1)
        root.addLayout(row_b)

        # Buttons
        brow = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_avg = QPushButton("Average → New Doc")
        self.btn_close = QPushButton("Close")
        brow.addStretch(1)
        brow.addWidget(self.btn_refresh)
        brow.addWidget(self.btn_avg)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

        self.btn_refresh.clicked.connect(self._populate)
        self.btn_avg.clicked.connect(self._do_average)
        self.btn_close.clicked.connect(self.reject)

        self._populate()

    def _populate(self):
        self.combo_a.clear()
        self.combo_b.clear()
        self._title_to_doc.clear()

        try:
            views = self.ctx.list_image_views()
        except Exception:
            views = []

        for title, doc in views:
            # if duplicate names exist, disambiguate slightly
            key = title
            if key in self._title_to_doc:
                # add uid or a counter suffix
                try:
                    uid = getattr(doc, "uid", "")[:6]
                    key = f"{title} [{uid}]"
                except Exception:
                    n = 2
                    while f"{title} ({n})" in self._title_to_doc:
                        n += 1
                    key = f"{title} ({n})"

            self._title_to_doc[key] = doc
            self.combo_a.addItem(key)
            self.combo_b.addItem(key)

        if self.combo_a.count() == 0:
            self.combo_a.addItem("<no image views>")
            self.combo_b.addItem("<no image views>")
            self.btn_avg.setEnabled(False)
        else:
            self.btn_avg.setEnabled(True)

    def _do_average(self):
        key_a = self.combo_a.currentText()
        key_b = self.combo_b.currentText()

        doc_a = self._title_to_doc.get(key_a)
        doc_b = self._title_to_doc.get(key_b)

        if doc_a is None or doc_b is None:
            QMessageBox.warning(self, "Average", "Please select two valid documents.")
            return

        img_a = getattr(doc_a, "image", None)
        img_b = getattr(doc_b, "image", None)

        if img_a is None or img_b is None:
            QMessageBox.warning(self, "Average", "One of the selected documents has no image.")
            return

        a = np.asarray(img_a, dtype=np.float32)
        b = np.asarray(img_b, dtype=np.float32)

        # reconcile mono/color
        if a.ndim == 2:
            a = a[..., None]
        if b.ndim == 2:
            b = b[..., None]
        if a.shape[2] == 1 and b.shape[2] == 3:
            a = np.repeat(a, 3, axis=2)
        if b.shape[2] == 1 and a.shape[2] == 3:
            b = np.repeat(b, 3, axis=2)

        if a.shape != b.shape:
            QMessageBox.warning(
                self, "Average",
                f"Shape mismatch:\nA: {a.shape}\nB: {b.shape}\n\n"
                "For this sample, images must match exactly."
            )
            return

        out = 0.5 * (a + b)

        # name the new doc based on view titles
        new_name = f"Average({key_a}, {key_b})"

        try:
            self.ctx.open_new_document(out, metadata={}, name=new_name)
            QMessageBox.information(self, "Average", f"Created new document:\n{new_name}")
        except Exception as e:
            QMessageBox.critical(self, "Average", f"Failed to create new doc:\n{e}")


def run(ctx):
    dlg = AverageTwoDocsDialog(ctx)
    dlg.exec()
