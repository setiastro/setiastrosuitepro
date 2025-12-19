from __future__ import annotations
import os
import glob
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QFileDialog, QHBoxLayout, QProgressBar, QMessageBox
)

from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image, save_image as legacy_save_image


# --- helpers ---------------------------------------------------------------

_ALL_INPUT_PATTERNS = [
    # science formats
    "*.fit", "*.fits", "*.fts", "*.fz", "*.xisf",
    # tiff/png/jpeg
    "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg",
    # common RAWs
    "*.cr2", "*.nef", "*.arw", "*.dng", "*.orf", "*.rw2", "*.pef",
]

# Allowed bit-depths per output format (labels match your legacy.save_image)
_ALLOWED_DEPTHS = {
    "png":  {"8-bit"},
    "jpg":  {"8-bit"},
    "jpeg": {"8-bit"},
    "fits": {"32-bit floating point"},
    "fit":  {"32-bit floating point"},
    "tiff": {"8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"},
    "tif":  {"8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"},
    "xisf": {"16-bit", "32-bit unsigned", "32-bit floating point"},
}

from setiastro.saspro.file_utils import _normalize_ext

def _format_token_for_save(ext: str) -> str:
    """
    Map UI extension to save_image's `original_format` token.
    """
    e = _normalize_ext(ext)
    if e == "tif":  return "tiff"
    if e == "jpg":  return "jpg"
    if e == "png":  return "png"
    if e in ("fit", "fits"): return e
    if e == "xisf": return "xisf"
    # default to fits if somehow unknown
    return "fits"


# --- worker ----------------------------------------------------------------

class _BatchWorker(QThread):
    progress = pyqtSignal(int, str)   # (percent, message)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, in_dir: str, out_dir: str, out_ext: str,
                 recurse: bool, skip_existing: bool, bit_depth_choice: str):
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.out_ext = out_ext  # like ".fits"
        self.recurse = recurse
        self.skip_existing = skip_existing
        self.bit_depth_choice = bit_depth_choice  # "Auto" or one of _ALLOWED_DEPTHS per format
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _collect_files(self) -> list[str]:
        files = []
        pats = _ALL_INPUT_PATTERNS
        if self.recurse:
            for pat in pats:
                files.extend(glob.glob(os.path.join(self.in_dir, "**", pat), recursive=True))
        else:
            for pat in pats:
                files.extend(glob.glob(os.path.join(self.in_dir, pat)))
        # unique + sorted
        files = sorted(set(files))
        return files

    def run(self):
        try:
            files = self._collect_files()
            n = len(files)
            if not n:
                self.failed.emit("No matching files in input directory.")
                return

            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            out_token = _format_token_for_save(self.out_ext)
            allowed = _ALLOWED_DEPTHS.get(_normalize_ext(self.out_ext), set())

            for i, src in enumerate(files, start=1):
                if self._cancel:
                    break

                base = Path(src).stem
                dst = os.path.join(self.out_dir, f"{base}{self.out_ext}")

                if self.skip_existing and os.path.exists(dst):
                    self.progress.emit(int(i * 100 / n), f"Skipping (exists): {os.path.basename(dst)}")
                    continue

                self.progress.emit(int((i - 1) * 100 / n), f"Loading {os.path.basename(src)}")
                try:
                    img, header, src_bit_depth, is_mono = legacy_load_image(src)
                except Exception as e:
                    self.progress.emit(int(i * 100 / n), f"Skipping (load failed): {os.path.basename(src)}")
                    continue

                if img is None:
                    self.progress.emit(int(i * 100 / n), f"Skipping (unreadable): {os.path.basename(src)}")
                    continue

                # Decide bit depth to use
                if self.bit_depth_choice == "Auto":
                    # Prefer source bit depth if valid for target, else pick a sane default from allowed
                    if src_bit_depth in allowed:
                        bit_depth = src_bit_depth
                    else:
                        # fallbacks by format
                        if out_token in ("png", "jpg"):
                            bit_depth = "8-bit"
                        elif out_token in ("tiff", "xisf"):
                            # prefer float if available, else first allowed
                            bit_depth = "32-bit floating point" if "32-bit floating point" in allowed else next(iter(allowed)) if allowed else None
                        else:  # fits/fit
                            bit_depth = "32-bit floating point"
                else:
                    bit_depth = self.bit_depth_choice
                    if allowed and bit_depth not in allowed:
                        # shouldn't happen because UI filters, but guard anyway
                        bit_depth = next(iter(allowed))

                # Write
                self.progress.emit(int((i - 1) * 100 / n), f"Saving {os.path.basename(dst)}")
                try:
                    legacy_save_image(
                        img_array=img,
                        filename=dst,
                        original_format=out_token,
                        bit_depth=bit_depth,
                        original_header=header,  # preserves FITS keywords when saving to FITS
                        is_mono=is_mono,
                        image_meta=None,
                        file_meta=None,
                    )
                except Exception as e:
                    self.progress.emit(int(i * 100 / n), f"ERROR: {os.path.basename(base)} → {e}")
                    continue

                self.progress.emit(int(i * 100 / n), f"Saved {os.path.basename(dst)}")

            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))


# --- dialog ----------------------------------------------------------------

class BatchConvertDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Convert"))
        self.setMinimumWidth(560)
        self.worker: _BatchWorker | None = None

        lay = QVBoxLayout(self)

        # in dir
        self.in_edit = QLineEdit()
        in_row = self._row("Input folder:", self.in_edit, self._browse_in)
        lay.addLayout(in_row)

        # out dir
        self.out_edit = QLineEdit()
        out_row = self._row("Output folder:", self.out_edit, self._browse_out)
        lay.addLayout(out_row)

        # options row
        opt_row = QHBoxLayout()

        self.recurse_cb = QCheckBox("Recurse subfolders")
        self.recurse_cb.setChecked(True)
        opt_row.addWidget(self.recurse_cb)

        self.skip_cb = QCheckBox("Skip existing")
        self.skip_cb.setChecked(True)
        opt_row.addWidget(self.skip_cb)

        opt_row.addStretch(1)
        lay.addLayout(opt_row)

        # output format + bit depth
        fmt_row = QHBoxLayout()
        self.fmt = QComboBox()
        self.fmt.addItems([".fits", ".fit", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".xisf"])
        self.fmt.currentIndexChanged.connect(self._refresh_depth_choices)

        self.depth = QComboBox()
        self.depth.addItem("Auto")  # always first
        # choices will be populated based on fmt

        fmt_row.addWidget(QLabel(self.tr("Output format:")))
        fmt_row.addWidget(self.fmt)
        fmt_row.addSpacing(16)
        fmt_row.addWidget(QLabel(self.tr("Bit depth:")))
        fmt_row.addWidget(self.depth)
        fmt_row.addStretch(1)
        lay.addLayout(fmt_row)

        # status/progress
        self.status = QLabel("")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)

        # buttons
        self.start_btn = QPushButton(self.tr("Start"))
        self.cancel_btn = QPushButton(self.tr("Cancel"))
        self.cancel_btn.setEnabled(False)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.cancel_btn)

        lay.addSpacing(8)
        lay.addWidget(self.status)
        lay.addWidget(self.bar)
        lay.addLayout(btns)

        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)

        # initialize bit-depth choices
        self._refresh_depth_choices()

    def _row(self, label: str, line: QLineEdit, browse_fn):
        hb = QHBoxLayout()
        hb.addWidget(QLabel(label))
        hb.addWidget(line, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(browse_fn)
        hb.addWidget(b)
        return hb

    def _browse_in(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Input Folder", self.in_edit.text().strip() or "")
        if d:
            self.in_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder", self.out_edit.text().strip() or "")
        if d:
            self.out_edit.setText(d)

    def _refresh_depth_choices(self):
        self.depth.blockSignals(True)
        cur_fmt = self.fmt.currentText().lstrip(".").lower()
        self.depth.clear()
        self.depth.addItem("Auto")
        # Map .tif/.tiff etc.
        key = _normalize_ext(cur_fmt)
        allowed = _ALLOWED_DEPTHS.get(key if key != "tif" else "tiff", set())
        for d in sorted(allowed):
            self.depth.addItem(d)
        # pick a sensible default
        self.depth.setCurrentIndex(0)
        self.depth.blockSignals(False)

    def _start(self):
        in_dir = self.in_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not in_dir or not out_dir:
            QMessageBox.warning(self, "Batch Convert", "Pick input and output folders.")
            return

        if os.path.abspath(in_dir) == os.path.abspath(out_dir):
            # you *can* convert in place, but warn if target ext might overwrite sources
            QMessageBox.information(
                self, "In-place Notice",
                "Input and output folders are the same. Existing files with the same base name and extension may be overwritten."
            )

        out_ext = self.fmt.currentText()
        recurse = self.recurse_cb.isChecked()
        skip_existing = self.skip_cb.isChecked()
        bit_depth_choice = self.depth.currentText()  # "Auto" or explicit

        self.worker = _BatchWorker(in_dir, out_dir, out_ext, recurse, skip_existing, bit_depth_choice)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Starting…")
        self.bar.setValue(0)
        self.worker.start()

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self.bar.setValue(pct)
        self.status.setText(msg)

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Done.")

    def _on_failed(self, err: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Batch Convert", err)
