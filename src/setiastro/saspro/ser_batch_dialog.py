# src/setiastro/saspro/ser_batch_dialog.py
from __future__ import annotations

import os
import traceback
from typing import Optional, List

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QColor, QBrush, QImage, QPixmap, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QProgressBar,
    QGroupBox, QFormLayout, QTextEdit, QMessageBox, QComboBox,
    QScrollArea, QAbstractItemView
)

from setiastro.saspro.ser_stack_config import SERStackConfig
from setiastro.saspro.ser_stacker import analyze_ser, stack_ser, AnalyzeResult
from setiastro.saspro.ser_stacker_dialog import _push_as_new_doc
from setiastro.saspro.imageops.serloader import open_planetary_source

# ── Status sentinels ─────────────────────────────────────────────────────────
_ST_PENDING = "pending"
_ST_RUNNING = "running"
_ST_DONE    = "done"
_ST_FAILED  = "failed"

_STATUS_COLORS = {
    _ST_PENDING: QColor("#888888"),
    _ST_RUNNING: QColor("#4499ff"),
    _ST_DONE:    QColor("#44cc66"),
    _ST_FAILED:  QColor("#ff5555"),
}

_VIDEO_EXTS = {".ser", ".avi", ".mp4", ".mov", ".mkv"}


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


# ── Surface-anchor picker ─────────────────────────────────────────────────────
class _AnchorPickerDialog(QDialog):
    def __init__(self, parent, ref_img01: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle("Set Surface Anchor — Shift+drag a patch on the reference frame")
        self.setModal(True)
        self.resize(900, 700)

        self._ref = np.asarray(ref_img01, dtype=np.float32)
        self._H, self._W = self._ref.shape[:2]
        self._anchor: Optional[tuple] = None

        self._zoom = 1.0
        self._fit_pending = True
        self._drag_start = None
        self._drag_cur   = None
        self._dragging   = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        hint = QLabel(
            "Shift+drag a patch on the reference frame to define the surface anchor.\n"
            "Choose a high-contrast area (crater rim, sunspot, etc.) present in all batch files.",
            self
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa; font-size:11px;")
        outer.addWidget(hint, 0)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.viewport().installEventFilter(self)
        self._scroll.viewport().setMouseTracking(True)

        self._pix_lbl = QLabel(self)
        self._pix_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pix_lbl.setStyleSheet("background:#111;")
        self._scroll.setWidget(self._pix_lbl)
        outer.addWidget(self._scroll, 1)

        zoom_row = QHBoxLayout()
        self.btn_zo  = QPushButton("–", self)
        self.btn_zi  = QPushButton("+", self)
        self.btn_fit = QPushButton("Fit", self)
        self.btn_100 = QPushButton("1:1", self)
        for b in (self.btn_zo, self.btn_zi, self.btn_fit, self.btn_100):
            b.setFixedWidth(44)
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        self.lbl_anchor_status = QLabel("No anchor set — Shift+drag to define", self)
        self.lbl_anchor_status.setStyleSheet("color:#c66;")
        zoom_row.addWidget(self.lbl_anchor_status)
        outer.addLayout(zoom_row, 0)

        btns = QHBoxLayout()
        self.btn_ok     = QPushButton("OK", self)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_ok.setEnabled(False)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        outer.addLayout(btns, 0)

        self.btn_fit.clicked.connect(self._fit_to_window)
        self.btn_100.clicked.connect(lambda: self._set_zoom(1.0))
        self.btn_zi.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zo.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._base_u8 = self._make_u8(self._ref)
        self._render()

    @staticmethod
    def _make_u8(img01: np.ndarray) -> np.ndarray:
        mono = img01 if img01.ndim == 2 else img01[..., 0]
        lo = float(np.percentile(mono, 1.0))
        hi = float(np.percentile(mono, 99.5))
        if hi <= lo + 1e-8:
            hi = lo + 1e-3
        v = np.clip((mono - lo) / (hi - lo), 0.0, 1.0)
        return (v * 255.0 + 0.5).astype(np.uint8)

    def _render(self):
        u8 = self._base_u8
        if not u8.flags["C_CONTIGUOUS"]:
            u8 = np.ascontiguousarray(u8)
        h, w = u8.shape
        qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        base_pm = QPixmap.fromImage(qimg)

        zw = max(1, int(round(w * self._zoom)))
        zh = max(1, int(round(h * self._zoom)))
        pm = base_pm.scaled(zw, zh,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)

        if self._anchor is not None:
            ax, ay, aw, ah = [int(v) for v in self._anchor]
            p = QPainter(pm)
            pen = QPen(QColor(0, 170, 255), 3)
            p.setPen(pen)
            p.setBrush(QColor(0, 170, 255, 30))
            p.drawRect(int(ax * self._zoom), int(ay * self._zoom),
                       int(aw * self._zoom), int(ah * self._zoom))
            p.end()

        if self._dragging and self._drag_start and self._drag_cur:
            wp = self._pix_lbl.pos()
            x1 = int(self._drag_start.x() - wp.x())
            y1 = int(self._drag_start.y() - wp.y())
            x2 = int(self._drag_cur.x()   - wp.x())
            y2 = int(self._drag_cur.y()   - wp.y())
            left,  right = (x1, x2) if x1 < x2 else (x2, x1)
            top,   bot   = (y1, y2) if y1 < y2 else (y2, y1)
            p = QPainter(pm)
            p.setPen(QPen(QColor(0, 255, 0), 2))
            p.setBrush(QColor(0, 255, 0, 20))
            p.drawRect(left, top, right - left, bot - top)
            p.end()

        self._pix_lbl.setPixmap(pm)
        self._pix_lbl.setFixedSize(pm.size())

    def _fit_to_window(self):
        vw = max(1, self._scroll.viewport().width() - 4)
        vh = max(1, self._scroll.viewport().height() - 4)
        self._set_zoom(min(vw / max(1, self._W), vh / max(1, self._H)))

    def _set_zoom(self, z: float):
        self._zoom = float(max(0.05, min(8.0, z)))
        self._render()

    def showEvent(self, e):
        super().showEvent(e)
        if self._fit_pending:
            self._fit_pending = False
            QTimer.singleShot(0, self._fit_to_window)

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        vp = self._scroll.viewport()
        try:
            if obj is vp:
                et = event.type()
                if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        self._dragging   = True
                        self._drag_start = event.position().toPoint()
                        self._drag_cur   = self._drag_start
                        vp.setCursor(Qt.CursorShape.CrossCursor)
                        event.accept()
                        return True

                elif et == QEvent.Type.MouseMove and self._dragging:
                    self._drag_cur = event.position().toPoint()
                    self._render()
                    event.accept()
                    return True

                elif (et == QEvent.Type.MouseButtonRelease
                      and event.button() == Qt.MouseButton.LeftButton
                      and self._dragging):
                    self._dragging = False
                    vp.setCursor(Qt.CursorShape.ArrowCursor)
                    self._drag_cur = event.position().toPoint()

                    wp = self._pix_lbl.pos()
                    x1 = (self._drag_start.x() - wp.x()) / max(1e-6, self._zoom)
                    y1 = (self._drag_start.y() - wp.y()) / max(1e-6, self._zoom)
                    x2 = (self._drag_cur.x()   - wp.x()) / max(1e-6, self._zoom)
                    y2 = (self._drag_cur.y()   - wp.y()) / max(1e-6, self._zoom)

                    left  = int(max(0, min(x1, x2)))
                    top   = int(max(0, min(y1, y2)))
                    right = int(min(self._W, max(x1, x2)))
                    bot   = int(min(self._H, max(y1, y2)))
                    rw, rh = right - left, bot - top

                    if rw >= 16 and rh >= 16:
                        self._anchor = (left, top, rw, rh)
                        self.lbl_anchor_status.setText(
                            f"Anchor: x={left}, y={top}, w={rw}, h={rh}"
                        )
                        self.lbl_anchor_status.setStyleSheet("color:#4a4;")
                        self.btn_ok.setEnabled(True)

                    self._drag_start = None
                    self._drag_cur   = None
                    self._render()
                    event.accept()
                    return True

                elif (et == QEvent.Type.Wheel
                      and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    dy = event.angleDelta().y()
                    self._set_zoom(self._zoom * (1.25 if dy > 0 else 1 / 1.25))
                    event.accept()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def anchor(self) -> Optional[tuple]:
        return self._anchor


# ── Batch worker ──────────────────────────────────────────────────────────────
class _BatchWorker(QThread):
    file_started  = pyqtSignal(int)
    file_progress = pyqtSignal(int, int, str)
    # KEY FIX: emit the array as object so GUI thread can push it to SASpro
    file_done     = pyqtSignal(int, str, object)   # index, out_path, np.ndarray
    file_failed   = pyqtSignal(int, str)
    all_done      = pyqtSignal()

    def __init__(
        self,
        files: List[str],
        out_dir: str,
        cfgs: List[SERStackConfig],
        *,
        debayer: bool,
        ref_mode: str,
        ref_count: int,
        # NOTE: main is intentionally NOT stored here — all Qt object
        # creation must happen on the GUI thread via signals, never here.
    ):
        super().__init__()
        self._files     = files
        self._out_dir   = out_dir
        self._cfgs      = cfgs
        self._debayer   = bool(debayer)
        self._ref_mode  = ref_mode
        self._ref_count = int(ref_count)
        self._cancel    = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        for i, (path, cfg) in enumerate(zip(self._files, self._cfgs)):
            if self._cancel:
                break

            self.file_started.emit(i)

            try:
                def _cb(done, total, phase):
                    self.file_progress.emit(int(done), int(total), str(phase))

                # ── Analyze ───────────────────────────────────────────────
                ar: AnalyzeResult = analyze_ser(
                    cfg,
                    debayer=self._debayer,
                    to_rgb=False,
                    bayer_pattern=getattr(cfg, "bayer_pattern", None),
                    ref_mode=self._ref_mode,
                    ref_count=self._ref_count,
                    progress_cb=_cb,
                )

                if self._cancel:
                    break

                # ── Stack ─────────────────────────────────────────────────
                out, diag = stack_ser(
                    cfg.source,
                    roi=cfg.roi,
                    debayer=self._debayer,
                    to_rgb=False,
                    bayer_pattern=getattr(cfg, "bayer_pattern", None),
                    keep_percent=float(cfg.keep_percent),
                    track_mode=str(cfg.track_mode),
                    surface_anchor=cfg.surface_anchor,
                    analysis=ar,
                    local_warp=True,
                    progress_cb=_cb,
                    drizzle_scale=float(cfg.drizzle_scale),
                    drizzle_pixfrac=float(cfg.drizzle_pixfrac),
                    drizzle_kernel=str(cfg.drizzle_kernel),
                    drizzle_sigma=float(cfg.drizzle_sigma),
                )

                if self._cancel:
                    break

                # ── Save (file I/O only — no Qt here) ─────────────────────
                stem = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(self._out_dir, f"{stem}_stack.tif")

                try:
                    from setiastro.saspro.legacy.image_manager import save_image
                    save_image(
                        img_array=out,
                        filename=out_path,
                        original_format="tiff",
                        bit_depth=None,
                        original_header=None,
                        is_mono=(out.ndim == 2),
                        image_meta=None,
                        file_meta=None,
                    )
                except Exception:
                    import tifffile
                    tifffile.imwrite(out_path, out.astype(np.float32))

                # Emit array to GUI thread — _push_as_new_doc runs there, not here
                self.file_done.emit(i, out_path, out)

            except Exception as e:
                self.file_failed.emit(i, f"{e}\n\n{traceback.format_exc()}")

        self.all_done.emit()


# ── Main batch dialog ─────────────────────────────────────────────────────────
class SERBatchDialog(QDialog):

    def __init__(
        self,
        parent=None,
        *,
        main,
        initial_files: Optional[List[str]] = None,
        base_cfg_kwargs: dict,
        debayer: bool,
        track_mode: str,
        roi,
        surface_anchor,
        bayer_pattern: Optional[str],
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Planetary Stacker")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(820, 660)

        self._main           = main
        self._base_cfg_kw    = base_cfg_kwargs
        self._debayer        = bool(debayer)
        self._track_mode     = track_mode
        self._roi            = roi
        self._surface_anchor = surface_anchor
        self._bayer_pattern  = bayer_pattern
        self._out_dir        = ""
        self._worker: Optional[_BatchWorker] = None
        self._files: List[str] = []

        self._build_ui()

        if initial_files:
            self._add_files(initial_files)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # ── File list ──────────────────────────────────────────────────────
        gbF = QGroupBox("Input Files  (SER / AVI / MP4 / MOV / MKV only)", self)
        vF  = QVBoxLayout(gbF)

        self.lst = QListWidget(self)
        self.lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst.setMinimumHeight(200)
        vF.addWidget(self.lst, 1)

        file_btns = QHBoxLayout()
        self.btn_add    = QPushButton("Add Files…", self)
        self.btn_remove = QPushButton("Remove Selected", self)
        self.btn_clear  = QPushButton("Clear All", self)
        file_btns.addWidget(self.btn_add)
        file_btns.addWidget(self.btn_remove)
        file_btns.addWidget(self.btn_clear)
        file_btns.addStretch(1)
        vF.addLayout(file_btns)
        outer.addWidget(gbF, 1)

        # ── Settings summary + drizzle override ───────────────────────────
        gbS = QGroupBox("Stacking Settings  (inherited from viewer)", self)
        fS  = QFormLayout(gbS)

        fS.addRow("Track mode:", QLabel(self._track_mode.capitalize(), self))

        kp = self._base_cfg_kw.get("keep_percent", 20.0)
        fS.addRow("Keep %:", QLabel(f"{kp:.1f}%", self))

        roi_str = str(self._roi) if self._roi else "(full frame)"
        fS.addRow("ROI:", QLabel(roi_str, self))

        if self._track_mode == "surface":
            anc_str = (str(self._surface_anchor) if self._surface_anchor
                       else "⚠️ Not set — will be picked from first file")
            self.lbl_anchor = QLabel(anc_str, self)
            self.lbl_anchor.setStyleSheet(
                "color:#4a4;" if self._surface_anchor else "color:#c66;"
            )
            fS.addRow("Surface anchor:", self.lbl_anchor)

        # Drizzle is the one setting users commonly want to change per batch
        self.cmb_drizzle = QComboBox(self)
        self.cmb_drizzle.addItems(["Off (1×)", "1.5×", "2×"])
        self.cmb_drizzle.setToolTip(
            "Drizzle scale for stacking.\n"
            "1.5× ≈ 2.25× compute cost, 2× ≈ 4× compute cost."
        )
        fS.addRow("Drizzle:", self.cmb_drizzle)

        outer.addWidget(gbS, 0)

        # ── Output folder ──────────────────────────────────────────────────
        gbO = QGroupBox("Output", self)
        hO  = QHBoxLayout(gbO)
        self.lbl_out = QLabel("(not set)", self)
        self.lbl_out.setStyleSheet("color:#888;")
        self.btn_out = QPushButton("Choose Folder…", self)
        hO.addWidget(self.lbl_out, 1)
        hO.addWidget(self.btn_out, 0)
        outer.addWidget(gbO, 0)

        # ── Progress ───────────────────────────────────────────────────────
        self.lbl_overall = QLabel("", self)
        self.lbl_overall.setStyleSheet("color:#aaa;")
        outer.addWidget(self.lbl_overall, 0)

        self.prog_overall = QProgressBar(self)
        self.prog_overall.setVisible(False)
        outer.addWidget(self.prog_overall, 0)

        self.prog_file = QProgressBar(self)
        self.prog_file.setVisible(False)
        outer.addWidget(self.prog_file, 0)

        self.lbl_phase = QLabel("", self)
        self.lbl_phase.setStyleSheet("color:#aaa; font-size:11px;")
        outer.addWidget(self.lbl_phase, 0)

        # ── Log ────────────────────────────────────────────────────────────
        gbL = QGroupBox("Log", self)
        vL  = QVBoxLayout(gbL)
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        vL.addWidget(self.log)
        outer.addWidget(gbL, 0)

        # ── Action buttons ─────────────────────────────────────────────────
        act = QHBoxLayout()
        self.btn_run    = QPushButton("▶  Run Batch", self)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_close  = QPushButton("Close", self)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        act.addWidget(self.btn_run)
        act.addWidget(self.btn_cancel)
        act.addStretch(1)
        act.addWidget(self.btn_close)
        outer.addLayout(act, 0)

        # ── Signals ────────────────────────────────────────────────────────
        self.btn_add.clicked.connect(self._on_add_files)
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_out.clicked.connect(self._on_choose_out)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_close.clicked.connect(self.close)

    # ── File management ───────────────────────────────────────────────────────
    def _add_files(self, paths: List[str]):
        existing = set(self._files)
        added, skipped = 0, []
        for p in paths:
            if not _is_video(p):
                skipped.append(os.path.basename(p))
                continue
            if p in existing:
                continue
            self._files.append(p)
            existing.add(p)
            item = QListWidgetItem(f"⏳  {os.path.basename(p)}")
            item.setForeground(QBrush(_STATUS_COLORS[_ST_PENDING]))
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.lst.addItem(item)
            added += 1

        if skipped:
            self._log(f"Skipped {len(skipped)} non-video file(s): {', '.join(skipped)}")
        if added:
            self._log(f"Added {added} file(s). Total: {len(self._files)}")
        self._update_run_button()

    def _on_add_files(self):
        settings = QSettings()
        last_dir = settings.value("BatchStacker/last_open_dir", "", type=str) or ""

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", last_dir,
            "Planetary Videos (*.ser *.avi *.mp4 *.mov *.mkv);;All Files (*)"
        )
        if paths:
            settings.setValue("BatchStacker/last_open_dir", os.path.dirname(paths[0]))
            self._add_files(paths)

    def _on_remove_selected(self):
        rows = sorted(
            {self.lst.row(it) for it in self.lst.selectedItems()},
            reverse=True
        )
        for r in rows:
            self.lst.takeItem(r)
            self._files.pop(r)
        self._update_run_button()

    def _on_clear(self):
        self.lst.clear()
        self._files.clear()
        self._update_run_button()

    def _on_choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder", self._out_dir or "")
        if d:
            self._out_dir = d
            self.lbl_out.setText(d)
            self.lbl_out.setStyleSheet("color:#ddd;")
            self._update_run_button()

    def _update_run_button(self):
        self.btn_run.setEnabled(bool(self._files) and bool(self._out_dir))

    # ── Surface anchor ────────────────────────────────────────────────────────
    def _acquire_surface_anchor(self) -> bool:
        if self._track_mode != "surface":
            return True
        if self._surface_anchor is not None:
            return True

        first_file = self._files[0]
        self._log(f"Surface mode: acquiring anchor from: {os.path.basename(first_file)}")

        try:
            src = open_planetary_source(first_file, cache_items=2)
            try:
                img = src.get_frame(0, roi=self._roi, debayer=self._debayer,
                                    to_float01=True, force_rgb=False,
                                    bayer_pattern=self._bayer_pattern)
            finally:
                src.close()
        except Exception as e:
            QMessageBox.critical(self, "Batch", f"Could not open first file for anchor:\n{e}")
            return False

        ref = img[..., 0] if img.ndim == 3 else img

        dlg = _AnchorPickerDialog(self, ref)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.anchor() is None:
            return False

        self._surface_anchor = dlg.anchor()
        if hasattr(self, "lbl_anchor"):
            x, y, w, h = self._surface_anchor
            self.lbl_anchor.setText(f"✅ x={x}, y={y}, w={w}, h={h}")
            self.lbl_anchor.setStyleSheet("color:#4a4;")
        self._log(f"Surface anchor set: {self._surface_anchor}")
        return True

    # ── Run ───────────────────────────────────────────────────────────────────
    def _on_run(self):
        if not self._files or not self._out_dir:
            return
        if not self._acquire_surface_anchor():
            return

        # Resolve drizzle from combo
        scale_text = self.cmb_drizzle.currentText()
        if "1.5" in scale_text:
            drizzle_scale = 1.5
        elif "2" in scale_text:
            drizzle_scale = 2.0
        else:
            drizzle_scale = 1.0

        # Build per-file configs
        cfgs = []
        for path in self._files:
            kw = dict(self._base_cfg_kw)
            kw["track_mode"]     = self._track_mode
            kw["roi"]            = self._roi
            kw["surface_anchor"] = self._surface_anchor
            kw["bayer_pattern"]  = self._bayer_pattern
            kw["drizzle_scale"]  = drizzle_scale
            cfgs.append(SERStackConfig(source=path, **kw))

        # Reset list
        for row in range(self.lst.count()):
            item = self.lst.item(row)
            item.setText(f"⏳  {os.path.basename(self._files[row])}")
            item.setForeground(QBrush(_STATUS_COLORS[_ST_PENDING]))

        # Lock UI
        self.btn_run.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_out.setEnabled(False)
        self.cmb_drizzle.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)

        n = len(self._files)
        self.prog_overall.setRange(0, n)
        self.prog_overall.setValue(0)
        self.prog_overall.setVisible(True)
        self.prog_file.setRange(0, 100)
        self.prog_file.setValue(0)
        self.prog_file.setVisible(True)
        self.lbl_overall.setText(f"0 / {n} files done")
        self.lbl_phase.setText("")
        self._log(f"Starting batch: {n} file(s) → {self._out_dir}")

        # No `main` passed to worker — doc push happens in GUI thread via signal
        self._worker = _BatchWorker(
            self._files,
            self._out_dir,
            cfgs,
            debayer=self._debayer,
            ref_mode="best_frame",
            ref_count=5,
        )
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.request_cancel()
            self._log("Cancel requested — finishing current file…")
            self.btn_cancel.setEnabled(False)

    # ── Worker signals — all run on GUI thread ────────────────────────────────
    def _on_file_started(self, i: int):
        item = self.lst.item(i)
        if item:
            item.setText(f"🔄  {os.path.basename(self._files[i])}")
            item.setForeground(QBrush(_STATUS_COLORS[_ST_RUNNING]))
            self.lst.scrollToItem(item)
        self._log(f"[{i+1}/{len(self._files)}] Processing: {os.path.basename(self._files[i])}")
        self.prog_file.setValue(0)

    def _on_file_progress(self, done: int, total: int, phase: str):
        total = max(1, total)
        pct = int(round(100.0 * done / total))
        self.prog_file.setValue(pct)
        self.lbl_phase.setText(f"{phase}: {done}/{total} ({pct}%)")

    def _on_file_done(self, i: int, out_path: str, arr: object):
        item = self.lst.item(i)
        if item:
            item.setText(f"✅  {os.path.basename(self._files[i])}")
            item.setForeground(QBrush(_STATUS_COLORS[_ST_DONE]))

        # Safe: this runs on the GUI thread because it's connected via signal
        try:
            _push_as_new_doc(
                self._main,
                None,
                arr,
                title_suffix="_stack",
                source="Batch Planetary Stacker",
                source_path=self._files[i],
            )
        except Exception:
            pass

        done = sum(
            1 for r in range(self.lst.count())
            if self.lst.item(r) and self.lst.item(r).text().startswith("✅")
        )
        self.prog_overall.setValue(done)
        self.lbl_overall.setText(f"{done} / {len(self._files)} files done")
        self._log(f"  ✅ Saved → {out_path}")

    def _on_file_failed(self, i: int, msg: str):
        item = self.lst.item(i)
        if item:
            item.setText(f"❌  {os.path.basename(self._files[i])}")
            item.setForeground(QBrush(_STATUS_COLORS[_ST_FAILED]))
        self._log(f"  ❌ Failed: {os.path.basename(self._files[i])}")
        self._log(f"     {msg.splitlines()[0]}")

    def _on_all_done(self):
        self.prog_file.setVisible(False)
        self.lbl_phase.setText("")
        self.btn_run.setEnabled(bool(self._files) and bool(self._out_dir))
        self.btn_add.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_out.setEnabled(True)
        self.cmb_drizzle.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

        done_count = sum(
            1 for r in range(self.lst.count())
            if self.lst.item(r) and self.lst.item(r).text().startswith("✅")
        )
        fail_count = len(self._files) - done_count
        self.lbl_overall.setText(
            f"Batch complete: {done_count} succeeded, {fail_count} failed"
        )
        self._log(f"Batch complete. {done_count}/{len(self._files)} succeeded → {self._out_dir}")

        if done_count > 0:
            QMessageBox.information(
                self, "Batch Complete",
                f"{done_count} stack(s) saved to:\n{self._out_dir}"
                + (f"\n\n{fail_count} file(s) failed — see log." if fail_count else "")
            )

    def _log(self, s: str):
        try:
            self.log.append(s)
        except Exception:
            pass