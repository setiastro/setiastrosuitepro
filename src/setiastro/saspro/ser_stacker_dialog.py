# src/setiastro/saspro/ser_stacker_dialog.py
from __future__ import annotations

import traceback
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFormLayout, QComboBox, QDoubleSpinBox, QCheckBox, QTextEdit, QProgressBar
)

from setiastro.saspro.ser_stack_config import SERStackConfig
from setiastro.saspro.ser_stacker import stack_ser


def _derive_view_base_title(main, doc) -> str:
    """
    Prefer the active view's title (respecting per-view rename/override),
    fallback to the document display name, then to doc.name, and finally 'Image'.
    Also strips any decorations if available.
    """
    # 1) Ask main for a subwindow for this document, if it exposes a helper
    try:
        if hasattr(main, "_subwindow_for_document"):
            sw = main._subwindow_for_document(doc)
            if sw:
                w = sw.widget() if hasattr(sw, "widget") else sw
                if hasattr(w, "_effective_title"):
                    t = w._effective_title() or ""
                else:
                    t = sw.windowTitle() if hasattr(sw, "windowTitle") else ""
                if hasattr(w, "_strip_decorations"):
                    t, _ = w._strip_decorations(t)
                if t.strip():
                    return t.strip()
    except Exception:
        pass

    # 2) Try scanning MDI for a subwindow whose widget holds this document
    try:
        mdi = (getattr(main, "mdi_area", None)
               or getattr(main, "mdiArea", None)
               or getattr(main, "mdi", None))
        if mdi and hasattr(mdi, "subWindowList"):
            for sw in mdi.subWindowList():
                w = sw.widget()
                if getattr(w, "document", None) is doc:
                    t = sw.windowTitle() if hasattr(sw, "windowTitle") else ""
                    if hasattr(w, "_strip_decorations"):
                        t, _ = w._strip_decorations(t)
                    if t.strip():
                        return t.strip()
    except Exception:
        pass

    # 3) Fallback to document's display name (then name, then generic)
    try:
        if hasattr(doc, "display_name"):
            t = doc.display_name()
            if t and t.strip():
                return t.strip()
    except Exception:
        pass

    return (getattr(doc, "name", "") or "Image").strip()


def _push_as_new_doc(main, source_doc, arr: np.ndarray, *, title_suffix="_stack", source="SER Stacking"):
    dm = getattr(main, "docman", None)
    if not dm or not hasattr(dm, "open_array"):
        return None

    try:
        base = "SER"
        if source_doc is not None:
            base = _derive_view_base_title(main, source_doc) or base

        title = base if (title_suffix and base.endswith(title_suffix)) else f"{base}{title_suffix}"

        x = np.asarray(arr)
        # keep mono mono
        if x.ndim == 3 and x.shape[2] == 1:
            x = x[..., 0]
        x = x.astype(np.float32, copy=False)

        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": bool(x.ndim == 2),
            "source": source,
        }

        newdoc = dm.open_array(x, metadata=meta, title=title)

        if hasattr(main, "_spawn_subwindow_for"):
            main._spawn_subwindow_for(newdoc)

        return newdoc
    except Exception:
        return None

class _StackWorker(QThread):
    finished_ok = pyqtSignal(object, object)   # out(np.ndarray), diag(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg: SERStackConfig, *, debayer: bool, to_rgb: bool):
        super().__init__()
        self.cfg = cfg
        self.debayer = bool(debayer)
        self.to_rgb = bool(to_rgb)

    def run(self):
        try:
            out, diag = stack_ser(self.cfg, debayer=self.debayer, to_rgb=self.to_rgb)
            self.finished_ok.emit(out, diag)   # GUI thread will handle doc creation
        except Exception as e:
            msg = f"{e}\n\n{traceback.format_exc()}"
            self.failed.emit(msg)


class SERStackerDialog(QDialog):
    """
    Dedicated stacking UI (AutoStakkert-like direction):
    - Keeps viewer separate from stacking.
    - V1: track mode, keep %, uses ROI + optional surface anchor from viewer.
    - Later: alignment points (manual/auto), quality graph, drizzle, etc.
    """

    # Main app can connect this to "push to new view"
    stackProduced = pyqtSignal(object, object)  # out(np.ndarray), diag(dict)

    def __init__(
        self,
        parent=None,
        *,
        main,             
        source_doc,               
        ser_path: str,
        roi=None,                 # (x,y,w,h) full-frame coords or None
        track_mode: str = "planetary",
        surface_anchor=None,      # (x,y,w,h) ROI-space or None
        debayer: bool = True,
        keep_percent: float = 20.0,
    ):
        super().__init__(parent)
        self.setWindowTitle("SER Stacker")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._main = main
        self._source_doc = source_doc
        self._ser_path = ser_path
        self._roi = roi
        self._surface_anchor = surface_anchor
        self._debayer = bool(debayer)

        self._worker: _StackWorker | None = None
        self._last_out: np.ndarray | None = None
        self._last_diag: dict | None = None

        self._build_ui()

        # defaults
        self.cmb_track.setCurrentText(
            "Planetary" if track_mode == "planetary" else ("Surface" if track_mode == "surface" else "Off")
        )
        self.spin_keep.setValue(float(keep_percent))
        self.chk_debayer.setChecked(bool(debayer))
        self._update_anchor_warning()

        self._append_log(f"SER: {ser_path}")
        self._append_log(f"ROI: {roi if roi is not None else '(full frame)'}")
        if track_mode == "surface":
            self._append_log(f"Surface anchor (ROI-space): {surface_anchor}")

    # ---------------- UI ----------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # --- Options ---
        gb = QGroupBox("Stack Settings", self)
        form = QFormLayout(gb)

        self.cmb_track = QComboBox(self)
        self.cmb_track.addItems(["Planetary", "Surface", "Off"])

        self.spin_keep = QDoubleSpinBox(self)
        self.spin_keep.setRange(0.1, 100.0)
        self.spin_keep.setDecimals(1)
        self.spin_keep.setSingleStep(1.0)
        self.spin_keep.setValue(20.0)

        self.chk_debayer = QCheckBox("Debayer (Bayer SER)", self)
        self.chk_debayer.setChecked(True)

        self.lbl_anchor = QLabel("", self)
        self.lbl_anchor.setWordWrap(True)

        form.addRow("Tracking", self.cmb_track)
        form.addRow("Keep %", self.spin_keep)
        form.addRow("", self.chk_debayer)
        form.addRow("Surface anchor", self.lbl_anchor)

        outer.addWidget(gb, 0)

        # --- Actions row ---
        row = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze (soon)", self)
        self.btn_analyze.setEnabled(False)  # placeholder: next milestone (quality graph + cutoff)
        self.btn_stack = QPushButton("Stack Now", self)
        self.btn_close = QPushButton("Close", self)

        row.addWidget(self.btn_analyze)
        row.addStretch(1)
        row.addWidget(self.btn_stack)
        row.addWidget(self.btn_close)

        outer.addLayout(row)

        # --- Progress + log ---
        self.prog = QProgressBar(self)
        self.prog.setRange(0, 0)
        self.prog.setVisible(False)
        outer.addWidget(self.prog)

        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        outer.addWidget(self.log, 1)

        # --- Signals ---
        self.btn_close.clicked.connect(self.close)
        self.btn_stack.clicked.connect(self._start_stack)
        self.cmb_track.currentIndexChanged.connect(self._update_anchor_warning)

    # ---------------- helpers ----------------

    def _append_log(self, s: str):
        try:
            self.log.append(s)
        except Exception:
            pass

    def _track_mode_value(self) -> str:
        t = self.cmb_track.currentText().strip().lower()
        if t.startswith("planet"):
            return "planetary"
        if t.startswith("surface"):
            return "surface"
        return "off"

    def _update_anchor_warning(self):
        mode = self._track_mode_value()
        if mode != "surface":
            self.lbl_anchor.setText("(not used)")
            self.lbl_anchor.setStyleSheet("color:#888;")
            return

        if self._surface_anchor is None:
            self.lbl_anchor.setText("REQUIRED (set in SER Viewer with Ctrl+Shift+drag)")
            self.lbl_anchor.setStyleSheet("color:#c66;")
        else:
            x, y, w, h = self._surface_anchor
            self.lbl_anchor.setText(f"x={x}, y={y}, w={w}, h={h} (ROI-space)")
            self.lbl_anchor.setStyleSheet("color:#4a4;")

    # ---------------- actions ----------------

    def _start_stack(self):
        if not self._ser_path:
            return

        mode = self._track_mode_value()
        if mode == "surface" and self._surface_anchor is None:
            self._append_log("Surface mode requires an anchor. Set it in the viewer (Ctrl+Shift+drag).")
            return

        cfg = SERStackConfig(
            ser_path=self._ser_path,
            roi=self._roi,
            track_mode=mode,
            surface_anchor=self._surface_anchor,
            keep_percent=float(self.spin_keep.value()),
        )

        debayer = bool(self.chk_debayer.isChecked())

        self.btn_stack.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.prog.setVisible(True)
        self._append_log("Stacking...")

        self._worker = _StackWorker(cfg, debayer=debayer, to_rgb=False)
        self._worker.finished_ok.connect(self._on_stack_ok)
        self._worker.failed.connect(self._on_stack_fail)
        self._worker.start()

    def _on_stack_ok(self, out, diag):
        self._last_out = out
        self._last_diag = diag

        self.prog.setVisible(False)
        self.btn_stack.setEnabled(True)
        self.btn_close.setEnabled(True)

        self._append_log(f"Done. Kept {diag.get('frames_kept')} / {diag.get('frames_total')}")
        self._append_log(f"Track: {diag.get('track_mode')}  ROI: {diag.get('roi_used')}")

        # ✅ Create the new stacked document (GUI thread)
        newdoc = _push_as_new_doc(
            self._main,
            self._source_doc,
            out,
            title_suffix="_stack",
            source="SER Stack"
        )

        # Optional: stash diag on the document metadata (handy later)
        if newdoc is not None:
            try:
                md = getattr(newdoc, "metadata", None)
                if md is None:
                    md = {}
                    setattr(newdoc, "metadata", md)
                md["ser_stack_diag"] = diag
            except Exception:
                pass

        # Keep emitting too (so other callers can hook it)
        self.stackProduced.emit(out, diag)


    def _on_stack_fail(self, msg: str):
        self.prog.setVisible(False)
        self.btn_stack.setEnabled(True)
        self.btn_close.setEnabled(True)
        self._append_log("FAILED:")
        self._append_log(msg)
