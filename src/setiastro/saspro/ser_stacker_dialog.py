# src/setiastro/saspro/ser_stacker_dialog.py
from __future__ import annotations

import traceback
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtWidgets import (QWidget, QSpinBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFormLayout, QComboBox, QDoubleSpinBox, QCheckBox, QTextEdit, QProgressBar
)
from PyQt6.QtGui import QPainter, QPen, QColor

from setiastro.saspro.ser_stack_config import SERStackConfig

from setiastro.saspro.ser_stacker import stack_ser, analyze_ser, AnalyzeResult



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


class QualityGraph(QWidget):
    """
    AS-style quality plot (sorted curve expected):
    - Curve: q[0] best ... q[N-1] worst
    - Vertical cutoff line at keep_k
    - Dashed horizontal median line (q50)
    - Minimal axis labels: Best/Worst, min/median/max
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: np.ndarray | None = None
        self._keep_k: int | None = None
        self.setMinimumHeight(160)

    def set_data(self, q: np.ndarray | None, keep_k: int | None = None):
        self._q = None if q is None else np.asarray(q, dtype=np.float32)
        self._keep_k = keep_k
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # room for labels
        r = self.rect().adjusted(34, 10, -10, -22)

        # frame
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawRect(r)

        if self._q is None or self._q.size < 2:
            p.setPen(QPen(QColor(160, 160, 160), 1))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Analyze to see quality graph")
            p.end()
            return

        q = self._q
        N = int(q.size)
        qmin = float(np.min(q))
        qmax = float(np.max(q))
        if qmax <= qmin + 1e-12:
            qmax = qmin + 1e-6

        def y_for(val: float) -> float:
            return r.bottom() - ((val - qmin) / (qmax - qmin)) * r.height()

        # ---- median dashed line ----
        qmed = qmin + 0.5 * (qmax - qmin)
        ymed = y_for(qmed)
        pen_med = QPen(QColor(120, 120, 120), 1)
        pen_med.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen_med)
        p.drawLine(int(r.left()), int(ymed), int(r.right()), int(ymed))

        # ---- curve ----
        p.setPen(QPen(QColor(0, 220, 0), 2))
        lastx = lasty = None
        for i in range(N):
            x = r.left() + (i / (N - 1)) * r.width()
            y = y_for(float(q[i]))
            if lastx is not None:
                p.drawLine(int(lastx), int(lasty), int(x), int(y))
            lastx, lasty = x, y

        # ---- cutoff line ----
        if self._keep_k is not None and N > 1:
            k = int(max(1, min(N, self._keep_k)))
            xcut = r.left() + ((k - 1) / (N - 1)) * r.width()
            p.setPen(QPen(QColor(255, 220, 0), 2))
            p.drawLine(int(xcut), int(r.top()), int(xcut), int(r.bottom()))

        # ---- labels ----
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawText(self.rect().adjusted(6, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, "Best")
        p.drawText(self.rect().adjusted(0, 0, -6, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, "Worst")

        # y labels: max, median, min
        p.drawText(4, int(r.top()) + 10, f"{qmax:.3g}")
        p.drawText(4, int(ymed) + 4,  f"{qmed:.3g}")
        p.drawText(4, int(r.bottom()), f"{qmin:.3g}")

        p.end()

class _AnalyzeWorker(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, phase
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cfg: SERStackConfig, *, debayer: bool, to_rgb: bool, ref_mode: str, ref_count: int):
        super().__init__()
        self.cfg = cfg
        self.debayer = bool(debayer)
        self.to_rgb = bool(to_rgb)
        self.ref_mode = ref_mode
        self.ref_count = int(ref_count)
        self._cancel = False

    def run(self):
        try:
            def cb(done: int, total: int, phase: str):
                self.progress.emit(int(done), int(total), str(phase))

            ar = analyze_ser(
                self.cfg,
                debayer=self.debayer,
                to_rgb=self.to_rgb,
                ref_mode=self.ref_mode,
                ref_count=self.ref_count,
                progress_cb=cb,
            )
            self.finished_ok.emit(ar)
        except Exception as e:
            msg = f"{e}\n\n{traceback.format_exc()}"
            self.failed.emit(msg)

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
            analysis = getattr(self, "_analysis", None)
            out, diag = stack_ser(self.cfg, debayer=self.debayer, to_rgb=self.to_rgb, analysis=analysis)
            self.finished_ok.emit(out, diag)
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
        self._analysis: AnalyzeResult | None = None
        self._worker_analyze: _AnalyzeWorker | None = None
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
        # --- Analysis / Reference / APs ---
        gbA = QGroupBox("Analyze", self)
        fA = QFormLayout(gbA)

        self.cmb_ref = QComboBox(self)
        self.cmb_ref.addItems(["Best frame", "Best stack (N)"])

        self.spin_refN = QSpinBox(self)
        self.spin_refN.setRange(2, 200)
        self.spin_refN.setValue(10)

        self.graph = QualityGraph(self)

        self.chk_multipoint = QCheckBox("Multi-point alignment (APs) — soon", self)
        self.chk_multipoint.setChecked(False)
        self.chk_multipoint.setEnabled(True)  # enable now just as UI; stacker will ignore for now

        self.spin_ap_size = QSpinBox(self)
        self.spin_ap_size.setRange(16, 256)
        self.spin_ap_size.setSingleStep(8)
        self.spin_ap_size.setValue(64)

        self.spin_ap_spacing = QSpinBox(self)
        self.spin_ap_spacing.setRange(8, 256)
        self.spin_ap_spacing.setSingleStep(8)
        self.spin_ap_spacing.setValue(48)

        fA.addRow("Reference", self.cmb_ref)
        fA.addRow("Ref stack N", self.spin_refN)
        fA.addRow("", self.chk_multipoint)
        fA.addRow("AP size (px)", self.spin_ap_size)
        fA.addRow("AP spacing (px)", self.spin_ap_spacing)

        outer.addWidget(gbA, 0)
        outer.addWidget(self.graph, 0)

        row = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze", self)
        self.btn_analyze.setEnabled(True)        
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
        self.lbl_prog = QLabel("", self)
        self.lbl_prog.setStyleSheet("color:#aaa;")
        self.lbl_prog.setVisible(False)
        outer.addWidget(self.lbl_prog)

        self.prog = QProgressBar(self)
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
        self.btn_analyze.clicked.connect(self._start_analyze)

        # When keep% changes, update cutoff line if analyzed
        self.spin_keep.valueChanged.connect(self._update_graph_cutoff)
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
    def _start_analyze(self):
        mode = self._track_mode_value()
        if mode == "surface" and self._surface_anchor is None:
            self._append_log("Surface mode requires an anchor. Set it in the viewer (Ctrl+Shift+drag).")
            return

        ref_mode = "best_stack" if self.cmb_ref.currentText().lower().startswith("best stack") else "best_frame"
        refN = int(self.spin_refN.value()) if ref_mode == "best_stack" else 1

        cfg = SERStackConfig(
            ser_path=self._ser_path,
            roi=self._roi,
            track_mode=mode,
            surface_anchor=self._surface_anchor,
            keep_percent=float(self.spin_keep.value()),
        )

        self.btn_analyze.setEnabled(False)
        self.btn_stack.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.lbl_prog.setVisible(True)
        self.lbl_prog.setText("Analyzing…")
        self.prog.setVisible(True)
        self.prog.setRange(0, 100)
        self.prog.setValue(0)

        self._worker_analyze = _AnalyzeWorker(cfg, debayer=bool(self.chk_debayer.isChecked()), to_rgb=False,
                                            ref_mode=ref_mode, ref_count=refN)
        self._worker_analyze.finished_ok.connect(self._on_analyze_ok)
        self._worker_analyze.failed.connect(self._on_analyze_fail)
        self._worker_analyze.progress.connect(self._on_analyze_progress)
        self._worker_analyze.start()

    def _on_analyze_progress(self, done: int, total: int, phase: str):
        total = max(1, int(total))
        done = max(0, min(total, int(done)))
        pct = int(round(100.0 * done / total))
        self.prog.setRange(0, 100)
        self.prog.setValue(pct)
        self.lbl_prog.setText(f"{phase}: {done}/{total} ({pct}%)")


    def _on_analyze_ok(self, ar: AnalyzeResult):
        self._analysis = ar
        self.prog.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_stack.setEnabled(True)
        self.btn_close.setEnabled(True)

        self._append_log(f"Analyze done. frames={ar.frames_total}  track={ar.track_mode}")
        self._append_log(f"Ref: {ar.ref_mode} (N={ar.ref_count})")

        # update graph (time-order) + cutoff marker based on keep%
        k = int(round(ar.frames_total * (float(self.spin_keep.value()) / 100.0)))
        k = max(1, min(ar.frames_total, k))
        q_sorted = ar.quality[ar.order]
        self.graph.set_data(q_sorted, keep_k=k)

    def _on_analyze_fail(self, msg: str):
        self.prog.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_stack.setEnabled(True)
        self.btn_close.setEnabled(True)
        self._append_log("ANALYZE FAILED:")
        self._append_log(msg)

    def _update_graph_cutoff(self):
        if self._analysis is None:
            return
        n = int(self._analysis.frames_total)
        k = int(round(n * (float(self.spin_keep.value()) / 100.0)))
        k = max(1, min(n, k))
        q_sorted = self._analysis.quality[self._analysis.order]
        self.graph.set_data(q_sorted, keep_k=k)


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
        self.btn_analyze.setEnabled(False)
        self.prog.setVisible(True)
        self._append_log("Stacking...")

        # pass analysis through the worker by storing on self (simple)
        self._worker = _StackWorker(cfg, debayer=debayer, to_rgb=False)
        self._worker._analysis = self._analysis  # attach dynamically (or extend worker ctor cleanly)
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
