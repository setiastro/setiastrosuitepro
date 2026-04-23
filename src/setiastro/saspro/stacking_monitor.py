# ============================================================
#  ____       _   _ _    _       _
# / ___|  ___| |_(_) |  / \  ___| |_ _ __ ___
# \___ \ / _ \ __| | | / _ \/ __| __| '__/ _ \
#  ___) |  __/ |_| | |/ ___ \__ \ |_| | | (_) |
# |____/ \___|\__|_|_/_/   \_\___/\__|_|  \___/
#
#  SASpro – Stacking Execution Monitor
#  Franklin Marek  |  www.setiastro.com
# ============================================================
from __future__ import annotations

import re
import time
from typing import Optional

from PyQt6.QtCore import Qt, QSettings, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHeaderView, QWidget,
)

# ── palette (matches SASpro dark theme) ──────────────────────────────────
_BG      = "#1a1a2e"
_PANEL   = "#16213e"
_ACCENT  = "#e94560"
_ACCENT2 = "#0f3460"
_FG      = "#eaeaea"
_DIM     = "#888888"
_GREEN   = "#4caf50"
_YELLOW  = "#ffc107"
_RED     = "#e94560"
_ORANGE  = "#ff9800"


# ── Column indices ────────────────────────────────────────────────────────
_COL_OP      = 0
_COL_GROUP   = 1
_COL_ELAPSED = 2
_COL_STATUS  = 3
_COL_NOTE    = 4
_NCOLS       = 5

# ── Status constants ──────────────────────────────────────────────────────
_ST_RUNNING = "running"
_ST_OK      = "success"
_ST_WARN    = "warning"
_ST_FAIL    = "failed"
_ST_INFO    = "info"


# ═══════════════════════════════════════════════════════════════════════════
#  Message classifier
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (regex_pattern, operation_label, status_hint, group_capture_group_or_None)
# Patterns are tried in order; first match wins.
_RULES: list[tuple[re.Pattern, str, str, Optional[int]]] = []

def _r(pat: str, op: str, st: str, grp: Optional[int] = None):
    _RULES.append((re.compile(pat, re.IGNORECASE), op, st, grp))

# ── Master dark ───────────────────────────────────────────────────────────
_r(r"Starting Master Dark",                      "Master Darks",     _ST_RUNNING)
_r(r"🟢 Processing \d+ darks for (.+?) in session", "Master Darks",  _ST_RUNNING, 1)
_r(r"✅ Master Dark saved and added to UI", "Master Darks", _ST_OK)
_r(r"⚠️.*dark.*skip",                            "Master Darks",     _ST_WARN)

# ── Master flat ───────────────────────────────────────────────────────────
_r(r"Starting Master Flat",                      "Master Flats",     _ST_RUNNING)
_r(r"🟢 Processing \d+ flats for .+?\[(.+?)\]",  "Master Flats",     _ST_RUNNING, 1)
_r(r"✅ Master Flat saved",                       "Master Flats",     _ST_OK)
_r(r"⚠️.*flat.*skip",                            "Master Flats",     _ST_WARN)

# ── Calibration ───────────────────────────────────────────────────────────
_r(r"Dark Subtracted",                           "Calibration",      _ST_RUNNING)
_r(r"Flat Applied",                              "Calibration",      _ST_RUNNING)
_r(r"✅ Calibration Complete",                   "Calibration",      _ST_OK)
_r(r"Saved: .+\.fit \(\d+/(\d+)\)",             "Calibration",      _ST_RUNNING)
_r(r"❌ (?:ERROR|CALIBRATION ERROR)",            "Calibration",      _ST_FAIL)



# Replace the existing Measurement, Normalization, Registration rules with:

# ── Measurements ─────────────────────────────────────────────────────────
_r(r"📏 Phase: Measurements starting",           "Measurements",     _ST_RUNNING)
_r(r"📏 Phase: Measurements complete",           "Measurements",     _ST_OK)

# ── Normalization ─────────────────────────────────────────────────────────
_r(r"📏 Phase: Normalization starting",          "Normalization",    _ST_RUNNING)
_r(r"📏 Phase: Normalization complete",          "Normalization",    _ST_OK)

# ── Reference frame ───────────────────────────────────────────────────────
_r(r"Auto-selected reference|Using user-specified reference", "Ref Frame Selection", _ST_OK)
_r(r"📌 Reference for alignment",               "Ref Frame Selection", _ST_INFO)

# ── Star alignment ────────────────────────────────────────────────────────
_r(r"📏 Phase: Star alignment starting",         "Registration",     _ST_RUNNING)
_r(r"📏 Phase: Star alignment complete",         "Registration",     _ST_OK)
_r(r"Alignment summary: (\d+ succeeded)",        "Registration",     _ST_OK,      1)
_r(r"🚨 Rejected \d+ frame",                    "Registration",     _ST_WARN)

# ── Integration ───────────────────────────────────────────────────────────
_r(r"📏 Phase: Integration starting",            "Integration",      _ST_RUNNING)
_r(r"Starting integration for group '(.+?)' with", "Integration",    _ST_RUNNING, 1)
_r(r"📊 Stacking group '(.+?)' with (.+)",       "Integration",      _ST_RUNNING, 1)
_r(r"Post-align finalize from prepass",          "Integration",      _ST_RUNNING)
_r(r"🔹 .* Finalizing '(.+?)' from prepass",     "Integration",      _ST_RUNNING, 1)
_r(r"✅ Saved integrated image.*for '(.+?)'",    "Integration",      _ST_OK,      1)
_r(r"📐 Drizzle for '(.+?)'",                    "Integration",      _ST_RUNNING, 1)

# ── Autocrop ──────────────────────────────────────────────────────────────
_r(r"✂️.*[Cc]rop",                              "Autocrop",         _ST_RUNNING)
_r(r"✂️ Saved auto-cropped",                    "Autocrop",         _ST_OK)

# ── Astrometric solution (SASD / dither) ─────────────────────────────────
_r(r"Transform file saved.*\.sasd \(v2\)", "Alignment Transforms", _ST_OK)

# ── Generic success / warning / error catch-alls ─────────────────────────
_r(r"^✅",   "Complete",  _ST_OK)
_r(r"^⚠️",  "Warning",   _ST_WARN)
_r(r"^❌",   "Error",     _ST_FAIL)

# ── MF Deconvolution ──────────────────────────────────────────────────────
_r(r"MFDeconv launched for (\d+) group",         "MF Deconvolution", _ST_RUNNING, 1)
_r(r"Deconvolving '(.+?)' \(\d+ frames\)",        "MF Deconvolution", _ST_RUNNING, 1)
_r(r"⚙️ MFDeconv engine",                        "MF Deconvolution", _ST_RUNNING)
_r(r"✅ MFDeconv complete for all groups",        "MF Deconvolution", _ST_OK)
_r(r"⚠️ MFDeconv finished with failures",        "MF Deconvolution", _ST_WARN)
_r(r"❌ MFDeconv failed for '(.+?)'",            "MF Deconvolution", _ST_FAIL,    1)
_r(r"✂️ \(MF\) Saved auto-cropped",              "MF Deconvolution", _ST_OK)
_r(r"Post-MF finalize failed",                   "MF Deconvolution", _ST_WARN)

# ── Comet stacking ────────────────────────────────────────────────────────
_r(r"🌠 Comet mode enabled",                     "Comet Stack",      _ST_RUNNING)
_r(r"🟢 Measuring comet centers",                "Comet Stack",      _ST_RUNNING)
_r(r"🟠 Comet-aligned integration",              "Comet Stack",      _ST_RUNNING)
_r(r"✅ Saved CometOnly",                        "Comet Stack",      _ST_OK)
_r(r"🌠 Comet anchor in reference frame",        "Comet Stack",      _ST_INFO)
_r(r"◦ user confirmed",                          "Comet Stack",      _ST_INFO)
_r(r"◦ seed xy=",                               "Comet Stack",      _ST_INFO)

# ── Comet star removal ────────────────────────────────────────────────────
_r(r"✨ Comet star removal enabled",             "Comet StarRemoval", _ST_RUNNING)
_r(r"✓ \[\d+/\d+\] starless saved",             "Comet StarRemoval", _ST_RUNNING)
_r(r"✨ Using comet-aligned STARLESS",           "Comet StarRemoval", _ST_OK)
_r(r"⚠️ star removal failed",                   "Comet StarRemoval", _ST_WARN)
_r(r"⚠️ Comet star removal pre-process aborted", "Comet StarRemoval", _ST_FAIL)

# ── Comet blend ───────────────────────────────────────────────────────────
_r(r"🟡 Blending Stars\+Comet",                 "Comet Blend",      _ST_RUNNING)
_r(r"✅ Saved CometBlend",                       "Comet Blend",      _ST_OK)

# ── Noise patterns to suppress (tile-level chatter) ──────────────────────
_SUPPRESS = re.compile(
    r"tile \d+/\d+"
    r"|Creating temp memmap"
    r"|memmap:"
    r"|chunk \d+/\d+.*frames"
    r"|Before saving:"
    r"|min ="
    r"|max ="
    r"|LIGHT final"
    r"|Aligning stars… \(\d"
    r"|🗂️"
    r"|📦 \d+ tiles"
    r"|🧭 Total tiles"
    r"|⚙️.*reducer"
    r"|🌍 Load"
    r"|🌍 Meas"
    r"|✅ Master Dark saved: "
    r"|scale_guess"
    r"|✅ Prepass '"
    r"|🔹 \[\d+/\d+\] Finalizing"
    r"|Rejection prepass:"
    r"|🔹 .* Finalizing '.+?' from prepass"    
    r"|preview median"
    r"|Debayer"
    r"|↻ 180"
    r"|🔧 Resamp"
    r"|📏 Pixel-scale norm"
    r"|🔎 CFA"
    r"|💡 For best"
    r"|refine_if"
    r"|Image Registration Started"
    r"|🔄 Image Registration"
    r"|Aligning stars"
    r"|Measuring chunk"
    r"|✅ All chunks complete"
    r"|🌀 Normalizing chunk"
    r"|Updated self\.light_files"    
    r"|Transform file saved.*alignment_transforms\.sasd(?! \(v2\))"
    r"|_n\.fit"
    r"|ABE Poly"
    r"|Gradient removal"
    r"|🧪 MF prepass payload"
    r"|🌟 MFDeconv star-mask"
    r"|MF prepass payload for"
    r"|Integrating comet tile \d+"
    r"|◦ applying rejection algorithm"
    r"|◦ Tile source: STARLESS"
    r"|◦ DarkStar comet"
    r"|◦ StarNet comet"
    r"|◦ using user seed"
    r"|◦ seeding first registered"
    r"|Comet High-Clip"
    r"|Comet Lower-Trim"
    r"|comet preview skipped"
    r"|🌟 MFDeconv star-mask reference"    
    r"|rejection_map="
    r"|seed_image="    
    r"|🧪 Rejection prepass: \d+ group"
    r"|🔹 \[\d+/\d+\] Rejection prepass",
    re.IGNORECASE
)

def _classify(msg: str) -> Optional[tuple[str, str, str]]:
    """
    Returns (operation, group_hint, status) or None if message should be suppressed.
    """
    # Strip ANSI escapes and leading whitespace
    clean = re.sub(r"\x1b\[[0-9;]*m", "", msg).strip()

    if _SUPPRESS.search(clean):
        return None

    for pat, op, st, grp_idx in _RULES:
        m = pat.search(clean)
        if m:
            group = ""
            if grp_idx is not None:
                try:
                    group = m.group(grp_idx).strip()
                except Exception:
                    group = ""
            return op, group, st

    return None   # unrecognised — suppress


# ═══════════════════════════════════════════════════════════════════════════
#  MonitorRow — one logical step tracked by the monitor
# ═══════════════════════════════════════════════════════════════════════════
class _MonitorRow:
    def __init__(self, operation: str, group: str, status: str, note: str = ""):
        self.operation  = operation
        self.group      = group
        self.status     = status
        self.note       = note
        self.t_start    = time.monotonic()
        self.t_end: Optional[float] = None

    def elapsed_str(self) -> str:
        t = (self.t_end or time.monotonic()) - self.t_start
        if t < 60:
            return f"{t:.0f}s"
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"

    def finish(self, status: str, note: str = ""):
        self.t_end  = time.monotonic()
        self.status = status
        if note:
            self.note = note


# ═══════════════════════════════════════════════════════════════════════════
#  StackingMonitorDialog
# ═══════════════════════════════════════════════════════════════════════════
class StackingMonitorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stacking Execution Monitor")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.setMinimumSize(960, 500)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._settings  = QSettings("SetiAstro", "SASpro")
        self._rows: list[_MonitorRow] = []
        self._open: dict[str, int] = {}
        self._run_start: Optional[float] = None
        self._log_bus   = None

        self._build_ui()
        self._restore_geometry()
        self._connect_bus()
        from PyQt6.QtCore import QTimer
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick_elapsed)
        self._tick_timer.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e6e6e6;
            }
            QLabel#monitor_title {
                font-size: 16px;
                font-weight: bold;
                color: #FF4500;
            }
            QTableWidget {
                background-color: #2b2b2b;
                color: #e6e6e6;
                gridline-color: #444444;
                border: 1px solid #555;
                border-radius: 6px;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-right: 1px solid #555;
                padding: 5px 8px;
            }
            QTableWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #333;
            }
            QTableWidget::item:selected {
                background-color: #FF4500;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #dddddd;
                border: 1px solid #666;
                border-radius: 5px;
                padding: 5px 14px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #FF4500;
                color: #ffffff;
                border: 1px solid #FF4500;
            }
            QLabel#elapsed_label {
                color: #bbbbbb;
                font-size: 11px;
            }
        """)

        # header row
        hdr_row = QHBoxLayout()
        title = QLabel("Stacking Execution Monitor")
        title.setObjectName("monitor_title")
        hdr_row.addWidget(title)
        hdr_row.addStretch(1)
        root.addLayout(hdr_row)

        # table
        self._table = QTableWidget(0, _NCOLS)
        self._table.setHorizontalHeaderLabels(
            ["Operation", "Group", "Elapsed", "Status", "Note"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(True)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            self._table.styleSheet() +
            "QTableWidget { alternate-background-color: #252525; }"
        )

        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(_COL_OP,      QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_GROUP,   QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(_COL_ELAPSED, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_STATUS,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_NOTE,    QHeaderView.ResizeMode.Stretch)

        vh = self._table.verticalHeader()
        vh.setDefaultSectionSize(28)

        root.addWidget(self._table, stretch=1)

        # bottom bar
        bot = QHBoxLayout()
        self._lbl_total = QLabel("Ready.")
        self._lbl_total.setObjectName("elapsed_label")
        bot.addWidget(self._lbl_total, stretch=1)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedHeight(26)
        btn_clear.clicked.connect(self.clear)
        bot.addWidget(btn_clear)

        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(26)
        btn_close.clicked.connect(self.hide)
        bot.addWidget(btn_close)
        root.addLayout(bot)

        footer = QLabel(
            '<a href="https://www.setiastro.com" '
            'style="color:#FF4500;text-decoration:none;">'
            'Franklin Marek  ·  www.setiastro.com</a>'
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setOpenExternalLinks(True)
        footer.setStyleSheet("font-size:10px; color:#888; padding-top:2px;")
        root.addWidget(footer)

    def _tick_elapsed(self):
        """Called every second to refresh the elapsed time on all running rows."""
        for op, idx in self._open.items():
            if idx < self._table.rowCount():
                item = self._table.item(idx, _COL_ELAPSED)
                if item is not None:
                    item.setText(self._rows[idx].elapsed_str())

    # ---------------------------------------------------------------- bus
    def _connect_bus(self):
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, "_sasd_log_bus"):
            self._log_bus = app._sasd_log_bus
            self._log_bus.posted.connect(
                self._on_message, Qt.ConnectionType.QueuedConnection
            )

    def _disconnect_bus(self):
        if self._log_bus is not None:
            try:
                self._log_bus.posted.disconnect(self._on_message)
            except Exception:
                pass
            self._log_bus = None

    # ----------------------------------------------------------- public API
    def start_run(self):
        self.clear()
        self._run_start = time.monotonic()
        self._tick_timer.start()
        self.show()
        self.raise_()
        self.activateWindow()


    def finish_run(self, ok: bool, summary: str = ""):
        # Close any rows that are still marked running
        for op, idx in list(self._open.items()):
            row = self._rows[idx]
            if row.status == _ST_RUNNING:
                row.finish(_ST_OK if ok else _ST_FAIL)
                self._refresh_row(idx)
        self._open.clear()

        # Only stop the timer if no new operations could still arrive.
        # For drizzle/MFD pipelines, finish_run is called after integration
        # but before the post-processing phase — so we leave the timer running
        # and let _on_message restart it if needed. We stop it here only as
        # a soft stop; _on_message will restart if new RUNNING ops appear.
        self._tick_timer.stop()

        total_s = ""
        if self._run_start is not None:
            t = time.monotonic() - self._run_start
            m, s = divmod(int(t), 60)
            total_s = f"{m:02d}:{s:02d}"

        # Show as "integration complete" rather than fully done,
        # since drizzle/MFD may still follow
        if ok:
            self._lbl_total.setText(f"Integration phase done ({total_s}) — post-processing may follow…")
            self._lbl_total.setStyleSheet(
                "color:#f0c040; font-size:11px; font-weight:bold;"
            )
        else:
            self._lbl_total.setText(f"Executed in {total_s}   ✗ Failed")
            self._lbl_total.setStyleSheet(
                "color:#ff4d4f; font-size:12px; font-weight:bold;"
            )

    def finish_all(self, ok: bool, summary: str = ""):
        """Call this when the entire pipeline is done including drizzle/MFD."""
        for op, idx in list(self._open.items()):
            row = self._rows[idx]
            if row.status == _ST_RUNNING:
                row.finish(_ST_OK if ok else _ST_FAIL)
                self._refresh_row(idx)
        self._open.clear()
        self._tick_timer.stop()

        total_s = ""
        if self._run_start is not None:
            t = time.monotonic() - self._run_start
            m, s = divmod(int(t), 60)
            total_s = f"{m:02d}:{s:02d}"

        if ok:
            self._lbl_total.setText(f"Executed in {total_s}   ✓ Complete")
            self._lbl_total.setStyleSheet(
                "color:#52c41a; font-size:12px; font-weight:bold;"
            )
        else:
            self._lbl_total.setText(f"Executed in {total_s}   ✗ Failed")
            self._lbl_total.setStyleSheet(
                "color:#ff4d4f; font-size:12px; font-weight:bold;"
            )

    def clear(self):
        self._table.setRowCount(0)
        self._rows.clear()
        self._open.clear()
        self._run_start = None
        self._lbl_total.setText("Ready.")
        self._lbl_total.setStyleSheet(f"color:{_DIM};font-size:10px;")

    # ---------------------------------------------------- message handler
    @pyqtSlot(str)
    def _on_message(self, msg: str):
        result = _classify(msg)
        if result is None:
            return

        op, group, status = result

        note = re.sub(
            r"^[\U0001F300-\U0001FFFF\u2600-\u26FF\u2700-\u27BF✅❌⚠️🔹📌✂️▶️🔄📊📐🧪🧱🔎💡↻⏭️⚡⏸]+\s*",
            "", msg
        ).strip()
        if len(note) > 120:
            note = note[:117] + "…"

        # ── If a new RUNNING op arrives after the run was marked complete,
        #    resume the timer — drizzle/MFD start after integration finishes
        if status == _ST_RUNNING and not self._tick_timer.isActive():
            self._tick_timer.start()
            # Clear the "Complete" label so it doesn't show as done while work continues
            self._lbl_total.setText("Running (post-integration phase)…")
            self._lbl_total.setStyleSheet(f"color:{_YELLOW};font-size:11px;")

        # ── finishing transitions ─────────────────────────────────────────
        if status in (_ST_OK, _ST_FAIL, _ST_WARN) and op in self._open:
            idx = self._open.pop(op)
            self._rows[idx].finish(status, note)
            self._refresh_row(idx)
            return

        # ── continuing a running row ──────────────────────────────────────
        if status == _ST_RUNNING and op in self._open:
            idx = self._open[op]
            row = self._rows[idx]
            if group and not row.group:
                row.group = group
            row.note = note
            self._refresh_row(idx)
            return

        # ── new row ───────────────────────────────────────────────────────
        r = _MonitorRow(op, group, status, note)
        self._rows.append(r)
        idx = len(self._rows) - 1

        if status == _ST_RUNNING:
            self._open[op] = idx

        self._insert_table_row(idx)
        self._table.scrollToBottom()

    # ---------------------------------------------------- table rendering
    def _insert_table_row(self, idx: int):
        row = self._rows[idx]
        self._table.insertRow(idx)
        self._write_row(idx, row)

    def _refresh_row(self, idx: int):
        row = self._rows[idx]
        self._write_row(idx, row)


    def _write_row(self, idx: int, row: _MonitorRow):
        def _item(text: str, color: Optional[str] = None,
                  bold: bool = False, center: bool = False) -> QTableWidgetItem:
            it = QTableWidgetItem(text)
            it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            if color:
                it.setForeground(QColor(color))
            if bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            if center:
                it.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
            return it

        status_map = {
            _ST_RUNNING: ("⟳ running", "#f0c040"),
            _ST_OK:      ("✓ success", "#52c41a"),
            _ST_WARN:    ("⚠ warning", "#ff9800"),
            _ST_FAIL:    ("✗ failed",  "#ff4d4f"),
            _ST_INFO:    ("ℹ info",    "#888888"),
        }
        st_text, st_color = status_map.get(row.status, ("?", "#888888"))

        self._table.setItem(idx, _COL_OP,
            _item(row.operation, color="#ffffff", bold=True))
        self._table.setItem(idx, _COL_GROUP,
            _item(row.group, color="#cccccc"))
        self._table.setItem(idx, _COL_ELAPSED,
            _item(row.elapsed_str(), color="#aaaaaa", center=True))
        self._table.setItem(idx, _COL_STATUS,
            _item(st_text, color=st_color, center=True))
        self._table.setItem(idx, _COL_NOTE,
            _item(row.note, color="#888888"))

        # subtle row tint for finished states
        tint = {
            _ST_OK:   "#1a2e1a",
            _ST_FAIL: "#2e1a1a",
            _ST_WARN: "#2e221a",
        }.get(row.status)
        if tint:
            for c in range(_NCOLS):
                item = self._table.item(idx, c)
                if item:
                    item.setBackground(QColor(tint))

    # ---------------------------------------------------------------- geo
    def _restore_geometry(self):
        g = self._settings.value("StackingMonitor/geometry")
        if g:
            self.restoreGeometry(g)

    def closeEvent(self, event):
        self._tick_timer.stop()
        self._settings.setValue("StackingMonitor/geometry", self.saveGeometry())
        self._disconnect_bus()
        event.accept()

    def hideEvent(self, event):
        self._settings.setValue("StackingMonitor/geometry", self.saveGeometry())
        super().hideEvent(event)