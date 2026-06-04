# pro/system_log_dock.py
"""
System Log Dock — echoes stdout/stderr with ANSI color rendering,
log-level filtering, search, and copy/clear controls.
"""
from __future__ import annotations

import re
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QColor, QFont, QTextCharFormat, QTextCursor, QGuiApplication
)
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLineEdit, QCheckBox, QLabel
)

# ---------------------------------------------------------------------------
# ANSI → QColor map (standard 8 + bright 8)
# ---------------------------------------------------------------------------
_ANSI_COLORS = {
    30: "#555555", 31: "#cc3333", 32: "#33cc33", 33: "#ccaa00",
    34: "#3399cc", 35: "#cc33cc", 36: "#33cccc", 37: "#cccccc",
    90: "#888888", 91: "#ff5555", 92: "#55ff55", 93: "#ffff55",
    94: "#5599ff", 95: "#ff55ff", 96: "#55ffff", 97: "#ffffff",
}

_ANSI_RE = re.compile(r'\033\[([0-9;]*)m')

# Simple heuristics to auto-tag log level from plain text
_LEVEL_RE = re.compile(
    r'\b(error|exception|traceback|failed|critical)\b', re.IGNORECASE
)
_WARN_RE  = re.compile(r'\b(warning|warn)\b', re.IGNORECASE)


def _parse_ansi(text: str) -> list[tuple[str, QTextCharFormat]]:
    """
    Split `text` into (chunk, format) pairs, respecting ANSI SGR codes.
    Returns a list ready to insert into a QTextEdit.
    """
    result: list[tuple[str, QTextCharFormat]] = []
    fmt = QTextCharFormat()
    pos = 0

    for m in _ANSI_RE.finditer(text):
        # plain chunk before this escape
        chunk = text[pos:m.start()]
        if chunk:
            result.append((chunk, QTextCharFormat(fmt)))
        pos = m.end()

        codes = [int(c) for c in m.group(1).split(";") if c.isdigit()] or [0]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0:
                fmt = QTextCharFormat()
            elif c == 1:
                fmt.setFontWeight(QFont.Weight.Bold)
            elif c == 3:
                fmt.setFontItalic(True)
            elif c == 4:
                fmt.setFontUnderline(True)
            elif 30 <= c <= 37 or 90 <= c <= 97:
                fmt.setForeground(QColor(_ANSI_COLORS.get(c, "#cccccc")))
            elif c == 38 and i + 4 < len(codes) and codes[i+1] == 2:
                # 24-bit RGB:  38;2;R;G;B
                r, g, b = codes[i+2], codes[i+3], codes[i+4]
                fmt.setForeground(QColor(r, g, b))
                i += 4
            elif c == 39:
                fmt.clearForeground()
            i += 1

    # trailing plain text
    tail = text[pos:]
    if tail:
        result.append((tail, QTextCharFormat(fmt)))

    return result

_DEFAULT_FMT = QTextCharFormat()

class SystemLogDock(QDockWidget):
    """
    Upgraded System Log dock.

    Drop-in for the inline QPlainTextEdit used in dock_mixin.py.
    Wire it the same way:

        self.log_text = self.system_log_dock.log_text   # for _append_log_text compat
        OR replace _append_log_text to call system_log_dock.append_text(msg)
    """

    MAX_BLOCKS = 5_000

    def __init__(self, parent=None):
        super().__init__("System Log", parent)
        self.setObjectName("SystemLogDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        # ── root widget ──────────────────────────────────────────────────
        root = QWidget(self)
        lay  = QVBoxLayout(root)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # ── toolbar row ──────────────────────────────────────────────────
        tb = QHBoxLayout()
        tb.setSpacing(6)

        # search
        self._search = QLineEdit(root)
        self._search.setPlaceholderText("Search…")
        self._search.setClearButtonEnabled(True)
        self._search.setMaximumWidth(220)
        self._search.textChanged.connect(self._on_search)
        tb.addWidget(self._search)

        # level filters
        tb.addWidget(QLabel("Show:", root))
        self._chk_info  = QCheckBox("Info",  root)
        self._chk_warn  = QCheckBox("Warn",  root)
        self._chk_error = QCheckBox("Error", root)
        for chk in (self._chk_info, self._chk_warn, self._chk_error):
            chk.setChecked(True)
            chk.toggled.connect(self._rebuild_visible)
            tb.addWidget(chk)

        tb.addStretch(1)

        # buttons
        self._btn_copy  = QPushButton("Copy All", root)
        self._btn_clear = QPushButton("Clear",    root)
        self._btn_copy.clicked.connect(self._copy_all)
        self._btn_clear.clicked.connect(self._clear)
        tb.addWidget(self._btn_copy)
        tb.addWidget(self._btn_clear)

        lay.addLayout(tb)

        # ── log view ─────────────────────────────────────────────────────
        self.log_text = QTextEdit(root)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_text.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1e1e1e;"
            "  color: #cccccc;"
            "  font-family: 'Cascadia Mono', 'Consolas', 'Courier New', monospace;"
            "  font-size: 11px;"
            "  padding: 4px;"
            "}"
        )
        lay.addWidget(self.log_text, 1)

        self.setWidget(root)

        # ── internal state ────────────────────────────────────────────────
        # Each entry: {"raw": str, "level": "info"|"warn"|"error", "ts": str}
        self._entries: list[dict] = []

        # debounce search/filter rebuilds
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(120)
        self._rebuild_timer.timeout.connect(self._do_rebuild)
        

    # ── public API ────────────────────────────────────────────────────────

    def append_text(self, text: str):
        """
        Main entry point.  Call this instead of log_text.insertPlainText().
        Handles ANSI codes, \r carriage-return overwrites, timestamps, and trimming.
        """
        if not text or not text.strip():
            return

        ts = datetime.now().strftime("%H:%M:%S")

        # handle carriage-return (progress line overwrite)
        if "\r" in text:
            parts = text.split("\r")
            # only the last part after final \r matters for display
            text = parts[-1]

        level = self._classify(text)

        self._entries.append({"raw": text, "level": level, "ts": ts})

        # trim old entries
        if len(self._entries) > self.MAX_BLOCKS:
            self._entries = self._entries[-self.MAX_BLOCKS:]

        # only append to view if this level is currently visible
        if self._level_visible(level):
            self._append_to_view(text, level, ts)

    # ── internals ─────────────────────────────────────────────────────────

    def _classify(self, text: str) -> str:
        plain = _ANSI_RE.sub("", text)
        if _LEVEL_RE.search(plain):
            return "error"
        if _WARN_RE.search(plain):
            return "warn"
        return "info"

    def _level_visible(self, level: str) -> bool:
        if level == "error":
            return self._chk_error.isChecked()
        if level == "warn":
            return self._chk_warn.isChecked()
        return self._chk_info.isChecked()

    def _color_for_level(self, level: str) -> QColor | None:
        if level == "error":
            return QColor("#ff6666")
        if level == "warn":
            return QColor("#ffcc44")
        return None  # use ANSI / default
    
    def _append_to_view(self, raw: str, level: str, ts: str):
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        ts_fmt = QTextCharFormat()
        ts_fmt.setForeground(QColor("#666666"))
        cursor.insertText(f"[{ts}] ", ts_fmt)

        chunks = _parse_ansi(raw)
        level_color = self._color_for_level(level)

        for chunk_text, chunk_fmt in chunks:
            if level_color and chunk_fmt.foreground() == _DEFAULT_FMT.foreground():
                chunk_fmt.setForeground(level_color)
            cursor.insertText(chunk_text, chunk_fmt)

        plain_fmt = QTextCharFormat()
        cursor.insertText("\n", plain_fmt)

        self.log_text.setTextCursor(cursor)

        sb = self.log_text.verticalScrollBar()
        if sb.value() >= sb.maximum() - 4:
            sb.setValue(sb.maximum())

    def _on_search(self, _text: str):
        self._rebuild_timer.start()

    def _rebuild_visible(self):
        self._rebuild_timer.start()

    def _do_rebuild(self):
        needle = self._search.text().strip().lower()
        self.log_text.clear()

        for entry in self._entries:
            level = entry["level"]
            raw   = entry["raw"]
            ts    = entry["ts"]

            if not self._level_visible(level):
                continue

            plain = _ANSI_RE.sub("", raw).lower()
            if needle and needle not in plain:
                continue

            self._append_to_view(raw, level, ts)

    def _copy_all(self):
        lines = []
        for e in self._entries:
            plain = _ANSI_RE.sub("", e["raw"])
            lines.append(f"[{e['ts']}] {plain}")
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _clear(self):
        self._entries.clear()
        self.log_text.clear()