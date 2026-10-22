#saspro/status_log_dock.py
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPlainTextEdit, QPushButton, QHBoxLayout
)

import os
from datetime import datetime

class StatusLogDock(QDockWidget):
    MAX_BLOCKS = 2000

    def __init__(self, parent=None):
        super().__init__(self.tr("Stacking Log"), parent)
        self.setObjectName("StackingLogDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        w = QWidget(self)
        lay = QVBoxLayout(w); lay.setContentsMargins(6,6,6,6)
        self.view = QPlainTextEdit(w)
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # Show scrollbars when content overflows (NoWrap guarantees horizontal
        # overflow on long lines). AsNeeded + the explicit QScrollBar styling
        # below is what makes the groove+handle actually paint — styling the
        # QPlainTextEdit alone leaves Qt drawing only the arrow buttons on some
        # platforms/themes.
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: black;
                color: white;
                font-family: Monospace;
                padding: 6px;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 14px;
                margin: 0px;
                border: none;
            }
            QScrollBar:horizontal {
                background: #1a1a1a;
                height: 14px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 24px;
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal {
                background: #4a4a4a;
                min-width: 24px;
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::handle:hover {
                background: #5e5e5e;
            }
            /* Step arrows at each end, kept visible alongside the handle. */
            QScrollBar::add-line:vertical {
                background: #2a2a2a;
                height: 14px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
                border: none;
            }
            QScrollBar::sub-line:vertical {
                background: #2a2a2a;
                height: 14px;
                subcontrol-position: top;
                subcontrol-origin: margin;
                border: none;
            }
            QScrollBar::add-line:horizontal {
                background: #2a2a2a;
                width: 14px;
                subcontrol-position: right;
                subcontrol-origin: margin;
                border: none;
            }
            QScrollBar::sub-line:horizontal {
                background: #2a2a2a;
                width: 14px;
                subcontrol-position: left;
                subcontrol-origin: margin;
                border: none;
            }
            QScrollBar::add-line:hover, QScrollBar::sub-line:hover {
                background: #3a3a3a;
            }
            /* Leave the arrow glyphs unstyled so Qt paints its native
               triangles inside the buttons above — restyling them with a
               plain background would render squares instead of arrows. */
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none;
            }
            """
        )
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        btn_clear = QPushButton("Clear", w)
        btn_clear.clicked.connect(self.view.clear)
        row.addWidget(btn_clear)
        row.addStretch(1)
        lay.addLayout(row)
        self.setWidget(w)

        # File logging state
        self._log_file = None      # open file handle
        self._log_path = None      # str path

    def open_log_file(self, directory: str) -> str:
        """
        Open a new timestamped log file in `directory`.
        Closes any previously open log file first.
        Returns the path of the new log file.
        """
        self.close_log_file()

        try:
            os.makedirs(directory, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(directory, f"stacking_log_{ts}.txt")
            self._log_file = open(path, "w", encoding="utf-8", buffering=1)  # line-buffered
            self._log_path = path
            # Write a header so the file is immediately identifiable
            self._log_file.write(
                f"# SASpro Stacking Log\n"
                f"# Started: {datetime.now().isoformat()}\n"
                f"# Directory: {directory}\n\n"
            )
            self._log_file.flush()
            return path
        except Exception as e:
            self._log_file = None
            self._log_path = None
            print(f"[StatusLogDock] Could not open log file: {e}")
            return ""

    def close_log_file(self):
        """Flush and close the current log file if open."""
        if self._log_file is not None:
            try:
                self._log_file.write(
                    f"\n# Log closed: {datetime.now().isoformat()}\n"
                )
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
            self._log_path = None

    def current_log_path(self) -> str:
        """Return the path of the currently open log file, or empty string."""
        return self._log_path or ""

    @pyqtSlot(str)
    def append_line(self, message: str):
        doc = self.view.document()

        # Remember where the user had the horizontal scroll BEFORE we touch the
        # document. Inserting text (and the setTextCursor calls below) would
        # otherwise drag the view to the caret at end-of-line. We restore this
        # exact position afterward so a newly appended line — however long —
        # never moves the horizontal scroll: left stays left, and a user who
        # scrolled right to read a column stays right there.
        hsb = self.view.horizontalScrollBar()
        saved_hscroll = hsb.value()

        # --- existing UI logic unchanged ---
        if message.startswith("\r"):
            display = message[1:]
            if doc.blockCount() > 0:
                cur = self.view.textCursor()
                cur.movePosition(QTextCursor.MoveOperation.End)
                cur.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                 QTextCursor.MoveMode.KeepAnchor)
                cur.removeSelectedText()
                cur.insertText(display)
                self.view.setTextCursor(cur)
            else:
                self.view.appendPlainText(display)
        elif message.startswith("🔄 Normalizing") and doc.blockCount() > 0:
            last = doc.findBlockByNumber(doc.blockCount() - 1)
            if last.isValid() and last.text().startswith("🔄 Normalizing"):
                cur = self.view.textCursor()
                cur.movePosition(QTextCursor.MoveOperation.End)
                cur.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                 QTextCursor.MoveMode.KeepAnchor)
                cur.removeSelectedText()
                cur.insertText(message)
                self.view.setTextCursor(cur)
            else:
                self.view.appendPlainText(message)
        else:
            self.view.appendPlainText(message)

        # trim earliest lines
        if doc.blockCount() > self.MAX_BLOCKS:
            extra = doc.blockCount() - self.MAX_BLOCKS
            cur = self.view.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.Start)
            cur.movePosition(QTextCursor.MoveOperation.Down,
                             QTextCursor.MoveMode.KeepAnchor, extra)
            cur.removeSelectedText()
            self.view.setTextCursor(self.view.textCursor())

        # Autoscroll vertically to follow new lines (unchanged), but keep the
        # horizontal scroll exactly where the user had it (captured above),
        # clamped to the new range in case trimming shrank the content width.
        vsb = self.view.verticalScrollBar()
        vsb.setValue(vsb.maximum())

        hsb.setValue(max(hsb.minimum(), min(saved_hscroll, hsb.maximum())))

        # --- write to file (strip \r lines to last value; write others as-is) ---
        if self._log_file is not None:
            try:
                ts = datetime.now().strftime("%H:%M:%S")
                line = message.lstrip("\r")  # \r lines are progress overwrites; write final value
                self._log_file.write(f"[{ts}] {line}\n")
                # buffering=1 (line-buffered) means flush is implicit on \n
            except Exception:
                pass

    def show_raise(self):
        self.setVisible(True)
        self.raise_()
        if self.widget():
            self.widget().setFocus()