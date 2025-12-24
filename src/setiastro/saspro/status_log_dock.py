# pro/status_log_dock.py
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPlainTextEdit, QPushButton, QHBoxLayout
)

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
        self.view.setStyleSheet(
            "background-color: black; color: white; font-family: Monospace; padding: 6px;"
        )
        lay.addWidget(self.view, 1)

        row = QHBoxLayout()
        btn_clear = QPushButton("Clear", w)
        btn_clear.clicked.connect(self.view.clear)
        row.addWidget(btn_clear)
        row.addStretch(1)
        lay.addLayout(row)

        self.setWidget(w)

    @pyqtSlot(str)
    def append_line(self, message: str):
        doc = self.view.document()

        # coalesce “Normalizing …” lines (replace last if same prefix)
        if message.startswith("🔄 Normalizing") and doc.blockCount() > 0:
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

        # autoscroll
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def show_raise(self):
        self.setVisible(True)
        self.raise_()
        if self.widget():
            self.widget().setFocus()
