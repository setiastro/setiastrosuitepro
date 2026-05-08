from __future__ import annotations

from PyQt6.QtGui import QClipboard, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from setiastro.saspro.diagnostics import collect_diagnostics, default_report_path, write_report


class DiagnosticsReportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Diagnostics Report"))
        self.resize(860, 620)

        self._report = collect_diagnostics()

        layout = QVBoxLayout(self)
        self.editor = QTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setAcceptRichText(False)
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.editor.setPlainText(self._report.markdown)
        layout.addWidget(self.editor)

        actions = QHBoxLayout()
        btn_copy = QPushButton(self.tr("Copy"), self)
        btn_copy.clicked.connect(self._copy_report)
        actions.addWidget(btn_copy)

        btn_save = QPushButton(self.tr("Save..."), self)
        btn_save.clicked.connect(self._save_report)
        actions.addWidget(btn_save)

        actions.addStretch(1)

        btn_close = QPushButton(self.tr("Close"), self)
        btn_close.clicked.connect(self.accept)
        actions.addWidget(btn_close)
        layout.addLayout(actions)

    def _copy_report(self) -> None:
        QGuiApplication.clipboard().setText(self._report.markdown, mode=QClipboard.Mode.Clipboard)

    def _save_report(self) -> None:
        suggested = default_report_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Diagnostics Report"),
            str(suggested),
            self.tr("Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"),
        )
        if not path:
            return
        try:
            saved = write_report(self._report.markdown, output_path=path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.tr("Save Diagnostics Report"),
                self.tr("Could not save the diagnostics report:\n{0}").format(exc),
            )
            return
        QMessageBox.information(
            self,
            self.tr("Save Diagnostics Report"),
            self.tr("Diagnostics report saved to:\n{0}").format(saved),
        )
