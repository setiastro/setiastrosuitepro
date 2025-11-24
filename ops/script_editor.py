# ops/script_editor.py
from __future__ import annotations
import io, sys, traceback
from pathlib import Path

from ops.scripts import get_scripts_dir  # your existing helper

from PyQt6.QtCore import Qt, QRect, QSize, QRegularExpression
from PyQt6.QtGui import QFont, QAction, QColor, QPainter, QTextCursor, QTextDocument, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPlainTextEdit, QPushButton,
    QLabel, QMessageBox, QFileDialog, QSplitter, QInputDialog, QDockWidget,
    QLineEdit, QToolButton, QCheckBox, QTextEdit
)

# -----------------------------------------------------------------------------
# Code editor with line numbers (QPlainTextEdit subclass)
# -----------------------------------------------------------------------------
class LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self.code_editor = editor

    def sizeHint(self):
        return QSize(self.code_editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.code_editor.line_number_area_paint_event(event)


class CodeEditor(QPlainTextEdit):
    INDENT = "    "  # 4 spaces; change to "\t" if you prefer tabs
    def __init__(self, parent=None):
        super().__init__(parent)
        self._line_number_area = LineNumberArea(self)

        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)

        self._update_line_number_area_width(0)
        self._highlight_current_line()

    def line_number_area_width(self):
        digits = max(1, len(str(self.blockCount())))
        space = 6 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def _update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy):
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor(30, 30, 30))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor(140, 140, 140))
                painter.drawText(
                    0, top, self._line_number_area.width() - 4,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    number
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def _highlight_current_line(self):
        # Subtle current-line highlight
        extra = []
        if not self.isReadOnly():
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(QColor(45, 45, 45))
            sel.format.setProperty(sel.format.Property.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            extra.append(sel)
        self.setExtraSelections(extra)

    def open_find_bar(self, replace=False):
        self.find_bar.set_replace_mode(replace)


    def contextMenuEvent(self, e):
        menu = self.createStandardContextMenu()

        menu.addSeparator()
        act_find = menu.addAction("Find…")
        act_replace = menu.addAction("Replace…")

        act_find.triggered.connect(lambda: self.open_find_bar(replace=False))
        act_replace.triggered.connect(lambda: self.open_find_bar(replace=True))

        menu.exec(e.globalPos())

    def toggle_find_bar(self, replace: bool = False):
        """Show/hide the find bar. If showing, optionally switch replace mode."""
        fb = getattr(self, "find_bar", None)
        if fb is None:
            # fallback: just open normal find UI
            return self.open_find_bar(replace=replace)

        vis = fb.isVisible()
        fb.setVisible(not vis)
        if not vis:
            try:
                fb.set_replace_mode(bool(replace))
            except Exception:
                pass
            try:
                fb.focus_find()
            except Exception:
                pass

    # -------------------------------
    # Indent / dedent helpers
    # -------------------------------
    def _indent_blocks(self, start_block: int, end_block: int):
        doc = self.document()
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for bn in range(start_block, end_block + 1):
            block = doc.findBlockByNumber(bn)
            if not block.isValid():
                continue
            c = QTextCursor(block)
            c.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            c.insertText(self.INDENT)
        cursor.endEditBlock()

    def _dedent_blocks(self, start_block: int, end_block: int):
        doc = self.document()
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for bn in range(start_block, end_block + 1):
            block = doc.findBlockByNumber(bn)
            if not block.isValid():
                continue
            text = block.text()
            c = QTextCursor(block)
            c.movePosition(QTextCursor.MoveOperation.StartOfBlock)

            # Remove indent if present
            if text.startswith(self.INDENT):
                for _ in range(len(self.INDENT)):
                    c.deleteChar()
            elif text.startswith("\t"):
                c.deleteChar()
        cursor.endEditBlock()

    def _selected_block_range(self):
        cur = self.textCursor()
        doc = self.document()
        start = cur.selectionStart()
        end = cur.selectionEnd()

        start_block = doc.findBlock(start).blockNumber()
        end_block = doc.findBlock(end).blockNumber()
        return start_block, end_block

    # -------------------------------
    # Key handling for Tab / Shift+Tab
    # -------------------------------
    def keyPressEvent(self, e):
        key = e.key()

        # Tab: indent selection or insert indent
        if key == Qt.Key.Key_Tab and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            cur = self.textCursor()
            if cur.hasSelection():
                sb, eb = self._selected_block_range()
                self._indent_blocks(sb, eb)
            else:
                cur.insertText(self.INDENT)
            return

        # Shift+Tab (Backtab): dedent selection
        if key == Qt.Key.Key_Backtab:
            cur = self.textCursor()
            if cur.hasSelection():
                sb, eb = self._selected_block_range()
                self._dedent_blocks(sb, eb)
            else:
                # single-line dedent
                sb, eb = self._selected_block_range()
                self._dedent_blocks(sb, sb)
            return

        super().keyPressEvent(e)

    def insertFromMimeData(self, source):
        """
        Normalize pasted indentation:
          - Convert tabs to 4 spaces
          - Replace weird NBSP with normal space
        """
        try:
            text = source.text()
            if text:
                text = text.replace("\u00A0", " ")  # NBSP → space
                text = text.replace("\t", self.INDENT)
                self.textCursor().insertText(text)
                return
        except Exception:
            pass
        super().insertFromMimeData(source)

# -----------------------------------------------------------------------------
# Find / Replace bar
# -----------------------------------------------------------------------------
class FindReplaceBar(QWidget):
    def __init__(self, editor: CodeEditor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self._last_find = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find…")

        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace with…")
        self.replace_edit.setVisible(False)

        self.chk_case = QCheckBox("Case")
        self.chk_word = QCheckBox("Word")
        self.chk_wrap = QCheckBox("Wrap")
        self.chk_wrap.setChecked(True)

        self.btn_prev = QToolButton(); self.btn_prev.setText("Prev")
        self.btn_next = QToolButton(); self.btn_next.setText("Next")
        self.btn_replace = QToolButton(); self.btn_replace.setText("Replace")
        self.btn_replace_all = QToolButton(); self.btn_replace_all.setText("All")
        self.btn_close = QToolButton(); self.btn_close.setText("✕")

        self.btn_replace.setVisible(False)
        self.btn_replace_all.setVisible(False)

        lay.addWidget(QLabel("Find:"))
        lay.addWidget(self.find_edit, 2)
        lay.addWidget(QLabel("Replace:"))
        lay.addWidget(self.replace_edit, 2)
        lay.addWidget(self.chk_case)
        lay.addWidget(self.chk_word)
        lay.addWidget(self.chk_wrap)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.btn_replace)
        lay.addWidget(self.btn_replace_all)
        lay.addWidget(self.btn_close)

        # wiring
        self.find_edit.returnPressed.connect(self.find_next)
        self.btn_next.clicked.connect(self.find_next)
        self.btn_prev.clicked.connect(self.find_prev)
        self.btn_replace.clicked.connect(self.replace_one)
        self.btn_replace_all.clicked.connect(self.replace_all)
        self.btn_close.clicked.connect(self.hide)

    def show_find(self):
        self.replace_edit.setVisible(False)
        self.btn_replace.setVisible(False)
        self.btn_replace_all.setVisible(False)
        self.show()
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def show_replace(self):
        self.replace_edit.setVisible(True)
        self.btn_replace.setVisible(True)
        self.btn_replace_all.setVisible(True)
        self.show()
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    # ---- internal flags ----
    def _flags(self, backward=False):
        flags = QTextDocument.FindFlag(0)
        if backward:
            flags |= QTextDocument.FindFlag.FindBackward
        if self.chk_case.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if self.chk_word.isChecked():
            flags |= QTextDocument.FindFlag.FindWholeWords
        return flags

    def _do_find(self, backward=False):
        text = self.find_edit.text()
        if not text:
            return False

        self._last_find = text
        ok = self.editor.find(text, self._flags(backward=backward))

        if not ok and self.chk_wrap.isChecked():
            # wrap to start/end
            cursor = self.editor.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start
            )
            self.editor.setTextCursor(cursor)
            ok = self.editor.find(text, self._flags(backward=backward))

        return ok

    def find_next(self):
        self._do_find(backward=False)

    def find_prev(self):
        self._do_find(backward=True)

    def replace_one(self):
        find_text = self.find_edit.text()
        if not find_text:
            return

        cursor = self.editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == find_text:
            cursor.insertText(self.replace_edit.text())
            self.editor.setTextCursor(cursor)

        self.find_next()

    def replace_all(self):
        find_text = self.find_edit.text()
        if not find_text:
            return
        replace_text = self.replace_edit.text()

        cursor = self.editor.textCursor()
        cursor.beginEditBlock()

        # start from top
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.editor.setTextCursor(cursor)

        count = 0
        while self.editor.find(find_text, self._flags(backward=False)):
            c = self.editor.textCursor()
            if c.hasSelection():
                c.insertText(replace_text)
                count += 1

        cursor.endEditBlock()

    def set_replace_mode(self, replace: bool):
        """Switch between find-only and find+replace UI."""
        if replace:
            self.show_replace()
        else:
            self.show_find()

    def focus_find(self):
        """Put focus in the find box."""
        self.find_edit.setFocus()
        self.find_edit.selectAll()

class _StdCapture:
    """Context manager to capture stdout/stderr into a StringIO."""
    def __init__(self):
        self.buf = io.StringIO()
        self._old_out = None
        self._old_err = None

    def __enter__(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self.buf
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout, sys.stderr = self._old_out, self._old_err

    def text(self) -> str:
        return self.buf.getvalue()

class PythonHighlighter(QSyntaxHighlighter):
    """
    Simple Python syntax highlighter for QPlainTextEdit/QTextDocument.
    Dark-theme friendly colors.
    """
    def __init__(self, document):
        super().__init__(document)

        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            if italic:
                f.setFontItalic(True)
            return f

        # ---- formats ----
        self.f_keyword   = fmt("#C586C0", bold=True)
        self.f_builtin   = fmt("#4FC1FF")
        self.f_number    = fmt("#B5CEA8")
        self.f_string    = fmt("#CE9178")
        self.f_comment   = fmt("#6A9955", italic=True)
        self.f_decorator = fmt("#DCDCAA")
        self.f_self      = fmt("#9CDCFE")
        self.f_defclass  = fmt("#569CD6", bold=True)

        # ---- keyword lists ----
        keywords = [
            "False","None","True","and","as","assert","async","await","break",
            "class","continue","def","del","elif","else","except","finally","for",
            "from","global","if","import","in","is","lambda","nonlocal","not",
            "or","pass","raise","return","try","while","with","yield","match","case"
        ]

        builtins = [
            "abs","all","any","ascii","bin","bool","breakpoint","bytearray","bytes",
            "callable","chr","classmethod","compile","complex","delattr","dict","dir",
            "divmod","enumerate","eval","exec","filter","float","format","frozenset",
            "getattr","globals","hasattr","hash","help","hex","id","input","int",
            "isinstance","issubclass","iter","len","list","locals","map","max",
            "memoryview","min","next","object","oct","open","ord","pow","print",
            "property","range","repr","reversed","round","set","setattr","slice",
            "sorted","staticmethod","str","sum","super","tuple","type","vars","zip"
        ]

        # ---- rules ----
        self.rules = []

        # keywords
        for kw in keywords:
            self.rules.append((QRegularExpression(rf"\b{kw}\b"), self.f_keyword))

        # builtins
        for bi in builtins:
            self.rules.append((QRegularExpression(rf"\b{bi}\b"), self.f_builtin))

        # def / class name highlighting
        self.rules.append((QRegularExpression(r"\bdef\s+([A-Za-z_]\w*)"), self.f_defclass))
        self.rules.append((QRegularExpression(r"\bclass\s+([A-Za-z_]\w*)"), self.f_defclass))

        # decorators
        self.rules.append((QRegularExpression(r"^\s*@\w+"), self.f_decorator))

        # numbers (int/float/hex/binary with underscores)
        self.rules.append((QRegularExpression(r"\b0[xX][0-9A-Fa-f_]+\b"), self.f_number))
        self.rules.append((QRegularExpression(r"\b0[bB][01_]+\b"), self.f_number))
        self.rules.append((QRegularExpression(r"\b0[oO][0-7_]+\b"), self.f_number))
        self.rules.append((QRegularExpression(r"\b\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d[\d_]*)?\b"), self.f_number))

        # self / cls
        self.rules.append((QRegularExpression(r"\bself\b"), self.f_self))
        self.rules.append((QRegularExpression(r"\bcls\b"), self.f_self))

        # single-line strings
        self.rules.append((QRegularExpression(r"(?<!\\)'.*?(?<!\\)'"), self.f_string))
        self.rules.append((QRegularExpression(r'(?<!\\)".*?(?<!\\)"'), self.f_string))

        # comments
        self.rules.append((QRegularExpression(r"#.*$"), self.f_comment))

        # multiline triple-quoted strings
        self.tri_single = QRegularExpression("'''")
        self.tri_double = QRegularExpression('"""')

    def highlightBlock(self, text: str):
        # apply normal single-line rules
        for pattern, form in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                start = m.capturedStart()
                length = m.capturedLength()
                self.setFormat(start, length, form)

        # handle multiline triple strings
        self.setCurrentBlockState(0)
        self._do_multiline(text, self.tri_single, 1, self.f_string)
        self._do_multiline(text, self.tri_double, 2, self.f_string)

    def _do_multiline(self, text, delimiter, in_state, style):
        start = 0
        if self.previousBlockState() != in_state:
            m = delimiter.match(text)
            start = m.capturedStart() if m.hasMatch() else -1

        while start >= 0:
            m = delimiter.match(text, start + 3)
            end = m.capturedStart() if m.hasMatch() else -1

            if end >= 0:
                length = end - start + 3
                self.setFormat(start, length, style)
                start = delimiter.match(text, start + length).capturedStart()
            else:
                self.setFormat(start, len(text) - start, style)
                self.setCurrentBlockState(in_state)
                return


class ScriptEditorDock(QDockWidget):
    def __init__(self, app_window, parent=None):
        super().__init__("Script Editor", parent or app_window)
        self.app = app_window
        self.scripts_dir = get_scripts_dir()

        self._current_path: Path | None = None
        self._dirty = False

        root = QWidget(self)
        self.setWidget(root)

        main = QVBoxLayout(root)

        # --- top bar (2 rows) ---
        barwrap = QVBoxLayout()
        barwrap.setContentsMargins(0, 0, 0, 0)
        barwrap.setSpacing(4)

        # Row 1: file label + file ops
        row1 = QHBoxLayout()
        self.lbl_file = QLabel("No script loaded")
        row1.addWidget(self.lbl_file, 0)
        row1.addSpacing(12)

        self.btn_new = QPushButton("New")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save As…")
        self.btn_delete = QPushButton("🗑 Delete")  # if you added it already

        row1.addWidget(self.btn_new)
        row1.addWidget(self.btn_save)
        row1.addWidget(self.btn_save_as)
        row1.addWidget(self.btn_delete)
        row1.addStretch(1)

        # Row 2: edit + run + help ops
        row2 = QHBoxLayout()

        self.btn_find = QPushButton("🔍 Find")
        self.btn_replace = QPushButton("🧩 Replace")

        self.btn_run = QPushButton("🟢▶ Run")
        self.btn_run_base = QPushButton("▶ Run on Base")

        self.btn_reload = QPushButton("Reload Scripts")
        self.btn_cmd_help = QPushButton("❓ Command Help")

        row2.addWidget(self.btn_find)
        row2.addWidget(self.btn_replace)
        row2.addSpacing(12)
        row2.addWidget(self.btn_run)
        row2.addWidget(self.btn_run_base)
        row2.addSpacing(12)
        row2.addWidget(self.btn_reload)
        row2.addSpacing(12)
        row2.addWidget(self.btn_cmd_help)
        row2.addStretch(1)

        barwrap.addLayout(row1)
        barwrap.addLayout(row2)
        main.addLayout(barwrap)


        # --- splitter: left list, right editor/output ---
        split = QSplitter(Qt.Orientation.Horizontal)

        # left script list
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0,0,0,0)
        left_lay.addWidget(QLabel("Scripts"))

        self.list_scripts = QListWidget()
        left_lay.addWidget(self.list_scripts, 1)

        # right: editor + output
        right = QSplitter(Qt.Orientation.Vertical)

        # editor
        editor_wrap = QWidget()
        editor_lay = QVBoxLayout(editor_wrap)
        editor_lay.setContentsMargins(0, 0, 0, 0)
        editor_lay.setSpacing(0)

        self.editor = CodeEditor()
        self.highlighter = PythonHighlighter(self.editor.document())
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(10)
        self.editor.setFont(f)
        self.editor.setTabStopDistance(4 * self.editor.fontMetrics().horizontalAdvance(' '))

        self.find_bar = FindReplaceBar(self.editor)
        self.find_bar.hide()

        editor_lay.addWidget(self.find_bar)
        editor_lay.addWidget(self.editor, 1)
        self.editor.find_bar = self.find_bar

        # Now wire toolbar buttons (editor exists now)
        self.btn_find.clicked.connect(lambda: self.editor.open_find_bar(replace=False))
        self.btn_replace.clicked.connect(lambda: self.editor.open_find_bar(replace=True))
        right.addWidget(editor_wrap)

        # output
        out_wrap = QWidget()
        out_lay = QVBoxLayout(out_wrap)
        out_lay.setContentsMargins(0,0,0,0)
        out_lay.addWidget(QLabel("Output / Traceback"))

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(f)
        out_lay.addWidget(self.output, 1)

        row2 = QHBoxLayout()
        self.btn_copy = QPushButton("Copy All")
        self.btn_clear = QPushButton("Clear")
        row2.addStretch(1)
        row2.addWidget(self.btn_copy)
        row2.addWidget(self.btn_clear)
        out_lay.addLayout(row2)

        right.addWidget(out_wrap)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        main.addWidget(split, 1)

        # --- wiring ---
        self.btn_new.clicked.connect(self.new_script)
        self.btn_save.clicked.connect(self.save_script)
        self.btn_save_as.clicked.connect(self.save_script_as)
        self.btn_run.clicked.connect(lambda: self.run_script(on_base=False))
        self.btn_run_base.clicked.connect(lambda: self.run_script(on_base=True))
        self.btn_reload.clicked.connect(self.reload_scripts)
        self.btn_cmd_help.clicked.connect(self.open_command_help)
        self.btn_copy.clicked.connect(self.copy_all_output)
        self.btn_clear.clicked.connect(lambda: self.output.setPlainText(""))
        self.btn_delete.clicked.connect(self.delete_script) 

        self.list_scripts.itemDoubleClicked.connect(
            lambda it: self.open_script(self.scripts_dir / it.text())
        )
        self.editor.textChanged.connect(self._mark_dirty)
        # --- Find / Replace shortcuts ---
        self.act_find = QAction("Find", self)
        self.act_find.setShortcut("Ctrl+F")
        self.act_find.triggered.connect(self.find_bar.show_find)
        self.addAction(self.act_find)

        self.act_replace = QAction("Replace", self)
        self.act_replace.setShortcut("Ctrl+H")
        self.act_replace.triggered.connect(self.find_bar.show_replace)
        self.addAction(self.act_replace)

        self.act_find_next = QAction("Find Next", self)
        self.act_find_next.setShortcut("F3")
        self.act_find_next.triggered.connect(self.find_bar.find_next)
        self.addAction(self.act_find_next)

        self.act_find_prev = QAction("Find Previous", self)
        self.act_find_prev.setShortcut("Shift+F3")
        self.act_find_prev.triggered.connect(self.find_bar.find_prev)
        self.addAction(self.act_find_prev)

        self.reload_scripts()


    # ------------------------------------------------------------------
    # list management
    def reload_scripts(self):
        self.list_scripts.clear()
        for p in sorted(self.scripts_dir.glob("*.py")):
            self.list_scripts.addItem(p.name)

        if hasattr(self.app, "scriptman"):
            self.app.scriptman.load_registry()
            if hasattr(self.app, "menu_scripts"):
                self.app.scriptman.rebuild_menu(self.app.menu_scripts)

        self._log(f"Reloaded scripts from {self.scripts_dir}")

    def open_command_help(self):
        try:
            from ops.command_help_dialog import CommandHelpDialog
        except Exception as e:
            QMessageBox.critical(self, "Command Help", f"Failed to open help dialog:\n{e}")
            return

        dlg = CommandHelpDialog(parent=self, editor=self.editor)
        dlg.exec()

    # ------------------------------------------------------------------
    # file operations
    def maybe_save_dirty(self) -> bool:
        if not self._dirty:
            return True
        r = QMessageBox.question(
            self, "Unsaved Changes",
            "This script has unsaved changes. Save now?",
            QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Cancel
        )
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.Yes:
            return self.save_script()
        return True

    def new_script(self):
        if not self.maybe_save_dirty():
            return
        name, ok = QInputDialog.getText(
            self, "New Script", "Script name (no extension):"
        )
        if not ok or not name.strip():
            return
        path = self.scripts_dir / f"{name.strip()}.py"
        if path.exists():
            QMessageBox.warning(self, "Exists", "A script with that name already exists.")
            return

        template = (
            "# SASpro user script\n"
            "SCRIPT_NAME = \"New Script\"\n"
            "SCRIPT_GROUP = \"User\"\n\n"
            "def run(ctx):\n"
            "    ctx.log(\"Hello from New Script\")\n"
            "    # img = ctx.get_image()\n"
            "    # ctx.set_image(img, step_name=\"Script\")\n"
        )
        path.write_text(template, encoding="utf-8")
        self.open_script(path)
        self.reload_scripts()

    def open_script(self, path: Path):
        if not self.maybe_save_dirty():
            return
        try:
            txt = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.editor.setPlainText(txt)
        self._current_path = path
        self._dirty = False
        self._update_title()

    def save_script(self) -> bool:
        if self._current_path is None:
            return self.save_script_as()
        try:
            txt = self.editor.toPlainText()

            # Normalize tabs and NBSP on save
            txt = txt.replace("\u00A0", " ")
            txt = txt.replace("\t", self.editor.INDENT)

            self._current_path.write_text(txt, encoding="utf-8")

            # If we modified text, reflect it in editor so user sees reality
            if txt != self.editor.toPlainText():
                self.editor.blockSignals(True)
                self.editor.setPlainText(txt)
                self.editor.blockSignals(False)

            self._dirty = False
            self._update_title()
            self.reload_scripts()  # refresh menu + list
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return False

    def save_script_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script As", str(self.scripts_dir),
            "Python Script (*.py)"
        )
        if not path:
            return False
        p = Path(path)
        if p.suffix.lower() != ".py":
            p = p.with_suffix(".py")
        self._current_path = p
        return self.save_script()



    def delete_script(self):
        """
        Permanently delete a script file from disk.
        Priority:
          1) currently selected item in list
          2) currently open script
        """
        # If current doc is dirty and we're about to delete it, give chance to save/cancel
        # Determine target
        target: Path | None = None

        it = self.list_scripts.currentItem()
        if it is not None:
            p = self.scripts_dir / it.text()
            if p.exists():
                target = p

        if target is None and self._current_path is not None and self._current_path.exists():
            target = self._current_path

        if target is None or not target.exists():
            QMessageBox.information(self, "Delete Script", "No script selected or loaded.")
            return

        # If deleting the currently open script and it's dirty, ask first
        try:
            if (
                self._current_path is not None
                and target.resolve() == self._current_path.resolve()
                and self._dirty
            ):
                if not self.maybe_save_dirty():
                    return
        except Exception:
            # resolve() can fail on weird paths; ignore and proceed
            if self._dirty:
                if not self.maybe_save_dirty():
                    return

        name = target.name
        r = QMessageBox.warning(
            self,
            "Delete Script",
            f"Delete '{name}' permanently?\n\n"
            "This will remove the file from disk and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        try:
            target.unlink()
        except Exception as e:
            QMessageBox.critical(self, "Delete Failed", f"Could not delete '{name}':\n{e}")
            return

        # If we deleted the open script, clear editor state
        try:
            if self._current_path is not None and target.resolve() == self._current_path.resolve():
                self.editor.blockSignals(True)
                self.editor.setPlainText("")
                self.editor.blockSignals(False)
                self._current_path = None
                self._dirty = False
                self._update_title()
        except Exception:
            pass

        self.reload_scripts()
        self._log(f"Deleted script: {name}")


    # ------------------------------------------------------------------
    # running
    def run_script(self, *, on_base: bool):
        if self._current_path is None:
            QMessageBox.information(self, "Run Script", "No script is loaded.")
            return

        # autosave before run
        if self._dirty:
            ok = self.save_script()
            if not ok:
                return

        self.output.setPlainText("")
        self._log(f"Running {self._current_path.name} (on_base={on_base})")

        # ---- PRE-FLIGHT: compile & indentation sanity ----
        src = self.editor.toPlainText()
        try:
            # 1) Python parser check (catches IndentationError immediately)
            compile(src, str(self._current_path), "exec")

            # 2) tabnanny mixed-indent check (more specific warnings)
            import tabnanny, tempfile, os
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(src)
                tmp_name = tf.name
            try:
                tabnanny.check(tmp_name)
            finally:
                try: os.remove(tmp_name)
                except Exception: pass

        except Exception as e:
            tb = traceback.format_exc()
            self.output.appendPlainText(tb)
            QMessageBox.critical(
                self,
                "Indentation / Syntax Error",
                "This script has a Python parse or indentation problem.\n\n"
                f"{e}\n\n"
                "Fix it before running."
            )
            return

        # ---- RUN ----
        try:
            man = getattr(self.app, "scriptman", None)
            if man is None:
                raise RuntimeError("ScriptManager not initialized on main window.")

            entry = man._load_one_script(self._current_path)
            if entry is None or entry.run is None:
                raise RuntimeError("Script has no run(ctx).")

            with _StdCapture() as cap:
                man.run_entry(entry, on_base=on_base)

            out = cap.text().strip()
            if out:
                self.output.appendPlainText(out)

        except Exception:
            tb = traceback.format_exc()
            self.output.appendPlainText(tb)
            self._log("Script ERROR:\n" + tb)



    # ------------------------------------------------------------------
    # ui helpers
    def _mark_dirty(self):
        if self._current_path is None:
            self._dirty = True
        else:
            self._dirty = True
        self._update_title()

    def _update_title(self):
        name = self._current_path.name if self._current_path else "Untitled"
        star = " *" if self._dirty else ""
        self.lbl_file.setText(f"{name}{star}")

    def copy_all_output(self):
        self.output.selectAll()
        self.output.copy()
        self.output.moveCursor(self.output.textCursor().End)

    def _log(self, s: str):
        try:
            self.app._log(f"[ScriptEditor] {s}")
        except Exception:
            print("[ScriptEditor]", s)
