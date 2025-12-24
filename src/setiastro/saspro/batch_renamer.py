# pro/batch_renamer.py
from __future__ import annotations

import os
import re
import shutil
from typing import List
from collections import defaultdict
from datetime import datetime, timezone

from astropy.io import fits

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QSplitter, QWidget, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QDialogButtonBox, QComboBox, QSpinBox, QCheckBox, QFileDialog, QListWidget, QMessageBox
)
from PyQt6.QtCore import QSettings


class BatchRenamerDialog(QDialog):
    r"""
    Batch rename files using a template like:
      LIGHT_{FILTER}_{EXPOSURE:.0f}s_{DATE-OBS:%Y%m%d}_{#03}.{ext}

    Supports:
      - Any FITS keyword in braces: {FILTER}, {EXPOSURE}, {OBJECT}, …
      - Optional format spec: {EXPOSURE:.1f}, {DATE-OBS:%Y%m%d}
      - Counter: {#} or {#03} (zero-padded width)
      - Extension placeholder: {ext} (original extension, no dot)
      - Filters with pipes, e.g. {OBJECT|re:(\w+)|upper}
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Rename from FITS"))
        self.settings = QSettings()
        self.files: list[str] = []
        self.headers: dict[str, fits.Header] = {}
        self.union_keys: list[str] = []

        # PyQt6-safe window flags
        self.setWindowFlag(Qt.WindowType.WindowSystemMenuHint, True)
        self.setWindowFlag(Qt.WindowType.WindowTitleHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setSizeGripEnabled(True)

        self._build_ui()
        self._load_settings()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Top: source and destination
        io_row = QHBoxLayout()
        self.src_edit = QLineEdit(self)
        self.src_edit.setPlaceholderText(self.tr("Select a folder or add files…"))
        btn_scan = QPushButton(self.tr("Scan Folder…"), self); btn_scan.clicked.connect(self._scan_folder)
        btn_add = QPushButton(self.tr("Add Files…"), self); btn_add.clicked.connect(self._add_files)
        btn_clear = QPushButton(self.tr("Clear Selections"), self); btn_clear.clicked.connect(self._clear_selection)
        io_row.addWidget(QLabel(self.tr("Source:")))
        io_row.addWidget(self.src_edit, 1)
        io_row.addWidget(btn_scan)
        io_row.addWidget(btn_add)
        io_row.addWidget(btn_clear)

        self.dest_edit = QLineEdit(self)
        self.dest_edit.setPlaceholderText(self.tr("(optional) Rename into this folder; leave empty to rename in place"))
        btn_dest = QPushButton(self.tr("Browse…"), self); btn_dest.clicked.connect(self._pick_dest)
        io_row2 = QHBoxLayout()
        io_row2.addWidget(QLabel(self.tr("Destination:")))
        io_row2.addWidget(self.dest_edit, 1)
        io_row2.addWidget(btn_dest)

        root.addLayout(io_row); root.addLayout(io_row2)

        # Middle: template & options
        pat_box = QGroupBox(self.tr("Filename pattern"))
        pat_lay = QHBoxLayout(pat_box)

        self.pattern_edit = QLineEdit(self)
        self.pattern_edit.setPlaceholderText(self.tr("e.g. LIGHT_{FILTER}_{EXPOSURE:.0f}s_{DATE-OBS:%Y%m%d}_{#03}.{ext}"))
        self.pattern_edit.textChanged.connect(self._refresh_preview)

        self.lower_cb = QCheckBox(self.tr("lowercase"), self);     self.lower_cb.toggled.connect(self._refresh_preview)
        self.slug_cb  = QCheckBox(self.tr("spaces→_"), self);      self.slug_cb.toggled.connect(self._refresh_preview)
        self.keep_ext_cb = QCheckBox(self.tr("append .{ext} if missing"), self); self.keep_ext_cb.setChecked(True)
        self.index_start = QSpinBox(self); self.index_start.setRange(0, 999999); self.index_start.setValue(1)
        self.index_start.valueChanged.connect(self._refresh_preview)

        self.token_combo = QComboBox(self)
        self.token_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.token_combo.setMinimumContentsLength(12)
        self.token_combo.setEditable(False)
        self.token_combo.setToolTip(self.tr("Insert token"))
        self.token_combo.activated.connect(
            lambda idx: self._insert_token(self.token_combo.itemText(idx))
        )

        insert_btn = QPushButton(self.tr("Insert"), self)
        insert_btn.clicked.connect(lambda: self._insert_token(self.token_combo.currentText()))

        pat_lay.addWidget(QLabel(self.tr("Template:"))); pat_lay.addWidget(self.pattern_edit, 1)
        pat_lay.addWidget(self.token_combo); pat_lay.addWidget(insert_btn)
        pat_lay.addWidget(self.lower_cb); pat_lay.addWidget(self.slug_cb)
        pat_lay.addWidget(QLabel(self.tr("Index start:"))); pat_lay.addWidget(self.index_start)
        pat_lay.addWidget(self.keep_ext_cb)

        root.addWidget(pat_box)

        # Splitter: keys list | table
        split = QSplitter(Qt.Orientation.Horizontal, self)

        # Keys list
        left = QWidget(self); lyt = QVBoxLayout(left)
        left.setFixedWidth(180)
        self.keys_list = QListWidget(self)
        self.keys_list.itemDoubleClicked.connect(self._insert_key_from_list)
        lyt.addWidget(QLabel(self.tr("Available FITS keywords (double-click to insert):")))
        lyt.addWidget(self.keys_list, 1)
        split.addWidget(left)

        # Table
        right = QWidget(self); rlyt = QVBoxLayout(right)
        self.table = QTableWidget(self); self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([self.tr("Old path"), "→", self.tr("New name"), self.tr("Status")])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        try:
            self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)  # PyQt6
        except AttributeError:
            pass
        rlyt.addWidget(QLabel(self.tr("Preview")))
        rlyt.addWidget(self.table, 1)
        split.addWidget(right)
        split.setSizes([250, 700])

        root.addWidget(split, 1)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, False)

        # Buttons
        btns = QDialogButtonBox(self)
        self.btn_preview = btns.addButton(self.tr("Preview"), QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_rename  = btns.addButton(self.tr("Rename"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_close   = btns.addButton(QDialogButtonBox.StandardButton.Close)
        self.btn_preview.clicked.connect(self._refresh_preview)
        self.btn_rename.clicked.connect(self._do_rename)
        self.btn_close.clicked.connect(self.close)
        root.addWidget(btns)

        self._populate_token_keywords()

    # ---------- settings ----------
    def _load_settings(self):
        self.settings.beginGroup("batchrename")
        self.src_edit.setText(self.settings.value("last_dir", "", type=str) or "")
        self.dest_edit.setText(self.settings.value("dest_dir", "", type=str) or "")
        self.pattern_edit.setText(self.settings.value(
            "pattern", "LIGHT_{FILTER}_{EXPOSURE:.0f}s_{DATE-OBS:%Y%m%d}_{#03}.{ext}", type=str))
        self.lower_cb.setChecked(self.settings.value("lower", False, type=bool))
        self.slug_cb.setChecked(self.settings.value("slug", True, type=bool))
        self.keep_ext_cb.setChecked(self.settings.value("keep_ext", True, type=bool))
        self.index_start.setValue(self.settings.value("index_start", 1, type=int))
        self.settings.endGroup()
        if self.src_edit.text():
            self._scan_existing(self.src_edit.text())

    def _save_settings(self):
        self.settings.beginGroup("batchrename")
        self.settings.setValue("last_dir", self.src_edit.text().strip())
        self.settings.setValue("dest_dir", self.dest_edit.text().strip())
        self.settings.setValue("pattern", self.pattern_edit.text().strip())
        self.settings.setValue("lower", self.lower_cb.isChecked())
        self.settings.setValue("slug", self.slug_cb.isChecked())
        self.settings.setValue("keep_ext", self.keep_ext_cb.isChecked())
        self.settings.setValue("index_start", self.index_start.value())
        self.settings.endGroup()

    # ---------- file loading ----------
    def _scan_folder(self):
        start = self.src_edit.text().strip()
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Folder"), start or "")
        if not path: return
        self.src_edit.setText(path)
        self._scan_existing(path)
        self._save_settings()

    def _scan_existing(self, path: str):
        paths = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith((".fits", ".fit", ".fts", ".fz")):
                    paths.append(os.path.join(root, f))
        paths.sort()
        self._set_files(paths)

    def _add_files(self):
        start = self.src_edit.text().strip() or ""
        files, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Add FITS files"), start, self.tr("FITS files (*.fit *.fits *.fts *.fz);;All files (*)")
        )
        if not files: return
        new = sorted(set(self.files) | set(files))
        self._set_files(new)
        if not self.src_edit.text() and files:
            self.src_edit.setText(os.path.dirname(files[0]))
        self._save_settings()

    def _set_files(self, paths: List[str]):
        self.files = paths
        self.headers.clear()
        union = set()
        ok, bad = 0, 0
        for p in self.files:
            try:
                with fits.open(p, memmap=False) as hdul:
                    h = hdul[0].header
                self.headers[p] = h
                union.update([str(k) for k in h.keys()])
                ok += 1
            except Exception:
                bad += 1
        self.union_keys = sorted(union)
        self._rebuild_keys_list()
        self._fill_table_rows()
        self._refresh_preview()

    def _pick_dest(self):
        start = self.dest_edit.text().strip() or self.src_edit.text().strip()
        d = QFileDialog.getExistingDirectory(self, self.tr("Choose Destination Folder"), start or "")
        if not d: return
        self.dest_edit.setText(d)
        self._save_settings()

    def _autosize_combo(self, combo: QComboBox, base_padding: int = 36):
        if combo.count() == 0:
            combo.setMinimumWidth(160)
            return
        fm = QFontMetrics(combo.font())
        maxw = 0
        for i in range(combo.count()):
            w = fm.horizontalAdvance(combo.itemText(i))
            if not combo.itemIcon(i).isNull():
                w += combo.iconSize().width() + 8
            maxw = max(maxw, w)
        width = maxw + base_padding
        combo.setMinimumWidth(width)
        if combo.view() is not None:
            combo.view().setMinimumWidth(width)
        combo.updateGeometry()

    # ---------- keys & template insertion ----------
    def _rebuild_keys_list(self):
        self.keys_list.clear()
        for k in self.union_keys:
            self.keys_list.addItem(QListWidgetItem(k))
        self._populate_token_keywords()

    def _insert_key_from_list(self, item: QListWidgetItem):
        if not item: return
        self._insert_text("{"+item.text()+"}")

    def _insert_token(self, token: str):
        if not token: return
        self._insert_text(token)

    def _populate_token_keywords(self):
        tokens = ["{#}", "{#03}", "{ext}"] + [f"{{{k}}}" for k in self.union_keys]
        self.token_combo.blockSignals(True)
        self.token_combo.clear()
        self.token_combo.addItems(sorted(tokens, key=str.lower))
        self.token_combo.blockSignals(False)
        QTimer.singleShot(0, lambda: self._autosize_combo(self.token_combo))

    def _insert_text(self, text: str):
        e = self.pattern_edit
        pos = e.cursorPosition()
        s = e.text()
        e.setText(s[:pos] + text + s[pos:])
        e.setCursorPosition(pos + len(text))
        self._refresh_preview()

    # ---------- preview/rename ----------
    def _fill_table_rows(self):
        self.table.setRowCount(len(self.files))
        for r, p in enumerate(self.files):
            self.table.setItem(r, 0, QTableWidgetItem(p))
            self.table.setItem(r, 1, QTableWidgetItem("→"))
            self.table.setItem(r, 2, QTableWidgetItem(""))
            self.table.setItem(r, 3, QTableWidgetItem(""))

    def _clear_selection(self):
        self.files = []
        self.headers.clear()
        self.union_keys = []
        self._rebuild_keys_list()
        self.table.setRowCount(0)
        self._refresh_preview()

    def _refresh_preview(self):
        pat = self.pattern_edit.text().strip()
        if not pat: return
        dest = (self.dest_edit.text().strip() or None)
        start_idx = self.index_start.value()
        lower = self.lower_cb.isChecked()
        slug  = self.slug_cb.isChecked()
        keep_ext = self.keep_ext_cb.isChecked()

        names = []
        for i, p in enumerate(self.files):
            hdr = self.headers.get(p, fits.Header())
            base = self._render_pattern(pat, hdr, i, start_idx, p)
            if keep_ext and "{ext}" not in pat:
                ext = os.path.splitext(p)[1]
                if ext:
                    base = f"{base}{ext}"
            if lower: base = base.lower()
            if slug:  base = self._slugify(base)
            folder = dest if dest else os.path.dirname(p)
            target = os.path.join(folder, base)
            names.append(target)

        seen = defaultdict(int)
        for t in names: seen[t] += 1

        for r, p in enumerate(self.files):
            newp = names[r]
            self._set_table_preview_row(r, p, newp, seen[newp])

    def _set_table_preview_row(self, r: int, old: str, new: str, count: int):
        self.table.item(r, 0).setText(old)
        self.table.item(r, 2).setText(new)
        status = ""
        conflict = (count > 1)
        if conflict: status = self.tr("name collision")
        elif os.path.exists(new): status = self.tr("will overwrite")
        else: status = self.tr("ok")
        it = QTableWidgetItem(status)
        if conflict or status == "will overwrite":
            it.setForeground(Qt.GlobalColor.red)
        self.table.setItem(r, 3, it)
        it_old = self.table.item(r, 0)
        it_new = self.table.item(r, 2)
        if it_old: it_old.setToolTip(old)
        if it_new: it_new.setToolTip(new)

    def _do_rename(self):
        n = self.table.rowCount()
        targets = [self.table.item(r, 2).text() for r in range(n)]
        counts = defaultdict(int)
        for t in targets: counts[t] += 1
        collisions = [t for t,c in counts.items() if c > 1]
        if collisions:
            QMessageBox.warning(self, self.tr("Collisions"),
                self.tr("Two or more files would map to the same name. Adjust your pattern."))
            return

        failures = []
        for r in range(n):
            oldp = self.table.item(r, 0).text()
            newp = self.table.item(r, 2).text()
            if oldp == newp:
                continue
            os.makedirs(os.path.dirname(newp), exist_ok=True)
            try:
                shutil.move(oldp, newp)
                self.table.item(r, 3).setText(self.tr("renamed"))
            except Exception as e:
                self.table.item(r, 3).setText(self.tr("ERROR: {0}").format(e))
                self.table.item(r, 3).setForeground(Qt.GlobalColor.red)
                failures.append((oldp, str(e)))

        if failures:
            QMessageBox.warning(self, self.tr("Done with errors"),
                self.tr("Some files could not be renamed ({0} errors).").format(len(failures)))
        else:
            QMessageBox.information(self, self.tr("Done"), self.tr("All files renamed."))
        self._save_settings()
        src = self.src_edit.text().strip()
        if src and not self.dest_edit.text().strip():
            self._scan_existing(src)

    # ---------- helpers ----------
    @staticmethod
    def _slugify(s: str) -> str:
        s = s.replace(" ", "_")
        return re.sub(r"[^A-Za-z0-9._-]+", "", s)

    def _render_pattern(self, pat: str, hdr: fits.Header, i: int, start_idx: int, file_path: str) -> str:
        def apply_filters(text: str, filters: list[str]) -> str:
            out = str(text)
            for f in filters:
                f = f.strip()
                if f.startswith("re:"):
                    pattern = f[3:]
                    m = re.search(pattern, out)
                    if not m:
                        out = ""
                    else:
                        out = m.group(1) if m.lastindex else m.group(0)
                elif f == "lower":
                    out = out.lower()
                elif f == "upper":
                    out = out.upper()
                elif f.startswith("slice:"):
                    try:
                        _, a, b = f.split(":", 2)
                        a = int(a) if a else None
                        b = int(b) if b else None
                        out = out[a:b]
                    except Exception:
                        pass
                elif f == "strip":
                    out = out.strip()
            return out

        def _split_top_level_pipes(s: str) -> List[str]:
            parts, buf = [], []
            depth = 0
            esc = False
            for ch in s:
                if esc:
                    buf.append(ch); esc = False; continue
                if ch == '\\':
                    buf.append(ch); esc = True; continue
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth = max(0, depth-1)
                if ch == '|' and depth == 0:
                    parts.append(''.join(buf)); buf = []
                else:
                    buf.append(ch)
            parts.append(''.join(buf))
            return parts

        token_re = re.compile(r"\{((?:[^{}]|\{[^{}]*\})+)\}")

        def repl(m):
            body = m.group(1)
            parts = _split_top_level_pipes(body)
            key_fmt = parts[0]
            filters = parts[1:] if len(parts) > 1 else []

            # counter?
            if key_fmt.startswith("#"):
                w = key_fmt[1:]
                try:
                    pad = int(w) if w else 0
                except Exception:
                    pad = 0
                num = i + start_idx
                return f"{num:0{pad}d}" if pad else str(num)

            # extension?
            if key_fmt.lower() == "ext":
                ext = os.path.splitext(file_path)[1]
                return ext.lstrip(".")

            # key[:fmt]
            if ":" in key_fmt:
                key, fmt = key_fmt.split(":", 1)
            else:
                key, fmt = key_fmt, ""
            key_up = key.upper()
            val = hdr.get(key_up, "")
            if val is None:
                val = ""

            # DATE-like with datetime fmt
            if fmt and key_up in ("DATE-OBS", "DATE"):
                s = str(val).strip().replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    out = dt.strftime(fmt)
                except Exception:
                    out = str(val)
                return apply_filters(out, filters)

            # TIME-only keys with time fmt
            if fmt and key_up in ("TIME-OBS", "UTSTART", "UTC-START"):
                s = str(val).strip()
                try:
                    if s.count(":") == 2:
                        tt = datetime.strptime(s, "%H:%M:%S").time()
                    else:
                        tt = datetime.strptime(s, "%H:%M").time()
                    out = tt.strftime(fmt)
                except Exception:
                    try:
                        tt = datetime.fromisoformat(f"1970-01-01T{s}").time()
                        out = tt.strftime(fmt)
                    except Exception:
                        out = str(val)
                return apply_filters(out, filters)

            # numeric with fmt
            if fmt:
                try:
                    out = format(float(val), fmt)
                    return apply_filters(out, filters)
                except Exception:
                    pass

            return apply_filters(str(val), filters)

        return token_re.sub(repl, pat)
