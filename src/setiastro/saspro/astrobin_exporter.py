# pro/astrobin_exporter.py
from __future__ import annotations

import os
import re
import io
import csv
import webbrowser
import shutil
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
from astropy.io import fits

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import (
    QFontMetrics, QIntValidator, QDoubleValidator,
    QStandardItemModel, QStandardItem
)
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QGridLayout, QSplitter, QListWidget, QListWidgetItem, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView, QTableWidget,
    QTableWidgetItem, QTextEdit, QDialogButtonBox, QComboBox, QSpinBox, QCheckBox,
    QSpacerItem, QSizePolicy, QStyledItemDelegate, QToolButton, QMessageBox, QCompleter
)
from PyQt6.QtGui import QGuiApplication

# Optional XISF support (skip .xisf if not importable)
try:
    from setiastro.saspro.legacy.xisf import XISF
except Exception:
    XISF = None  # handled gracefully below

# ---- constants --------------------------------------------------------------

ASTROBIN_FILTER_URL = "https://app.astrobin.com/equipment/explorer/filter?page=1"

# Try to honor your project path name if present; otherwise, fall back next to this file.


# ---- delegates / completer -------------------------------------------------

class _IdOnlyCompleter(QCompleter):
    """
    Shows 'ID — Brand — Name' in the popup, but inserts only the numeric ID.
    """
    def pathFromIndex(self, index):
        return index.data(Qt.ItemDataRole.UserRole) or super().pathFromIndex(index)

class _AstrobinIdDelegate(QStyledItemDelegate):
    """
    QLineEdit with int validator + optional completer for the AstroBin ID column.
    """
    def __init__(self, parent=None, completer: Optional[QCompleter] = None):
        super().__init__(parent)
        self._completer = completer

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setPlaceholderText("e.g. 4408")
        editor.setValidator(QIntValidator(1, 999_999_999, editor))
        if self._completer is not None:
            editor.setCompleter(self._completer)
        return editor

# ---- Filter ID editor ------------------------------------------------------



class FilterIdDialog(QDialog):
    """
    Editable table: local filter name ↔ AstroBin numeric ID.
    Loads/saves mapping in QSettings key: astrobin_exporter/filter_map
    Also supports an offline CSV for ID lookup / completion.
    """

    BLANK_ROWS = 6

    def __init__(self, parent, filters_in_data: List[str], settings: QSettings,
                 current_map: Optional[Dict[str, str]] = None,
                 offline_csv_default: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("AstroBin Filter IDs")
        self.settings = settings
        self._offline_csv_default = offline_csv_default
        base_names = sorted({f for f in (filters_in_data or []) if f and f != "Unknown"}, key=str.lower)
        stored_map = self._load_mapping()
        if current_map:
            stored_map = {**stored_map, **current_map}
        all_names = sorted(set(base_names) | set(stored_map.keys()), key=str.lower)

        root = QVBoxLayout(self)

        # Help row
        help_row = QHBoxLayout()
        help_label = QLabel("Edit filter names and their AstroBin numeric IDs.")
        help_btn = QToolButton(self)
        help_btn.setText("?")
        help_btn.setToolTip("Open AstroBin Equipment Explorer (Filters)")
        help_btn.clicked.connect(lambda: webbrowser.open(ASTROBIN_FILTER_URL))

        self.load_db_btn = QPushButton(self)
        self.load_db_btn.setToolTip("Search or load the offline filters database.")
        self.load_db_btn.clicked.connect(self._on_offline_action)

        help_row.addWidget(help_label)
        help_row.addStretch(1)
        help_row.addWidget(self.load_db_btn)
        help_row.addWidget(help_btn)
        root.addLayout(help_row)

        # Table
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Filter name", "AstroBin ID"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)

        # Offline DB → completer
        self._load_offline_db()
        self._id_completer = self._make_id_completer()
        self.table.setItemDelegateForColumn(1, _AstrobinIdDelegate(self.table, completer=self._id_completer))
        self._update_offline_button_text()

        # Fill rows
        rows = len(all_names) + self.BLANK_ROWS
        self.table.setRowCount(rows)
        r = 0
        for name in all_names:
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsEditable)
            id_item = QTableWidgetItem(str(stored_map.get(name, "")))
            id_item.setFlags(id_item.flags() | Qt.ItemFlag.ItemIsEditable)
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, id_item)
            r += 1

        while r < rows:
            self.table.setItem(r, 0, QTableWidgetItem(""))
            self.table.setItem(r, 1, QTableWidgetItem(""))
            self.table.item(r, 0).setFlags(self.table.item(r, 0).flags() | Qt.ItemFlag.ItemIsEditable)
            self.table.item(r, 1).setFlags(self.table.item(r, 1).flags() | Qt.ItemFlag.ItemIsEditable)
            self.table.item(r, 1).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            r += 1

        root.addWidget(self.table, 1)

        # Row actions
        row_actions = QHBoxLayout()
        self.btn_add = QPushButton("Add row");    self.btn_add.clicked.connect(self._add_row)
        self.btn_del = QPushButton("Delete selected"); self.btn_del.clicked.connect(self._delete_selected_rows)
        row_actions.addWidget(self.btn_add); row_actions.addWidget(self.btn_del); row_actions.addStretch(1)
        root.addLayout(row_actions)

        # OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.table.setCurrentCell(0, 0)

    # --- settings map helpers ---
    def _load_mapping(self) -> Dict[str, str]:
        self.settings.beginGroup("astrobin_exporter")
        raw = self.settings.value("filter_map", "")
        self.settings.endGroup()
        mapping: Dict[str, str] = {}
        if isinstance(raw, str) and raw:
            for chunk in raw.split(";"):
                if "=" in chunk:
                    k, v = chunk.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k:
                        mapping[k] = v
        return mapping

    def mapping(self) -> Dict[str, str]:
        mp: Dict[str, str] = {}
        rows = self.table.rowCount()
        for r in range(rows):
            name_item = self.table.item(r, 0)
            id_item = self.table.item(r, 1)
            name = (name_item.text().strip() if name_item else "")
            fid = (id_item.text().strip() if id_item else "")
            if not name:
                continue
            if fid and fid.isdigit():
                mp[name] = fid
        return mp

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(""))
        self.table.setItem(r, 1, QTableWidgetItem(""))
        # make editable + center the ID
        self.table.item(r, 0).setFlags(self.table.item(r, 0).flags() | Qt.ItemFlag.ItemIsEditable)
        self.table.item(r, 1).setFlags(self.table.item(r, 1).flags() | Qt.ItemFlag.ItemIsEditable)
        self.table.item(r, 1).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.scrollToBottom()
        self.table.setCurrentCell(r, 0)

    def _delete_selected_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def save_to_settings(self):
        mp = self.mapping()
        blob = ";".join(f"{k}={v}" for k, v in mp.items())
        self.settings.beginGroup("astrobin_exporter")
        self.settings.setValue("filter_map", blob)
        self.settings.endGroup()

    # --- offline DB (CSV) ---
    def _find_offline_csv(self) -> Optional[str]:
        # 0) explicit override from parent
        if self._offline_csv_default and os.path.isfile(self._offline_csv_default):
            return self._offline_csv_default

        # 1) user-specified in settings
        self.settings.beginGroup("astrobin_exporter")
        saved = self.settings.value("offline_filters_csv", "")
        self.settings.endGroup()
        if isinstance(saved, str) and saved and os.path.isfile(saved):
            return saved

        # 2) module default (if you kept it)
        if os.path.isfile(self._offline_csv_default):
            return self._offline_csv_default
        return None

    def _update_offline_button_text(self):
        self.load_db_btn.setText("Search offline DB…" if getattr(self, "_offline_rows", None) else "Load offline DB…")

    def _on_offline_action(self):
        if getattr(self, "_offline_rows", None):
            self._open_offline_search()
        else:
            self._browse_offline_db()
            self._id_completer = self._make_id_completer()
            self.table.setItemDelegateForColumn(1, _AstrobinIdDelegate(self.table, completer=self._id_completer))
            self._update_offline_button_text()

    def _browse_offline_db(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select AstroBin Filters CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        self._load_offline_db(path)
        self._id_completer = self._make_id_completer()
        self.table.setItemDelegateForColumn(1, _AstrobinIdDelegate(self.table, completer=self._id_completer))

    def _load_offline_db(self, csv_path: Optional[str] = None) -> List[Dict]:
        if not csv_path:
            csv_path = self._find_offline_csv()
        self._offline_rows = []
        if not csv_path:
            return self._offline_rows

        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    fid = (row.get("id") or "").strip()
                    if not fid.isdigit():
                        continue
                    self._offline_rows.append({
                        "id": fid,
                        "brand": (row.get("brand") or "").strip(),
                        "name": (row.get("name") or "").strip()
                    })
            self.settings.beginGroup("astrobin_exporter")
            self.settings.setValue("offline_filters_csv", csv_path)
            self.settings.endGroup()
        except Exception as e:
            print(f"[WARN] Failed to load offline CSV: {e}")
        return self._offline_rows

    def _make_id_completer(self) -> Optional[QCompleter]:
        rows = getattr(self, "_offline_rows", None) or []
        if not rows:
            return None
        model = QStandardItemModel()
        for r in rows:
            fid = r["id"]; brand = r.get("brand") or ""; name = r.get("name") or ""
            disp = f"{fid} — {brand} — {name}".strip(" —")
            it = QStandardItem(disp)
            it.setData(fid, Qt.ItemDataRole.UserRole)
            model.appendRow(it)
        comp = _IdOnlyCompleter(model, self)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setCompletionRole(Qt.ItemDataRole.DisplayRole)
        return comp

    def _open_offline_search(self):
        if not getattr(self, "_offline_rows", None):
            QMessageBox.information(self, "No DB", "Offline filters database not loaded yet.")
            return

        dlg = QDialog(self); dlg.setWindowTitle("Search AstroBin Filters (offline)")
        v = QVBoxLayout(dlg)
        q = QLineEdit(dlg); q.setPlaceholderText("Search ID, brand, or name…")
        v.addWidget(q)

        tbl = QTableWidget(dlg); tbl.setColumnCount(3)
        tbl.setHorizontalHeaderLabels(["ID", "Brand", "Name"])
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        v.addWidget(tbl, 1)

        rows = sorted(self._offline_rows, key=lambda r: (r.get("brand","").lower(), r.get("name","").lower()))
        tbl.setRowCount(len(rows))
        for r, data in enumerate(rows):
            for c, key in enumerate(("id","brand","name")):
                it = QTableWidgetItem(data.get(key,""))
                if c == 0: it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                tbl.setItem(r, c, it)

        dlg.resize(520, 360)
        tbl.resizeColumnsToContents()
        hdr.resizeSection(0, 90)
        hdr.resizeSection(1, 120)

        def apply_filter(text: str):
            t = (text or "").lower()
            for r in range(tbl.rowCount()):
                row_txt = " ".join((tbl.item(r, c).text() if tbl.item(r, c) else "") for c in range(3)).lower()
                tbl.setRowHidden(r, t not in row_txt)

        q.textChanged.connect(apply_filter)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dlg)
        v.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        tbl.doubleClicked.connect(lambda *_: dlg.accept())

        if dlg.exec() == QDialog.DialogCode.Accepted:
            r = tbl.currentRow()
            if r >= 0:
                fid = tbl.item(r, 0).text()
                cur = self.table.currentRow()
                if cur < 0: cur = 0
                item = QTableWidgetItem(fid)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(cur, 1, item)
                self.table.setCurrentCell(cur, 1)
                self.table.editItem(item)

# ---- Main Exporter ---------------------------------------------------------

class AstrobinExportTab(QWidget):
    """
    Left: file picker + tree (Object→Filter→Exposure)
    Right: global inputs + aggregated table + CSV preview + copy CSV.
    """
    def __init__(self, parent=None, offline_filters_csv: Optional[str] = None):
        super().__init__(parent)
        self.settings = QSettings()
        self.file_paths: List[str] = []
        self.records: List[dict] = []
        self.rows: List[dict] = []
        self._filter_map: Dict[str, str] = self._load_filter_map()
        self._offline_csv_default = offline_filters_csv 

        self._build_ui()
        self._load_defaults()

    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter)

        # LEFT
        left = QWidget(self); lyt = QVBoxLayout(left)
        self.info_lbl = QLabel("Load FITS via 'Select Folder…' or 'Add Files…' to begin.")
        lyt.addWidget(self.info_lbl)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Select Folder…"); self.btn_open.clicked.connect(self.open_directory)
        self.btn_add_files = QPushButton("Add Files…"); self.btn_add_files.clicked.connect(self.open_files)
        self.btn_clear = QPushButton("Clear"); self.btn_clear.clicked.connect(self.clear_images)
        btn_row.addWidget(self.btn_open); btn_row.addWidget(self.btn_add_files); btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        lyt.addLayout(btn_row)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(1)
        self.tree.setHeaderLabels(["Files (Object → Filter → Exposure)"])
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        lyt.addWidget(self.tree)

        splitter.addWidget(left)

        # RIGHT
        right = QWidget(self); rlyt = QVBoxLayout(right)

        form_box = QGroupBox("Global inputs (used if FITS headers are missing/zero)")
        grid = QGridLayout(form_box)

        # row 0
        grid.addWidget(QLabel("f/number"), 0, 0)
        self.fnum_edit = QLineEdit(self); self.fnum_edit.setPlaceholderText("e.g. 4.0")
        self.fnum_edit.setValidator(QDoubleValidator(0.0, 999.0, 2, self)); self.fnum_edit.textChanged.connect(self._recompute)
        grid.addWidget(self.fnum_edit, 0, 1)

        grid.addWidget(QLabel("Darks (#)"), 0, 2)
        self.darks_edit = QLineEdit(self); self._setup_int_line(self.darks_edit, 0, 999999)
        grid.addWidget(self.darks_edit, 0, 3)

        grid.addWidget(QLabel("Flats (#)"), 0, 4)
        self.flats_edit = QLineEdit(self); self._setup_int_line(self.flats_edit, 0, 999999)
        grid.addWidget(self.flats_edit, 0, 5)

        # row 1
        grid.addWidget(QLabel("Flat-darks (#)"), 1, 0)
        self.flatdarks_edit = QLineEdit(self); self._setup_int_line(self.flatdarks_edit, 0, 999999)
        grid.addWidget(self.flatdarks_edit, 1, 1)

        grid.addWidget(QLabel("Bias (#)"), 1, 2)
        self.bias_edit = QLineEdit(self); self._setup_int_line(self.bias_edit, 0, 999999)
        grid.addWidget(self.bias_edit, 1, 3)

        grid.addWidget(QLabel("Bortle"), 1, 4)
        self.bortle_edit = QLineEdit(self); self.bortle_edit.setPlaceholderText("0–9")
        self.bortle_edit.setValidator(QIntValidator(0, 9, self)); self.bortle_edit.textChanged.connect(self._recompute)
        grid.addWidget(self.bortle_edit, 1, 5)

        # row 2
        grid.addWidget(QLabel("Mean SQM"), 2, 0)
        self.mean_sqm_edit = QLineEdit(self); self.mean_sqm_edit.setPlaceholderText("e.g. 21.30")
        self.mean_sqm_edit.setValidator(QDoubleValidator(0.0, 25.0, 2, self)); self.mean_sqm_edit.textChanged.connect(self._recompute)
        grid.addWidget(self.mean_sqm_edit, 2, 1)

        grid.addWidget(QLabel("Mean FWHM"), 2, 2)
        self.mean_fwhm_edit = QLineEdit(self); self.mean_fwhm_edit.setPlaceholderText("e.g. 2.10")
        self.mean_fwhm_edit.setValidator(QDoubleValidator(0.0, 50.0, 2, self)); self.mean_fwhm_edit.textChanged.connect(self._recompute)
        grid.addWidget(self.mean_fwhm_edit, 2, 3)

        self.noon_cb = QCheckBox("Group nights noon → noon (local time)")
        self.noon_cb.setToolTip("Prevents splitting a single observing night at midnight.")
        self.noon_cb.setChecked(self.settings.value("astrobin_exporter/noon_to_noon", True, type=bool))
        self.noon_cb.toggled.connect(self._recompute)
        grid.addWidget(self.noon_cb, 2, 4, 1, 2)

        grid.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 2, 4, 1, 2)

        # Filter mapping row
        map_row = QHBoxLayout()
        self.filter_summary = QLabel(self._filters_summary_text()); map_row.addWidget(self.filter_summary)
        self.btn_edit_filters = QPushButton("Manage Filter IDs…"); self.btn_edit_filters.clicked.connect(self._edit_filters)
        map_row.addWidget(self.btn_edit_filters)
        qmark = QToolButton(self); qmark.setText("?")
        qmark.setToolTip("Open AstroBin Equipment Explorer (Filters)")
        qmark.clicked.connect(lambda: webbrowser.open(ASTROBIN_FILTER_URL))
        map_row.addWidget(qmark); map_row.addStretch(1)
        map_wrap = QWidget(self); map_wrap.setLayout(map_row)
        grid.addWidget(map_wrap, 3, 0, 1, 6)

        rlyt.addWidget(form_box)

        # Aggregated table
        self.table = QTableWidget(self)
        cols = ['date','filter','number','duration','gain','iso','binning','sensorCooling',
                'fNumber','darks','flats','flatDarks','bias','bortle','meanSqm','meanFwhm','temperature']
        self.table.setColumnCount(len(cols)); self.table.setHorizontalHeaderLabels(cols)
        hdr_tbl = self.table.horizontalHeader()
        hdr_tbl.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr_tbl.setStretchLastSection(False)
        hdr_tbl.setMinimumSectionSize(50)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        rlyt.addWidget(self.table, 1)

        # CSV preview
        rlyt.addWidget(QLabel("CSV Preview:"))
        self.csv_view = QTextEdit(self); self.csv_view.setReadOnly(True)
        rlyt.addWidget(self.csv_view, 1)

        # Actions
        act_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Recompute"); self.btn_refresh.clicked.connect(self._recompute)
        self.btn_copy_csv = QPushButton("Copy CSV"); self.btn_copy_csv.clicked.connect(self._copy_csv_to_clipboard)
        act_row.addWidget(self.btn_refresh); act_row.addWidget(self.btn_copy_csv); act_row.addStretch(1)
        rlyt.addLayout(act_row)

        splitter.addWidget(right)
        splitter.setSizes([360, 680])
        self.setLayout(root)

        self.info_lbl.setText("Load FITS/XISF via 'Select Folder…' or 'Add Files…' to begin.")

    def _setup_int_line(self, line: QLineEdit, lo: int, hi: int):
        line.setValidator(QIntValidator(lo, hi, self))
        line.setPlaceholderText("0")
        line.textChanged.connect(self._recompute)

    # ---------- settings ----------
    def _load_defaults(self):
        self.settings.beginGroup("astrobin_exporter")
        self.fnum_edit.setText(str(self.settings.value("fnumber", "")))
        self.darks_edit.setText(str(self.settings.value("darks", "")))
        self.flats_edit.setText(str(self.settings.value("flats", "")))
        self.flatdarks_edit.setText(str(self.settings.value("flatdarks", "")))
        self.bias_edit.setText(str(self.settings.value("bias", "")))
        self.bortle_edit.setText(str(self.settings.value("bortle", "")))
        self.mean_sqm_edit.setText(str(self.settings.value("mean_sqm", "")))
        self.mean_fwhm_edit.setText(str(self.settings.value("mean_fwhm", "")))
        self.noon_cb.setChecked(self.settings.value("noon_to_noon", True, type=bool))
        self._last_dir = str(self.settings.value("last_dir", "")) or ""
        self.settings.endGroup()

    def _save_defaults(self):
        self.settings.beginGroup("astrobin_exporter")
        self.settings.setValue("fnumber", self.fnum_edit.text().strip())
        self.settings.setValue("darks", self.darks_edit.text().strip())
        self.settings.setValue("flats", self.flats_edit.text().strip())
        self.settings.setValue("flatdarks", self.flatdarks_edit.text().strip())
        self.settings.setValue("bias", self.bias_edit.text().strip())
        self.settings.setValue("bortle", self.bortle_edit.text().strip())
        self.settings.setValue("mean_sqm", self.mean_sqm_edit.text().strip())
        self.settings.setValue("mean_fwhm", self.mean_fwhm_edit.text().strip())
        self.settings.setValue("noon_to_noon", self.noon_cb.isChecked())
        self.settings.endGroup()

    def _get_last_dir(self) -> str:
        return getattr(self, "_last_dir", "") or ""

    def _save_last_dir(self, path: str):
        if not path: return
        self._last_dir = path
        self.settings.beginGroup("astrobin_exporter")
        self.settings.setValue("last_dir", path)
        self.settings.endGroup()

    # ---------- file I/O ----------
    def clear_images(self):
        self.file_paths.clear(); self.records.clear(); self.rows.clear()
        self.tree.blockSignals(True); self.tree.clear(); self.tree.blockSignals(False)
        self.table.setRowCount(0); self.csv_view.clear()
        self.info_lbl.setText("Cleared. Load FITS via 'Select Folder…' or 'Add Files…' to begin.")

    def open_directory(self):
        start = self._get_last_dir() or ""
        directory = QFileDialog.getExistingDirectory(self, "Select Folder Containing FITS/XISF Files", start)
        if not directory: return
        self._save_last_dir(directory)

        paths = []
        for root, _, files in os.walk(directory):
            for fn in files:
                if fn.lower().endswith((".fit", ".fits", ".xisf")):
                    paths.append(os.path.join(root, fn))
        paths.sort(key=self._natural_key)
        if not paths:
            QMessageBox.information(self, "No Images", "No .fit/.fits/.xisf files found.")
            return
        self.file_paths = paths
        self._read_headers(); self._build_tree(); self._recompute()

    def open_files(self):
        start = self._get_last_dir() or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS/XISF Files", start, "FITS/XISF (*.fit *.fits *.xisf);;All Files (*)"
        )
        if not paths: return
        self._save_last_dir(os.path.dirname(paths[0]))

        new_paths = [p for p in paths if p not in self.file_paths]
        if not new_paths:
            QMessageBox.information(self, "No New Files", "All selected files are already in the list.")
            return

        self.file_paths = sorted(set(self.file_paths + new_paths), key=self._natural_key)
        self._read_headers(); self._build_tree(); self._recompute()

    # ---------- header reading ----------
    def _xisf_first_kw(self, image_meta: dict, key: str, default=None):
        try:
            vals = (image_meta.get("FITSKeywords") or {}).get(key, [])
            if vals: return vals[0].get("value", default)
        except Exception:
            pass
        return default

    def _xisf_search_props(self, image_meta: dict, substrings, default=None):
        props = image_meta.get("XISFProperties") or {}
        if isinstance(substrings, str):
            substrings = [substrings]
        for k, v in props.items():
            lk = k.lower()
            if any(s in lk for s in substrings):
                return v.get("value", default)
        return default

    def _xisf_flatten_fits_keywords(self, image_meta: dict) -> dict:
        out = {}
        try:
            kmap = image_meta.get("FITSKeywords") or {}
            for k, arr in kmap.items():
                if arr:
                    out[k] = arr[0].get("value", None)
        except Exception:
            pass
        return out

    def _read_headers(self):
        from pathlib import Path
        self.records.clear()
        ok, bad, skipped_xisf = 0, 0, 0

        for fp in self.file_paths:
            try:
                ext = Path(fp).suffix.lower()
                if ext in (".fit", ".fits"):
                    with fits.open(fp, memmap=False) as hdul:
                        h = hdul[0].header

                    exposure = h.get("EXPOSURE", h.get("EXPTIME", 0.0))
                    binning = self._derive_binning(h)
                    rec = {
                        "PATH": fp, "NAME": os.path.basename(fp),
                        "OBJECT": str(h.get("OBJECT", "Unknown")),
                        "FILTER": str(h.get("FILTER", "Unknown")),
                        "EXPOSURE": self._safe_float(exposure),
                        "GAIN": str(h.get("GAIN", "0")),
                        "ISO": str(h.get("ISO", "0")),
                        "BINNING": binning,
                        "CCD_TEMP": self._safe_float(h.get("CCD-TEMP", 0.0)),
                        "FOCTEMP": self._safe_float(h.get("FOCTEMP", 0.0)),
                        "DARK": str(h.get("DARK", "0")),
                        "FLAT": str(h.get("FLAT", "0")),
                        "FLATDARK": str(h.get("FLATDARK", "0")),
                        "BIAS": str(h.get("BIAS", "0")),
                        "BORTLE": str(h.get("BORTLE", "0")),
                        "MEAN_SQM": str(h.get("MEAN_SQM", "0")),
                        "MEAN_FWHM": str(h.get("MEAN_FWHM", "0")),
                        "DATE": self._to_date_only(str(h.get("DATE-OBS", "0"))),
                        "DATEOBS": str(h.get("DATE-OBS", "")),
                    }
                    self.records.append(rec); ok += 1

                elif ext == ".xisf":
                    if XISF is None:
                        skipped_xisf += 1
                        continue
                    xisf = XISF(fp)
                    image_meta = xisf.get_images_metadata()[0]
                    flat = self._xisf_flatten_fits_keywords(image_meta)
                    exposure = flat.get("EXPOSURE", flat.get("EXPTIME", 0.0))
                    filt_name = str(flat.get("FILTER", "")) or \
                                str(self._xisf_search_props(image_meta, ["filter", "channel", "band"], default="Unknown"))
                    gain_val = flat.get("GAIN", None)
                    if gain_val is None:
                        gain_val = self._xisf_search_props(image_meta, "gain", default="0")
                    iso_val = flat.get("ISO", None)
                    if iso_val is None:
                        iso_val = self._xisf_search_props(image_meta, "iso", default="0")
                    binning = self._derive_binning(flat)
                    ccd_temp = self._safe_float(
                        flat.get("CCD-TEMP", self._xisf_search_props(image_meta, ["ccd-temp","sensor","temperature"], default=0.0))
                    )
                    foc_temp = self._safe_float(flat.get("FOCTEMP", 0.0))
                    dateobs = str(flat.get("DATE-OBS", "")) or ""

                    rec = {
                        "PATH": fp, "NAME": os.path.basename(fp),
                        "OBJECT": str(flat.get("OBJECT", self._xisf_search_props(image_meta, "object", default="Unknown"))),
                        "FILTER": filt_name or "Unknown",
                        "EXPOSURE": self._safe_float(exposure),
                        "GAIN": str(gain_val if gain_val is not None else "0"),
                        "ISO": str(iso_val if iso_val is not None else "0"),
                        "BINNING": binning,
                        "CCD_TEMP": ccd_temp,
                        "FOCTEMP": foc_temp,
                        "DARK": str(flat.get("DARK", "0")),
                        "FLAT": str(flat.get("FLAT", "0")),
                        "FLATDARK": str(flat.get("FLATDARK", "0")),
                        "BIAS": str(flat.get("BIAS", "0")),
                        "BORTLE": str(flat.get("BORTLE", "0")),
                        "MEAN_SQM": str(flat.get("MEAN_SQM", "0")),
                        "MEAN_FWHM": str(flat.get("MEAN_FWHM", "0")),
                        "DATE": self._to_date_only(dateobs) if dateobs else "0",
                        "DATEOBS": dateobs,
                    }
                    self.records.append(rec); ok += 1

            except Exception as e:
                print(f"[WARN] Failed to read {fp}: {e}")
                bad += 1

        msg = f"Loaded {ok} file(s)"
        if bad: msg += f" ({bad} failed)"
        if skipped_xisf: msg += f" — skipped {skipped_xisf} XISF (reader unavailable)"
        self.info_lbl.setText(msg + ".")

    # ---------- helpers ----------
    @staticmethod
    def _natural_key(path: str):
        name = os.path.basename(path)
        return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', name)]

    @staticmethod
    def _safe_float(x, default=0.0) -> float:
        try: return float(x)
        except Exception: return float(default)

    @staticmethod
    def _derive_binning(h) -> str:
        for k in ("XBINNING", "XBIN", "CCDXBIN"):
            if k in h:
                try: return str(int(float(h[k])))
                except Exception: return str(h[k])
        return "0"

    @staticmethod
    def _to_date_only(date_obs: str) -> str:
        if not date_obs or date_obs == "0":
            return "0"
        return date_obs.split("T")[0].strip()

    def _parse_date_obs(self, s: str) -> Optional[datetime]:
        s = (s or "").strip()
        if not s or s == "0": return None
        s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _night_date_str(self, date_obs: str, noon_to_noon: bool) -> str:
        dt = self._parse_date_obs(date_obs)
        if not dt:
            return (date_obs.split("T")[0] if date_obs else "0")
        local_tz = datetime.now().astimezone().tzinfo
        ldt = dt.astimezone(local_tz)
        if noon_to_noon:
            ldt = ldt - timedelta(hours=12)
        return ldt.date().isoformat()

    def _build_tree(self):
        self.tree.blockSignals(True); self.tree.clear()

        grouped: Dict[Tuple[str, str, float], List[dict]] = defaultdict(list)
        for rec in self.records:
            key = (rec["OBJECT"], rec["FILTER"], rec["EXPOSURE"])
            grouped[key].append(rec)

        by_obj: Dict[str, Dict[str, Dict[float, List[dict]]]] = defaultdict(lambda: defaultdict(dict))
        for (obj, filt, exp), lst in grouped.items():
            by_obj[obj].setdefault(filt, {})
            by_obj[obj][filt][exp] = lst

        for obj in sorted(by_obj, key=str.lower):
            obj_item = QTreeWidgetItem([f"Object: {obj}"])
            obj_item.setFlags(obj_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            obj_item.setCheckState(0, Qt.CheckState.Checked)
            self.tree.addTopLevelItem(obj_item); obj_item.setExpanded(True)

            for filt in sorted(by_obj[obj], key=str.lower):
                filt_item = QTreeWidgetItem([f"Filter: {filt}"])
                filt_item.setFlags(filt_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                filt_item.setCheckState(0, Qt.CheckState.Checked)
                obj_item.addChild(filt_item); filt_item.setExpanded(True)

                for exp in sorted(by_obj[obj][filt].keys(), key=lambda e: str(e)):
                    exp_item = QTreeWidgetItem([f"Exposure: {exp}"])
                    exp_item.setData(0, Qt.ItemDataRole.UserRole, float(exp))
                    exp_item.setFlags(exp_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    exp_item.setCheckState(0, Qt.CheckState.Checked)
                    filt_item.addChild(exp_item)
                    for rec in by_obj[obj][filt][exp]:
                        leaf = QTreeWidgetItem([rec["NAME"]])
                        leaf.setData(0, Qt.ItemDataRole.UserRole, rec["PATH"])
                        leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        leaf.setCheckState(0, Qt.CheckState.Checked)
                        exp_item.addChild(leaf)

        self.tree.blockSignals(False)
        self.tree.header().setStretchLastSection(False)
        self.tree.resizeColumnToContents(0)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)

    def _has_gain(self, v) -> bool:
        s = str(v).strip()
        if not s: return False
        try: return float(s) > 0.0
        except Exception: return False

    def _on_tree_item_changed(self, item: QTreeWidgetItem, _col: int):
        state = item.checkState(0)
        for i in range(item.childCount()):
            ch = item.child(i)
            ch.setCheckState(0, state)
        self._recompute()

    # ---------- aggregation & CSV ----------
    def _included_paths(self) -> set:
        paths = set()
        def recurse(node: QTreeWidgetItem):
            if node.childCount() == 0:
                if node.checkState(0) == Qt.CheckState.Checked:
                    p = node.data(0, Qt.ItemDataRole.UserRole)
                    if isinstance(p, str): paths.add(p)
                return
            for i in range(node.childCount()):
                recurse(node.child(i))
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            recurse(root.child(i))
        return paths

    def _fallback(self, header_val: str, global_val: str) -> str:
        hv = (header_val or "").strip(); gv = (global_val or "").strip()
        if hv in ("", "0", "0.0") and gv != "": return gv
        return hv or "0"

    def _recompute(self):
        self._save_defaults()
        noon_to_noon = self.noon_cb.isChecked()

        selected = self._included_paths()
        if not selected:
            self.rows = []; self._refresh_table(); self._refresh_csv_text(); return

        agg = defaultdict(lambda: {
            'date': '0', 'filter': '0', 'number': 0, 'duration': 0, 'gain': '0', 'iso': '0',
            'binning': '0', 'sensorCooling': 0, 'fNumber': '0', 'darks': '0', 'flats': '0',
            'flatDarks': '0', 'bias': '0', 'bortle': '0', 'meanSqm': '0', 'meanFwhm': '0',
            'temperature_sum': 0.0, 'temp_count': 0
        })

        fnum = self.fnum_edit.text().strip() or "0"
        g_darks = self.darks_edit.text().strip()
        g_flats = self.flats_edit.text().strip()
        g_flatdarks = self.flatdarks_edit.text().strip()
        g_bias = self.bias_edit.text().strip()
        g_bortle = self.bortle_edit.text().strip()
        g_sqm = self.mean_sqm_edit.text().strip()
        g_fwhm = self.mean_fwhm_edit.text().strip()

        for rec in self.records:
            if rec["PATH"] not in selected: continue
            date = self._night_date_str(rec.get("DATEOBS",""), noon_to_noon)
            filt_name = rec["FILTER"] or "0"
            filt_id = self._filter_map.get(filt_name, filt_name)
            exposure = rec["EXPOSURE"] or 0.0
            key = (date, str(filt_id), float(exposure))

            item = agg[key]
            item['date'] = date
            item['filter'] = str(filt_id)
            item['duration'] = exposure
            item['gain'] = rec["GAIN"]
            item['iso'] = rec["ISO"]
            item['binning'] = rec["BINNING"]
            item['sensorCooling'] = int(round(rec["CCD_TEMP"])) if rec["CCD_TEMP"] else 0
            item['fNumber'] = fnum

            item['darks'] = self._fallback(rec["DARK"], g_darks)
            item['flats'] = self._fallback(rec["FLAT"], g_flats)
            item['flatDarks'] = self._fallback(rec["FLATDARK"], g_flatdarks)
            item['bias'] = self._fallback(rec["BIAS"], g_bias)
            item['bortle'] = self._fallback(rec["BORTLE"], g_bortle)
            item['meanSqm'] = self._fallback(rec["MEAN_SQM"], g_sqm)
            item['meanFwhm'] = self._fallback(rec["MEAN_FWHM"], g_fwhm)

            if rec["FOCTEMP"]:
                item['temperature_sum'] += float(rec["FOCTEMP"])
                item['temp_count'] += 1
            item['number'] += 1

        out = []
        for (_date, _fid, _exp), v in agg.items():
            temp = int(round(v['temperature_sum'] / v['temp_count'])) if v['temp_count'] > 0 else 0
            row = {
                'date': v['date'], 'filter': v['filter'], 'number': v['number'],
                'duration': v['duration'], 'gain': v['gain'], 'iso': v['iso'],
                'binning': v['binning'], 'sensorCooling': v['sensorCooling'], 'fNumber': v['fNumber'],
                'darks': v['darks'], 'flats': v['flats'], 'flatDarks': v['flatDarks'],
                'bias': v['bias'], 'bortle': v['bortle'], 'meanSqm': v['meanSqm'],
                'meanFwhm': v['meanFwhm'], 'temperature': temp
            }
            if self._has_gain(row['gain']):
                row['iso'] = ""  # if gain is present, blank ISO
            out.append(row)

        out.sort(key=lambda r: (r['date'], r['filter'], float(r['duration'])))
        self.rows = out
        self._refresh_table(); self._refresh_csv_text()

    def _refresh_table(self):
        cols = ['date','filter','number','duration','gain','iso','binning','sensorCooling',
                'fNumber','darks','flats','flatDarks','bias','bortle','meanSqm','meanFwhm','temperature']
        self.table.setRowCount(len(self.rows))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(self.rows):
            for c, key in enumerate(cols):
                item = QTableWidgetItem(str(row.get(key, "")))
                if key == "filter" and not str(row.get(key, "")).isdigit():
                    item.setForeground(Qt.GlobalColor.red)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def _rows_to_csv_str(self) -> str:
        base_fields = ['date','filter','number','duration','gain','iso','binning',
                       'sensorCooling','fNumber','darks','flats','flatDarks','bias',
                       'bortle','meanSqm','meanFwhm','temperature']
        drop_iso = any(self._has_gain(r.get('gain', '')) for r in (self.rows or []))
        fieldnames = [f for f in base_fields if f != 'iso'] if drop_iso else base_fields
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader(); writer.writerows(self.rows or [])
        return buf.getvalue()

    def _refresh_csv_text(self):
        self.csv_view.setPlainText(self._rows_to_csv_str())

    def _copy_csv_to_clipboard(self):
        txt = self._rows_to_csv_str()
        if not txt.strip():
            QMessageBox.information(self, "Nothing to copy", "There is no CSV content yet.")
            return
        QGuiApplication.clipboard().setText(txt)
        QMessageBox.information(self, "Copied", "CSV copied to clipboard.")

    # ---------- filter map ----------
    def _load_filter_map(self) -> Dict[str, str]:
        defaults = {"Ha":"4408", "OIII":"4413", "SII":"4418", "L":"4450", "R":"4455", "G":"4445", "B":"4440"}
        self.settings.beginGroup("astrobin_exporter")
        raw = self.settings.value("filter_map", "")
        self.settings.endGroup()
        if not raw:
            blob = ";".join(f"{k}={v}" for k, v in defaults.items())
            self.settings.beginGroup("astrobin_exporter")
            self.settings.setValue("filter_map", blob)
            self.settings.endGroup()
            return defaults.copy()
        mapping: Dict[str, str] = {}
        for chunk in str(raw).split(";"):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and v.isdigit():
                    mapping[k] = v
        return mapping

    def _filters_summary_text(self) -> str:
        if not self._filter_map: return "No mappings set"
        pairs = sorted(self._filter_map.items(), key=lambda kv: kv[0].lower())
        return ", ".join(f"{k}→{v}" for k, v in pairs)

    def _edit_filters(self):
        names_in_data = sorted({rec.get("FILTER","Unknown") for rec in self.records if rec.get("FILTER")}, key=str.lower)
        dlg = FilterIdDialog(
            self,
            names_in_data,
            self.settings,
            current_map=self._filter_map,
            offline_csv_default=self._offline_csv_default  # <— pass override
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.save_to_settings()
            self._filter_map = self._load_filter_map()
            self.filter_summary.setText(self._filters_summary_text())
            self._recompute()

# ---- wrapper dialog ---------------------------------------------------------

class AstrobinExporterDialog(QDialog):
    def __init__(self, parent=None, offline_filters_csv: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("AstroBin Exporter")
        self.resize(980, 640)
        v = QVBoxLayout(self)
        self.tab = AstrobinExportTab(self, offline_filters_csv=offline_filters_csv)
        v.addWidget(self.tab, 1)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        v.addWidget(btns)
