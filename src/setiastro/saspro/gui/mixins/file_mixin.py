# pro/gui/mixins/file_mixin.py
"""
File operations mixin for AstroSuiteProMainWindow.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QProgressDialog, QApplication,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox
)

if TYPE_CHECKING:
    pass

from typing import Sequence



class FitsBundleExportDialog(QDialog):
    """
    Lets the user choose which open image/table views to include in a bundled FITS export.
    Each row is checkable. Checked items will be written as HDUs.
    """

    def __init__(self, parent, entries: Sequence[dict], active_doc=None):
        super().__init__(parent)
        self.setWindowTitle("Export Bundled FITS")
        self.resize(620, 460)

        self._entries = list(entries)
        self._active_doc = active_doc

        lay = QVBoxLayout(self)

        info = QLabel(
            "Choose which open views to include in the FITS bundle.\n"
            "Checked items will be written as HDUs.\n"
            "Image views become image HDUs. Table views become FITS table HDUs."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        self.list_docs = QListWidget(self)
        lay.addWidget(self.list_docs, 1)

        for ent in self._entries:
            doc = ent.get("doc")
            title = str(ent.get("title") or "Untitled")
            tag = str(ent.get("tag") or "View")

            label = f"[{tag}] {title}"
            if doc is active_doc:
                label += "  [Active]"

            item = QListWidgetItem(label, self.list_docs)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, ent)

        row = QHBoxLayout()

        self.btn_all = QPushButton("Select All", self)
        self.btn_none = QPushButton("Select None", self)
        self.btn_active = QPushButton("Active Only", self)

        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        self.btn_active.clicked.connect(self._select_active_only)

        row.addWidget(self.btn_all)
        row.addWidget(self.btn_none)
        row.addWidget(self.btn_active)
        row.addStretch(1)
        lay.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _select_all(self):
        for i in range(self.list_docs.count()):
            self.list_docs.item(i).setCheckState(Qt.CheckState.Checked)

    def _select_none(self):
        for i in range(self.list_docs.count()):
            self.list_docs.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _select_active_only(self):
        for i in range(self.list_docs.count()):
            item = self.list_docs.item(i)
            ent = item.data(Qt.ItemDataRole.UserRole) or {}
            doc = ent.get("doc")
            item.setCheckState(
                Qt.CheckState.Checked if doc is self._active_doc else Qt.CheckState.Unchecked
            )

    def _accept_if_valid(self):
        if not self.selected_entries():
            return
        self.accept()

    def selected_entries(self):
        out = []
        for i in range(self.list_docs.count()):
            item = self.list_docs.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ent = item.data(Qt.ItemDataRole.UserRole)
                if ent is not None:
                    out.append(dict(ent))
        return out

_PROC_SCAN_RE = re.compile(r"^(?P<prefix>.+)_proc(?P<n>\d+)(?P<tag>_[^.]*)?$", re.IGNORECASE)

def _split_stars_suffix(name_no_ext: str) -> Tuple[str, bool]:
    if name_no_ext.lower().endswith("_stars"):
        return name_no_ext[:-6], True
    return name_no_ext, False

def _strip_proc_tail(base_no_ext: str) -> str:
    """
    If base already ends in _procN or _procN_tag, strip that whole tail so we build from the true prefix.
    """
    m = _PROC_SCAN_RE.match(base_no_ext)
    if not m:
        return base_no_ext
    return m.group("prefix")

def _safe_ext_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1] or ""
    return ext.lower()

def _next_proc_index_for(prefix: str, workdir: str, stars: bool, tag: str, ext: str) -> int:
    """
    Find next N by scanning the directory for files that match:
      prefix_procN{tag}{_stars}{ext}
    """
    # normalize
    tag_part = tag or ""          # e.g. "_dbe" or ""
    stars_part = "_stars" if stars else ""

    patt = re.compile(
        r"^" + re.escape(prefix)
        + r"_proc(\d+)"
        + re.escape(tag_part)
        + re.escape(stars_part)
        + re.escape(ext)
        + r"$",
        re.IGNORECASE
    )

    max_n = 0
    try:
        for fn in os.listdir(workdir):
            m = patt.match(fn)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except Exception:
        pass

    return max_n + 1

class _FitsBundleDialog(QDialog):
    def __init__(self, parent, entries: list[dict], active_index: int = -1):
        super().__init__(parent)
        self.setWindowTitle("Export FITS Bundle")

        self._entries = list(entries)

        lay = QVBoxLayout(self)

        lbl = QLabel(
            "Choose which open image views to bundle into the FITS file.\n"
            "The first checked item becomes the Primary HDU."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self.listw = QListWidget(self)
        lay.addWidget(self.listw, 1)

        for i, ent in enumerate(self._entries):
            title = ent.get("title") or f"Image {i+1}"
            item = QListWidgetItem(title)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            # default: check all, but put active item first in selection focus
            item.setCheckState(Qt.CheckState.Checked)
            self.listw.addItem(item)

        if 0 <= active_index < self.listw.count():
            self.listw.setCurrentRow(active_index)

        row = QHBoxLayout()
        btn_all = QPushButton("Select All", self)
        btn_none = QPushButton("Select None", self)
        btn_up = QPushButton("Move Up", self)
        btn_down = QPushButton("Move Down", self)

        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_up.clicked.connect(self._move_up)
        btn_down.clicked.connect(self._move_down)

        row.addWidget(btn_all)
        row.addWidget(btn_none)
        row.addStretch(1)
        row.addWidget(btn_up)
        row.addWidget(btn_down)
        lay.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.resize(520, 420)

    def _select_all(self):
        for i in range(self.listw.count()):
            self.listw.item(i).setCheckState(Qt.CheckState.Checked)

    def _select_none(self):
        for i in range(self.listw.count()):
            self.listw.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _move_up(self):
        row = self.listw.currentRow()
        if row <= 0:
            return
        item = self.listw.takeItem(row)
        self.listw.insertItem(row - 1, item)
        self.listw.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.listw.currentRow()
        if row < 0 or row >= self.listw.count() - 1:
            return
        item = self.listw.takeItem(row)
        self.listw.insertItem(row + 1, item)
        self.listw.setCurrentRow(row + 1)

    def selected_entries(self) -> list[dict]:
        out = []
        for i in range(self.listw.count()):
            item = self.listw.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ent = dict(item.data(Qt.ItemDataRole.UserRole) or {})
                ent["title"] = item.text()
                out.append(ent)
        return out

@dataclass
class _CheckpointSpec:
    mode_all_open: bool = False            # default active only
    format_override: Optional[str] = None  # e.g. ".fits" or ".xisf" or None meaning same-as-source
    tag: str = ""                          # e.g. "_dbe" or "" (must start with "_" if non-empty)
    include_paired_stars: bool = True      # if base exists and paired stars exists, save both


class FileMixin:
    """
    Mixin for file operations.
    
    Provides methods for opening, saving, importing, and exporting files.
    """
    
    def _load_recent_lists(self):
        """Load recent files and projects from settings."""
        if not hasattr(self, "settings"):
            return
        
        self._recent_image_paths = self.settings.value("recent/images", [], type=list) or []
        self._recent_project_paths = self.settings.value("recent/projects", [], type=list) or []
    
    def _save_recent_lists(self):
        """Save recent files and projects to settings."""
        if not hasattr(self, "settings"):
            return
        
        self.settings.setValue("recent/images", self._recent_image_paths[:self._recent_max])
        self.settings.setValue("recent/projects", self._recent_project_paths[:self._recent_max])
    
    def _open_recent_image(self, path: str):
        """
        Open a recent image file.
        
        Args:
            path: Path to the image file
        """
        if hasattr(self, "open_files"):
            self.open_files([path])
    
    def _open_recent_project(self, path: str):
        """
        Open a recent project file.
        
        Args:
            path: Path to the project file
        """
        if hasattr(self, "load_project"):
            self.load_project(path)
    
    def _add_to_recent_images(self, path: str):
        """
        Add a path to recent images list.
        
        Args:
            path: Path to add
        """
        if path in self._recent_image_paths:
            self._recent_image_paths.remove(path)
        self._recent_image_paths.insert(0, path)
        self._recent_image_paths = self._recent_image_paths[:self._recent_max]
        self._save_recent_lists()
        if hasattr(self, "_rebuild_recent_menus"):
            self._rebuild_recent_menus()
    
    def _collect_open_bundle_entries(self) -> tuple[list[dict], object | None]:
        """
        Return (entries, active_doc) for currently open bundle-capable views.
        Supports:
        - image docs
        - table docs

        Each entry:
            {
                "title": str,
                "doc": object,
                "subwindow": QMdiSubWindow | None,
                "kind": "image" | "table",
                "tag": "Image" | "Table" | "Vector Table",
            }
        """
        entries: list[dict] = []
        active_doc = None

        mdi = getattr(self, "mdi", None)
        if mdi is None:
            return entries, active_doc

        active_sw = None
        try:
            active_sw = mdi.activeSubWindow()
        except Exception:
            active_sw = None

        for sw in mdi.subWindowList():
            try:
                w = sw.widget()
            except Exception:
                w = None

            doc = None
            try:
                doc = getattr(w, "document", None) or getattr(sw, "document", None)
            except Exception:
                doc = None

            if doc is None:
                continue

            meta = getattr(doc, "metadata", {}) or {}
            img = getattr(doc, "image", None)
            rows = getattr(doc, "rows", None)
            headers = getattr(doc, "headers", None)

            kind = None
            tag = None

            if img is not None:
                kind = "image"
                tag = "Image"
            elif rows is not None and headers is not None:
                kind = "table"
                if str(meta.get("doc_subtype", "")).lower() == "vector":
                    tag = "Vector Table"
                else:
                    tag = "Table"
            else:
                continue

            try:
                title = (sw.windowTitle() or "").strip()
            except Exception:
                title = ""

            if not title:
                try:
                    if hasattr(doc, "display_name") and callable(doc.display_name):
                        title = str(doc.display_name())
                    else:
                        title = str(meta.get("display_name", "Untitled"))
                except Exception:
                    title = "Untitled"

            ent = {
                "title": title,
                "doc": doc,
                "subwindow": sw,
                "kind": kind,
                "tag": tag,
            }
            entries.append(ent)

            if sw is active_sw:
                active_doc = doc

        return entries, active_doc

    def create_table(self):
        from setiastro.saspro.subwindow import CreateTableDialog
        import traceback

        dlg = CreateTableDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        specs = dlg.column_specs()
        headers = [c.name for c in specs]
        kinds = [c.kind for c in specs]
        nrows = dlg.row_count()

        rows = []
        for _ in range(nrows):
            row = []
            for kind in kinds:
                if kind == "bool":
                    row.append(False)
                else:
                    row.append("")
            rows.append(row)

        meta = {
            "doc_type": "table",
            "editable": True,
            "column_kinds": kinds,
            "display_name": dlg.table_name(),
            "original_format": "internal",
        }

        try:
            doc = self.docman.create_table_document(
                headers=headers,
                rows=rows,
                metadata=meta,
                title=dlg.table_name(),
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                self.tr("Create Table"),
                f"Failed to create table document:\n\n{type(e).__name__}: {e}",
            )
            return

        self._log(
            f"Created table: {dlg.table_name()} "
            f"(rows={len(rows)}, cols={len(headers)}, editable=True)"
        )

        # Only spawn manually if your app really needs it.
        # But DO NOT swallow errors.
        try:
            if hasattr(self, "_spawn_subwindow_for"):
                self._spawn_subwindow_for(doc)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                self.tr("Create Table"),
                f"Table document was created, but the table view failed to open:\n\n"
                f"{type(e).__name__}: {e}",
            )
            return
    
    def export_fits_bundle(self):
        from setiastro.saspro.main_helpers import (
            best_doc_name as _best_doc_name,
            normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
        )
        from setiastro.saspro.file_utils import _sanitize_filename
        from setiastro.saspro.save_options import ExportDialog

        entries, active_doc = self._collect_open_bundle_entries()
        if not entries:
            QMessageBox.information(
                self,
                self.tr("Export FITS Bundle"),
                self.tr("There are no open image or table views to export.")
            )
            return

        chooser = FitsBundleExportDialog(
            self,
            entries,
            active_doc=active_doc,
        )
        if chooser.exec() != chooser.DialogCode.Accepted:
            return

        selected_entries = chooser.selected_entries()
        if not selected_entries:
            QMessageBox.information(
                self,
                self.tr("Export FITS Bundle"),
                self.tr("No views were selected.")
            )
            return

        candidate_dir = self.settings.value("paths/last_save_dir", "", type=str) or ""
        if not candidate_dir or not os.path.isdir(candidate_dir):
            try:
                doc = active_doc or (selected_entries[0].get("doc") if selected_entries else None)
                if doc is not None:
                    orig_path = (getattr(doc, "metadata", {}) or {}).get("file_path", "") or ""
                    if "::" in orig_path:
                        orig_path = orig_path.split("::", 1)[0]
                    if orig_path:
                        pdir = os.path.dirname(orig_path)
                        if pdir and os.path.isdir(pdir):
                            candidate_dir = pdir
            except Exception:
                pass

        if not candidate_dir or not os.path.isdir(candidate_dir):
            from pathlib import Path
            candidate_dir = str(Path.home())

        try:
            base_doc = active_doc or selected_entries[0].get("doc")
            suggested = _best_doc_name(base_doc) if base_doc else "fits_bundle"
        except Exception:
            suggested = "fits_bundle"

        suggested = os.path.splitext(suggested)[0] + "_bundle"
        suggested = _sanitize_filename(suggested)
        suggested_path = os.path.join(candidate_dir, suggested)

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("Export FITS Bundle"),
            suggested_path,
            "FITS (*.fits *.fit)"
        )
        if not path:
            return

        before = path
        path, ext_norm = _normalize_save_path_chosen_filter(path, selected_filter)

        if before != path:
            self._log(f"Adjusted filename for safety:\n  {before}\n-> {path}")

        dlg = ExportDialog(
            self,
            ext_norm,
            "32-bit floating point",
            current_jpeg_quality=None,
            settings=self.settings,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        chosen_bd = dlg.selected_bit_depth()
        export_opts = dlg.export_options()

        try:
            self.docman.save_fits_bundle(
                selected_entries,
                path,
                bit_depth_override=chosen_bd,
                export_opts=export_opts,
            )
            self._log(
                f"Exported FITS bundle: {path} "
                f"({chosen_bd}, {len(selected_entries)} view(s))"
            )
            self.settings.setValue("paths/last_save_dir", os.path.dirname(path))
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Export FITS Bundle failed"),
                str(e),
            )

    def _add_to_recent_projects(self, path: str):
        """
        Add a path to recent projects list.
        
        Args:
            path: Path to add
        """
        if path in self._recent_project_paths:
            self._recent_project_paths.remove(path)
        self._recent_project_paths.insert(0, path)
        self._recent_project_paths = self._recent_project_paths[:self._recent_max]
        self._save_recent_lists()
        if hasattr(self, "_rebuild_recent_menus"):
            self._rebuild_recent_menus()

    def open_files(self):
        # One-stop "All Supported" plus focused groups the user can switch to
        filters = (
            "All Supported (*.png *.jpg *.jpeg *.tif *.tiff "
            "*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz *.xisf *.pdf "
            "*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef);;"
            "PDF (*.pdf);;"
            "Astro (FITS/XISF) (*.xisf *.fits *.fit *.fts *.fits.gz *.fit.gz *.fz);;"
            "RAW Images (*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef);;"
            "Common Images (*.png *.jpg *.jpeg *.tif *.tiff);;"
            "All Files (*)"
        )

        # read last dir; validate it still exists
        last_dir = self.settings.value("paths/last_open_dir", "", type=str) or ""
        if last_dir and not os.path.isdir(last_dir):
            last_dir = ""

        paths, _ = QFileDialog.getOpenFileNames(self, self.tr("Open Images"), last_dir, filters)
        if not paths:
            return

        # store the directory of the first picked file
        try:
            self.settings.setValue("paths/last_open_dir", os.path.dirname(paths[0]))
        except Exception:
            pass

        # ---- BEGIN batch open (stable placement) ----
        try:
            self._mdi_begin_open_batch(mode="cascade")
        except Exception:
            pass

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # open each path (doc_manager should emit documentAdded; no manual spawn)
            for p in paths:
                try:
                    if p.lower().endswith(".pdf"):
                        self._open_pdf(p)
                        continue
                    _ = self.docman.open_path(p)
                    self._log(f"Opened: {p}")
                    self._add_recent_image(p)

                    # Increment statistics
                    try:
                        count = self.settings.value("stats/opened_images_count", 0, type=int)
                        self.settings.setValue("stats/opened_images_count", count + 1)
                    except Exception:
                        pass

                    # Let Qt paint newly spawned subwindows as we go
                    QApplication.processEvents()

                except Exception as e:
                    QMessageBox.warning(self, self.tr("Open failed"), f"{p}\n\n{e}")
                    QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
            try:
                self._mdi_end_open_batch()
            except Exception:
                pass

    def _open_pdf(self, path: str):
        try:
            import pypdfium2 as pdfium
        except ImportError:
            QMessageBox.warning(
                self, "PDF Support",
                "pypdfium2 is required to open PDF files.\n\n"
                "Install it with:  pip install pypdfium2"
            )
            return

        try:
            pdf = pdfium.PdfDocument(path)
        except Exception as e:
            QMessageBox.critical(self, "Open PDF", f"Failed to open PDF:\n{e}")
            return

        n_pages = len(pdf)
        stem = os.path.splitext(os.path.basename(path))[0]

        # Ask DPI if multi-page so user knows what they're getting into
        dpi = 150
        if n_pages > 1:
            from PyQt6.QtWidgets import QInputDialog
            dpi, ok = QInputDialog.getInt(
                self, "Open PDF",
                f"PDF has {n_pages} pages. Render DPI:",
                150, 72, 600, 50
            )
            if not ok:
                return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for i, page in enumerate(pdf):
                try:
                    bitmap = page.render(scale=dpi / 72.0)
                    pil_img = bitmap.to_pil()
                    import numpy as np
                    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
                    # ensure HWC RGB
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    elif arr.shape[2] == 4:
                        arr = arr[:, :, :3]

                    title = f"{stem}_page{i + 1}" if n_pages > 1 else stem
                    meta = {
                        "display_name": title,
                        "file_path": f"{path}::page{i + 1}",
                        "original_format": "pdf",
                        "bit_depth": "8-bit",
                        "is_mono": False,
                        "pdf_page": i + 1,
                        "pdf_dpi": dpi,
                    }
                    self.docman.open_array(arr, metadata=meta, title=title)
                    QApplication.processEvents()
                except Exception as e:
                    print(f"[PDF] Page {i + 1} failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def save_active(self):
        from setiastro.saspro.main_helpers import (
            best_doc_name as _best_doc_name,
            normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
        )
        from setiastro.saspro.file_utils import _sanitize_filename
        
        doc = self._active_doc()
        if not doc:
            return

        filters = (
            "FITS (*.fits *.fit);;"
            "XISF (*.xisf);;"
            "TIFF (*.tif *.tiff);;"
            "PNG (*.png);;"
            "JPEG (*.jpg *.jpeg)"
        )

        # --- Determine initial directory nicely -----------------------------
        # 1) Try the document's original file path (strip any "::HDU" or "::XISF[...]" suffix)
        orig_path = (doc.metadata or {}).get("file_path", "") or ""
        if "::" in orig_path:
            # e.g. "/foo/bar/file.fits::HDU 2" or "...::XISF[3]"
            orig_path_fs = orig_path.split("::", 1)[0]
        else:
            orig_path_fs = orig_path

        candidate_dir = ""
        try:
            if orig_path_fs:
                pdir = os.path.dirname(orig_path_fs)
                if pdir and os.path.isdir(pdir):
                    candidate_dir = pdir
        except Exception:
            candidate_dir = ""

        # 2) Else, fall back to last save dir setting
        if not candidate_dir:
            candidate_dir = self.settings.value("paths/last_save_dir", "", type=str) or ""

        # 3) Else, home directory
        if not candidate_dir or not os.path.isdir(candidate_dir):
            from pathlib import Path
            candidate_dir = str(Path.home())

        # --- Suggest a sane filename ---------------------------------------
        suggested = _best_doc_name(doc)
        suggested = os.path.splitext(suggested)[0]               # remove any ext
        suggested_safe = _sanitize_filename(suggested)
        suggested_path = os.path.join(candidate_dir, suggested_safe)

        # --- Open dialog ----------------------------------------
        path, selected_filter = QFileDialog.getSaveFileName(self, self.tr("Save As"), suggested_path, filters)
        if not path:
            return

        before = path
        path, ext_norm = _normalize_save_path_chosen_filter(path, selected_filter)

        # If we changed the path (e.g., sanitized), inform once
        if before != path:
            self._log(f"Adjusted filename for safety:\n  {before}\n-> {path}")

        # --- Bit depth selection ----------------------------------------
        from setiastro.saspro.save_options import ExportDialog
        current_bd = doc.metadata.get("bit_depth")
        current_jq = (doc.metadata or {}).get("jpeg_quality", None)

        dlg = ExportDialog(
            self, ext_norm, current_bd,
            current_jpeg_quality=current_jq,
            settings=self.settings,   # so it remembers per-format defaults
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        chosen_bd = dlg.selected_bit_depth()
        chosen_jq = dlg.selected_jpeg_quality()
        export_opts = dlg.export_options()

        # --- Save & remember folder ----------------------------------------
        try:
            self.docman.save_document(
                doc, path,
                bit_depth_override=chosen_bd,
                jpeg_quality=chosen_jq,
                export_opts=export_opts,
            )
            self._log(f"Saved: {path} ({chosen_bd})")
            self.settings.setValue("paths/last_save_dir", os.path.dirname(path))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Save failed"), str(e))

    def _load_recent_lists(self):
        """Load MRU lists from QSettings."""
        def _as_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return [str(v) for v in val if v]
            if isinstance(val, str):
                if not val:
                    return []
                # allow ";;" separated fallback if ever needed
                return [s for s in val.split(";;") if s]
            return []

        self._recent_image_paths = _as_list(
            self.settings.value("recent/image_paths", [])
        )
        self._recent_project_paths = _as_list(
            self.settings.value("recent/project_paths", [])
        )

        # Enforce max + uniqueness (most recent first)
        def _dedupe_keep_order(seq):
            seen = set()
            out = []
            for p in seq:
                if p in seen:
                    continue
                seen.add(p)
                out.append(p)
            return out[: self._recent_max]

        self._recent_image_paths = _dedupe_keep_order(self._recent_image_paths)
        self._recent_project_paths = _dedupe_keep_order(self._recent_project_paths)

    def _save_recent_lists(self):
        try:
            self.settings.setValue("recent/image_paths", self._recent_image_paths)
            self.settings.setValue("recent/project_paths", self._recent_project_paths)
        except Exception:
            pass

    def _open_recent_image(self, path: str):
        if not path:
            return
        if not os.path.exists(path):
            if QMessageBox.question(
                self,
                self.tr("File not found"),
                self.tr("The file does not exist:\n{path}\n\nRemove it from the recent images list?").replace("{path}", path),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes:
                self._recent_image_paths = [p for p in self._recent_image_paths if p != path]
                self._save_recent_lists()
                self._rebuild_recent_menus()
            return

        try:
            self.docman.open_path(path)
            self._log(f"Opened (recent): {path}")
            # bump to front
            self._add_recent_image(path)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Open failed"), f"{path}\n\n{e}")

    def _open_recent_project(self, path: str):
        if not path:
            return
        if not os.path.exists(path):
            if QMessageBox.question(
                self,
                self.tr("Project not found"),
                self.tr("The project file does not exist:\n{path}\n\nRemove it from the recent projects list?").replace("{path}", path),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes:
                self._recent_project_paths = [p for p in self._recent_project_paths if p != path]
                self._save_recent_lists()
                self._rebuild_recent_menus()
            return

        if not self._prepare_for_project_load("Load Project"):
            return

        self._do_load_project_path(path)

    def _add_recent_image(self, path: str):
        p = os.path.abspath(path)
        self._recent_image_paths = [p] + [
            x for x in self._recent_image_paths if x != p
        ]
        self._recent_image_paths = self._recent_image_paths[: self._recent_max]
        self._save_recent_lists()
        self._rebuild_recent_menus()

    def _add_recent_project(self, path: str):
        p = os.path.abspath(path)
        self._recent_project_paths = [p] + [
            x for x in self._recent_project_paths if x != p
        ]
        self._recent_project_paths = self._recent_project_paths[: self._recent_max]
        self._save_recent_lists()
        self._rebuild_recent_menus()

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save Project"), "", "SetiAstro Project (*.sas)"
        )
        if not path:
            return
        if not path.lower().endswith(".sas"):
            path += ".sas"

        docs = self._collect_open_documents()
        if not docs:
            QMessageBox.warning(self, self.tr("Save Project"), self.tr("No documents to save."))
            return

        try:
            compress = self._ask_project_compress()  # your existing yes/no dialog

            # Busy dialog (indeterminate)
            dlg = QProgressDialog(self.tr("Saving project..."), "", 0, 0, self)
            dlg.setWindowTitle(self.tr("Saving"))
            # PyQt6 (with PyQt5 fallback if you ever run it there)
            try:
                dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            except AttributeError:
                dlg.setWindowModality(Qt.ApplicationModal)  # PyQt5

            # Hide the cancel button (API differs across versions)
            try:
                dlg.setCancelButton(None)
            except TypeError:
                dlg.setCancelButtonText("")

            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.show()

            # Threaded save
            from setiastro.saspro.widgets.common_utilities import ProjectSaveWorker as _ProjectSaveWorker
            
            self._proj_save_worker = _ProjectSaveWorker(
                path,
                docs,
                getattr(self, "shortcuts", None),
                getattr(self, "mdi", None),
                compress,
                parent=self,
            )

            def _on_proj_save_ok():
                dlg.close()
                self._log("Project saved.")
                self._add_recent_project(path)

            self._proj_save_worker.ok.connect(_on_proj_save_ok)
            self._proj_save_worker.error.connect(
                lambda msg: (
                    dlg.close(),
                    QMessageBox.critical(self, self.tr("Save Project"), self.tr("Failed to save:\n{msg}").replace("{msg}", msg)),
                )
            )
            self._proj_save_worker.finished.connect(
                lambda: setattr(self, "_proj_save_worker", None)
            )
            self._proj_save_worker.start()

        except Exception as e:
            QMessageBox.critical(self, self.tr("Save Project"), self.tr("Failed to save:\n{e}").replace("{e}", str(e)))

    def _load_project(self):
        # warn / clear current desktop
        if not self._prepare_for_project_load("Load Project"):
            return

        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Load Project"), "", "SetiAstro Project (*.sas)"
        )
        if not path:
            return

        self._do_load_project_path(path)

    def _new_project(self):
        if not self._confirm_discard(title=self.tr("New Project"),
                                    msg=self.tr("Start a new project? This closes all views and clears desktop shortcuts.")):
            return

        # Close views + docs + shelf
        self._close_all_subwindows()
        self._clear_all_documents()
        self._clear_minimized_shelf()

        # Clear desktop shortcuts (widgets + persisted positions)
        try:
            if getattr(self, "shortcuts", None):
                self.shortcuts.clear()
            else:
                # Fallback: wipe persisted layout so nothing reloads later
                from PyQt6.QtCore import QSettings
                from setiastro.saspro.shortcuts import SET_KEY_V1, SET_KEY_V2
                s = QSettings()
                s.setValue(SET_KEY_V2, "[]")
                s.remove(SET_KEY_V1)
                s.sync()
        except Exception:
            pass

        # (Optional) keep canvas ready for fresh adds
        try:
            if getattr(self, "shortcuts", None):
                self.shortcuts.canvas.raise_()
                self.shortcuts.canvas.show()
                self.shortcuts.canvas.setFocus()
        except Exception:
            pass

        self._log("New project workspace ready.")

    def _collect_open_documents(self):
        # Prefer DocManager if present
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm is not None and getattr(dm, "_docs", None) is not None:
            return list(dm._docs)

        # Fallback: harvest from open subwindows
        docs = []
        for sw in self.mdi.subWindowList():
            try:
                view = sw.widget()
                doc = getattr(view, "document", None)
                if doc is not None:
                    docs.append(doc)
            except Exception:
                pass
        return docs

    def checkpoint_save(self):
        """
        Public entrypoint bound to QAction. Default behavior = active-only, same format, no tag.
        If you later add a small dialog, it can call _checkpoint_save_with_spec(spec).
        """
        spec = _CheckpointSpec(
            mode_all_open=False,
            format_override=None,
            tag="",
            include_paired_stars=True
        )
        self._checkpoint_save_with_spec(spec)

    def _checkpoint_save_with_spec(self, spec: _CheckpointSpec):
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm is None:
            QMessageBox.warning(self, self.tr("Checkpoint Save"), self.tr("Document manager not available."))
            return

        # Collect docs to save
        docs = []
        if spec.mode_all_open:
            docs = self._collect_open_documents()
        else:
            d = self._active_doc()
            if d:
                docs = [d]

        if not docs:
            return

        # Build a quick lookup for paired stars docs by "base title"
        # We'll use best_doc_name(doc) for stable naming.
        from setiastro.saspro.main_helpers import best_doc_name as _best_doc_name

        name_to_doc = {}
        for d in docs if spec.mode_all_open else self._collect_open_documents():
            try:
                nm = os.path.splitext(_best_doc_name(d))[0]
                name_to_doc[nm] = d
            except Exception:
                pass

        saved_any = False
        for doc in docs:
            try:
                ok = self._checkpoint_save_one(doc, spec, name_to_doc=name_to_doc)
                saved_any = saved_any or ok
            except Exception as e:
                QMessageBox.warning(self, self.tr("Checkpoint Save"), str(e))

        if saved_any:
            self._log("Checkpoint Save complete.")

    def _checkpoint_save_one(self, doc, spec: _CheckpointSpec, name_to_doc: Optional[dict] = None) -> bool:
        from setiastro.saspro.main_helpers import best_doc_name as _best_doc_name

        # Determine source file path (real filesystem path, not "::HDU" etc)
        meta = getattr(doc, "metadata", {}) or {}
        orig_path = meta.get("file_path", "") or ""
        if "::" in orig_path:
            orig_path_fs = orig_path.split("::", 1)[0]
        else:
            orig_path_fs = orig_path

        if not orig_path_fs:
            # Official tool could prompt for folder; for now we follow your style and warn.
            self._log(f"Checkpoint skipped (unsaved doc): {_best_doc_name(doc)}")
            return False

        workdir = os.path.dirname(orig_path_fs)
        if not workdir or not os.path.isdir(workdir):
            self._log(f"Checkpoint skipped (bad folder): {orig_path_fs}")
            return False

        base = os.path.splitext(os.path.basename(orig_path_fs))[0]  # file base, not window title
        ext_src = _safe_ext_from_path(orig_path_fs)                 # includes dot, e.g. ".fits"
        ext = (spec.format_override or ext_src or ".fits").lower()

        # normalize tag
        tag = (spec.tag or "").strip()
        if tag and not tag.startswith("_"):
            tag = "_" + tag

        # stars suffix (based on current doc name, not path base) — but your pipeline expects _stars in title commonly
        # We'll infer from best_doc_name(doc) to match your window naming.
        doc_title_no_ext = os.path.splitext(_best_doc_name(doc))[0]
        doc_base_no_stars, is_stars = _split_stars_suffix(doc_title_no_ext)

        # Use the *filesystem base* as prefix source, but strip any proc tail
        # This keeps checkpoint files grouped with the original file on disk.
        prefix = _strip_proc_tail(base)

        n_next = _next_proc_index_for(prefix, workdir, is_stars, tag, ext)
        out_base = f"{prefix}_proc{n_next}{tag}" + ("_stars" if is_stars else "")
        out_path = os.path.join(workdir, out_base + ext)

        # Save using DocManager so headers/bit depth handling remains consistent
        # Use doc's current bit depth unless you want "same as save_active dialog" later.
        chosen_bd = meta.get("bit_depth", None)

        self.docman.save_document(
            doc, out_path,
            bit_depth_override=chosen_bd,
            jpeg_quality=(meta.get("jpeg_quality", None) if ext in (".jpg", ".jpeg") else None),
            export_opts=None,
        )
        self._log(f"Checkpoint saved: {out_path}")

        # Optional: also save paired stars doc if requested
        if spec.include_paired_stars and (not is_stars) and name_to_doc:
            paired_name = doc_title_no_ext + "_stars"
            paired_doc = name_to_doc.get(paired_name)
            if paired_doc is not None:
                # Save paired as well with same N
                out_base2 = f"{prefix}_proc{n_next}{tag}_stars"
                out_path2 = os.path.join(workdir, out_base2 + ext)
                meta2 = getattr(paired_doc, "metadata", {}) or {}
                bd2 = meta2.get("bit_depth", None)
                self.docman.save_document(
                    paired_doc, out_path2,
                    bit_depth_override=bd2,
                    jpeg_quality=(meta2.get("jpeg_quality", None) if ext in (".jpg", ".jpeg") else None),
                    export_opts=None,
                )
                self._log(f"Checkpoint saved (paired stars): {out_path2}")

        return True
    
    def save_active_as_format(self, fmt: str):
        """
        Save active document directly in a specific format,
        pre-populating the file dialog with the right extension.
        fmt: 'fits', 'xisf', 'tiff', 'png', 'jpeg'
        """
        from setiastro.saspro.main_helpers import (
            best_doc_name as _best_doc_name,
            normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
        )
        from setiastro.saspro.file_utils import _sanitize_filename

        doc = self._active_doc()
        if not doc:
            return

        fmt_map = {
            "fits":  ("FITS (*.fits *.fit)",  ".fits"),
            "xisf":  ("XISF (*.xisf)",        ".xisf"),
            "tiff":  ("TIFF (*.tif *.tiff)",  ".tiff"),
            "png":   ("PNG (*.png)",           ".png"),
            "jpeg":  ("JPEG (*.jpg *.jpeg)",   ".jpg"),
        }
        if fmt not in fmt_map:
            self.save_active()
            return

        selected_filter, ext = fmt_map[fmt]

        # Build all filters with the chosen one first
        all_filters = [
            "FITS (*.fits *.fit)",
            "XISF (*.xisf)",
            "TIFF (*.tif *.tiff)",
            "PNG (*.png)",
            "JPEG (*.jpg *.jpeg)",
        ]
        # Move chosen filter to top
        ordered = [selected_filter] + [f for f in all_filters if f != selected_filter]
        filters = ";;".join(ordered)

        # Directory
        meta = getattr(doc, "metadata", {}) or {}
        orig_path = meta.get("file_path", "") or ""
        if "::" in orig_path:
            orig_path = orig_path.split("::", 1)[0]

        candidate_dir = ""
        try:
            if orig_path:
                pdir = os.path.dirname(orig_path)
                if pdir and os.path.isdir(pdir):
                    candidate_dir = pdir
        except Exception:
            pass

        if not candidate_dir:
            candidate_dir = self.settings.value("paths/last_save_dir", "", type=str) or ""
        if not candidate_dir or not os.path.isdir(candidate_dir):
            from pathlib import Path
            candidate_dir = str(Path.home())

        # Suggested filename with pre-applied extension
        suggested = _best_doc_name(doc)
        suggested = os.path.splitext(suggested)[0]
        suggested = _sanitize_filename(suggested) + ext
        suggested_path = os.path.join(candidate_dir, suggested)

        path, chosen_filter = QFileDialog.getSaveFileName(
            self, self.tr("Save As"), suggested_path, filters
        )
        if not path:
            return

        path, ext_norm = _normalize_save_path_chosen_filter(path, chosen_filter)

        from setiastro.saspro.save_options import ExportDialog
        current_bd = meta.get("bit_depth")
        current_jq = meta.get("jpeg_quality", None)

        dlg = ExportDialog(
            self, ext_norm, current_bd,
            current_jpeg_quality=current_jq,
            settings=self.settings,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        try:
            self.docman.save_document(
                doc, path,
                bit_depth_override=dlg.selected_bit_depth(),
                jpeg_quality=dlg.selected_jpeg_quality(),
                export_opts=dlg.export_options(),
            )
            self._log(f"Saved: {path} ({dlg.selected_bit_depth()})")
            self.settings.setValue("paths/last_save_dir", os.path.dirname(path))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Save failed"), str(e))    