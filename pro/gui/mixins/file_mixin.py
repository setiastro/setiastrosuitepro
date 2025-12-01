# pro/gui/mixins/file_mixin.py
"""
File operations mixin for AstroSuiteProMainWindow.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

if TYPE_CHECKING:
    pass


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
# Extracted FILE methods

    def open_files(self):
        # One-stop "All Supported" plus focused groups the user can switch to
        filters = (
            "All Supported (*.png *.jpg *.jpeg *.tif *.tiff "
            "*.fits *.fit *.fits.gz *.fit.gz *.fz *.xisf "
            "*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef);;"
            "Astro (FITS/XISF) (*.xisf *.fits *.fit *.fits.gz *.fit.gz *.fz);;"
            "RAW Images (*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef);;"
            "Common Images (*.png *.jpg *.jpeg *.tif *.tiff);;"
            "All Files (*)"
        )

        # read last dir; validate it still exists
        last_dir = self.settings.value("paths/last_open_dir", "", type=str) or ""
        if last_dir and not os.path.isdir(last_dir):
            last_dir = ""

        paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", last_dir, filters)
        if not paths:
            return

        # store the directory of the first picked file
        try:
            self.settings.setValue("paths/last_open_dir", os.path.dirname(paths[0]))
        except Exception:
            pass

        # open each path (doc_manager should emit documentAdded; no manual spawn)
        for p in paths:
            try:
                doc = self.docman.open_path(p)   # this emits documentAdded
                self._log(f"Opened: {p}")
                self._add_recent_image(p)        # âœ... track in MRU
            except Exception as e:
                QMessageBox.warning(self, "Open failed", f"{p}\n\n{e}")

    def save_active(self):
        from pro.main_helpers import (
            best_doc_name as _best_doc_name,
            normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
        )
        from pro.file_utils import _sanitize_filename
        
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
        path, selected_filter = QFileDialog.getSaveFileName(self, "Save As", suggested_path, filters)
        if not path:
            return

        before = path
        path, ext_norm = _normalize_save_path_chosen_filter(path, selected_filter)

        # If we changed the path (e.g., sanitized), inform once
        if before != path:
            self._log(f"Adjusted filename for safety:\n  {before}\n-> {path}")

        # --- Bit depth selection ----------------------------------------
        from pro.save_options import SaveOptionsDialog
        current_bd = doc.metadata.get("bit_depth")
        dlg = SaveOptionsDialog(self, ext_norm, current_bd)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        chosen_bd = dlg.selected_bit_depth()

        # --- Save & remember folder ----------------------------------------
        try:
            self.docman.save_document(doc, path, bit_depth_override=chosen_bd)
            self._log(f"Saved: {path} ({chosen_bd})")
            self.settings.setValue("paths/last_save_dir", os.path.dirname(path))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

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
                "File not found",
                f"The file does not exist:\n{path}\n\n"
                "Remove it from the recent images list?",
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
            QMessageBox.warning(self, "Open failed", f"{path}\n\n{e}")

    def _open_recent_project(self, path: str):
        if not path:
            return
        if not os.path.exists(path):
            if QMessageBox.question(
                self,
                "Project not found",
                f"The project file does not exist:\n{path}\n\n"
                "Remove it from the recent projects list?",
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
            self, "Save Project", "", "SetiAstro Project (*.sas)"
        )
        if not path:
            return
        if not path.lower().endswith(".sas"):
            path += ".sas"

        docs = self._collect_open_documents()
        if not docs:
            QMessageBox.warning(self, "Save Project", "No documents to save.")
            return

        try:
            compress = self._ask_project_compress()  # your existing yes/no dialog

            # Busy dialog (indeterminate)
            dlg = QProgressDialog("Saving project...", "", 0, 0, self)
            dlg.setWindowTitle("Saving")
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
            from pro.widgets.common_utilities import ProjectSaveWorker as _ProjectSaveWorker
            
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
                    QMessageBox.critical(self, "Save Project", f"Failed to save:\n{msg}"),
                )
            )
            self._proj_save_worker.finished.connect(
                lambda: setattr(self, "_proj_save_worker", None)
            )
            self._proj_save_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Save Project", f"Failed to save:\n{e}")

    def _load_project(self):
        # warn / clear current desktop
        if not self._prepare_for_project_load("Load Project"):
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "SetiAstro Project (*.sas)"
        )
        if not path:
            return

        self._do_load_project_path(path)

    def _new_project(self):
        if not self._confirm_discard(title="New Project",
                                    msg="Start a new project? This closes all views and clears desktop shortcuts."):
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
                from pro.shortcuts import SET_KEY_V1, SET_KEY_V2
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

