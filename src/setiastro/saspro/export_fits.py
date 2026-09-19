from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QProgressDialog, QPushButton, QVBoxLayout, QWidget,
)

from setiastro.saspro.file_utils import sanitize_filename

_HEADER_KEYS = ("fits_header", "original_header", "header")
_WCS_STRUCTURAL = {
    "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4",
    "BSCALE", "BZERO", "EXTEND", "END", "PCOUNT", "GCOUNT",
}
_SAVE_MUTATED_KEYS = (
    "file_path", "original_format", "bit_depth", "original_header", "fits_header",
)

COMMAND_ID = "export_fits"
_SETTINGS_KEY = "paths/last_export_fits_dir"

try:
    from setiastro.saspro.resources import disk_path
except Exception:
    disk_path = None


@dataclass
class ExportFitsResult:
    ok: bool
    path: str | None = None
    skipped: bool = False
    reason: str = ""


def _source_fs_path(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    orig = str(meta.get("file_path", "") or "")
    if "::" in orig:
        orig = orig.split("::", 1)[0]
    return orig


def export_fits_stem(doc) -> str:
    orig = _source_fs_path(doc)
    if orig:
        stem = os.path.splitext(os.path.basename(orig))[0]
        if stem:
            return sanitize_filename(stem) or "untitled"
    try:
        from setiastro.saspro.main_helpers import best_doc_name
        name = best_doc_name(doc)
    except Exception:
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("display_name") or meta.get("name") or "untitled"
    stem = os.path.splitext(str(name or "untitled"))[0]
    return sanitize_filename(stem) or "untitled"


def export_fits_path(doc, out_dir: str) -> str:
    return os.path.join(os.path.abspath(str(out_dir)), f"{export_fits_stem(doc)}.fits")


def normalize_export_fits_preset(preset: dict | None) -> dict:
    p = dict(preset or {})
    out_dir = str(p.get("out_dir") or p.get("folder") or p.get("output_dir") or "").strip()
    return {
        "out_dir": out_dir,
        "overwrite": bool(p.get("overwrite", True)),
    }


def _as_fits_header(value):
    if value is None:
        return None
    try:
        from astropy.io.fits import Header
    except Exception:
        return None
    if isinstance(value, Header):
        return value
    if isinstance(value, dict):
        hdr = Header()
        for key, val in value.items():
            if isinstance(key, str) and key.startswith("XISF:"):
                continue
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            try:
                hdr[str(key)] = val
            except Exception:
                continue
        return hdr if len(hdr) else None
    return None


def _header_from_snapshot(snap):
    if not isinstance(snap, dict):
        return None
    try:
        from setiastro.saspro.project_io import _deserialize_header_any
    except Exception:
        return None
    try:
        return _as_fits_header(_deserialize_header_any(snap))
    except Exception:
        return None


def _wcs_header_from_doc(doc, meta: dict):
    sources = (
        meta.get("wcs_header"),
        meta.get("wcs"),
        getattr(doc, "wcs", None),
    )
    try:
        from astropy.wcs import WCS
    except Exception:
        WCS = None
    for src in sources:
        if src is None:
            continue
        if WCS is not None:
            try:
                if isinstance(src, WCS):
                    return src.to_header(relax=True)
            except Exception:
                pass
        hdr = _as_fits_header(src)
        if hdr is not None:
            return hdr
    return None


def _merge_wcs_cards(hdr, wcs_hdr):
    if hdr is None:
        return wcs_hdr
    if wcs_hdr is None:
        return hdr
    out = hdr.copy() if hasattr(hdr, "copy") else hdr
    for key, val in wcs_hdr.items():
        if key in _WCS_STRUCTURAL:
            continue
        try:
            out[key] = val
        except Exception:
            continue
    return out


def resolve_header_for_export(doc):
    """Best FITS header for export: live cards, then snapshot, then WCS."""
    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
    hdr = None
    for key in _HEADER_KEYS:
        hdr = _as_fits_header(meta.get(key))
        if hdr is not None:
            break
    if hdr is None:
        hdr = _as_fits_header(getattr(doc, "original_header", None))
    if hdr is None:
        hdr = _header_from_snapshot(meta.get("__header_snapshot__"))
    wcs_hdr = _wcs_header_from_doc(doc, meta)
    merged = _merge_wcs_cards(hdr, wcs_hdr)
    if merged is None:
        return None
    return merged.copy() if hasattr(merged, "copy") else merged


def export_document_as_fits(docman, doc, preset: dict | None = None) -> ExportFitsResult:
    """
    Write `doc` as a FITS file into the preset folder.

    Does not retarget the in-memory document (file_path / dirty stay unchanged).
    """
    spec = normalize_export_fits_preset(preset)
    if doc is None or getattr(doc, "image", None) is None:
        return ExportFitsResult(ok=False, skipped=True, reason="No image to export")
    out_dir = spec["out_dir"]
    if not out_dir:
        return ExportFitsResult(ok=False, skipped=True, reason="No output folder in preset")

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        return ExportFitsResult(ok=False, skipped=True, reason=f"Cannot create folder: {e}")

    path = export_fits_path(doc, out_dir)
    if (not spec["overwrite"]) and os.path.exists(path):
        return ExportFitsResult(ok=False, skipped=True, path=path, reason="File exists")

    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
        if hasattr(doc, "metadata"):
            doc.metadata = meta
    dirty = bool(getattr(doc, "dirty", False))
    restore = [(key, key in meta, meta.get(key)) for key in _SAVE_MUTATED_KEYS]
    bit_depth = meta.get("bit_depth")
    hdr = resolve_header_for_export(doc)
    if hdr is not None:
        meta["original_header"] = hdr
        meta["fits_header"] = hdr
    try:
        docman.save_document(doc, path, bit_depth_override=bit_depth)
    except Exception as e:
        return ExportFitsResult(ok=False, skipped=False, path=path, reason=str(e))
    finally:
        for key, existed, val in restore:
            if existed:
                meta[key] = val
            else:
                meta.pop(key, None)
        if hasattr(doc, "metadata"):
            doc.metadata = meta
        if hasattr(doc, "dirty"):
            doc.dirty = dirty
    return ExportFitsResult(ok=True, path=path)


def _last_export_dir(mw, preset: dict | None = None) -> str:
    spec = normalize_export_fits_preset(preset)
    if spec["out_dir"] and os.path.isdir(spec["out_dir"]):
        return spec["out_dir"]
    settings = getattr(mw, "settings", None)
    if settings is not None:
        for key in (_SETTINGS_KEY, "paths/last_save_dir"):
            try:
                v = settings.value(key, "", type=str) or ""
            except Exception:
                v = ""
            if v and os.path.isdir(v):
                return v
    return spec["out_dir"]


def _remember_export_dir(mw, out_dir: str) -> None:
    settings = getattr(mw, "settings", None)
    if settings is None or not out_dir:
        return
    try:
        settings.setValue(_SETTINGS_KEY, out_dir)
    except Exception:
        pass


def _active_doc_from_main(mw):
    getter = getattr(mw, "_active_doc", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
    if dm is not None and hasattr(dm, "get_active_document"):
        try:
            return dm.get_active_document()
        except Exception:
            pass
    return None


class ExportFitsPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export FITS — Preset")
        init = normalize_export_fits_preset(initial)
        self.edit_dir = QLineEdit(init["out_dir"])
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.edit_dir, 1)
        row.addWidget(btn_browse)
        folder = QWidget()
        folder.setLayout(row)
        self.chk_overwrite = QCheckBox("Overwrite existing files")
        self.chk_overwrite.setChecked(init["overwrite"])
        form = QFormLayout(self)
        form.addRow("Output folder:", folder)
        form.addRow("", self.chk_overwrite)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _browse(self):
        start = self.edit_dir.text().strip() or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if chosen:
            self.edit_dir.setText(chosen)

    def result_dict(self) -> dict:
        return {
            "out_dir": self.edit_dir.text().strip(),
            "overwrite": bool(self.chk_overwrite.isChecked()),
        }


class ExportFitsDialog(QDialog):
    def __init__(self, main_window, preset: dict | None = None):
        super().__init__(main_window)
        self._mw = main_window
        self.setWindowTitle("Export FITS")
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(520, 260)
        init = normalize_export_fits_preset(preset)
        init["out_dir"] = _last_export_dir(main_window, init)

        self.edit_dir = QLineEdit(init["out_dir"])
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.edit_dir, 1)
        row.addWidget(btn_browse)

        self.chk_overwrite = QCheckBox("Overwrite existing files")
        self.chk_overwrite.setChecked(init["overwrite"])

        hint = QLabel(
            "Saves the current view as .fits into this folder.\n"
            "Use Apply to View Bundle, or drop a shortcut on a bundle, to export many views."
        )
        hint.setWordWrap(True)

        v = QVBoxLayout(self)
        v.addWidget(hint)
        form = QFormLayout()
        folder = QWidget()
        folder.setLayout(row)
        form.addRow("Output folder:", folder)
        form.addRow("", self.chk_overwrite)
        v.addLayout(form)

        btns = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply)
        self.btn_apply_vbundle = QPushButton("Apply to View Bundle…")
        self.btn_apply_vbundle.clicked.connect(self._apply_to_view_bundle)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btns.addWidget(btn_apply)
        btns.addWidget(self.btn_apply_vbundle)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        v.addLayout(btns)

        from setiastro.saspro.shortcuts import PresetDragHandle
        grip = QHBoxLayout()
        icon = QIcon(disk_path) if disk_path else QIcon()
        self.preset_drag_handle = PresetDragHandle(
            COMMAND_ID,
            self.current_preset,
            icon=icon,
            tooltip=(
                "Drag to the canvas to create an Export FITS shortcut\n"
                "with this folder. Drop on an image, or add to a Function Bundle,\n"
                "to export headlessly."
            ),
            parent=self,
        )
        grip.addWidget(self.preset_drag_handle)
        grip.addStretch(1)
        v.addLayout(grip)
        self.resize(560, 280)

    def current_preset(self) -> dict:
        return {
            "out_dir": self.edit_dir.text().strip(),
            "overwrite": bool(self.chk_overwrite.isChecked()),
        }

    def seed_from_preset(self, preset: dict | None):
        spec = normalize_export_fits_preset(preset)
        if spec["out_dir"]:
            self.edit_dir.setText(spec["out_dir"])
        self.chk_overwrite.setChecked(spec["overwrite"])

    def _browse(self):
        start = self.edit_dir.text().strip() or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if chosen:
            self.edit_dir.setText(chosen)

    def _require_folder(self) -> dict | None:
        spec = self.current_preset()
        if not spec["out_dir"]:
            self._browse()
            spec = self.current_preset()
        if not spec["out_dir"]:
            QMessageBox.warning(self, "Export FITS", "Choose an output folder first.")
            return None
        return spec

    def _docman(self):
        return getattr(self._mw, "docman", None) or getattr(self._mw, "doc_manager", None)

    def _export_docs(self, docs, *, close_when_done: bool = False):
        spec = self.current_preset()
        dm = self._docman()
        if dm is None:
            return 0, ["Document manager not available."]
        log = getattr(self._mw, "_log", None)
        applied = 0
        errors = []
        total = max(1, len(docs))
        pd = None
        if len(docs) > 1:
            pd = QProgressDialog("Exporting FITS…", None, 0, total, self)
            pd.setWindowTitle("Export FITS")
            pd.setWindowFlag(Qt.WindowType.Window, True)
            pd.setMinimumWidth(420)
            pd.setMinimumHeight(110)
            pd.setMinimumDuration(0)
            pd.setWindowModality(Qt.WindowModality.ApplicationModal)
            pd.setAutoClose(False)
            pd.setAutoReset(False)
            pd.setCancelButton(None)
            pd.resize(420, 120)
            pd.show()
            QApplication.processEvents()
        try:
            for i, doc in enumerate(docs, start=1):
                if pd is not None:
                    pd.setValue(i - 1)
                    pd.setLabelText(f"Exporting FITS ({i}/{total})…")
                    QApplication.processEvents()
                result = export_document_as_fits(dm, doc, spec)
                if result.ok:
                    applied += 1
                    if callable(log):
                        log(f"Exported FITS: {result.path}")
                else:
                    reason = result.reason or "Export failed."
                    errors.append(reason)
                    if callable(log):
                        log(f"Export FITS skipped: {reason}")
            if pd is not None:
                pd.setValue(total)
        finally:
            if pd is not None:
                pd.close()
                pd.deleteLater()
                QApplication.processEvents()
        if spec.get("out_dir"):
            _remember_export_dir(self._mw, spec["out_dir"])
        if close_when_done:
            self.close()
        return applied, errors

    def _apply(self):
        if self._require_folder() is None:
            return
        doc = _active_doc_from_main(self._mw)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Export FITS", "No image in the active view.")
            return
        if self._docman() is None:
            QMessageBox.warning(self, "Export FITS", "Document manager not available.")
            return
        applied, errors = self._export_docs([doc], close_when_done=False)
        if errors:
            QMessageBox.warning(self, "Export FITS", errors[0])
            return
        if applied:
            self.close()

    def _load_view_bundle_choices(self):
        settings = QSettings()
        settings.sync()
        raw = ""
        for key in ("viewbundles/v3", "viewbundles/v2", "viewbundles/v1"):
            raw = settings.value(key, "", type=str) or ""
            if raw:
                break
        try:
            data = json.loads(raw or "[]")
        except Exception:
            data = []
        choices = []
        for bundle in data:
            if not isinstance(bundle, dict):
                continue
            name = str(bundle.get("name") or "Bundle").strip() or "Bundle"
            ptrs = []
            for x in (bundle.get("doc_ptrs") or []):
                try:
                    ptrs.append(int(x))
                except Exception:
                    pass
            files = [str(p) for p in (bundle.get("file_paths") or []) if p]
            choices.append((name, ptrs, files))
        return choices

    def _pick_view_bundle(self):
        choices = self._load_view_bundle_choices()
        if not choices:
            QMessageBox.information(self, "Export FITS", "No View Bundles found.")
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle("Apply to View Bundle…")
        dlg.setMinimumSize(420, 280)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Select a View Bundle:"))
        lb = QListWidget(dlg)
        for name, ptrs, files in choices:
            it = QListWidgetItem(f"{name}  ({len(ptrs)} views, {len(files)} files)")
            it.setData(Qt.ItemDataRole.UserRole, (ptrs, files))
            lb.addItem(it)
        if lb.count():
            lb.setCurrentRow(0)
        v.addWidget(lb, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        cur = lb.currentItem()
        if not cur:
            return None
        return cur.data(Qt.ItemDataRole.UserRole)

    def _doc_from_file(self, path: str):
        from setiastro.saspro.doc_manager import ImageDocument
        from setiastro.saspro.legacy.image_manager import load_image
        img, header, bit_depth, is_mono = load_image(path)
        if img is None:
            raise RuntimeError(f"Could not load: {path}")
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "fits"
        return ImageDocument(img, metadata={
            "file_path": path,
            "original_header": header,
            "bit_depth": bit_depth,
            "is_mono": is_mono,
            "original_format": ext,
        })

    def _apply_to_view_bundle(self):
        if self._require_folder() is None:
            return
        picked = self._pick_view_bundle()
        if not picked:
            return
        ptrs, files = picked
        from setiastro.saspro.view_bundle import _find_main_window, _resolve_doc_and_subwindow
        mw = _find_main_window(self) or self._mw
        docs = []
        for ptr in ptrs or []:
            doc = None
            if mw is not None:
                doc, sw = _resolve_doc_and_subwindow(mw, ptr)
                if sw is not None and hasattr(mw, "mdi"):
                    try:
                        mw.mdi.setActiveSubWindow(sw)
                        QApplication.processEvents()
                    except Exception:
                        pass
            if doc is not None and getattr(doc, "image", None) is not None:
                docs.append(doc)
        for path in files or []:
            try:
                docs.append(self._doc_from_file(path))
            except Exception as e:
                QMessageBox.warning(self, "Export FITS", str(e))
                return
        if not docs:
            QMessageBox.information(self, "Export FITS", "No valid targets in the selected bundle.")
            return
        applied, errors = self._export_docs(docs, close_when_done=True)
        if applied == 0 and errors:
            QMessageBox.warning(self, "Export FITS", errors[0])
        elif errors:
            QMessageBox.warning(
                self,
                "Export FITS",
                f"Exported {applied} file(s).\n\nErrors:\n" + "\n".join(errors[:12]),
            )


def open_export_fits_with_preset(main_window, preset: dict | None = None):
    dlg = getattr(main_window, "_export_fits_dialog", None)
    if dlg is None:
        dlg = ExportFitsDialog(main_window, preset or {})
        try:
            main_window._export_fits_dialog = dlg
            dlg.destroyed.connect(
                lambda *_: setattr(main_window, "_export_fits_dialog", None)
                if getattr(main_window, "_export_fits_dialog", None) is dlg
                else None
            )
        except Exception:
            pass
    else:
        try:
            dlg.seed_from_preset(preset or {})
        except Exception:
            pass
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
