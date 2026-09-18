from __future__ import annotations

import os
from dataclasses import dataclass

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
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
        self.setModal(False)
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
            "In a Function Bundle the folder comes from this preset — no dialog per image."
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
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btns.addWidget(btn_apply)
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

    def _apply(self):
        spec = self.current_preset()
        if not spec["out_dir"]:
            self._browse()
            spec = self.current_preset()
        if not spec["out_dir"]:
            QMessageBox.warning(self, "Export FITS", "Choose an output folder first.")
            return
        doc = _active_doc_from_main(self._mw)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Export FITS", "No image in the active view.")
            return
        dm = getattr(self._mw, "docman", None) or getattr(self._mw, "doc_manager", None)
        if dm is None:
            QMessageBox.warning(self, "Export FITS", "Document manager not available.")
            return
        result = export_document_as_fits(dm, doc, spec)
        _remember_export_dir(self._mw, spec["out_dir"])
        log = getattr(self._mw, "_log", None)
        if result.ok:
            if callable(log):
                log(f"Exported FITS: {result.path}")
            self.close()
        else:
            if callable(log):
                log(f"Export FITS skipped: {result.reason}")
            QMessageBox.warning(self, "Export FITS", result.reason or "Export failed.")


def open_export_fits_with_preset(main_window, preset: dict | None = None):
    dlg = ExportFitsDialog(main_window, preset or {})
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
