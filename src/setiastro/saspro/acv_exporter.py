# src/setiastro/saspro/acv_exporter.py
from __future__ import annotations

import os
import re
from pathlib import Path

from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QMessageBox, QWidget
)

# If you have a canonical place for "global save_as", wire it here.
# Example placeholders (change to your actual import):
# from setiastro.saspro.actions.save_as import save_as
# from setiastro.saspro.fileio.save_as import save_as


def _tr(s: str) -> str:
    return QCoreApplication.translate("AstroCatalogueViewerExporter", s)


class AstroCatalogueViewerExporterDialog(QDialog):
    """
    Astro Catalogue Viewer Exporter
    - Persist folders in QSettings
    - Export current active document/view via your global save_as
    - Route by name prefix: M -> Messier, NGC -> NGC, IC -> IC, C -> Caldwell, else Master
    """

    # QSettings keys
    K_MASTER = "acv_export/master_folder"
    K_M      = "acv_export/messier_folder"
    K_NGC    = "acv_export/ngc_folder"
    K_IC     = "acv_export/ic_folder"
    K_C      = "acv_export/caldwell_folder"
    K_FMT    = "acv_export/format"

    def __init__(self, parent, dm, doc):
        super().__init__(parent)
        self.dm = dm
        self.doc = doc
        self.setObjectName("acv_exporter_dialog")
        self.setWindowTitle(_tr("Astro Catalogue Viewer Exporter"))

        # Prefer not using WA_DeleteOnClose if that’s a Linux landmine for you.
        # (You can still set it from caller if you want.)
        self._main = parent
        self._settings = getattr(parent, "settings", None)  # typical SASpro pattern

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # -------------------------
        # Folder section
        # -------------------------
        gb = QGroupBox(_tr("Image folders"), self)
        g = QGridLayout(gb)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        self.ed_master = QLineEdit(self)
        self.btn_master = QPushButton(_tr("Browse…"), self)
        self.btn_master.clicked.connect(lambda: self._browse_folder(self.ed_master))

        g.addWidget(QLabel(_tr("Master Image Folder")), 0, 0)
        g.addWidget(self.ed_master, 0, 1)
        g.addWidget(self.btn_master, 0, 2)

        gb2 = QGroupBox(_tr("Image folder per catalog"), self)
        gg = QGridLayout(gb2)
        gg.setContentsMargins(10, 10, 10, 10)
        gg.setHorizontalSpacing(10)
        gg.setVerticalSpacing(8)

        self.ed_m = QLineEdit(self)
        self.ed_ngc = QLineEdit(self)
        self.ed_ic = QLineEdit(self)
        self.ed_c = QLineEdit(self)

        bm = QPushButton(_tr("Browse…"), self); bm.clicked.connect(lambda: self._browse_folder(self.ed_m))
        bn = QPushButton(_tr("Browse…"), self); bn.clicked.connect(lambda: self._browse_folder(self.ed_ngc))
        bi = QPushButton(_tr("Browse…"), self); bi.clicked.connect(lambda: self._browse_folder(self.ed_ic))
        bc = QPushButton(_tr("Browse…"), self); bc.clicked.connect(lambda: self._browse_folder(self.ed_c))

        gg.addWidget(QLabel(_tr("Messier")), 0, 0); gg.addWidget(self.ed_m,   0, 1); gg.addWidget(bm, 0, 2)
        gg.addWidget(QLabel(_tr("NGC")),     1, 0); gg.addWidget(self.ed_ngc, 1, 1); gg.addWidget(bn, 1, 2)
        gg.addWidget(QLabel(_tr("IC")),      2, 0); gg.addWidget(self.ed_ic,  2, 1); gg.addWidget(bi, 2, 2)
        gg.addWidget(QLabel(_tr("Caldwell")),3, 0); gg.addWidget(self.ed_c,   3, 1); gg.addWidget(bc, 3, 2)

        root.addWidget(gb)
        root.addWidget(gb2)

        # -------------------------
        # Export controls
        # -------------------------
        row = QHBoxLayout()
        row.setSpacing(10)

        self.ed_name = QLineEdit(self)
        self.ed_name.setPlaceholderText(_tr("e.g. M31, NGC5060, M31_HaOnly…"))

        self.cmb_fmt = QComboBox(self)
        # keep these lower-case as “extensions”
        self.cmb_fmt.addItems(["jpg", "png", "tif", "fit"])

        self.btn_export = QPushButton(_tr("Export"), self)
        self.btn_export.clicked.connect(self._on_export)

        row.addWidget(QLabel(_tr("Image Name")), 0)
        row.addWidget(self.ed_name, 2)
        row.addWidget(QLabel(_tr("Type")), 0)
        row.addWidget(self.cmb_fmt, 0)
        row.addWidget(self.btn_export, 0)

        root.addLayout(row)

        # Load persisted settings
        self._load_settings()

        # Save on edits (lightweight)
        self.ed_master.textChanged.connect(self._save_settings)
        self.ed_m.textChanged.connect(self._save_settings)
        self.ed_ngc.textChanged.connect(self._save_settings)
        self.ed_ic.textChanged.connect(self._save_settings)
        self.ed_c.textChanged.connect(self._save_settings)
        self.cmb_fmt.currentTextChanged.connect(self._save_settings)

    # -------------------------
    # Settings
    # -------------------------
    def _load_settings(self):
        s = self._settings
        if s is None:
            return
        self.ed_master.setText(s.value(self.K_MASTER, "", type=str) or "")
        self.ed_m.setText(s.value(self.K_M, "", type=str) or "")
        self.ed_ngc.setText(s.value(self.K_NGC, "", type=str) or "")
        self.ed_ic.setText(s.value(self.K_IC, "", type=str) or "")
        self.ed_c.setText(s.value(self.K_C, "", type=str) or "")
        fmt = (s.value(self.K_FMT, "png", type=str) or "png").lower()
        idx = self.cmb_fmt.findText(fmt)
        if idx >= 0:
            self.cmb_fmt.setCurrentIndex(idx)

    def _save_settings(self):
        s = self._settings
        if s is None:
            return
        s.setValue(self.K_MASTER, self.ed_master.text().strip())
        s.setValue(self.K_M, self.ed_m.text().strip())
        s.setValue(self.K_NGC, self.ed_ngc.text().strip())
        s.setValue(self.K_IC, self.ed_ic.text().strip())
        s.setValue(self.K_C, self.ed_c.text().strip())
        s.setValue(self.K_FMT, self.cmb_fmt.currentText().strip().lower())
        try:
            s.sync()
        except Exception:
            pass

    # -------------------------
    # Folder helpers
    # -------------------------
    def _browse_folder(self, target_edit: QLineEdit):
        start = target_edit.text().strip() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, _tr("Select folder"), start)
        if path:
            target_edit.setText(path)

    # -------------------------
    # Active doc resolution (ROI aware)
    # -------------------------
    def _get_active_doc(self):
        mw = self._main
        mdi = getattr(mw, "mdi", None)
        dm  = getattr(mw, "doc_manager", None) or getattr(mw, "docman", None)

        if mdi is None or not hasattr(mdi, "activeSubWindow"):
            return None

        sw = mdi.activeSubWindow()
        if not sw:
            return None

        view = sw.widget()

        # Prefer ROI-aware doc lookup via DocManager
        if dm is not None and hasattr(dm, "get_document_for_view"):
            try:
                doc = dm.get_document_for_view(view)
                if doc is not None:
                    return doc
            except Exception:
                pass

        # Fallback: view.document
        return getattr(view, "document", None)

    # -------------------------
    # Routing logic
    # -------------------------
    @staticmethod
    def _pick_target_folder(name: str, master: str, m: str, ngc: str, ic: str, c: str) -> str:
        raw = (name or "").strip().upper()

        # routing normalization: remove spaces (optionally also "_" and "-" if you want)
        u = re.sub(r"\s+", "", raw)

        if re.match(r"^M\d+", u):
            return m or master
        if re.match(r"^NGC\d+", u):     # optional tighten
            return ngc or master
        if re.match(r"^IC\d+", u):      # optional tighten
            return ic or master
        if re.match(r"^C\d+", u):
            return c or master
        return master

    def _normalize_for_routing(self, name: str) -> str:
        # Uppercase + remove spaces only (routing only)
        u = (name or "").strip().upper()
        u = re.sub(r"\s+", "", u)
        return u

    def _sanitize_filename(self, s: str) -> str:
        # Keep it simple + safe across OSes
        s = (s or "").strip()
        s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
        s = re.sub(r"\s+", " ", s)                 # keep single spaces
        s = re.sub(r"[^\w\-\.\(\) ]+", "", s)      # drop odd chars
        return s.strip() or "export"

    # -------------------------
    # Export action
    # -------------------------
    def _on_export(self):
        fmt = self.cmb_fmt.currentText().strip().lower()

        # -------------------------
        # Name (raw + routing-normalized)
        # -------------------------
        raw_name = self.ed_name.text().strip()
        if not raw_name:
            QMessageBox.information(
                self, _tr("Export"),
                _tr("Please enter an Image Name (e.g. M31, NGC5060…).")
            )
            return

        route_name = self._normalize_for_routing(raw_name)

        # -------------------------
        # Active doc (passed in)
        # -------------------------
        doc = self.doc
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, _tr("Export"), _tr("No active image. Open an image first."))
            return

        # -------------------------
        # Folder config
        # -------------------------
        master_dir = self.ed_master.text().strip()
        if not master_dir:
            QMessageBox.information(self, _tr("Export"), _tr("Please set a Master Image Folder."))
            return

        messier_dir  = self.ed_m.text().strip()   or master_dir
        ngc_dir      = self.ed_ngc.text().strip() or master_dir
        ic_dir       = self.ed_ic.text().strip()  or master_dir
        caldwell_dir = self.ed_c.text().strip()   or master_dir

        # -------------------------
        # Route by prefix (digit-guard)
        # -------------------------
        if route_name.startswith("M") and route_name[1:2].isdigit():
            base_dir = messier_dir
        elif route_name.startswith("NGC") and route_name[3:4].isdigit():
            base_dir = ngc_dir
        elif route_name.startswith("IC") and route_name[2:3].isdigit():
            base_dir = ic_dir
        elif route_name.startswith("C") and route_name[1:2].isdigit():
            base_dir = caldwell_dir
        else:
            base_dir = master_dir

        # Ensure folder exists
        try:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self, _tr("Export"),
                _tr("Could not create folder:\n{0}\n\n{1}").format(base_dir, str(e))
            )
            return

        # Filename uses user's raw name (sanitized for filesystem)
        file_stem = self._sanitize_filename(raw_name)
        out_path = str(Path(base_dir) / f"{file_stem}.{fmt}")

        # -------------------------
        # Save
        # -------------------------
        ok, err = self._save_current_doc_to_path(doc, out_path, fmt)
        if ok:
            QMessageBox.information(self, _tr("Export"), _tr("Exported:\n{0}").format(out_path))
        else:
            QMessageBox.critical(self, _tr("Export"), _tr("Export failed:\n{0}").format(err or "Unknown error"))


    def _save_current_doc_to_path(self, doc, out_path: str, fmt: str) -> tuple[bool, str]:
        try:
            import numpy as np
            from astropy.io import fits
            from astropy.wcs import WCS

            from setiastro.saspro.legacy.image_manager import save_image as legacy_save_image

            img = getattr(doc, "image", None)
            if img is None:
                return False, "Active document has no image data."

            # Ensure numpy array
            img_array = np.asarray(img)

            # Mono detection (your saver uses this for RAW/XISF paths; safe to provide anyway)
            is_mono = (img_array.ndim == 2) or (img_array.ndim == 3 and img_array.shape[2] == 1)

            # Preserve bit depth for tif/fit if the doc has it; PNG/JPG ignore bit depth anyway.
            bit_depth = (
                getattr(doc, "bit_depth", None)
                or getattr(doc, "bitdepth", None)
                or (getattr(doc, "metadata", None) or {}).get("bit_depth")
                or (getattr(doc, "metadata", None) or {}).get("bitDepth")
            )

            # Pull headers/WCS from metadata (matches your SASv2->SASpro WCS preference)
            md = getattr(doc, "metadata", None) or {}

            original_header = md.get("original_header", None)
            wcs_header = None

            # If wcs is stored as astropy.wcs.WCS, convert to Header
            wcs_obj = md.get("wcs", None)
            if wcs_obj is not None:
                try:
                    if isinstance(wcs_obj, WCS):
                        wcs_header = wcs_obj.to_header(relax=True)
                    elif isinstance(wcs_obj, fits.Header):
                        wcs_header = wcs_obj
                except Exception:
                    wcs_header = None

            # Optional passthroughs (safe if missing)
            image_meta = md.get("image_meta", None) or md.get("image_metadata", None)
            file_meta  = md.get("file_meta", None)  or md.get("xisf_metadata", None)

            # IMPORTANT:
            # legacy_save_image expects original_format to be the desired output format,
            # and will normalize extension itself.
            legacy_save_image(
                img_array=img_array,
                filename=out_path,
                original_format=fmt,          # "jpg","png","tif","fit"
                bit_depth=bit_depth,          # keep same depth for tif/fit when possible
                original_header=original_header,
                is_mono=is_mono,
                image_meta=image_meta,
                file_meta=file_meta,
                wcs_header=wcs_header,        # merged into FITS header (your saver supports this)
            )
            return True, ""

        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
