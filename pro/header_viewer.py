# pro/header_viewer.py
from __future__ import annotations
import os, csv
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

from astropy.io import fits
from xisf import XISF

# we’ll reuse your loader helper for FITS headers
from legacy.image_manager import get_valid_header
from pro.doc_manager import ImageDocument


class HeaderViewerDock(QDockWidget):
    """
    Dock that shows metadata for the currently active ImageDocument.
    Supports FITS headers and XISF file & image metadata.
    """
    def __init__(self, parent=None):
        super().__init__("Header Viewer", parent)
        self._doc: Optional[ImageDocument] = None
        self._doc_conn = False

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Key", "Value"])
        self._tree.setColumnWidth(0, 220)

        self._save_btn = QPushButton("Save Metadata…")
        self._save_btn.clicked.connect(self._save_metadata)

        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._tree)
        lay.addWidget(self._save_btn)
        self.setWidget(w)

    # ---- public API ----
    def set_document(self, doc: Optional[ImageDocument]):
        # disconnect old
        if self._doc and hasattr(self._doc, "changed"):
            try:
                self._doc.changed.disconnect(self._on_doc_changed)
            except Exception:
                pass

        self._doc = doc
        if self._doc and hasattr(self._doc, "changed"):
            try:
                self._doc.changed.connect(self._on_doc_changed)
            except Exception:
                pass

        self._rebuild()

    def _on_doc_changed(self):
        # metadata changed → rebuild view
        self._rebuild()

    def _rebuild(self):
        self._tree.clear()
        if not self._doc:
            self.setWindowTitle("Header Viewer")
            return

        meta = self._doc.metadata or {}
        path = meta.get("file_path", "") or ""
        base = os.path.basename(path) if path else "Untitled"
        self.setWindowTitle(f"Header: {base}")

        try:
            # 1) If metadata already has a real fits.Header, show that first.
            hdr = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
            if isinstance(hdr, fits.Header):
                self._populate_fits_header(hdr)

            # 2) If no header in metadata and the file itself is FITS-like, read it.
            elif path.lower().endswith((".fits", ".fit", ".fz", ".fits.fz", ".fit.fz")):
                file_hdr = meta.get("original_header")
                if file_hdr is None or not isinstance(file_hdr, fits.Header):
                    file_hdr, _ = get_valid_header(path)
                self._populate_fits_header(file_hdr)

            # 3) If there is a real astropy.wcs.WCS object, expand it as proper keys.
            try:
                from astropy.wcs import WCS as _WCS
                wcs_obj = meta.get("wcs")
                if isinstance(wcs_obj, _WCS):
                    self._populate_wcs(wcs_obj)
            except Exception:
                pass

            # 4) Show the remaining lightweight metadata (skip heavy objects).
            info_root = QTreeWidgetItem(["Metadata"])
            self._tree.addTopLevelItem(info_root)
            for k, v in meta.items():
                if k in ("original_header", "fits_header", "header", "wcs"):
                    continue  # avoid giant repr blobs; we render them above
                info_root.addChild(QTreeWidgetItem([str(k), str(v)]))

            self._tree.expandAll()

        except Exception as e:
            QMessageBox.warning(self, "Metadata", f"Failed to read metadata:\n{e}")


    # ---- population helpers ----
    def _populate_fits_header(self, header: Any):
        root = QTreeWidgetItem(["FITS Header"])
        self._tree.addTopLevelItem(root)

        items = []
        if isinstance(header, fits.Header):
            items = header.items()
        elif isinstance(header, dict):
            items = header.items()

        for k, v in items:
            root.addChild(QTreeWidgetItem([str(k), str(v)]))

    def _populate_wcs(self, wcs_obj):
        """Show a real astropy.wcs.WCS as header-like key/values."""
        root = QTreeWidgetItem(["WCS"])
        self._tree.addTopLevelItem(root)
        try:
            # Use relax=True so SIP/etc. are included if present.
            wcs_hdr = wcs_obj.to_header(relax=True)
            for k, v in wcs_hdr.items():
                root.addChild(QTreeWidgetItem([str(k), str(v)]))
        except Exception:
            # Fallback: parse the repr into lines (better than a single blob).
            for line in str(wcs_obj).splitlines():
                s = line.strip()
                if not s:
                    continue
                if ":" in s:
                    a, b = s.split(":", 1)
                    root.addChild(QTreeWidgetItem([a.strip(), b.strip()]))
                else:
                    root.addChild(QTreeWidgetItem(["", s]))


    def _populate_from_xisf(self, path: str):
        x = XISF(path)
        file_meta: Dict[str, Any] = x.get_file_metadata()
        img_meta_list = x.get_images_metadata()
        img_meta: Dict[str, Any] = img_meta_list[0] if img_meta_list else {}

        # File-level metadata
        froot = QTreeWidgetItem(["XISF File Metadata"])
        self._tree.addTopLevelItem(froot)
        for k, v in file_meta.items():
            vstr = v.get("value", "") if isinstance(v, dict) else v
            froot.addChild(QTreeWidgetItem([str(k), str(vstr)]))

        # Image-level metadata
        iroot = QTreeWidgetItem(["XISF Image Metadata"])
        self._tree.addTopLevelItem(iroot)

        # FITS-like keywords (nested)
        if "FITSKeywords" in img_meta:
            fits_item = QTreeWidgetItem(["FITSKeywords"])
            iroot.addChild(fits_item)
            for kw, entries in img_meta["FITSKeywords"].items():
                for ent in entries:
                    fits_item.addChild(QTreeWidgetItem([kw, str(ent.get("value", ""))]))

        # XISFProperties (nested)
        if "XISFProperties" in img_meta:
            props_item = QTreeWidgetItem(["XISFProperties"])
            iroot.addChild(props_item)
            for prop_name, prop in img_meta["XISFProperties"].items():
                props_item.addChild(QTreeWidgetItem([prop_name, str(prop.get("value", ""))]))

        # Any remaining flat fields
        for k, v in img_meta.items():
            if k in ("FITSKeywords", "XISFProperties"):
                continue
            iroot.addChild(QTreeWidgetItem([k, str(v)]))

        self._tree.expandAll()

    # ---- export ----
    def _save_metadata(self):
        if not self._doc:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "CSV (*.csv)")
        if not path:
            return

        # Flatten the QTreeWidget contents into key/value rows
        rows = []
        def walk(item: QTreeWidgetItem, prefix: str = ""):
            key = item.text(0)
            val = item.text(1)
            full = f"{prefix}.{key}" if prefix else key
            if key and val:
                rows.append((full, val))
            for i in range(item.childCount()):
                walk(item.child(i), full)

        for i in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(i))

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Key", "Value"])
                w.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Save Metadata", f"Failed to save:\n{e}")
