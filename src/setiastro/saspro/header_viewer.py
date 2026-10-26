#saspro/header_viewer.py
from __future__ import annotations
import os
import csv
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication

from astropy.io import fits
try:
    from astropy.io.fits.verify import VerifyError
except Exception:
    class VerifyError(Exception):
        pass

from setiastro.saspro.xisf import XISF

# we’ll reuse your loader helper for FITS headers
from setiastro.saspro.legacy.image_manager import get_valid_header, _drop_invalid_cards
from setiastro.saspro.doc_manager import ImageDocument


class HeaderViewerDock(QDockWidget):
    """
    Dock that shows metadata for the currently active ImageDocument.
    Supports FITS headers and XISF file & image metadata.
    """
    def __init__(self, parent=None):
        super().__init__(self.tr("Header Viewer"), parent)
        self._doc: Optional[ImageDocument] = None
        self._doc_conn = False

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels([self.tr("Key"), self.tr("Value")])
        self._tree.setColumnWidth(0, 220)

        self._save_btn = QPushButton(self.tr("Save Metadata…"))
        self._save_btn.clicked.connect(self._save_metadata)
        self._copy_btn = QPushButton(self.tr("Copy Metadata"))
        self._copy_btn.clicked.connect(self._copy_metadata)
        self._dm = None              # <-- NEW: DocManager to query "active"
        self._follow_hover = False   # <-- optional toggle if you ever want hover-follow
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._tree)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._copy_btn)
        btn_row.addWidget(self._save_btn)
        lay.addLayout(btn_row)
        self.setWidget(w)

    def _same_base(self, a, b) -> bool:
        return self._unwrap_base_doc(a) is self._unwrap_base_doc(b)

    def attach_doc_manager(self, dm):
        self._dm = dm
        try:
            # When docs are added/removed, re-evaluate the focused base
            dm.documentAdded.connect(lambda _doc: self._maybe_refresh_for_active())
            dm.documentRemoved.connect(lambda _doc: self._maybe_refresh_for_active())

            # DO NOT use imageRegionUpdated to retarget; it can fire from hover-driven previews.
            # If you want to repaint the same doc on region changes, _on_doc_changed handles that.

            mdi = getattr(dm, "_mdi", None)
            if mdi and hasattr(mdi, "subWindowActivated"):
                mdi.subWindowActivated.connect(lambda _sw: self._maybe_refresh_for_active())

            # NEW: snap to truly active base (sticky, click-activated only)
            if hasattr(dm, "activeBaseChanged"):
                dm.activeBaseChanged.connect(lambda _doc: self._maybe_refresh_for_active())
        except Exception:
            pass

        self._maybe_refresh_for_active()


    def set_follow_hover(self, enabled: bool):
        self._follow_hover = bool(enabled)

    # Prefer base (true) doc over transient wrappers/proxies
    def _unwrap_base_doc(self, d):
        if d is None:
            return None
        # ROI preview wrapper → parent
        p = getattr(d, "_parent_doc", None)
        if isinstance(p, ImageDocument):
            return p
        # LiveViewDocument proxy → base
        b = getattr(d, "_base", None)
        if isinstance(b, ImageDocument):
            return b
        return d

    def _active_base_doc(self):
        if not self._dm:
            return None
        # Prefer DocManager’s sticky focused base if available
        if hasattr(self._dm, "get_focused_base_document"):
            try:
                return self._dm.get_focused_base_document()
            except Exception:
                pass
        # Fallback: unwrap whatever get_active_document returns
        try:
            cur = self._dm.get_active_document()
            return self._unwrap_base_doc(cur)
        except Exception:
            return None


    def _maybe_refresh_for_active(self):
        """Rebuild only if our bound document == current active base document."""
        active_base = self._active_base_doc()
        if active_base is None:
            # No active base. Only force-clear when there are genuinely no
            # documents left (e.g. the last view was just closed); otherwise keep
            # the current view so a momentary focus gap can't blank the dock.
            has_docs = False
            if self._dm is not None:
                try:
                    has_docs = bool(self._dm.all_documents())
                except Exception:
                    has_docs = False
            if not has_docs:
                if self._doc is not None and hasattr(self._doc, "changed"):
                    try:
                        self._doc.changed.disconnect(self._on_doc_changed)
                    except Exception:
                        pass
                self._doc = None
                self._rebuild()
            return
        # If we already show the same base doc, just ignore
        if self._unwrap_base_doc(self._doc) is active_base:
            return
        # Else bind to the active base doc
        self.set_document(active_base)

    # ---- public API ----
    def set_document(self, doc: Optional[ImageDocument]):
        """
        Hard-lock behavior:
        - If attached to a DocManager AND hover-follow is OFF, ignore the caller's 'doc'
            and always bind to the DocManager's *active base* doc.
        - Otherwise, behave like a normal setter.
        """
        if self._dm and not self._follow_hover:
            # Caller cannot hijack focus: resolve from DM every time
            doc = self._active_base_doc()

        # Always resolve to base (true) document for internal storage
        base_doc = self._unwrap_base_doc(doc)

        # No-op if unchanged
        if self._same_base(self._doc, base_doc):
            return

        # Disconnect old
        if self._doc and hasattr(self._doc, "changed"):
            try:
                self._doc.changed.disconnect(self._on_doc_changed)
            except Exception:
                pass

        self._doc = base_doc

        # Listen for internal changes on the *bound* doc
        if self._doc and hasattr(self._doc, "changed"):
            try:
                self._doc.changed.connect(self._on_doc_changed)
            except Exception:
                pass

        self._rebuild()


    def _on_doc_changed(self):
        """
        Only rebuild if our bound doc is STILL the active base doc.
        Prevents spurious rebuilds when focus changed between signal emit and slot run.
        """
        if self._dm and not self._follow_hover:
            active_base = self._active_base_doc()
            if not self._same_base(self._doc, active_base):
                # We got a change from an old/hover doc — ignore and snap to active.
                self._maybe_refresh_for_active()
                return
        self._rebuild()


    # --- helpers ---------------------------------------------------------
    def _populate_header_dict(self, d: dict, title="Header (dict)"):
        # We translate the default title if it's the default, but often title is passed in.
        # If title is passed in English from other methods, we should translate it at the call site or here if possible.
        # Since title is variable, we'll leave it as is, but ensure call sites pass translated strings.
        root = QTreeWidgetItem([title])
        self._tree.addTopLevelItem(root)
        for k, v in d.items():
            root.addChild(QTreeWidgetItem([str(k), str(v)]))

    def _populate_header_snapshot(self, snap: dict):
        fmt = (snap or {}).get("format", "")
        if fmt == "fits-cards":
            cards = snap.get("cards") or []
            hdr = fits.Header()
            for k, v, c in cards:
                try:
                    hdr[str(k)] = (v, c)
                except Exception:
                    # extremely defensive: skip bad card entries
                    pass
            try:
                hdr = _drop_invalid_cards(hdr)
            except Exception:
                pass
            self._populate_fits_header(hdr)
        elif fmt == "dict":
            self._populate_header_dict(snap.get("items") or {}, self.tr("Header (snapshot)"))
        else:
            # generic repr fallback
            txt = (snap or {}).get("text", "")
            node = QTreeWidgetItem([self.tr("Header (snapshot)")])
            self._tree.addTopLevelItem(node)
            node.addChild(QTreeWidgetItem(["repr", str(txt)]))


    def _try_populate_from_doc(self, meta: dict) -> bool:
        """Return True if we showed any header from the document metadata."""
        # 1) direct astropy header
        hdr = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
        if isinstance(hdr, fits.Header):
            try:
                hdr = _drop_invalid_cards(hdr.copy())
            except Exception:
                pass
            self._populate_fits_header(hdr)
            return True

        # 2) dict-style header (e.g., XISF-style properties captured as dict)
        if isinstance(hdr, dict):
            self._populate_header_dict(hdr, self.tr("Header (dict from document)"))
            return True

        # 3) JSON-safe snapshot captured by DocManager
        snap = meta.get("__header_snapshot__")
        if isinstance(snap, dict):
            self._populate_header_snapshot(snap)
            return True

        # 4) XISF properties stored in metadata (common keys)
        for k in ("xisf_header", "xisf_properties"):
            if isinstance(meta.get(k), dict):
                self._populate_header_dict(meta[k], self.tr("XISF Properties (document)"))
                return True

        return False

    def _try_populate_from_file(self, path: str, meta: dict) -> bool:
        """Return True if we read & showed header from the backing file."""
        if not path:
            return False
        p = path.lower()

        # FITS (and MEF and .fz) via legacy helper
        if p.endswith((".fits", ".fit", ".fz", ".fits.fz", ".fit.fz")):
            # prefer the on-disk header if not already in meta
            file_hdr = meta.get("original_header")
            if isinstance(file_hdr, fits.Header):
                try:
                    file_hdr = _drop_invalid_cards(file_hdr.copy())
                except Exception:
                    pass
            else:
                file_hdr, _ = get_valid_header(path)

            if isinstance(file_hdr, fits.Header):
                self._populate_fits_header(file_hdr)
                return True

        # XISF: try to open and show basic properties if available
        if p.endswith(".xisf"):
            try:
                xisf = XISF(path)
                props = getattr(xisf, "properties", None)
                if isinstance(props, dict):
                    self._populate_header_dict(props, self.tr("XISF Properties"))
                    return True
            except Exception:
                pass

        return False


    # --- main ------------------------------------------------------------
    def _rebuild(self):
        self._tree.clear()
        base_doc = self._unwrap_base_doc(self._doc)
        if not base_doc:
            self.setWindowTitle(self.tr("Header Viewer"))
            return
        self._doc = base_doc

        meta = self._doc.metadata or {}
        path = (meta.get("file_path") or "") if isinstance(meta.get("file_path"), str) else ""
        base = os.path.basename(path) if path else (meta.get("display_name") or self.tr("Untitled"))
        self.setWindowTitle(self.tr("Header: {0}").format(base))

        try:
            # 1) Prefer header data already stored with the document
            shown_any = self._try_populate_from_doc(meta)

            # 2) If we didn't render anything yet, fall back to the file on disk
            if not shown_any:
                shown_any = self._try_populate_from_file(path, meta)

            # 3) If there is a real astropy.wcs.WCS object, render it as key/value rows
            try:
                from astropy.wcs import WCS as _WCS
                wcs_obj = meta.get("wcs")
                if isinstance(wcs_obj, _WCS):
                    self._populate_wcs(wcs_obj)
            except Exception:
                pass

            # 4) XISF metadata as a browsable tree rather than one huge row.
            #    str(image_meta) on a WBPP master is ~50 KB of stringified dict;
            #    the standard properties are now promoted into the FITS header
            #    above, and the full detail lives here, nested and exportable.
            self._populate_xisf_meta(meta.get("xisf_meta"))

            # 5) Always show remaining lightweight metadata (skip heavy blobs we already rendered)
            info_root = QTreeWidgetItem([self.tr("Metadata")])
            self._tree.addTopLevelItem(info_root)
            for k, v in meta.items():
                if k in ("original_header", "fits_header", "header", "wcs", "__header_snapshot__",
                         "xisf_header", "xisf_properties", "xisf_meta", "image_meta"):
                    continue
                info_root.addChild(QTreeWidgetItem([str(k), self._short_value(v)]))

            self._tree.expandAll()

        except Exception:
            # per request: fail silently on final exception
            pass


    # ---- population helpers ----
    def _populate_fits_header(self, header: Any):
        root = QTreeWidgetItem([self.tr("FITS Header")])
        self._tree.addTopLevelItem(root)

        # FITS Header: sanitize and iterate cards defensively
        if isinstance(header, fits.Header):
            try:
                header = _drop_invalid_cards(header)
            except Exception:
                pass

            for card in header.cards:
                try:
                    k = str(card.keyword)
                    v = str(card.value)
                except VerifyError as e:
                    # Skip invalid/unparsable card
                    print(f"[HeaderViewer] Skipping invalid FITS card {getattr(card, 'keyword', '?')!r}: {e}")
                    continue
                except Exception as e:
                    print(f"[HeaderViewer] Error reading FITS card: {e}")
                    continue
                root.addChild(QTreeWidgetItem([k, v]))

        # Plain dict fallback (e.g., XISF-style dict)
        elif isinstance(header, dict):
            for k, v in header.items():
                try:
                    root.addChild(QTreeWidgetItem([str(k), str(v)]))
                except Exception:
                    continue


    def _populate_wcs(self, wcs_obj):
        """Show a real astropy.wcs.WCS as header-like key/values."""
        root = QTreeWidgetItem([self.tr("WCS")])
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


    # ---- XISF metadata already held by the document ----
    _MAX_VALUE_CHARS = 300

    @classmethod
    def _short_value(cls, v) -> str:
        """
        Render a metadata value as one readable line.

        Guards the panel against any oversized value: arrays are summarised by
        shape, and long text is clipped with its real length noted, so a blob
        can never make a row unreadable again.
        """
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray):
                return f"ndarray{tuple(v.shape)} {v.dtype}"
        except Exception:
            pass
        if isinstance(v, (bytes, bytearray)):
            return f"<{len(v)} bytes>"
        if isinstance(v, dict):
            return f"<dict, {len(v)} keys>"
        if isinstance(v, (list, tuple)) and len(v) > 12:
            return f"<{type(v).__name__}, {len(v)} items>"
        try:
            s = str(v)
        except Exception:
            return "<unprintable>"
        s = s.replace("\n", " ").replace("\r", " ")
        if len(s) > cls._MAX_VALUE_CHARS:
            return f"{s[:cls._MAX_VALUE_CHARS]}… ({len(s)} chars)"
        return s

    def _populate_xisf_meta(self, image_meta) -> bool:
        """
        Render in-memory XISF image metadata as nested nodes.

        Works off what the loader already stored -- no reopening the file.
        Tolerates the stringified-dict shape some paths produce.
        """
        if isinstance(image_meta, str):
            if not image_meta.strip():
                return False
            parsed = None
            try:
                import ast as _ast
                parsed = _ast.literal_eval(image_meta)
            except Exception:
                # Real blobs embed repr-only values such as dtype('float32'),
                # which literal_eval cannot read. Show it clipped rather than
                # dropping it on the floor -- this key is skipped below.
                parsed = None
            if isinstance(parsed, dict):
                image_meta = parsed
            else:
                root = QTreeWidgetItem([self.tr("XISF Image Metadata")])
                self._tree.addTopLevelItem(root)
                root.addChild(QTreeWidgetItem([self.tr("raw"), self._short_value(image_meta)]))
                return True

        if isinstance(image_meta, (list, tuple)) and image_meta:
            image_meta = image_meta[0]
        if not isinstance(image_meta, dict):
            return False

        root = QTreeWidgetItem([self.tr("XISF Image Metadata")])
        self._tree.addTopLevelItem(root)

        fk = image_meta.get("FITSKeywords")
        if isinstance(fk, dict) and fk:
            fits_item = QTreeWidgetItem(["FITSKeywords"])
            root.addChild(fits_item)
            for kw, entries in fk.items():
                if not isinstance(entries, (list, tuple)):
                    entries = [entries]
                for ent in entries:
                    if isinstance(ent, dict):
                        val = ent.get("value", "")
                        com = ent.get("comment", "")
                        # COMMENT/HISTORY carry their text in the comment field
                        text = val if (val not in (None, "")) else com
                    else:
                        text = ent
                    fits_item.addChild(QTreeWidgetItem([str(kw), self._short_value(text)]))

        props = image_meta.get("XISFProperties")
        if isinstance(props, dict) and props:
            props_item = QTreeWidgetItem(["XISFProperties"])
            root.addChild(props_item)
            for prop_name in sorted(props.keys()):
                prop = props[prop_name]
                if isinstance(prop, dict):
                    # Lazy/attachment-backed grids: describe, never stringify.
                    if prop.get("_lazy") or prop.get("location") or prop.get("value") is None:
                        r = prop.get("rows") or prop.get("Rows")
                        c = prop.get("columns") or prop.get("Columns")
                        if r is not None and c is not None:
                            shown = f"<{prop.get('type', 'array')} {r}x{c}, not loaded>"
                        else:
                            shown = f"<{prop.get('type', 'property')}, not loaded>"
                    else:
                        shown = self._short_value(prop.get("value"))
                else:
                    shown = self._short_value(prop)
                props_item.addChild(QTreeWidgetItem([str(prop_name), shown]))

        other = QTreeWidgetItem([self.tr("Image")])
        added = False
        for k, v in image_meta.items():
            if k in ("FITSKeywords", "XISFProperties"):
                continue
            other.addChild(QTreeWidgetItem([str(k), self._short_value(v)]))
            added = True
        if added:
            root.addChild(other)

        return True

    def _populate_from_xisf(self, path: str):
        x = XISF(path)
        file_meta: Dict[str, Any] = x.get_file_metadata()
        img_meta_list = x.get_images_metadata()
        img_meta: Dict[str, Any] = img_meta_list[0] if img_meta_list else {}

        # File-level metadata
        froot = QTreeWidgetItem([self.tr("XISF File Metadata")])
        self._tree.addTopLevelItem(froot)
        for k, v in file_meta.items():
            vstr = v.get("value", "") if isinstance(v, dict) else v
            froot.addChild(QTreeWidgetItem([str(k), str(vstr)]))

        # Image-level metadata
        iroot = QTreeWidgetItem([self.tr("XISF Image Metadata")])
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
    def _metadata_rows(self):
        """Flatten the tree into [(dotted_key, value), ...] — shared by copy/save."""
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
        return rows

    def _copy_metadata(self):
        if not self._doc:
            return
        rows = self._metadata_rows()
        if not rows:
            return
        # Tab-separated: pastes as two columns into a spreadsheet and stays
        # readable as plain text. Header row matches the CSV export.
        text = "Key\tValue\n" + "\n".join(f"{k}\t{v}" for k, v in rows)
        try:
            QGuiApplication.clipboard().setText(text)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Copy Metadata"),
                                 self.tr("Failed to copy:\n{0}").format(e))
            return
        # brief confirmation on the button itself
        try:
            self._copy_btn.setText(self.tr("Copied!"))
            QTimer.singleShot(1200, lambda: self._copy_btn.setText(self.tr("Copy Metadata")))
        except Exception:
            pass

    def _save_metadata(self):
        if not self._doc:
            return
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save Metadata"), "", self.tr("CSV (*.csv)"))
        if not path:
            return

        rows = self._metadata_rows()

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Key", "Value"])
                w.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Save Metadata"), self.tr("Failed to save:\n{0}").format(e))