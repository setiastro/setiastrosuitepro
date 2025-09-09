# pro/copyastro.py
# pro/m_header.py
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QMessageBox, QCheckBox, QMdiSubWindow
)

class CopyAstrometryDialog(QDialog):
    """
    Modeless picker that copies the WCS/SIP solution from a source doc
    into the target doc (explicitly passed active view).
    """
    def __init__(self, parent=None, target=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Astrometric Solution")
        self.setMinimumWidth(420)

        self._mw = parent
        self._dm = getattr(parent, "doc_manager", None) or getattr(parent, "docman", None)

        # --- resolve target doc from the passed-in active subwindow/view/doc
        self._tgt = self._doc_from_target(target)
        if self._tgt is None:
            # fallback to active doc helpers, just in case
            try:
                self._tgt = self._dm.get_active_document() if self._dm else None
            except Exception:
                self._tgt = None
            if self._tgt is None and hasattr(parent, "_active_doc"):
                try:
                    self._tgt = parent._active_doc()
                except Exception:
                    pass

        lay = QVBoxLayout(self)

        tgt_name = getattr(self._tgt, "display_name", lambda: None)() or "Active View"
        lay.addWidget(QLabel(f"Target: <b>{tgt_name}</b>"))

        lay.addWidget(QLabel("Choose a source image that already has a WCS/SIP solution:"))
        self.combo = QComboBox(self)
        lay.addWidget(self.combo)

        self.chk_ignore_sip = QCheckBox("Ignore SIP terms (copy TAN only)")
        self.chk_ignore_sip.setChecked(False)
        lay.addWidget(self.chk_ignore_sip)

        row = QHBoxLayout(); row.addStretch(1)
        self.btn_copy = QPushButton("Copy")
        self.btn_close = QPushButton("Close")
        row.addWidget(self.btn_copy); row.addWidget(self.btn_close)
        lay.addLayout(row)

        self.btn_copy.clicked.connect(self._do_copy)
        self.btn_close.clicked.connect(self.close)

        self._candidates = []  # list[(doc, name, wcs_dict)]
        self._load_sources()

    # --- helpers --------------------------------------------------------
    def _doc_from_target(self, target):
        """Accept QMdiSubWindow, ImageSubWindow, or ImageDocument."""
        try:
            if target is None:
                return None
            # QMdiSubWindow → widget() → .document
            if isinstance(target, QMdiSubWindow):
                w = target.widget()
                return getattr(w, "document", None)
            # ImageSubWindow-like
            if hasattr(target, "document"):
                return getattr(target, "document", None)
            # Already a document
            if hasattr(target, "image") and hasattr(target, "metadata"):
                return target
        except Exception:
            pass
        return None

    def _extract_wcs_dict_for(self, doc):
        # Prefer the MW helper you already have (returns a flat dict of FITS cards)
        if hasattr(self._mw, "_extract_wcs_dict"):
            try:
                d = self._mw._extract_wcs_dict(doc)
                if d: return dict(d)
            except Exception:
                pass

        # Fallback: read from original_header
        meta = getattr(doc, "metadata", {}) or {}
        hdr  = meta.get("original_header") or {}
        try:
            keys = [str(k).upper() for k in getattr(hdr, "keys", lambda: hdr.keys())()]
            if "CRVAL1" in keys and "CRVAL2" in keys:
                return {k: hdr[k] for k in getattr(hdr, "keys", lambda: hdr.keys())()}
        except Exception:
            pass
        return {}

    def _load_sources(self):
        self.combo.clear()
        self._candidates.clear()

        if not self._dm or not self._tgt:
            self.combo.addItem("No target image.")
            self.btn_copy.setEnabled(False)
            return

        try:
            docs = self._dm.all_documents()
        except Exception:
            docs = []

        found_any = False
        for d in docs:
            if d is self._tgt:
                continue
            w = self._extract_wcs_dict_for(d)
            if not w:
                continue
            name = getattr(d, "display_name", lambda: None)() or (getattr(d, "metadata", {}).get("file_path") or "Untitled")
            # hint text (RA/Dec)
            ra, dec = w.get("CRVAL1"), w.get("CRVAL2")
            hint = f"  (RA={ra:.5f}, Dec={dec:.5f})" if isinstance(ra, (int, float)) and isinstance(dec, (int, float)) else ""
            self.combo.addItem(name + hint)
            self._candidates.append((d, name, w))
            found_any = True

        if not found_any:
            self.combo.addItem("No other images with WCS found")
            self.btn_copy.setEnabled(False)

    # --- action ---------------------------------------------------------
    def _do_copy(self):
        if self._tgt is None:
            QMessageBox.information(self, "Copy Astrometry", "No target image.")
            return

        idx = self.combo.currentIndex()
        if idx < 0 or idx >= len(self._candidates):
            return

        _, src_name, wcs = self._candidates[idx]

        # Optionally strip SIP → TAN only
        if self.chk_ignore_sip.isChecked():
            wcs = {
                k: v for k, v in wcs.items()
                if not str(k).upper().startswith(("A_", "B_", "AP_", "BP_"))
                and str(k).upper() not in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}
            }
            # enforce TAN
            c1 = str(wcs.get("CTYPE1", "RA---TAN"))
            c2 = str(wcs.get("CTYPE2", "DEC--TAN"))
            if c1.endswith("-SIP"): wcs["CTYPE1"] = "RA---TAN"
            if c2.endswith("-SIP"): wcs["CTYPE2"] = "DEC--TAN"

        ok = False
        if hasattr(self._mw, "_apply_wcs_dict_to_doc"):
            try:
                ok = bool(self._mw._apply_wcs_dict_to_doc(self._tgt, dict(wcs)))
            except Exception:
                ok = False

        if not ok:
            QMessageBox.warning(self, "Copy Astrometry", "Failed to apply astrometric solution.")
            return

        # refresh header dock + listeners immediately
        try:
            if hasattr(self._mw, "_refresh_header_viewer"):
                self._mw._refresh_header_viewer(self._tgt)
            if hasattr(self._mw, "currentDocumentChanged"):
                self._mw.currentDocumentChanged.emit(self._tgt)
        except Exception:
            pass

        try:
            tgt_name = getattr(self._tgt, "display_name", lambda: None)() or "Target"
            QMessageBox.information(self, "Copy Astrometry",
                                    f"Copied solution from “{src_name}” to “{tgt_name}”.")
        except Exception:
            pass

        self.close()
