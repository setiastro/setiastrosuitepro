# pro/gui/mixins/header_mixin.py
"""
Header management mixin for AstroSuiteProMainWindow.

This mixin contains all functionality for viewing and managing FITS headers,
metadata, and WCS information.
"""
from __future__ import annotations
import re
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    pass


class HeaderMixin:
    """
    Mixin for header/metadata management.
    
    Provides methods for viewing, extracting, and manipulating document headers
    and WCS (World Coordinate System) information.
    """

    # Exact keys we always consider WCS
    _WCS_KEY_SET = {
        "WCSAXES", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
        "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "CDELT1", "CDELT2",
        "LONPOLE", "LATPOLE",
        "RADESYS", "RADECSYS", "EQUINOX", "EPOCH",
        "NAXIS1", "NAXIS2"  # useful context for UIs/solvers
    }

    def _ensure_header_map(self, doc):
        """Ensure doc has a header dictionary in metadata, return it."""
        meta = getattr(doc, "metadata", None)
        if meta is None:
            return None
        hdr = meta.get("original_header")
        if not isinstance(hdr, dict):
            hdr = {}
            meta["original_header"] = hdr
        return hdr

    def _coerce_wcs_numbers(self, d: dict) -> dict:
        """Convert common WCS/SIP values to int/float where appropriate."""
        numeric = {
            "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
            "CROTA1", "CROTA2", "EQUINOX", "WCSAXES", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
            "LONPOLE", "LATPOLE"
        }
        out = {}
        for k, v in d.items():
            K = str(k).upper()
            try:
                if K in numeric or re.match(r"^(A|B|AP|BP)_\d+_\d+$", K or ""):
                    if isinstance(v, str):
                        s = v.strip()
                        # int if clean integer, else float
                        out[K] = int(s) if re.fullmatch(r"[+-]?\d+", s) else float(s)
                    else:
                        out[K] = v
                else:
                    out[K] = v
            except Exception:
                out[K] = v
        return out

    def _extract_wcs_dict(self, doc) -> dict:
        """Collect a complete WCS/SIP dict from the doc's header/meta."""
        if doc is None:
            return {}
        src = (getattr(doc, "metadata", {}) or {}).get("original_header")

        wcs = {}
        if src is None:
            pass
        else:
            try:
                for k, v in dict(src).items():
                    K = str(k).upper()
                    if (K.startswith(("CRPIX", "CRVAL", "CDELT", "CD", "PC", "CROTA", "CTYPE", "CUNIT",
                                      "WCSAXES", "LONPOLE", "LATPOLE", "EQUINOX", "PV")) or
                        K in {"RADECSYS", "RADESYS", "NAXIS1", "NAXIS2"} or
                        K.startswith(("A_", "B_", "AP_", "BP_"))):
                        wcs[K] = v
            except Exception:
                pass

        # Also accept any mirror you previously stored
        meta = getattr(doc, "metadata", {}) or {}
        imgmeta = meta.get("image_meta") or meta.get("WCS") or {}
        if isinstance(imgmeta, dict):
            sub = imgmeta.get("WCS", imgmeta)
            if isinstance(sub, dict):
                for k, v in sub.items():
                    K = str(k).upper()
                    if (K.startswith(("CRPIX", "CRVAL", "CDELT", "CD", "PC", "CROTA", "CTYPE", "CUNIT",
                                      "WCSAXES", "LONPOLE", "LATPOLE", "EQUINOX", "PV")) or
                        K in {"RADECSYS", "RADESYS", "NAXIS1", "NAXIS2"} or
                        K.startswith(("A_", "B_", "AP_", "BP_"))):
                        wcs.setdefault(K, v)

        # sensible defaults/parity
        if any(k.startswith(("A_", "B_", "AP_", "BP_")) for k in wcs):
            wcs.setdefault("CUNIT1", "deg")
            wcs.setdefault("CUNIT2", "deg")
            # TAN-SIP labels if SIP present
            c1 = str(wcs.get("CTYPE1", "RA---TAN"))
            c2 = str(wcs.get("CTYPE2", "DEC--TAN"))
            if not c1.endswith("-SIP"):
                wcs["CTYPE1"] = "RA---TAN-SIP"
            if not c2.endswith("-SIP"):
                wcs["CTYPE2"] = "DEC--TAN-SIP"

        if "RADECSYS" in wcs and "RADESYS" not in wcs:
            wcs["RADESYS"] = wcs["RADECSYS"]
        if "WCSAXES" not in wcs and {"CTYPE1", "CTYPE2"} <= wcs.keys():
            wcs["WCSAXES"] = 2

        return self._coerce_wcs_numbers(wcs)

    def _ensure_header_for_doc(self, doc):
        """Return an astropy Header for doc.metadata['original_header'] (creating one if needed)."""
        from astropy.io.fits import Header
        import numpy as np
        
        meta = getattr(doc, "metadata", None)
        if not isinstance(meta, dict):
            setattr(doc, "metadata", {})
            meta = doc.metadata

        hdr_like = meta.get("original_header")
        
        # Already a Header?
        if isinstance(hdr_like, Header):
            hdr = hdr_like
        elif isinstance(hdr_like, dict):
            # coerce dict -> Header
            hdr = Header()
            int_keys = {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER", "WCSAXES", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"}
            for k, v in dict(hdr_like).items():
                K = str(k).upper()
                try:
                    if K in int_keys:
                        hdr[K] = int(float(str(v).strip().split()[0]))
                    elif re.match(r"^(?:A|B|AP|BP)_\d+_\d+$", K) or \
                         re.match(r"^(?:CRPIX|CRVAL|CDELT|CD|PC|CROTA|LATPOLE|LONPOLE|EQUINOX)\d?_?\d*$", K):
                        hdr[K] = float(str(v).strip().split()[0])
                    else:
                        hdr[K] = v
                except Exception:
                    pass
        else:
            hdr = Header()

        # Ensure basic axis cards exist (needed for non-FITS sources)
        try:
            img = getattr(doc, "image", None)
            if img is not None:
                a = np.asarray(img)
                H = int(a.shape[0]) if a.ndim >= 2 else 1
                W = int(a.shape[1]) if a.ndim >= 2 else 1
                C = int(a.shape[2]) if a.ndim == 3 else 1
                # Only set when missing (don't clobber real FITS headers)
                if "NAXIS" not in hdr:
                    hdr["NAXIS"] = 2 if a.ndim != 3 else 3
                if "NAXIS1" not in hdr:
                    hdr["NAXIS1"] = W
                if "NAXIS2" not in hdr:
                    hdr["NAXIS2"] = H
                if a.ndim == 3 and "NAXIS3" not in hdr:
                    hdr["NAXIS3"] = C
        except Exception:
            pass

        meta["original_header"] = hdr
        return hdr

    def _refresh_header_viewer(self, doc=None):
        """Rebuild the header dock from the given (or active) doc -- never raises."""
        try:
            doc = doc or self._active_doc()
            hv = getattr(self, "header_viewer", None)

            # If your dock widget has a native API, try it but don't trust it.
            if hv and hasattr(hv, "set_document"):
                try:
                    hv.set_document(doc)
                    return
                except Exception as e:
                    print("[header] set_document suppressed:", e)

            # Fallback path: extract -> populate, all guarded.
            rows = self._extract_header_pairs(doc)
            if not rows:
                self._clear_header_viewer(self.tr("No header") if doc else self.tr("No image"))
            else:
                self._populate_header_viewer(rows)
        except Exception as e:
            print("[header] refresh suppressed:", e)
            self._clear_header_viewer("")

    def _extract_header_pairs(self, doc):
        """
        Return list[(key, value, comment)].
        Prefers a JSON-safe snapshot if present, otherwise best-effort parsing.
        Never raises.
        """
        try:
            if not doc:
                return []

            meta = getattr(doc, "metadata", {}) or {}

            # 1) Prefer a snapshot if any writer/loader provided it.
            snap = meta.get("__header_snapshot__")
            if isinstance(snap, dict):
                fmt = snap.get("format")
                if fmt == "fits-cards":
                    cards = snap.get("cards") or []
                    out = []
                    for it in cards:
                        try:
                            k, v, c = it
                        except Exception:
                            # tolerate weird shapes
                            k, v, c = (str(it[0]) if it else "",
                                       "" if len(it) < 2 else str(it[1]),
                                       "" if len(it) < 3 else str(it[2]))
                        out.append((str(k), str(v), str(c)))
                    return out
                if fmt == "dict":
                    items = snap.get("items") or {}
                    out = []
                    for k, v in items.items():
                        if isinstance(v, dict):
                            out.append((str(k), str(v.get("value", "")), str(v.get("comment", ""))))
                        else:
                            out.append((str(k), str(v), ""))
                    return out
                if fmt == "repr":
                    return [(self.tr("Header"), str(snap.get("text", "")), "")]

            # 2) Live header object(s) (can be astropy, dict, or random).
            hdr = (meta.get("original_header")
                   or meta.get("fits_header")
                   or meta.get("header"))

            if hdr is None:
                return []

            # astropy.io.fits.Header (optional; no hard dependency)
            try:
                from astropy.io.fits import Header
            except Exception:
                Header = None

            if Header is not None:
                try:
                    if isinstance(hdr, Header):
                        out = []
                        for k in hdr.keys():
                            try:
                                val = hdr[k]
                            except Exception:
                                val = ""
                            try:
                                cmt = hdr.comments[k]
                            except Exception:
                                cmt = ""
                            out.append((str(k), str(val), str(cmt)))
                        return out
                except Exception as e:
                    print("[header] astropy parse suppressed:", e)

            # dict-ish header (e.g., XISF-like)
            if isinstance(hdr, dict):
                out = []
                for k, v in hdr.items():
                    if isinstance(v, dict):
                        out.append((str(k), str(v.get("value", "")), str(v.get("comment", ""))))
                    else:
                        # avoid huge array dumps
                        try:
                            import numpy as _np
                            if isinstance(v, _np.ndarray):
                                v = f"ndarray{tuple(v.shape)}"
                        except Exception:
                            pass
                        out.append((str(k), str(v), ""))
                return out

            # Fallback: string repr
            return [(self.tr("Header"), str(hdr), "")]
        except Exception as e:
            print("[header] extract suppressed:", e)
            return []

    def _populate_header_viewer(self, rows):
        """Render rows into whatever widget you expose; never raises."""
        try:
            w = self._header_widget()
        except Exception as e:
            print("[header] _header_widget suppressed:", e)
            return
        if w is None:
            return

        # Table widget path
        try:
            from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
            if isinstance(w, QTableWidget):
                try:
                    w.setRowCount(0)
                    w.setColumnCount(3)
                    w.setHorizontalHeaderLabels([self.tr("Key"), self.tr("Value"), self.tr("Comment")])
                    for r, (k, v, c) in enumerate(rows):
                        w.insertRow(r)
                        w.setItem(r, 0, QTableWidgetItem(k))
                        w.setItem(r, 1, QTableWidgetItem(v))
                        w.setItem(r, 2, QTableWidgetItem(c))
                    return
                except Exception as e:
                    print("[header] table populate suppressed:", e)
        except Exception:
            pass

        # List widget path
        try:
            from PyQt6.QtWidgets import QListWidget
            if isinstance(w, QListWidget):
                try:
                    w.clear()
                    for k, v, c in rows:
                        w.addItem(f"{k} = {v}" + (f"  / {c}" if c else ""))
                    return
                except Exception as e:
                    print("[header] list populate suppressed:", e)
        except Exception:
            pass

        # Plain text-ish
        try:
            if hasattr(w, "setPlainText"):
                w.setPlainText("\n".join(
                    f"{k} = {v}" + (f"  / {c}" if c else "") for (k, v, c) in rows
                ))
                return
            if hasattr(w, "setText"):
                w.setText("\n".join(
                    f"{k} = {v}" + (f"  / {c}" if c else "") for (k, v, c) in rows
                ))
                return
        except Exception as e:
            print("[header] text populate suppressed:", e)

    def _clear_header_viewer(self, message: str = ""):
        """Clear header viewer content quietly--no dialogs."""
        w = self._header_widget()
        if w is None:
            return
        try:
            from PyQt6.QtWidgets import QTableWidget
            if isinstance(w, QTableWidget):
                w.setRowCount(0)
                w.setColumnCount(3)
                w.setHorizontalHeaderLabels([self.tr("Key"), self.tr("Value"), self.tr("Comment")])
                return
        except Exception:
            pass
        try:
            from PyQt6.QtWidgets import QListWidget
            if isinstance(w, QListWidget):
                w.clear()
                if message:
                    w.addItem(message)
                return
        except Exception:
            pass
        # QTextEdit-like
        if hasattr(w, "setPlainText"):
            try:
                w.setPlainText(message or "")
            except Exception:
                pass

    def _header_widget(self):
        """
        Find the concrete widget used to display header text/table.
        Never raises; returns None if nothing sensible is found.
        """
        hv = getattr(self, "header_viewer", None) or getattr(self, "metadata_dock", None)
        if hv is None:
            return None

        # If it's a dock widget (QDockWidget-like), pull its child widget
        try:
            if hasattr(hv, "widget") and callable(hv.widget):
                inner = hv.widget()
                if inner is not None:
                    return inner
        except Exception:
            pass

        # It might already be the actual widget
        return hv

    def _on_doc_added_for_header_sync(self, doc):
        """Update header when the *active* doc changes."""
        try:
            doc.changed.connect(self._maybe_refresh_header_on_doc_change)
        except Exception:
            pass

    def _on_doc_removed_for_header_sync(self, doc):
        """If the removed doc was the active one, clear header."""
        if doc is self._active_doc():
            self._clear_header_viewer(self.tr("No image"))
            hv = getattr(self, "header_viewer", None)
            if hv and hasattr(hv, "set_document"):
                try:
                    hv.set_document(None)
                except Exception:
                    pass

        # If there are no more subwindows, force a global clear too
        if not self.mdi.subWindowList():
            self.currentDocumentChanged.emit(None)
            self._hdr_refresh_timer.start(0)

    def _maybe_refresh_header_on_doc_change(self):
        """Refresh header if sender is the active doc."""
        sender = self.sender()
        if sender is self._active_doc():
            self._hdr_refresh_timer.start(0)
