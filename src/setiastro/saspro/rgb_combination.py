#saspro/rgb_combination.py
from __future__ import annotations
import os
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QGroupBox, QFileDialog, QComboBox, QButtonGroup, QMessageBox, QFormLayout
)
from PyQt6.QtCore import Qt

try:
    from setiastro.saspro.legacy.image_manager import load_image
except Exception:
    load_image = None

from setiastro.saspro.help_support import make_help_button

def _to_f01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32)
    elif a.dtype.kind == "f" and a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.size:
        m = float(np.nanmax(a))
        if m > 1.0 and np.isfinite(m):
            a = a / m
    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def _mono_from_any(arr: np.ndarray) -> np.ndarray:
    a = _to_f01(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] >= 1:
        return a[:, :, 0]
    raise ValueError("Expected mono (HxW) or RGB (HxWx3) image.")


class RGBCombinationDialogPro(QDialog):
    """
    Combine R/G/B from open views or files and ALWAYS create a new document.
    """
    def __init__(self, parent, list_open_docs_fn=None, doc_manager=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("RGB Combination"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._list_open_docs = list_open_docs_fn or (lambda: [])
        self._docman = doc_manager

        # file-mode state
        self.r_path = None
        self.g_path = None
        self.b_path = None

        # ── mode choose
        self.load_files_radio = QRadioButton(self.tr("Load Individual Files"))
        self.use_views_radio  = QRadioButton(self.tr("Use Open Views"))
        self.use_views_radio.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.load_files_radio)
        self.mode_group.addButton(self.use_views_radio)

        mode_box = QGroupBox(self.tr("Select RGB Combination Mode"))
        ml = QVBoxLayout(mode_box)
        ml.addWidget(self.use_views_radio)
        ml.addWidget(self.load_files_radio)

        # ── file mode widgets
        self.r_label = QLabel(self.tr("Red: Not selected"))
        self.g_label = QLabel(self.tr("Green: Not selected"))
        self.b_label = QLabel(self.tr("Blue: Not selected"))
        self.btn_load_r = QPushButton(self.tr("Load Red…"));   self.btn_load_r.clicked.connect(lambda: self._pick_file("R"))
        self.btn_load_g = QPushButton(self.tr("Load Green…")); self.btn_load_g.clicked.connect(lambda: self._pick_file("G"))
        self.btn_load_b = QPushButton(self.tr("Load Blue…"));  self.btn_load_b.clicked.connect(lambda: self._pick_file("B"))

        file_box = QGroupBox(self.tr("Files"))
        fl = QVBoxLayout(file_box)
        for lab, btn in [(self.r_label, self.btn_load_r),
                         (self.g_label, self.btn_load_g),
                         (self.b_label, self.btn_load_b)]:
            row = QHBoxLayout(); row.addWidget(lab); row.addStretch(); row.addWidget(btn)
            fl.addLayout(row)

        # ── open-views widgets
        views_box = QGroupBox(self.tr("Select Open Views for R / G / B"))
        vf = QFormLayout(views_box)

        self.cmb_r = QComboBox()
        self.cmb_g = QComboBox()
        self.cmb_b = QComboBox()

        # optional: make long titles usable without widening the dialog
        for cmb in (self.cmb_r, self.cmb_g, self.cmb_b):
            cmb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(24)  # tweak if you want

        vf.addRow(self.tr("Red:"), self.cmb_r)
        vf.addRow(self.tr("Green:"), self.cmb_g)
        vf.addRow(self.tr("Blue:"), self.cmb_b)


        # ── buttons
        self.btn_combine = QPushButton(self.tr("Combine"))
        self.btn_cancel  = QPushButton(self.tr("Cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_combine.clicked.connect(self._combine)

        btns = QHBoxLayout()
        self.btn_help = make_help_button("rgb_combination", self)
        btns.addWidget(self.btn_help)
        btns.addStretch(); btns.addWidget(self.btn_combine); btns.addWidget(self.btn_cancel)

        # ── layout
        L = QVBoxLayout(self)
        L.addWidget(mode_box)
        L.addWidget(views_box)
        L.addWidget(file_box)
        L.addLayout(btns)

        self.mode_group.buttonClicked.connect(self._refresh_mode)
        self._populate_views()
        self._refresh_mode()

    # --- helpers

    # ─── Filter → channel map ──────────────────────────────────────────
    # Used by _filter_bonus below to score the FITS FILTER keyword, which is
    # the DEFINITIVE signal for which channel a stack belongs to — unlike
    # filename patterns, a FILTER card can't masquerade as something else.
    #
    # Weights: direct broadband R/G/B get 100, Sloan variants 90, narrowband
    # aliases 50-60. Lower narrowband weight means "if both an R and an Ha
    # file exist, prefer R for the red role" — the Ha file still beats a
    # file with no FILTER card, but a true R beats it.
    #
    # OIII is intentionally NOT mapped: users split it G or B depending on
    # palette (HOO, SHO), so we leave it to the filename regex / user choice.
    @staticmethod
    def _map_filter(f: str):
        """
        Return (channel, weight) for a FITS FILTER keyword value, or None if
        it maps to nothing we can act on. Case/space/underscore-insensitive.
        """
        if not f:
            return None
        s = str(f).strip().upper().replace(" ", "").replace("_", "").replace("-", "")
        EXACT = {
            "R":   ("R", 100), "RED":   ("R", 100),
            "G":   ("G", 100), "GREEN": ("G", 100),
            "B":   ("B", 100), "BLUE":  ("B", 100),
            # Sloan short and prime notations
            "SR":  ("R", 90),  "SG": ("G", 90),  "SB": ("B", 90),
            "R'":  ("R", 90),  "G'": ("G", 90),  "B'": ("B", 90),
            # Common narrowband mappings (weaker than broadband so true R wins)
            "HA":  ("R", 60),  "HALPHA": ("R", 60),
            "SII": ("R", 55),  "S2":     ("R", 55),
            # L / Lum / Luminance intentionally return None — not an RGB channel
        }
        if s in EXACT:
            return EXACT[s]
        # Filter-wheel entries like "Ha_7nm", "Halpha6", "S2narrow"
        if s.startswith(("HALPHA", "HA")):
            return ("R", 55)
        if s.startswith(("SII", "S2")):
            return ("R", 50)
        return None

    @staticmethod
    def _filter_from_doc(doc) -> str:
        """
        Pull the FITS FILTER keyword value off a doc's metadata. Returns the
        raw string (not uppercased — _map_filter normalises). Checks the
        common storage patterns SASpro uses: a flat metadata key, or a
        nested astropy Header under one of a few names.
        """
        if doc is None:
            return ""
        try:
            meta = getattr(doc, "metadata", None) or {}
        except Exception:
            return ""
        # Direct flat key
        for key in ("FILTER", "filter", "Filter"):
            v = meta.get(key)
            if v:
                return str(v)
        # Nested FITS/WCS header
        for hkey in ("original_header", "fits_header", "header", "wcs_header"):
            h = meta.get(hkey)
            if h is None:
                continue
            try:
                v = h.get("FILTER") if hasattr(h, "get") else h["FILTER"]
                if v:
                    return str(v)
            except Exception:
                continue
        return ""

    def _guess_channel(self, titles: list[str], docs: list, channel: str) -> int:
        """
        Score each (title, doc) pair for how likely it is to be the given
        channel (R/G/B), and return the index of the best match. Falls back
        to 0 only if nothing scored positive.

        Scoring is two-layer:
          1. FITS FILTER keyword — definitive when present. A card reading
             "G" gets +100 for green and -50 for red/blue.
          2. Filename regex — fallback for files whose FILTER card is
             missing or non-standard.

        The filename regex used to have a bug: `[_\\-\\s\\.]g[_\\-\\s\\.\\d]`
        allowed `\\d` as the trailing boundary, which let the SASpro gain
        prefix (`..._G121_...` on every master-light file) false-match as a
        green channel. Every file in a field tied at the same score, and the
        first one in the list won — which is why Green rarely auto-assigned
        correctly while R and B almost always did (there's no `_R<n>_` or
        `_B<n>_` convention to collide with). Removing `\\d` fixes it: only
        the real `_G_` separator matches now.
        """
        import re
        channel = channel.upper()

        def filter_bonus(doc) -> int:
            entry = self._map_filter(self._filter_from_doc(doc))
            if entry is None:
                return 0
            mapped, weight = entry
            if mapped == channel:
                return int(weight)
            # Mapped to a DIFFERENT channel — penalise so it doesn't win
            # the role it isn't right for. Half-weight: a FILTER=Ha file
            # still scores -30 for green, which the real _G_ filename
            # pattern (+8) will overcome, so the right file still wins.
            return -int(weight) // 2

        def filename_score(title: str) -> int:
            t = title.lower()
            s = 0

            if channel == "R":
                if re.search(r'\bred\b', t):               s += 10
                if re.search(r'[_\-\s\.]r[_\-\s\.]', t):   s += 8   # was [...\d] — see docstring
                if re.search(r'[_\-\s\.]ha[_\-\s\.]', t):  s += 6
                if re.search(r'[_\-\s\.]sii[_\-\s\.]', t): s += 5
                if t.endswith('_r') or t.endswith('-r'):   s += 7
                if re.search(r'[_\-\s\.]r$', t):           s += 7
                if re.search(r'\bgreen\b|\bblue\b', t):    s -= 10
                if re.search(r'[_\-\s\.]g[_\-\s\.]|[_\-\s\.]b[_\-\s\.]', t): s -= 5

            elif channel == "G":
                if re.search(r'\bgreen\b', t):             s += 10
                if re.search(r'[_\-\s\.]g[_\-\s\.]', t):   s += 8   # dropped \d — see docstring
                if t.endswith('_g') or t.endswith('-g'):   s += 7
                if re.search(r'[_\-\s\.]g$', t):           s += 7
                if re.search(r'[_\-\s\.]oiii[_\-\s\.]', t): s += 5
                if re.search(r'\bred\b|\bblue\b', t):      s -= 10
                if re.search(r'[_\-\s\.]r[_\-\s\.]|[_\-\s\.]b[_\-\s\.]', t): s -= 5

            elif channel == "B":
                if re.search(r'\bblue\b', t):              s += 10
                if re.search(r'[_\-\s\.]b[_\-\s\.]', t):   s += 8   # dropped \d — see docstring
                if t.endswith('_b') or t.endswith('-b'):   s += 7
                if re.search(r'[_\-\s\.]b$', t):           s += 7
                if re.search(r'[_\-\s\.]oiii[_\-\s\.]', t): s += 4
                if re.search(r'\bred\b|\bgreen\b', t):     s -= 10
                if re.search(r'[_\-\s\.]r[_\-\s\.]|[_\-\s\.]g[_\-\s\.]', t): s -= 5

            return s

        # If docs weren't supplied (older callers), fall back to filename-only
        # with the regex fix still in place.
        if not docs or len(docs) != len(titles):
            scores = [filename_score(t) for t in titles]
        else:
            scores = [filename_score(t) + filter_bonus(d)
                      for t, d in zip(titles, docs)]

        best_score = max(scores)
        if best_score <= 0:
            return 0
        return scores.index(best_score)

    def _labeled(self, text, w):
        box = QVBoxLayout()
        box.addWidget(QLabel(text))
        box.addWidget(w)
        return box

    def _populate_views(self):
        self.cmb_r.clear(); self.cmb_g.clear(); self.cmb_b.clear()
        titles = []
        docs = []
        for title, doc in self._list_open_docs():
            self.cmb_r.addItem(title, doc)
            self.cmb_g.addItem(title, doc)
            self.cmb_b.addItem(title, doc)
            titles.append(title)
            docs.append(doc)

        if len(titles) >= 3:
            # Pass docs through so _guess_channel can read the FITS FILTER
            # keyword — the definitive signal when it's present. Filename
            # patterns are the fallback.
            self.cmb_r.setCurrentIndex(self._guess_channel(titles, docs, "R"))
            self.cmb_g.setCurrentIndex(self._guess_channel(titles, docs, "G"))
            self.cmb_b.setCurrentIndex(self._guess_channel(titles, docs, "B"))

    def _refresh_mode(self):
        use_views = self.use_views_radio.isChecked()
        for w in (self.cmb_r, self.cmb_g, self.cmb_b):
            w.setEnabled(use_views)
        for w in (self.r_label, self.g_label, self.b_label, self.btn_load_r, self.btn_load_g, self.btn_load_b):
            w.setEnabled(not use_views)
        if not load_image:
            self.load_files_radio.setEnabled(False)
            self.use_views_radio.setChecked(True)

    def _pick_file(self, which):
        fp, _ = QFileDialog.getOpenFileName(
            self, f"Select {'Red' if which=='R' else 'Green' if which=='G' else 'Blue'} Image", "",
            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg);;All Files (*)"
        )
        if not fp: return
        base = os.path.basename(fp)
        if which == "R":
            self.r_path = fp; self.r_label.setText(f"Red: {base}")
        elif which == "G":
            self.g_path = fp; self.g_label.setText(f"Green: {base}")
        else:
            self.b_path = fp; self.b_label.setText(f"Blue: {base}")

    def _load_file_mono(self, path: str) -> np.ndarray:
        if load_image is None:
            raise RuntimeError("File loading not available.")
        arr, _, _, _ = load_image(path)
        if arr is None:
            raise RuntimeError(f"Failed to load: {path}")
        return _mono_from_any(arr)

    # --- combine → ALWAYS new document
    def _combine(self):
        try:
            if self.use_views_radio.isChecked():
                rdoc = self.cmb_r.currentData()
                gdoc = self.cmb_g.currentData()
                bdoc = self.cmb_b.currentData()
                if not (rdoc and gdoc and bdoc):
                    QMessageBox.information(self, "RGB Combination", "Please select three views.")
                    return
                r = _mono_from_any(getattr(rdoc, "image", None))
                g = _mono_from_any(getattr(gdoc, "image", None))
                b = _mono_from_any(getattr(bdoc, "image", None))
                title = "RGB_Combined"
                wcs_donor_docs = [gdoc, rdoc, bdoc]  # G → R → B priority
            else:
                if not (self.r_path and self.g_path and self.b_path):
                    QMessageBox.information(self, "RGB Combination", "Please choose three files.")
                    return
                r = self._load_file_mono(self.r_path)
                g = self._load_file_mono(self.g_path)
                b = self._load_file_mono(self.b_path)
                title = f"RGB_{os.path.splitext(os.path.basename(self.r_path))[0]}"
                wcs_donor_docs = []  # file mode: no open docs to pull WCS from

            if r.shape != g.shape or r.shape != b.shape:
                raise ValueError("All three images must have identical dimensions.")

            rgb = np.stack([r, g, b], axis=2).astype(np.float32, copy=False)
            rgb = np.clip(rgb, 0.0, 1.0)

            if not self._docman:
                raise RuntimeError("No document manager available.")

            if hasattr(self._docman, "open_array"):
                newdoc = self._docman.open_array(rgb, metadata={"step_name": "RGB Combination"}, title=title)
            elif hasattr(self._docman, "open_numpy"):
                newdoc = self._docman.open_numpy(rgb, metadata={"step_name": "RGB Combination"}, title=title)
            else:
                newdoc = self._docman.create_document(image=rgb, metadata={"step_name": "RGB Combination"}, name=title)

            # Inherit WCS from G → R → B, whichever has one first
            mw = self.parent()
            if wcs_donor_docs and hasattr(mw, "_extract_wcs_dict") and hasattr(mw, "_apply_wcs_dict_to_doc"):
                for donor in wcs_donor_docs:
                    try:
                        wcs = mw._extract_wcs_dict(donor)
                        if wcs:
                            mw._apply_wcs_dict_to_doc(newdoc, dict(wcs))
                            break
                    except Exception:
                        continue

            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(newdoc)

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "RGB Combination", f"Failed to combine:\n{e}")